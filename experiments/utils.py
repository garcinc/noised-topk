import torch
import random
import numpy as np
import os
from torch.nn import Parameter
import torch.nn.functional as F

from torchvision.models import resnet18, resnet50
from models.densenet import DenseNet3
import torch.nn as nn
from topk.svm import SmoothTopkSVM
from torch.nn import CrossEntropyLoss

from pytopk import BalNoisedTopK, ImbalNoisedTopK

from losses.focal import FocalLoss

from losses.ldam import LDAMLoss


def set_seed(args, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def count_correct_top_k(scores, labels, k):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()


def update_correct_per_class(batch_output, batch_y, d):
    """
    :param batch_output: predictions of the network, of size (n_batch, n_classes)
    :param batch_y: true labels, 1d tensor of size n_batch
    :param d: dictionary containing ths mapping between the class_id and the count of already correctly classified examples for that class
    :return: the updated dictionary with the results of the current batch
    """
    predicted_class = torch.argmax(batch_output, dim=-1)
    for true_label, predicted_label in zip(batch_y, predicted_class):
        if true_label == predicted_label:
            d[true_label.item()] += 1
        else:
            d[true_label.item()] += 0


def update_correct_per_class_topk(batch_output, batch_y, d, k):
    topk_labels_pred = torch.argsort(batch_output, axis=-1, descending=True)[:, :k]
    for true_label, predicted_labels in zip(batch_y, topk_labels_pred):
        d[true_label.item()] += torch.sum(true_label == predicted_labels).item()


def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    return d['epoch']


def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    optimizer.load_state_dict(d['optimizer'])


def save(model, optimizer, epoch, location):
    dir = os.path.dirname(location)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d = {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict()}
    torch.save(d, location)


def decay_lr(optimizer, decay_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_decay, epoch, decay_factor):
    if epoch in lr_decay:
        optimizer = decay_lr(optimizer, decay_factor)
    return optimizer


def get_model(args, n_classes):
    resnets = {'resnet18': resnet18, 'resnet50': resnet50}

    if args.normalize_ll:
        print('NORMALIZING LOGITS AND LAST LAYER WEIGHTS')
        LastLinearLayer = NormedLinear
    else:
        LastLinearLayer = nn.Linear

    if 'resnet' in args.model:
        model = resnets[args.model](pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = LastLinearLayer(num_ftrs, n_classes)

    elif args.model == 'densenet':
        # no pretraining for DenseNet 40-40
        model = DenseNet3(depth=40, num_classes=n_classes, growth_rate=40)
    else:
        raise NotImplementedError

    return model


def get_loss(args, n_classes, **kwargs):
    if args.loss == 'smooth_topk':
        criteria = SmoothTopkSVM(n_classes=n_classes, k=args.k, tau=args.tau)
    elif args.loss == 'ce':
        criteria = CrossEntropyLoss()
    elif args.loss == 'focal':
        criteria = FocalLoss(gamma=args.gamma_foc)
    elif args.loss == 'ldam':
        criteria = LDAMLoss(cls_num_list=kwargs['cls_num_list'], max_m=args.max_m, scale=args.scale)
    elif args.loss == 'bal_topk':
        criteria = BalNoisedTopK(k=args.k, epsilon=args.epsilon, n_sample=args.n_sample)
    elif args.loss == 'imbal_topk':
        criteria = ImbalNoisedTopK(k=args.k, epsilon=args.epsilon, max_m=args.max_m,
                                   cls_num_list=kwargs['cls_num_list'], scale=args.scale, n_sample=args.n_sample)
    else:
        raise NotImplementedError

    return criteria


def keep_best(early_stop, best_metric_value, model, optimizer, epoch, save_dir_root, save_file_name, locals_dict):

    var_mapping = {'topk': ('epoch_top_k_accuracy_val', '_best_topk_acc.tar'),
                   'top1': ('epoch_accuracy_val', '_best_top1_acc.tar'),
                   'mean_top1': ('val_mean_top1_acc', '_best_mean_top1_acc.tar'),
                   'mean_topk': ('val_mean_topk_acc', '_best_mean_topk_acc.tar')}
    curr_metric_value = locals_dict[var_mapping[early_stop][0]]

    if curr_metric_value > best_metric_value:
        best_metric_value = curr_metric_value
        save(model, optimizer, epoch,
             os.path.join(save_dir_root, 'checkpoints', save_file_name + var_mapping[early_stop][1]))

    return best_metric_value

