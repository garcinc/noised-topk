import os
from tqdm import tqdm
import pickle
import argparse
import time
import numpy as np
import torch
from torch.optim import SGD

from dataloader.get_data import get_data
from utils import set_seed, load_model, load_optimizer, get_model, get_loss, update_optimizer, keep_best
from cli import add_all_parsers
from epoch import train_epoch, val_epoch, test_epoch


def train(args):
    set_seed(args, use_gpu=torch.cuda.is_available())
    train_loader, val_loader, test_loader, dataset_attributes = get_data(args)
    print('args : ', args.__dict__)

    model = get_model(args, n_classes=dataset_attributes['n_classes'])
    start_epoch = 0
    if args.load_path:
        args.load_path = args.load_path.strip()
   
    if args.load_path:
        print(f'loading model from {args.load_path}')
        start_epoch = load_model(model, args.load_path, args.use_gpu)
    print('** start_epoch : ', start_epoch)

    criteria = get_loss(args, n_classes=dataset_attributes['n_classes'], cls_num_list=dataset_attributes.get('cls_num_list'))

    if args.use_gpu:
        print('USING GPU')
        if args.parallel_gpu:
            print('USING MULTIPLE GPUS')
            model = torch.nn.DataParallel(model).cuda()
        else:
            print('USING SINGLE GPU')
            torch.cuda.set_device(0)
            model.cuda()
        criteria.cuda()
    else:
        print('USING CPU')

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.mu, nesterov=True)

    if args.load_path:
        print(f'loading optimizer from {args.load_path}')
        load_optimizer(optimizer, args.load_path, args.use_gpu)

    # Containers for storing statistics over epochs
    loss_train, train_accuracy, topk_train_accuracy = [], [], []
    loss_val, val_accuracy, topk_val_accuracy = [], [], []
    class_accuracy_over_epochs, class_accuracy_over_epochs_topk = [], []

    best_metric = np.float('-inf')
    save_dir_name = args.save_dir_name.strip()
    save_file_name = args.save_file_name.strip()
    early_stop = args.early_stop.strip()

    save_dir_root = os.path.join(os.getcwd(), 'results', save_dir_name)

    time_epochs = []
    for epoch in tqdm(range(start_epoch, args.n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_decay=args.lr_decay, epoch=epoch, decay_factor=args.decay_factor)

        loss_epoch_train, epoch_accuracy_train, epoch_top_k_accuracy_train, = train_epoch(model, optimizer, train_loader,
                                                                                          criteria, loss_train,
                                                                                          train_accuracy,
                                                                                          topk_train_accuracy, args.k,
                                                                                          dataset_attributes['n_train'],
                                                                                          args.use_gpu)

        loss_epoch_val, epoch_accuracy_val, epoch_top_k_accuracy_val = val_epoch(model, val_loader, criteria,
                                                                                 loss_val, val_accuracy,
                                                                                 topk_val_accuracy,
                                                                                 args.k, dataset_attributes, args.use_gpu,
                                                                                 class_accuracy_over_epochs,
                                                                                 class_accuracy_over_epochs_topk,
                                                                                )

        val_mean_top1_acc = float(np.mean(list(class_accuracy_over_epochs[-1].values())))
        val_mean_topk_acc = float(np.mean(list(class_accuracy_over_epochs_topk[-1].values())))

        best_metric = keep_best(early_stop, best_metric, model, optimizer, epoch, save_dir_root, save_file_name, locals_dict=locals())

        print()
        t_f = time.time()
        epoch_duration = t_f-t
        time_epochs.append(epoch_duration)
        print(f'epoch {epoch} took {epoch_duration:.2f}')
        print(f'loss_epoch_train : {loss_epoch_train}')
        print(f'loss_epoch_val : {loss_epoch_val}')
        print(f'train accuracy : {epoch_accuracy_train} / train top_{args.k} accuracy : {epoch_top_k_accuracy_train}')
        print(f'val accuracy : {epoch_accuracy_val} / val top_{args.k} accuracy : {epoch_top_k_accuracy_val}')
        print(f'macro-average top-1 accuracy : {val_mean_top1_acc} / macro-average top-{args.k} accuracy : {val_mean_topk_acc}')

    file_ext_mapping = {'topk': '_best_topk_acc.tar', 'top1': '_best_top1_acc.tar',
                        'mean_top1':  '_best_mean_top1_acc.tar', 'mean_topk': '_best_mean_topk_acc.tar'}
    file_ext = file_ext_mapping[early_stop]

    print('Loading best validation model')
    load_model(model, os.path.join(save_dir_root, 'checkpoints', save_file_name + file_ext), args.use_gpu)

    loss_test, accuracy_test, top_k_accuracy_test, test_class_acc = test_epoch(model, test_loader, criteria, dataset_attributes, args.k,
                                                                           args.use_gpu, dataset_attributes['n_test'])

    test_mean_top1_acc = float(np.mean(list(test_class_acc['class_accuracy'].values())))
    test_mean_topk_acc = float(np.mean(list(test_class_acc['class_accuracy_topk'].values())))
    print(f'loss_test : {loss_test}')
    print(f'test accuracy : {accuracy_test} / test top_{args.k} accuracy : {top_k_accuracy_test}')
    print(f"test macro-average top-1 accuracy : {test_mean_top1_acc} / test macro-average top-{args.k} accuracy : {test_mean_topk_acc}")
    print()


    results = {'loss_train': loss_train, 'train_accuracy': train_accuracy, 'topk_train_accuracy': topk_train_accuracy,
               'loss_val': loss_val, 'val_accuracy': val_accuracy, 'topk_val_accuracy': topk_val_accuracy,
               'test_results':  {'loss': loss_test,
                                 'accuracy': accuracy_test,
                                 'topk-accuracy': top_k_accuracy_test,
                                 'class_accuracy': test_class_acc},
               'time_epochs': time_epochs}

    results['val_class_accuracy'] = class_accuracy_over_epochs
    results['params'] = args.__dict__

    mapping_save_name = {'_best_topk_acc.tar': '_best_topk_acc.pkl', '_best_top1_acc.tar': '_best_top1_acc.pkl',
                         '_best_mean_top1_acc.tar': '_best_mean_top1_acc.pkl', '_best_mean_topk_acc.tar': '_best_mean_topk_acc.pkl'}

    abs_save_file = os.path.join(save_dir_root, save_file_name + mapping_save_name[file_ext])
    print(f'Saving results to {abs_save_file}')
    with open(abs_save_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parsers(parser)
    args = parser.parse_args()
    train(args)
