import torch
import os


def add_all_parsers(parser):
    _add_loss_parser(parser)
    _add_training_parser(parser)
    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_hardware_parser(parser)
    _add_misc_parser(parser)


def _add_loss_parser(parser):
    group_loss = parser.add_argument_group('Loss parameters')
    group_loss.add_argument('--loss', choices=['smooth_topk', 'ce', 'ldam', 'balanced_noise_topk', 'imbalanced_noise_topk', 'focal'])
    group_loss.add_argument('--k', type=int, help='value of k for computing the topk loss and calculating topk accuracy')

    group_loss.add_argument('--n_sample', default=5, type=int, help='number of sampled noise vectors for the noised balanced and imbalanced losses')
    group_loss.add_argument('--epsilon', default=1.0, help='noise parameter if you choose the noise loss', type=float)
    group_loss.add_argument('--gamma_foc', default=1.0, help='gamma parameter for focal loss', type=float)
    group_loss.add_argument('--tau', default=1.0, help='smoothing parameter if you choose the smooth loss', type=float)
    group_loss.add_argument('--mu', type=float, default=0., help='weight decay parameter')
    group_loss.add_argument('--max_m', default=1.0, help='max margin for um topk loss', type=float)
    group_loss.add_argument('--exp', default=0.25, help='exponent for um topk loss', type=float)
    group_loss.add_argument('--scale', default=1.0, help='scale for um topk loss and ldam', type=float)


def _add_training_parser(parser):
    group_training = parser.add_argument_group('Training parameters')
    group_training.add_argument('--lr', type=float, help='learning rate to use')
    group_training.add_argument('--batch_size', type=int, default=256, help='default is 256')
    group_training.add_argument('--parallel_gpu', action='store_true')
    group_training.add_argument('--n_epochs', type=int)
    group_training.add_argument('--train_size', type=int, default=None)
    group_training.add_argument('--lr_decay', nargs='+', type=int, default=[])
    group_training.add_argument('--decay_factor', type=float, default=0.1)
    group_training.add_argument('--pretrained', action='store_true')
    group_training.add_argument('--early_stop', choices=['topk', 'top1', 'mean_top1', 'mean_topk'])
    group_training.add_argument('--normalize_ll', action='store_true', help='whether to normalize last layer for margin losses or not')


def _add_dataset_parser(parser):
    group_dataset = parser.add_argument_group('Dataset parameters')
    group_dataset.add_argument('--dataset',
                               choices=['cifar100', 'big_plantnet'],
                               help='choose the dataset you want to train on')
    group_dataset.add_argument('--noise_cifar', type=float, default=0.0, help='introduce noise in labels (experiment cifar100)')


def _add_model_parser(parser):
    group_model = parser.add_argument_group('Model parameters')
    group_model.add_argument('--model', choices=['resnet18', 'resnet50', 'densenet'],
                             help='choose one of the models used in our experiments')
    group_model.add_argument('--load_path', default=None, type=str, help='file path for restoring model and optimizer')


def _add_hardware_parser(parser):
    group_hardware = parser.add_argument_group('Hardware parameters')
    group_hardware.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available())


def _add_misc_parser(parser):
    group_misc = parser.add_argument_group('Miscellaneous parameters')
    group_misc.add_argument('--seed', type=int, help='set the seed for reproductible experiments')
    group_misc.add_argument('--num_workers', type=int, default=6,
                            help='number of workers for the data loader. Default is one. You can bring it up. '
                                 'If you have memory errors go back to one')
    group_misc.add_argument('--root', default=os.getcwd(), help='where to look for the data folder')
    group_misc.add_argument('--save_dir_name', help='name of the directory where all xp data are stored', required=True)
    group_misc.add_argument('--save_file_name', help='name of the saved file (used for tensorboard, saved pickle and checkpoints)', required=True)


