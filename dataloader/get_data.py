import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from dataloader.data_utils import Plantnet, LabelNoise, split_dataset
from collections import Counter


def get_data(args):
    root = os.path.join(args.root, 'data')
    if not os.path.exists(root):
        os.mkdir(root)

    if args.dataset == 'plantnet':
        return get_plantnet(root=os.path.join(root, 'plantnet'), batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'cifar100':
        # if not os.path.exists(os.path.join(root, 'cifar100')):
        #     os.mkdir(os.path.join(root, 'cifar100'))

        return get_cifar_100(root=os.path.join(root, 'cifar100'), batch_size=args.batch_size,
                             num_workers=args.num_workers, noise_cifar=args.noise_cifar)
    else:
        raise NotImplementedError


def get_cifar_100(root, batch_size, num_workers, noise_cifar, augment=True):
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    dataset_train = datasets.CIFAR100(root=root, train=True,
                                      transform=transform_train, download=True)
    dataset_val = datasets.CIFAR100(root=root, train=True,
                                    transform=transform_test, download=True)
    dataset_test = datasets.CIFAR100(root=root, train=False,
                                     transform=transform_test, download=True)


    if noise_cifar:
        dataset_train = LabelNoise(dataset_train, k=5, n_labels=100, p=noise_cifar)
        print(f'Running CIFAR100 with label noise set at {noise_cifar}')

    dataset_train, dataset_val = split_dataset(dataset_train, dataset_val, train_size=45000, val_size=5000)

    test_class_to_num_instances = Counter(dataset_test.targets)
    val_class_to_num_instances = Counter(dataset_val[i][1] for i in range(len(dataset_val)))


    train_loader = data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = data.DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset_attributes = {'n_train': len(dataset_train), 'n_val': len(dataset_val), 'n_test': len(dataset_test),
                          'n_classes': 100, 'class2num_instances': {'val': val_class_to_num_instances,
                                                                    'test': test_class_to_num_instances}}

    return train_loader, val_loader, test_loader, dataset_attributes


def get_plantnet(root, batch_size, num_workers):
    transform_train = transforms.Compose([transforms.Resize(size=256), transforms.RandomCrop(size=224),
                                          transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize(size=256), transforms.CenterCrop(size=224),
                                         transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = Plantnet(root, 'train', transform=transform_train)

    train_class_to_num_instances = Counter(trainset.targets)
    cls_num_list = [train_class_to_num_instances[i] for i in range(len(train_class_to_num_instances))]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    valset = Plantnet(root, 'val', transform=transform_test)
    val_class_to_num_instances = Counter(valset.targets)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = Plantnet(root, 'test', transform=transform_test)
    test_class_to_num_instances = Counter(testset.targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    n_classes = len(trainset.classes)

    dataset_attributes = {'n_train': len(trainset), 'n_val': len(valset), 'n_test': len(testset), 'n_classes': n_classes,
                          'class2num_instances': {'train': train_class_to_num_instances,
                                                  'val': val_class_to_num_instances,
                                                  'test': test_class_to_num_instances},
                          'class_to_idx': trainset.class_to_idx,
                          'cls_num_list': cls_num_list}

    return trainloader, valloader, testloader, dataset_attributes