import torch
from tqdm import tqdm
from collections import defaultdict
from utils import count_correct_top_k, update_correct_per_class, update_correct_per_class_topk
import torch.nn.functional as F


def train_epoch(model, optimizer, train_loader, criteria, loss_train, train_accuracy, topk_train_accuracy,
                k, n_train, use_gpu):
    model.train()
    loss_epoch_train = 0
    n_correct_train = 0
    n_correct_top_k_train = 0
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=1)):
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()

        optimizer.zero_grad()
        batch_output_train = model(batch_x_train)

        loss_batch_train = criteria(batch_output_train, batch_y_train)

        loss_epoch_train += loss_batch_train.item()
        loss_batch_train.backward()
        optimizer.step()
        with torch.no_grad():

            n_correct_train += torch.sum(torch.eq(batch_y_train, torch.argmax(batch_output_train, dim=-1))).item()
            n_correct_top_k_train += count_correct_top_k(scores=batch_output_train, labels=batch_y_train,
                                                         k=k).item()
    # At the end of epoch compute average of statistics over batches and store them
    with torch.no_grad():
        loss_epoch_train /= batch_idx
        epoch_accuracy_train = n_correct_train / n_train
        epoch_top_k_accuracy_train = n_correct_top_k_train / n_train

        loss_train.append(loss_epoch_train), train_accuracy.append(epoch_accuracy_train), topk_train_accuracy.append(
            epoch_top_k_accuracy_train)

    return loss_epoch_train, epoch_accuracy_train, epoch_top_k_accuracy_train


def val_epoch(model, val_loader, criteria, loss_val, val_accuracy, topk_val_accuracy, k, dataset_attributes,
              use_gpu, class_accuracy_over_epochs, class_accuracy_over_epochs_topk, append=True):

    print()
    model.eval()
    with torch.no_grad():

        class_acuracy = defaultdict(int)
        class_acuracy_topk = defaultdict(int)

        loss_epoch_val = 0
        n_correct_val = 0
        n_correct_top_k_val = 0
        for batch_idx, (batch_x_val, batch_y_val) in enumerate(tqdm(val_loader, desc='val', position=1)):
            if use_gpu:
                batch_x_val, batch_y_val = batch_x_val.cuda(), batch_y_val.cuda()
            batch_output_val = model(batch_x_val)

            loss_batch_val = criteria(batch_output_val, batch_y_val)
            loss_epoch_val += loss_batch_val.item()

            n_correct_val += torch.sum(torch.eq(batch_y_val, torch.argmax(batch_output_val, dim=-1))).item()
            n_correct_top_k_val += count_correct_top_k(scores=batch_output_val, labels=batch_y_val, k=k).item()

            update_correct_per_class(batch_output_val, batch_y_val, class_acuracy)
            update_correct_per_class_topk(batch_output_val, batch_y_val, class_acuracy_topk, k)

        # After seeing val update the statistics over batches and store them
        loss_epoch_val /= batch_idx
        epoch_accuracy_val = n_correct_val / dataset_attributes['n_val']
        epoch_top_k_accuracy_val = n_correct_top_k_val / dataset_attributes['n_val']

        loss_val.append(loss_epoch_val), val_accuracy.append(epoch_accuracy_val), topk_val_accuracy.append(
                epoch_top_k_accuracy_val)

        # If specified store the group accuracy and class accuracy
        for class_id in class_acuracy:
            class_acuracy[class_id] = class_acuracy[class_id] / dataset_attributes['class2num_instances']['val'][
                class_id]
            class_acuracy_topk[class_id] = class_acuracy_topk[class_id] / dataset_attributes['class2num_instances']['val'][
                class_id]
        if append:
            class_accuracy_over_epochs.append(class_acuracy)
            class_accuracy_over_epochs_topk.append(class_acuracy_topk)

    return loss_epoch_val, epoch_accuracy_val, epoch_top_k_accuracy_val


def test_epoch(model, test_loader, criteria, dataset_attributes, k, use_gpu, n_test):

    print()
    model.eval()
    with torch.no_grad():

        class_acuracy = defaultdict(int)
        class_acuracy_topk = defaultdict(int)

        loss_epoch_test = 0
        n_correct_test = 0
        n_correct_top_k_test = 0
        for batch_idx, (batch_x_test, batch_y_test) in enumerate(tqdm(test_loader, desc='test', position=1)):
            if use_gpu:
                batch_x_test, batch_y_test = batch_x_test.cuda(), batch_y_test.cuda()
            batch_output_test = model(batch_x_test)
            loss_batch_test = criteria(batch_output_test, batch_y_test)
            loss_epoch_test += loss_batch_test.item()

            n_correct_test += torch.sum(torch.eq(batch_y_test, torch.argmax(batch_output_test, dim=-1))).item()
            n_correct_top_k_test += count_correct_top_k(scores=batch_output_test, labels=batch_y_test, k=k).item()

            update_correct_per_class(batch_output_test, batch_y_test, class_acuracy)
            update_correct_per_class_topk(batch_output_test, batch_y_test, class_acuracy_topk, k)


        # After seeing test test update the statistics over batches and store them
        loss_epoch_test /= batch_idx
        epoch_accuracy_test = n_correct_test / n_test
        epoch_top_k_accuracy_test = n_correct_top_k_test / n_test

        for class_id in class_acuracy:
            class_acuracy[class_id] = class_acuracy[class_id] / dataset_attributes['class2num_instances']['test'][
                class_id]
            class_acuracy_topk[class_id] = class_acuracy_topk[class_id] / dataset_attributes['class2num_instances']['test'][
                class_id]
        opt = {}
        opt['class_accuracy'] = class_acuracy
        opt['class_accuracy_topk'] = class_acuracy_topk

    return loss_epoch_test, epoch_accuracy_test, epoch_top_k_accuracy_test, opt