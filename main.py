"""
file - main.py
Main script to train the aesthetic model on the AVA dataset.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

from tensorboardX import SummaryWriter

from dataset.dataset import AVADataset
from dataset.dataset import TENCENT

from model.model import *

import metrics as metrics_selector


def main(config, fold=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    metric = metrics_selector.get_metrics()

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d-%d.pth' % (config.warm_start_epoch, config.best_fold))))
        print('Successfully loaded model epoch-%d-%d.pth' % (config.warm_start_epoch, config.best_fold))
    else :
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'pre-epoch-%d.pth' % config.warm_start_epoch)))
        print('Successfully loaded model pre-epoch-%d.pth' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        
        trainset = TENCENT(type='cv_train', fold=fold, transform=train_transform)
        valset = TENCENT(type='cv_val', fold=fold, transform=val_transform)
        
        # trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.img_path, transform=train_transform)
        # valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.img_path, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        init_val_plcc = 0

        train_losses = []
        val_losses = []
        # for epoch in range(config.warm_start_epoch, config.epochs):
        for epoch in range(0, config.epochs):
            batch_losses = []
            for i, data in enumerate(train_loader):
                # images = data['image'].to(device)
                # labels = data['annotations'].to(device).float()
                images = data[0].to(device)
                labels = data[1].to(device).float()
                outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                # 10 classes to 5 classes
                outputs = outputs.view(-1, 5, 2, 1)
                # shape = (-1, 5, 1)
                outputs = outputs.sum(dim=2)

                optimizer.zero_grad()

                loss = emd_loss(labels, outputs)
                batch_losses.append(loss.item())

                loss.backward()

                optimizer.step()

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.data[0]))
                writer.add_scalar('batch train loss', loss.data[0], i + epoch * (len(trainset) // config.train_batch_size + 1))

            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if config.decay:
                if (epoch + 1) % 10 == 0:
                    conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                    optimizer = optim.SGD([
                        {'params': model.features.parameters(), 'lr': conv_base_lr},
                        {'params': model.classifier.parameters(), 'lr': dense_lr}],
                        momentum=0.9
                    )

            # do validation after each epoch
            batch_val_losses = []
            for data in val_loader:
                images = data[0].to(device)
                labels = data[1].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.view(-1, 10, 1)
                # 10 classes to 5 classes
                outputs = outputs.view(-1, 5, 2, 1)
                # shape = (-1, 5, 1)
                outputs = outputs.sum(dim=2)
                val_loss = emd_loss(labels, outputs)
                batch_val_losses.append(val_loss.item())
                metric.update(outputs, labels)

            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)
            print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

            # writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
            metric_results = metric.get_result()
            metric_str = ''
            for metric_key in metric_results:
                # writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
                metric_str += '{} = {}  '.format(metric_key, metric_results[metric_key])
            metric.reset()
            # metric_str += 'loss = {}'.format(tot_loss[t]/num_val_batches)
            print(metric_str)
            # writer.add_scalar('validation_loss', tot_loss['all']/len(val_dst), n_iter)

            # Use early stopping to monitor training
            plcc = metric_results['plcc']
            if init_val_plcc < plcc:
                init_val_loss = plcc
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d-%d.pth' % (epoch + 1, fold)))
                print('Done.\n')
                # reset count
                count = 0
            elif init_val_plcc >= plcc:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')
        return init_val_plcc, epoch+1
        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('EMD loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        # testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform)
        testset = TENCENT(type='test', transform=train_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            # 10 classes to 5 classes
            outputs = outputs.view(5, 2, 1)
            # shape = (5, 1)
            outputs = outputs.sum(dim=1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--img_path', type=str, default='./data/images')
    parser.add_argument('--train_csv_file', type=str, default='./data/train_labels.csv')
    parser.add_argument('--val_csv_file', type=str, default='./data/val_labels.csv')
    parser.add_argument('--test_csv_file', type=str, default='./data/test_labels.csv')

    # training parameters
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--decay', action='store_true')
    # parser.add_argument('--conv_base_lr', type=float, default=5e-3)
    # parser.add_argument('--dense_lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    # parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--train_batch_size', type=int, default=64)
    # parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=82)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')

    parser.add_argument('--conv_base_lr', type=float, default=3e-3)
    parser.add_argument('--dense_lr', type=float, default=3e-3)

    parser.add_argument('--best_fold', type=int, default=1)

    config = parser.parse_args()

    def _5foldcv() :
        config.train = True
        plcc_list = []
        epoch_list = []
        for i in range(0, 5) :
            print(i+1, ' fold')
            plcc, epoch = main(config, i+1)
            plcc_list.append(plcc)
            epoch_list.append(epoch)
        print(plcc_list)
        print(epoch)
        best_plcc = max(plcc_list)
        index = plcc_list.index(best_plcc)
        best_epoch = epoch_list[index]
        config.train = False
        config.test = True
        config.warm_start_epoc = best_epoch
        config.best_fold = index + 1
        print('test')
        main(config)
    # main(config, fold)

    _5foldcv()
