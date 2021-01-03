"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        annotations = self.annotations.iloc[idx, 1:].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class TENCENT(data.Dataset):
    def __init__(self, type, task='o', fold=0, root='Qomex_2020_mobile_game_imges', transform=None):
        self.root = root
        self.type = type
        self.transform = transform
        self.fold = fold
        self.task = task

        data = pd.read_csv(root + '/subjective_scores_v2/all.csv')
        data = data[data['type']!='train']
        cv = pd.read_csv(root + '/subjective_scores_v2/5fold.csv')

        dataset = {
            'cv_train' : [], 
            'cv_val' : [], 
            'val' : [],
            'test' : []
        }

        for index, row in data.iterrows():
            item = {
                'filename' : row['filename'],
                'label_h' : torch.tensor(self.distribution(row['h_0':'h_19'])),
                'label_c' : torch.tensor(self.distribution(row['c_0':'c_19'])),
                'label_f' : torch.tensor(self.distribution(row['f_0':'f_19'])),
                'label_o' : torch.tensor(self.distribution(row['o_0':'o_19']))
            }
            if row['type'] == 'validation' :
                dataset['val'].append(item)
            else :
                dataset['test'].append(item)
                
        for index, row in cv.iterrows():
            item = {
                'filename' : row['filename'],
                'label_h' : torch.tensor(self.distribution(row['h_0':'h_19'])),
                'label_c' : torch.tensor(self.distribution(row['c_0':'c_19'])),
                'label_f' : torch.tensor(self.distribution(row['f_0':'f_19'])),
                'label_o' : torch.tensor(self.distribution(row['o_0':'o_19']))
            }
        
            if row['5fold'] == self.fold :
                dataset['cv_val'].append(item)
            else :
                dataset['cv_train'].append(item)

        self.dataset = dataset


    def distribution(self, row):
        dis = np.zeros(5)
        for v in row :
            v = int(v)
            dis[v-1] = dis[v-1] + 1
        dis = dis / 20
        dis = dis.reshape(-1, 1)
        return dis


    def __len__(self):
        return len(self.dataset[self.type])

    def __getitem__(self, index):
        filename = self.dataset[self.type][index]['filename']
        label_h = self.dataset[self.type][index]['label_h']
        label_c = self.dataset[self.type][index]['label_c']
        label_f = self.dataset[self.type][index]['label_f']
        label_o = self.dataset[self.type][index]['label_o']
        img = Image.open(self.root + '/original_images/' + filename).convert('RGB')
    
        if self.transform:
            img = self.transform(img)
        if (self.task == 'o') :
            label = label_o 
        elif (self.task == 'h') :
            label = label_h 
        elif (self.task == 'c') :
            label = label_c 
        elif (self.task == 'f') :
            label = label_f
        else :
            print('error task')
        # return img, label_h, label_c, label_f, label_o
        return img, label


if __name__ == '__main__':

    # sanity check
    root = './data/images'
    csv_file = './data/train_labels.csv'
    train_transform = transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
    ])
    dset = AVADataset(csv_file=csv_file, root_dir=root, transform=train_transform)
    train_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader):
        images = data['image']
        print(images.size())
        labels = data['annotations']
        print(labels.size())
