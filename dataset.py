import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import math
import cv2



def load_cell_image(fname):

    src=Image.open(fname).convert('RGB')



    return src
def load_image(fname):

    src=Image.open(fname).convert('RGB')

    return src

class Imbalanced_Dataset(Dataset):
    def __init__(self, fname,transform=None,train=True):

        if train:
            filename = os.path.join(fname, 'train_imbalanced.txt')

        else:
            filename = os.path.join(fname, 'test_imbalanced.txt')

        fh = open(filename, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs

        self.transform = transform

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img = load_cell_image(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDataset(Dataset):
    def __init__(self, fname,transform=None,train=True):

        if train:
            filename = os.path.join(fname, 'train.txt')

        else:
            filename = os.path.join(fname, 'test.txt')

        fh = open(filename, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs

        self.transform = transform

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img = load_cell_image(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

