from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import copy
import pandas as pd
import os
import sys

current = os.path.dirname(os.path.realpath('DataLoader.py'))
parent = os.path.dirname(current)
parent_parent = os.path.dirname(os.path.realpath(parent))
print('import:     ', parent_parent)
sys.path.append(parent_parent)
from DataLoader import OpenBHBDataset


class LazyLoader:
    def __init__(self, initializer, *args, **kwargs):
        # print('---------------------------- lazy loader ------------------------')
        self.initializer = initializer
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        # print('---------------------------- lazy loader ------------------------')
        return self.initializer(*self.args, **self.kwargs)

    
    

class ImageSet(torch.utils.data.Dataset):
    def __init__(self, path_f, parent_dir=''):
        # print('---------------------------- imageSet ------------------------')
        lines = [p.strip().split() for p in open(path_f, 'r')]
        self.paths = [f'{parent_dir}/{path}' for path, _ in lines]
        self.labels = [int(label) - 1 for _, label in lines]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        img = Image.open(path)
        return {'img': img, 'label': label, 'path': path}


class Mixed(torch.utils.data.Dataset):
    def __init__(self, *args):
        # assumes domains are lazy loaded for efficiency
        # print('---------------------------- Mixed ------------------------')

        self.domains = [domain() for domain in args]
        self.lengths = [len(d) for d in self.domains]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):

        for d, domain in enumerate(self.domains):
            if index >= len(domain):
                index -= len(domain)
            else:
                x = domain[index]
                x['domain'] = d
                return x


class Augmentation(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 baseline,
                 augmentation=lambda x: x,
                 augment_half=False,
                 use_rgb_convert=False):
        self.dataset = dataset
        self.baseline = baseline
        self.augmentation = augmentation
        self.use_rgb_convert = use_rgb_convert
        self.augment_half = augment_half

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        if self.use_rgb_convert and x['img'].mode is not 'RGB':
            x['img'] = x['img'].convert('RGB')
        augment = True
        if self.augment_half and random.random() < 0.5:
            augment = False
        if augment:
            x['img'] = self.augmentation(x['img'])
        x['augmented'] = augment
        x['img'] = self.baseline(x['img'])
        return x