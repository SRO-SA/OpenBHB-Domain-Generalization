from DataLoader import OpenBHBDataset
import argparse
import os, shutil
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models
from torch import optim
import numpy as np
import tqdm
from torch.autograd import Variable
from torchsample.transforms import *
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from box import Box
from Train import start_train
from wilds import get_dataset


from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, choices=[0, 1], default=5)

    args = parser.parse_args()
    return args

def run(args):
    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)
    # data augmentation pipeline
    # augmentor = Compose([
    #     transforms.Lambda(lambda x: torch.Tensor(x)),
    #     RandomRotate(25),
    #     RandomTranslate([0.11, 0.11]),
    #     RandomFlip(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    # ])
    dataset = OpenBHBDataset()
  
    train_dataset = dataset.get_subset('train')
    val_dataset = dataset.get_subset('val')
    id_val_dataset = dataset.get_subset('id_val')
    test_dataset = dataset.get_subset('test')
    id_test_dataset = dataset.get_subset('id_test')
    
    
    train_loader = get_train_loader("standard", train_dataset, batch_size=1)
    validation_loader = get_eval_loader("standard", val_dataset, batch_size=1)
    id_validation_loader = get_eval_loader('standard', id_val_dataset, batch_size=1)
    test_loader = get_eval_loader('standard', test_dataset, batch_size=1)

    
    start_train(train_loader, validation_loader, test_loader, writer, args)
    
    
if __name__ == '__main__':
  args = parse_arguments()
  run(args)