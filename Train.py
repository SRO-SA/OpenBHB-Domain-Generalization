from DataLoader import Dataset
import argparse
import os, shutil
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models
from torch import optim
import numpy as np
import pandas as pd
import tqdm
from torch.autograd import Variable
from torchsample.transforms import *
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from box import Box


from brain_cancer import Model
from sklearn import metrics



# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-t', '--task', type=str, required=True,
#                         choices=['abnormal', 'acl', 'meniscus'])
#     parser.add_argument('-p', '--plane', type=str, required=True,
#                         choices=['sagittal', 'coronal', 'axial'])
#     parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
#     parser.add_argument('--lr_scheduler', type=int, choices=[0, 1], default=1)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--lr', type=float, default=1e-5)
#     parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
#     parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
#     parser.add_argument('--patience', type=int, choices=[0, 1], default=5)

#     args = parser.parse_args()
#     return args

def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, log_every=100):
    _ = model.train()
    if torch.cuda.is_available():
        model.cuda()
    y_preds = []
    y_trues = []
    losses = []
    aucs = []

    for i, (image, label, metadata) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            # weight = weight.cuda()

        prediction = model.forward(image.float())
        label = torch.squeeze(label, dim=[1])
        # print(prediction['y_pred'])
        lable_box = Box({'y_trues': label})
        # print("prediction: ", prediction['y_pred'].shape)
        # print("lable: ", lable_box['y_trues'])
        loss = nn.MSELoss()(
            prediction['y_pred'], lable_box['y_trues'])
        
        # print('prediction: ', prediction['y_pred'], "actual: ", lable_box['y_trues'])
        
        # print('loss is :', float(loss))
        # print(prediction['y_pred'].shape, lable_box['y_trues'].shape)
        loss.backward()
        optimizer.step()
        
        y_pred = int(prediction['y_pred'].item())
        y_true = int(lable_box['y_trues'].item())

        y_preds.append(y_pred)
        y_trues.append(y_true)

        # print('prediction :', y_preds)
        # print("accuracy is : {0:.16f}".format( metrics.accuracy_score(y_trues, y_preds)))
        try:
            auc = metrics.mean_absolute_error(y_trues, y_preds)
        except:
            auc = 0.5

        loss_value = loss.item()
        losses.append(loss_value)
        aucs.append(auc)

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                    | avg train loss {4} | train auc : {5}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4)
                  )
                  )
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(np.mean(aucs), 4)
    return train_loss_epoch, train_auc_epoch

def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []
    aucs = []

    for i, (image, label, metadata) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            # weight = weight.cuda()

        # print(label)
        # print(weight)
        label = label
        label = torch.squeeze(label, dim=[1])
        # weight = weight

        prediction = model.forward(image.float())['y_pred']
        # print(prediction.shape, label.shape)
        # loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss = torch.nn.MSELoss()(prediction, label)

        # print("-----------------------------")
        # print("loss is: ", loss)
        
        
        loss_value = loss.item()
        losses.append(loss_value)

        probas = prediction

        y_trues.append(int(label[0]))
        y_preds.append(int(probas[0].item()))

        # print(metrics.accuracy_score(y_trues, y_preds))
        try:
            auc = metrics.mean_absolute_error(y_trues, y_preds)
        except:
            auc = 0.5

        aucs.append(auc)

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(np.mean(aucs), 4)
    return val_loss_epoch, val_auc_epoch


def test_model(model, test_loader, writer, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    with torch.no_grad():
      for i, (image, label, metadata) in enumerate(test_loader):

          if torch.cuda.is_available():
              image = image.cuda()
              label = label.cuda()
              # weight = weight.cuda()

          # print(label)
          # print(weight)
          label = label
          label = torch.squeeze(label, dim=[1])
          # weight = weight

          prediction = model.forward(image.float())['y_pred']
          # print(prediction.shape, label.shape)
          # loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
          loss = torch.nn.MSELoss()(prediction, label)

          # print("-----------------------------")
          # print("loss is: ", loss)
          
          
          loss_value = loss.item()
          losses.append(loss_value)

          probas = prediction

          y_trues.append(int(label[0]))
          y_preds.append(int(probas[0].item()))

          # print(metrics.accuracy_score(y_trues, y_preds))
          try:
              auc = metrics.mean_absolute_error(y_trues, y_preds)
          except:
              auc = 0.5

          writer.add_scalar('Test/Loss', loss_value)
          writer.add_scalar('Test/AUC', auc)

          if (i % log_every == 0) & (i > 0):
              print('''[Single batch number : {0} / {1} ] | avg test loss {2} | test auc : {3}'''.
                    format(
                        i,
                        len(test_loader),
                        np.round(np.mean(losses), 4),
                        np.round(auc, 4),
                    )
                    )

      writer.add_scalar('Test/AUC_epoch', auc)

      test_loss_epoch = np.round(np.mean(losses), 4)
      test_auc_epoch = np.round(auc, 4)
      return test_loss_epoch, test_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def start_train(train_loader, validation_loader, test_loader, writer, args):
    mrnet = Model()
    mrnet = mrnet.cuda()
    
    lr=1e-2
    optimizer = optim.Adam(mrnet.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #   optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True
    # )
    best_val_loss = float('inf')
    best_val_auc = float(0)
    num_epochs = args.epochs
    iteraion_change_loss = 0
    patience = args.patience
    
    data = {'epoch': [i for i in range(num_epochs)],
            'train_loss': [],
            'train_auc': [],
            'validation_loss': [],
            'validation_auc': [],
            'test_loss': [],
            'test_auc': []
            }
    

    for epoch in range(num_epochs):
      train_loss, train_auc = train_model(
        mrnet, train_loader, epoch, num_epochs, optimizer, writer
      )
      val_loss, val_auc = evaluate_model(
        mrnet, validation_loader, epoch, num_epochs, writer, lr
      )
      data['train_loss'].append(train_loss)
      data['train_auc'].append(train_auc)
      data['validation_loss'].append(val_loss)
      data['validation_auc'].append(val_auc)
      data['test_loss'].append(0.0)
      data['test_auc'].append(0.0)
      
      
      print("train loss : {0} | train auc {1} | val loss {2} | val auc {3}".format(
        train_loss, train_auc, val_loss, val_auc
      ))
      
      # if args.lr_scheduler == 1:
      #   scheduler.step(val_loss)
        
      # iteration_change_loss +=1
      print('-'* 30)
      
      if val_auc > best_val_auc:
        best_val_auc = val_auc
        if bool(args.save_model):
          file_name = f'model_{args.task}_{args.plane}_val_auc_\
                      {val_auc:0.4f}_train_auc_{train_auc:0.4f}\
                      _epoch_{epoch+1}.pth'
          for f in os.listdir('./models/') :
            if (args.task in f) and (args.plane in f):
              os.remove(f'./models/{f}')
          torch.save(mrnet, f'./models/{file_name}')
      
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        iteraion_change_loss = 0
      if iteraion_change_loss == patience:
        print('Early stopping after {0} iterations without the decrease of the val loss'.format(
          iteraion_change_loss
        ))
        break
      
    test_loss, test_auc = test_model(
        mrnet, test_loader, writer
      )
    data['test_loss'][-1] = test_loss
    data['test_auc'][-1] = test_auc
    
    print("test loss : {0} | test auc {1}".format(
        test_loss, test_auc
      ))
    
    df = pd.DataFrame(data)
    df.set_index('epoch', inplace=True)
    csv_filename = 'training_results.csv'
    df.to_csv(csv_filename)
    print(f"Data written to '{csv_filename}'")
# if __name__ == '__main__':
#   args = parse_arguments()
#   run(args)