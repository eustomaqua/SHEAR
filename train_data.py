# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

import copy
import sys
import os

from reproduce.data_model import mlp
from reproduce.data_read import Adult, Credit
import argparse
from reproduce.parameters import raw_data_path, train_save_path


def train_epoch(model, train_loader, criterion, optimizer):
  model.train()
  for x, y, _ in train_loader:
    y_ = model(x)
    # print(y_.shape, y.shape)

    loss_value = criterion(y_, y)
    optimizer.zero_grad()
    loss_value.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=0.1, norm_type=2)  # necessary!!
    optimizer.step()


def validate(model, val_loader, criterion, epoch):
  correct_counting = 0
  counting = 0
  loss_value = 0
  model.eval()

  with torch.no_grad():
    for x, y, _ in val_loader:
      y_ = model(x)
      y_hat = y_.argmax(axis=1)
      # loss_value = criterion(y_, y)
      correct_counting += accuracy_score(y_hat, y, normalize=False)
      counting += y.shape[0]

  acc_val = correct_counting * 1. / counting
  print("Epoch {}, Acc {}".format(epoch, acc_val))
  return acc_val, loss_value


def save_checkpoint(fname, **kwargs):
  checkpoint = {}

  for key, value in kwargs.items():
    checkpoint[key] = value
    # setattr(self, key, value)

  torch.save(checkpoint, fname)


def load_checkpoint(fname):
  return torch.load(fname)


def train(model, train_loader, val_loader, criterion, optimizer,
          max_epoch=50):  # , round_num=0, checkpoint=None, save_path="./"):
  best_acc = 0
  best_state_dict = None

  for epoch in range(max_epoch):
    train_epoch(model, train_loader, criterion, optimizer)
    val_acc, _ = validate(model, val_loader, criterion, epoch)

    if val_acc > best_acc:
      best_acc = val_acc
      best_state_dict = copy.deepcopy(model.state_dict())
  return best_state_dict


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-data', '--dataset', type=str, default='adult')
  parser.add_argument('-iter', '--max-epoch', type=int, default=20)
  args = parser.parse_args()
  round_num = 0

  if args.dataset == 'adult':
    case = Adult()
  elif args.dataset == 'credit':
    case = Credit()
  path = raw_data_path(args.dataset)

  '''
  if args.dataset == 'adult':
    case = Adult()
    path = './adult_dataset/adult.csv'
    # ckpt = "./adult_dataset/model_adult_m_1_1_5_r_"
  elif args.dataset == 'credit':
    case = Credit()
    path = './credit_dataset/bank-full-dataset.csv'
    # ckpt = "./credit_dataset/model_credit_m_1_r_"

  ckpt = './ckpts/model_{}_m_1_r_'.format(args.dataset)
  '''

  datasets_torch, cate_attrib_book, dense_feat_index, \
      sparse_feat_index = case.load_data(
          path, val_size=0.2, test_size=0.2, run_num=round_num)
  x_train, y_train, z_train, x_val, y_val, z_val, \
      x_test, y_test, z_test = datasets_torch

  print("Train:")
  print(x_train.shape)
  print("Val:")
  print(x_val.shape)
  print("Test:")
  print(x_test.shape)

  model = mlp(input_dim=x_train.shape[1], output_dim=2,
              layer_num=3, hidden_dim=64,
              activation="torch.nn.functional.softplus")

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.CrossEntropyLoss()
  train_loader = DataLoader(TensorDataset(x_train, y_train, z_train),
                            batch_size=256, shuffle=True,
                            drop_last=True, pin_memory=True)
  val_loader = DataLoader(TensorDataset(x_val, y_val, z_val),
                          batch_size=256, shuffle=False,
                          drop_last=False, pin_memory=True)

  # checkpoint = {}
  print(x_train[:, 0:5])
  print(z_train[:, 0:5])

  best_state_dict = train(model, train_loader, val_loader,
                          criterion, optimizer,  # max_epoch=20)
                          max_epoch=args.max_epoch)

  # save_checkpoint(ckpt + str(round_num) + ".pth.tar",
  ckpt = train_save_path(args.dataset, round_num)
  save_checkpoint(ckpt,
                  round_index=round_num,
                  state_dict=best_state_dict,
                  layer_num=model.layer_num,
                  input_dim=model.input_dim,
                  hidden_dim=model.hidden_dim,
                  output_dim=model.output_dim,
                  activation=model.activation_str,
                  test_data_x = x_test,
                  test_data_y = y_test,
                  test_data_z = z_test,
                  cate_attrib_book=cate_attrib_book,
                  dense_feat_index=dense_feat_index,
                  sparse_feat_index=sparse_feat_index)


# python train_data.py -data adult -iter 20
