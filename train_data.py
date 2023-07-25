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


def save_checkpoint(fname, **kwargs):
  checkpoint = {}

  for key, value in kwargs.items():
    checkpoint[key] = value
    # setattr(self, key, value)

  torch.save(checkpoint, fname)


def load_checkpoint(fname):
  return torch.load(fname)


def train(model, train_loader, val_loader, criterion, optimizer,
          max_epoch=50):
  pass
