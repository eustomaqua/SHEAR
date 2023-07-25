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


class CustomDataset:
  def __init__(self):
    pass

  def load_data(self, path, val_size, test_size, run_num=0):
    X, Y, Z, X_raw, cate_attrib_book, dense_feat_index, \
        sparse_feat_index = self.load_raw_data(path)

    print("Majority num:", Z[Z == 1].shape[0])
    print("Minority num:", Z[Z == 0].shape[0])
    print("Positive num:", Y[Y == 1].shape[0])
    print("Negative num:", Y[Y == 0].shape[0])

    x_train_all, x_test, y_train_all, y_test, \
        z_train_all, z_test = train_test_split(
            X, Y, X_raw, test_size=test_size, random_state=run_num)
    x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(
        x_train_all, y_train_all, z_train_all, test_size=val_size,
        random_state=0)  # , random_state=fold)

    datasets_np = [(x_train, y_train, z_train), (
        x_val, y_val, z_val), (x_test, y_test, z_test)]
    datasets_torch = ()
    for dataset in datasets_np:
      x, y, z = dataset
      x = torch.from_numpy(x).type(torch.float)
      y = torch.from_numpy(y).type(torch.long).squeeze(dim=1)
      z = torch.from_numpy(z).type(torch.float)
      datasets_torch += (x, y, z)

    cate_attrib_book = [
        torch.from_numpy(x).type(torch.float) for x in cate_attrib_book]
    dense_feat_index = torch.from_numpy(dense_feat_index).type(torch.long)
    sparse_feat_index = torch.from_numpy(sparse_feat_index).type(torch.long)

    return datasets_torch, cate_attrib_book, dense_feat_index, \
        sparse_feat_index


class Adult(CustomDataset):
  def __init__(self):
    super().__init__()

  def load_raw_data(self, path):
    column_names = ['age', 'workclass', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'class']
    input_data = pd.read_csv(path, na_values = "?")

    sensitive_attribs = "sex"
    sensitive_value = input_data.loc[:, sensitive_attribs].values
    Z = pd.DataFrame((sensitive_value == "Male").astype(int), columns=["sex"])
    y = pd.DataFrame((
        input_data['class'] == ">50K").astype(int), columns=["class"])

    categorical_attribs = ["workclass", "education", "marital-status",
                           "occupation", "relationship", "race",
                           "native-country", "sex"]
    categorical_value = input_data.loc[:, categorical_attribs].pipe(
        pd.get_dummies, drop_first=False)
    real_attribs = ["age", "education-num", "capital-gain", "capital-loss",
                    "hours-per-week"]
    real_value = StandardScaler().fit_transform(
        input_data.loc[:, real_attribs].values)

    cate_attrib_encoded_buf = []
    cate_attrib_raw_buf = []
    cate_attrib_index_buf = []
    real_attrib_index_buf = np.arange(real_value.shape[-1])
    cate_attrib_book_buf = []

    for cate_attrib_index, cate_attrib in enumerate(categorical_attribs):
      enc = OneHotEncoder()  # handle_unknown='ignore')
      enc.fit(input_data.loc[:, cate_attrib].values.reshape(-1, 1))
      cate_attrib_encoded = enc.transform(
          input_data.loc[:, cate_attrib].values.reshape(-1, 1)).toarray()
      cate_attrib_book = np.eye(cate_attrib_encoded.shape[-1])

      scaler = StandardScaler()
      scaler.fit(cate_attrib_encoded)
      cate_attrib_scaled = scaler.transform(cate_attrib_encoded)
      cate_attrib_book_scaled = scaler.transform(cate_attrib_book)
      reference = cate_attrib_encoded.mean(axis=0).reshape(1, -1)
      cate_attrib_book_scaled = np.concatenate((
          cate_attrib_book_scaled, reference), axis=0)

      cate_attrib_raw = cate_attrib_encoded.argmax(axis=1).reshape(-1, 1)
      cate_attrib_encoded_buf.append(cate_attrib_scaled)
      cate_attrib_raw_buf.append(cate_attrib_raw)
      cate_attrib_index_buf.append(real_value.shape[-1] + cate_attrib_index)
      cate_attrib_book_buf.append(cate_attrib_book_scaled)

    cate_attrib_index_buf = np.array(
        cate_attrib_index_buf, dtype=np.compat.long)
    # print(real_value.values)
    # print(cate_attrib_encoded_buf)

    X_raw = np.concatenate([real_value] + cate_attrib_raw_buf, axis=1)
    X = np.concatenate([real_value] + cate_attrib_encoded_buf, axis=1)

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y.values, Z.values, X_raw, cate_attrib_book_buf, \
        real_attrib_index_buf, cate_attrib_index_buf


class Credit(CustomDataset):
  def __init__(self):
    super().__init__()

  def load_raw_data(self, path):
    input_data = pd.read_csv(path, na_values="?")  # , names=column_names)

    sensitive_attribs = "age"
    sensitive_value = input_data.loc[:, sensitive_attribs].values
    Z = pd.DataFrame((sensitive_value > 30).astype(int), columns=["age"])
    y = pd.DataFrame((input_data['y'] == "yes").astype(int), columns=["y"])

    categorical_attribs = ["job", "marital", "education", "default",
                           "housing", "loan", "contact", "month", "poutcome"]
    categorical_value = input_data.loc[
        :, categorical_attribs].pipe(pd.get_dummies, drop_first=False)
    real_attribs = [
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    real_value = StandardScaler().fit_transform(
        input_data.loc[:, real_attribs].values)

    cate_attrib_encoded_buf = []
    cate_attrib_raw_buf = []
    cate_attrib_index_buf = []
    real_attrib_index_buf = np.arange(real_value.shape[-1])
    cate_attrib_book_buf = []

    for cate_attrib_index, cate_attrib in enumerate(categorical_attribs):
      enc = OneHotEncoder()  # handle_unknown='ignore')
      enc.fit(input_data.loc[:, cate_attrib].values.reshape(-1, 1))
      cate_attrib_encoded = enc.transform(
          input_data.loc[:, cate_attrib].values.reshape(-1, 1)).toarray()
      cate_attrib_book = np.eye(cate_attrib_encoded.shape[-1])

      scaler = StandardScaler()
      scaler.fit(cate_attrib_encoded)
      cate_attrib_scaled = scaler.transform(cate_attrib_encoded)
      cate_attrib_book_scaled = scaler.transform(cate_attrib_book)
      reference = cate_attrib_encoded.mean(axis=0).reshape(1, -1)
      cate_attrib_book_scaled = np.concatenate((
          cate_attrib_book_scaled,
          reference), axis=0)  # -1 for background noise

      cate_attrib_raw = cate_attrib_encoded.argmax(axis=1).reshape(-1, 1)
      cate_attrib_encoded_buf.append(cate_attrib_scaled)
      cate_attrib_raw_buf.append(cate_attrib_raw)
      cate_attrib_index_buf.append(real_value.shape[-1] + cate_attrib_index)
      cate_attrib_book_buf.append(cate_attrib_book_scaled)

    cate_attrib_index_buf = np.array(
        cate_attrib_index_buf, dtype=np.compat.long)
    # print(real_value.values)
    # print(cate_attrib_encoded_buf)

    X_raw = np.concatenate([real_value] + cate_attrib_raw_buf, axis=1)
    X = np.concatenate([real_value] + cate_attrib_encoded_buf, axis=1)

    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y.values, Z.values, X_raw, cate_attrib_book_buf, \
        real_attrib_index_buf, cate_attrib_index_buf
