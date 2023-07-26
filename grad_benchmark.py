# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd

import numpy as np
import sys
import os
import tqdm
import argparse

from reproduce.data_model import mlp, Model_for_shap
from reproduce.data_read import Adult, Credit
from train_data import save_checkpoint
from captum.attr import *
import shap
from shap_utils import eff_shap


def get_second_order_grad(model, x, device=None):
  with torch.set_grad_enabled(True):

    if x.nelement() < 2:
      return np.array([])

    x.requires_grad = True

    y = model(x)
    grads = autograd.grad(y, x, create_graph=True)[0].squeeze()

    grad_list = []
    for j, grad in enumerate(grads):
      grad2 = autograd.grad(grad, x, retain_graph=True)[0].squeeze()
      grad_list.append(grad2)

    grad_matrix = torch.stack(grad_list)
    return torch.abs(grad_matrix)


@torch.no_grad()
def distance_estimation(x, reference,
                        dense_feat_index, cate_attrib_book):

  reference_sparse = torch.cat(
      [x[-1] for x in cate_attrib_book], dim=0)
  dense_feat_num = dense_feat_index.shape[0]
  reference = torch.cat((
      reference[0: dense_feat_num], reference_sparse), dim=0)

  distance_vector = torch.abs(
      x - reference.unsqueeze(dim=0)).squeeze(dim=0)

  distance_i, distance_j = torch.meshgrid(
      distance_vector, distance_vector)
  distance_matrix = distance_i * distance_j

  return distance_matrix


@torch.no_grad()
def error_term_estimation(mlp, x, reference, dense_feat_index,
                          sparse_feat_index, cate_attrib_book):
  interaction_scores = {}

  device = torch.device("cpu")

  grad_matrix_1 = get_second_order_grad(mlp, x, device)
  distance_matrix = distance_estimation(
      x, reference, dense_feat_index, cate_attrib_book)

  error_matrix = grad_matrix_1 * distance_matrix  # > 0

  dense_feat_num = dense_feat_index.shape[0]
  sparse_feat_num = sparse_feat_index.shape[0]

  error_matrix_comb = torch.zeros((dense_feat_num + sparse_feat_num,
                                   dense_feat_num + sparse_feat_num))

  error_matrix_comb[0: dense_feat_num,
                    0: dense_feat_num] = error_matrix[
      0: dense_feat_num, 0: dense_feat_num]

  block_len = torch.tensor([1 for x in dense_feat_index] + [
      x.shape[-1] for x in cate_attrib_book]).type(torch.int)
  # print(block_len)

  for feature_index_i in range(dense_feat_num + sparse_feat_num):
    for feature_index_j in range(dense_feat_num + sparse_feat_num):

      if feature_index_i == feature_index_j:
        error_matrix_comb[feature_index_i, feature_index_j] = -1
        continue

      elif (feature_index_i < dense_feat_num) and (
              feature_index_j < dense_feat_num):
        continue

      feature_i_dim = block_len[feature_index_i]
      feature_j_dim = block_len[feature_index_j]

      index_strt_i = block_len[0: feature_index_i].sum()
      index_strt_j = block_len[0: feature_index_j].sum()

      error_matrix_comb[feature_index_i, feature_index_j] = torch.max(
          error_matrix[index_strt_i: index_strt_i + feature_i_dim,
                       index_strt_j: index_strt_j + feature_j_dim])

  return error_matrix_comb


@torch.no_grad()
def grad_estimation(model_grad, data_loader, reference,
                    dense_feat_index, sparse_feat_index,
                    cate_attrib_book):
  # model.eval()

  feat_num = len(dense_feat_index) + len(sparse_feat_index)
  error_matrix_buf = torch.zeros((
      len(data_loader), feat_num, feat_num))
  for index, (x, _, _, _, _) in enumerate(data_loader):
    x = x[0].unsqueeze(dim=0)

    error_matrix = error_term_estimation(
        model_grad, x, reference, dense_feat_index, sparse_feat_index,
        cate_attrib_book).detach()
    error_matrix_buf[index] = error_matrix
    print(index)

    # if index == 10:
    #   break
  return error_matrix_buf


parser = argparse.ArgumentParser()
parser.add_argument(
    "--softmax", action='store_true', help="softmax model output.")

parser.add_argument('-data', '--dataset', type=str, default='adult')
# parser.add_argument('-iter', '--max-epoch', type=int, default=20)

args = parser.parse_args()


if __name__ == "__main__":
  # if args.dataset == 'adult':
  #   path = "./"
  # elif args.dataset == 'credit':
  #   path = "./credit_dataset"

  if args.softmax:
    model_checkpoint_fname = (
        "./{}_dataset/model_softmax_{}_m_1_r_0.pth.tar".format(
            args.dataset, args.dataset))
  else:
    model_checkpoint_fname = (
        "./{}_dataset/model_{}_m_1_r_0.pth.tar".format(
            args.dataset, args.dataset))

  checkpoint = torch.load(model_checkpoint_fname)

  dense_feat_index = checkpoint["dense_feat_index"]
  sparse_feat_index = checkpoint["sparse_feat_index"]
  cate_attrib_book = checkpoint["cate_attrib_book"]

  model = mlp(input_dim=checkpoint["input_dim"],
              output_dim=checkpoint["output_dim"],
              hidden_dim=checkpoint["hidden_dim"],
              layer_num=checkpoint["layer_num"],
              activation=checkpoint["activation"])

  model.load_state_dict(checkpoint["state_dict"])

  Model_for_shap = Model_for_shap(
      model, dense_feat_index, sparse_feat_index, cate_attrib_book)

  x_test = checkpoint["test_data_x"]
  y_test = checkpoint["test_data_y"]
  z_test = checkpoint["test_data_z"]
  shapley_value_gt = checkpoint["test_shapley_value"]
  shapley_ranking_gt = checkpoint["test_shapley_ranking"]
  reference = checkpoint["reference"]

  N = shapley_value_gt.shape[0]
  data_loader = DataLoader(TensorDataset(
      x_test[0: N], y_test[0: N], z_test[0: N],
      shapley_value_gt, shapley_ranking_gt),
      batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

  print(x_test[:, 0: 5])
  print(z_test[:, 0: 5])

  error_matrix = grad_estimation(
      model.forward_1, data_loader, reference, dense_feat_index,
      sparse_feat_index, cate_attrib_book)

  save_checkpoint(
      model_checkpoint_fname,
      round_index=checkpoint["round_index"],
      state_dict=checkpoint["state_dict"],
      layer_num=checkpoint["layer_num"],
      input_dim=checkpoint["input_dim"],
      hidden_dim=checkpoint["hidden_dim"],
      output_dim=checkpoint["output_dim"],
      activation=checkpoint["activation"],
      test_data_x=checkpoint["test_data_x"],
      test_data_y=checkpoint["test_data_y"],
      test_data_z=checkpoint["test_data_z"],
      reference=checkpoint["reference"],
      test_shapley_value=checkpoint["test_shapley_value"],
      test_shapley_ranking=checkpoint["test_shapley_ranking"],
      cate_attrib_book=cate_attrib_book,
      dense_feat_index=dense_feat_index,
      sparse_feat_index=sparse_feat_index,
      error_matrix=error_matrix)
