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
import time

from reproduce.data_model import mlp, Model_for_shap
from reproduce.data_read import Adult, Credit
from train_data import save_checkpoint
from captum.attr import *
import shap
from shap_utils import eff_shap
from eff_shap_utils import Efficient_shap

sys.path.append("./adult_dataset")
sys.path.append("./credit_dataset")
sys.path.append("./")


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
def distance_estimation(x, reference, dense_feat_index, cate_attrib_book):

  reference_sparse = torch.cat([x[-1] for x in cate_attrib_book], dim=0)
  dense_feat_num = dense_feat_index.shape[0]
  reference = torch.cat((
      reference[0: dense_feat_num], reference_sparse), dim=0)

  distance_vector = torch.abs(x - reference.unsqueeze(dim=0)).squeeze(dim=0)

  distance_i, distance_j = torch.meshgrid(distance_vector, distance_vector)
  distance_matrix = distance_i * distance_j

  return distance_matrix


@torch.no_grad()
def error_term_estimation(mlp, x, z, reference, dense_feat_index,
                          sparse_feat_index, cate_attrib_book):
  interaction_scores = {}

  device = torch.device("cpu")

  grad_matrix_1 = get_second_order_grad(mlp, x, device)
  distance_matrix = distance_estimation(
      x, reference, dense_feat_index, cate_attrib_book)

  error_matrix = grad_matrix_1 * distance_matrix

  dense_feat_num = dense_feat_index.shape[0]
  sparse_feat_num = sparse_feat_index.shape[0]

  error_matrix_comb = torch.zeros((dense_feat_num + sparse_feat_num,
                                   dense_feat_num + sparse_feat_num))

  error_matrix_comb[0: dense_feat_num, 0: dense_feat_num
                    ] = error_matrix[0: dense_feat_num, 0: dense_feat_num]

  block_len = torch.tensor(
      [1 for x in dense_feat_index] + 
      [x.shape[-1] for x in cate_attrib_book]).type(torch.int)

  for feature_index_i in range(dense_feat_num + sparse_feat_num):
    for feature_index_j in range(dense_feat_num + sparse_feat_num):

      if feature_index_i == feature_index_j:
        error_matrix_comb[feature_index_i, feature_index_j] = 0
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


# @torch.no_grad()
# def efficient_shap(model_for_shap, data_loader, reference,
#                    dense_feat_index, sparse_feat_index, cate_attrib_book):
@torch.no_grad()
def efficient_shap(model_grad, model_for_shap, data_loader, reference,
                   dense_feat_index, sparse_feat_index, cate_attrib_book):

  shapley_value_buf = []
  shapley_rank_buf = []
  topK = torch.log2(torch.tensor(args.sample_num)).type(torch.long) - 1
  MSE_buf = []
  mAP_buf = []
  total_time = 0

  eff_shap_agent = Efficient_shap(model_for_shap, reference, topK)

  for index, (x, y, z, sh_gt, rk_gt, error_matrix) in enumerate(data_loader):
    x = x[0].unsqueeze(dim=0)
    z = z[0].unsqueeze(dim=0)
    shapley_gt = sh_gt
    ranking_gt = rk_gt
    feat_num = z.shape[-1]
    error_matrix = error_matrix.squeeze(dim=0).numpy()

    eff_shap_agent.feature_selection(error_matrix)

    attr_buf = []
    for idx in range(args.circ_num):
      t0 = time.time()
      attr = eff_shap_agent.forward(z)
      total_time += time.time() - t0
      attr_buf.append(attr)

    attr_buf = torch.cat(attr_buf, dim=0)
    attr = attr_buf.mean(dim=0).reshape(1, -1)

    ranking = attr.argsort(dim=1)
    shapley_value_buf.append(attr)
    shapley_rank_buf.append(ranking)

    rank_mAP = ((ranking == ranking_gt).sum(dim=1).type(
        torch.float) / ranking_gt.shape[-1]).mean(dim=0)
    mAP_buf.append(rank_mAP)

    if args.dataset == 'adult':
      MSE = torch.square(attr - shapley_gt).mean(dim=0)
    elif args.dataset == 'credit':
      MSE = torch.abs(attr - shapley_gt).mean(dim=0)
    MSE_sum = torch.square(attr - shapley_gt).sum(dim=1).mean(dim=0)
    MSE_buf.append(MSE_sum)
    MMSE = torch.tensor(MSE_buf).mean(dim=0)
    mAP = torch.tensor(mAP_buf).mean(dim=0)
    mAP_std = np.array(mAP_buf).std(axis=0)

    print("Index: {}, mAP: {}, MMSE {}".format(index, rank_mAP, MMSE))

  shapley_value_buf = torch.cat(shapley_value_buf, dim=0)
  shapley_rank_buf = torch.cat(shapley_rank_buf, dim=0)
  return shapley_value_buf, shapley_rank_buf, total_time


parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, help="number of samples",
                    default=16)  # for good shapley value ranking
parser.add_argument("--circ_num", type=int,
                    help="number of circle for average",
                    default=1)  # for approaching y
parser.add_argument('--antithetical', action='store_true',
                    help='antithetical sampling.')
parser.add_argument("--softmax", action='store_true',
                    help="softmax model output.")
parser.add_argument("--save", action='store_true',
                    help="save estimated shapley value.")

parser.add_argument("-data", "--dataset", type=str, default='adult')
args = parser.parse_args()


if __name__ == "__main__":

  if args.softmax:
    checkpoint_fname = (
        "./ckpts/model_softmax_{}_m_1_r_0.pth.tar".format(
            args.dataset))
  else:
    checkpoint_fname = (
        "./ckpts/model_{}_m_1_r_0.pth.tar".format(args.dataset))
  checkpoint = torch.load(checkpoint_fname)
  del checkpoint_fname

  '''
  if args.softmax:
    checkpoint = torch.load(
        "./{}_dataset/model_softmax_{}_m_1_r_0.pth.tar".format(
            args.dataset, args.dataset))
  else:
    checkpoint = torch.load(
        "./{}_dataset/model_{}_m_1_r_0.pth.tar".format(
            args.dataset, args.dataset))
  '''

  dense_feat_index = checkpoint["dense_feat_index"]
  sparse_feat_index = checkpoint["sparse_feat_index"]
  cate_attrib_book = checkpoint["cate_attrib_book"]

  model = mlp(input_dim=checkpoint["input_dim"],
              hidden_dim=checkpoint["hidden_dim"],
              output_dim=checkpoint["output_dim"],
              layer_num=checkpoint["layer_num"],
              activation=checkpoint["activation"])

  model.load_state_dict(checkpoint["state_dict"])

  model_for_shap = Model_for_shap(model, dense_feat_index,
                                  sparse_feat_index, cate_attrib_book)

  x_test = checkpoint["test_data_x"]
  y_test = checkpoint["test_data_y"]
  z_test = checkpoint["test_data_z"]
  shapley_value_gt = checkpoint["test_shapley_value"]
  shapley_ranking_gt = checkpoint["test_shapley_ranking"]
  reference = checkpoint["reference"]
  error_matrix = checkpoint["error_matrix"]

  N = error_matrix.shape[0]
  data_loader = DataLoader(TensorDataset(
      x_test[0: N], y_test[0: N], z_test[0: N], shapley_value_gt[0: N],
      shapley_ranking_gt[0: N], error_matrix[0: N]),
      batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

  print(x_test[:, 0: 5])
  print(z_test[:, 0: 5])

  if args.softmax:
    shapley_value, shapley_rank, total_time = efficient_shap(
        model_for_shap.forward_softmax_1,
        model_for_shap.forward_softmax_1,
        data_loader, reference, dense_feat_index, sparse_feat_index,
        cate_attrib_book)

    '''
    save_checkpoint_name = (
        "./{}_dataset/softmax/efficient_shap_{}_m_1_s_".format(
            args.dataset, args.dataset)) + str(args.sample_num) + "_r_" + str(
        checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"
    '''

    save_checkpoint_name = (
        "./ckpts/softmax/efficient_shap_{}_m_1_s_".format(
            args.dataset))

  else:
    shapley_value, shapley_rank, total_time = efficient_shap(
        model_for_shap.forward_1,
        model_for_shap.forward_1,
        data_loader, reference, dense_feat_index,
        sparse_feat_index, cate_attrib_book)

    '''
    save_checkpoint_name = (
        "./{}_dataset/wo_softmax/efficient_shap_{}_m_1_s_".format(
            args.dataset, args.dataset)) + str(args.sample_num) + "_r_" + str(
        checkpoint["round_index"]) + "_c_" + str(args.circ_num) + ".pth.tar"
    '''

    save_checkpoint_name = (
        "./ckpts/wo_softmax/efficient_shap_{}_m_1_s_".format(
            args.dataset))
  save_checkpoint_name += (str(args.sample_num) + "_r_" +
                           str(checkpoint["round_index"]) + "_c_" +
                           str(args.circ_num) + ".pth.tar")

  if args.save:
    save_checkpoint(save_checkpoint_name,
                    round_index=checkpoint["round_index"],
                    shapley_value_gt=shapley_value_gt,
                    shapley_ranking_gt=shapley_ranking_gt,
                    shapley_value=shapley_value,
                    shapley_rank=shapley_rank,
                    sample_num=args.sample_num,
                    total_time=total_time)
