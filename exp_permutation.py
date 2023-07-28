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
from shapreg import (removal, games, shapley, shapley_unbiased,
                     shapley_sampling, stochastic_games)
from shapreg.utils import crossentropyloss

from shap_utils import permutation_sample_parallel
# from permutation_utils import PermutationGame
from reproduce.parameters import (train_save_path,
                                  empirical_params,
                                  checkpoint_save_locat)
sys.path.append("./")
sys.path.append("./adult_dataset")
sys.path.append("./credit_dataset")


@torch.no_grad()
def shapreg_shapley(model, data_loader, reference):

  shapley_value_buf = []
  shapley_rank_buf = []
  MSE_buf = []
  mAP_buf = []
  total_time = 0

  for index, (x, y, z, sh_gt, rk_gt) in enumerate(data_loader):
    x = x  # .numpy()
    y = y  # .numpy()
    z = z  # .numpy()
    shapley_gt = sh_gt.numpy()
    ranking_gt = rk_gt.numpy()
    feat_num = z.shape[-1]
    attr_buf = []

    # rank_mAP_buf = []
    for idx in range(args.circ_num):
      t0 = time.time()
      attr = permutation_sample_parallel(
          model, z, reference, batch_size=args.sample_num,
          antithetical=args.antithetical)
      # t1 = time.time()
      # print(index, t1 - t0)
      total_time += time.time() - t0
      attr_buf.append(attr.reshape(1, -1))

    attr_buf = np.concatenate(attr_buf, axis=0)
    attr = attr_buf.mean(axis=0).reshape(1, -1)

    ranking = attr.argsort(axis=1)
    shapley_value_buf.append(attr)
    shapley_rank_buf.append(ranking)

    rank_mAP = ((ranking == ranking_gt).astype(float).sum(
        axis=1) / ranking_gt.shape[-1]).mean(axis=0)
    mAP_buf.append(rank_mAP)

    mAP = np.array(mAP_buf).mean(axis=0)
    mAP_std = np.array(mAP_buf).std(axis=0)

    print("Index: {}, Rank Prediction: {}, mAP: {}, mAP std: {}"
          "".format(index, rank_mAP, mAP, mAP_std))

  shapley_value_buf = np.concatenate(shapley_value_buf, axis=0)
  shapley_rank_buf = np.concatenate(shapley_rank_buf, axis=0)

  shapley_value_buf = torch.from_numpy(shapley_value_buf).type(torch.float)
  shapley_rank_buf = torch.from_numpy(shapley_rank_buf).type(torch.int)

  return shapley_value_buf, shapley_rank_buf, total_time


args = empirical_params()
if __name__ == "__main__":
  checkpoint_fname = train_save_path(args.dataset, 0, args.softmax)
  checkpoint = torch.load(checkpoint_fname)
  del checkpoint_fname

  dense_feat_index = checkpoint["dense_feat_index"]
  sparse_feat_index = checkpoint["sparse_feat_index"]
  cate_attrib_book = checkpoint["cate_attrib_book"]

  model = mlp(input_dim=checkpoint["input_dim"],
              output_dim=checkpoint["output_dim"],
              hidden_dim=checkpoint["hidden_dim"],
              layer_num=checkpoint["layer_num"],
              activation=checkpoint["activation"])

  model.load_state_dict(checkpoint["state_dict"])

  model_for_shap = Model_for_shap(
      model, dense_feat_index, sparse_feat_index, cate_attrib_book)

  x_test = checkpoint["test_data_x"]
  y_test = checkpoint["test_data_y"]
  z_test = checkpoint["test_data_z"]
  shapley_value_gt = checkpoint["test_shapley_value"]
  shapley_ranking_gt = checkpoint["test_shapley_ranking"]
  reference = checkpoint["reference"]

  N = shapley_value_gt.shape[0]
  data_loader = DataLoader(TensorDataset(
      x_test[0: N], y_test[0: N], z_test[0: N], shapley_value_gt,
      shapley_ranking_gt), batch_size=1, shuffle=False,
      drop_last=False, pin_memory=True)

  if args.softmax:
    shapley_value, shapley_rank, total_time = shapreg_shapley(
        model_for_shap.forward_softmax_1, data_loader, reference)
  else:
    shapley_value, shapley_rank, total_time = shapreg_shapley(
        model_for_shap.forward_1, data_loader, reference)

  save_checkpoint_name = checkpoint_save_locat(
      args.dataset, args.sample_num, checkpoint["round_index"],
      args.circ_num, args.softmax,
      "permutation_" + "att_" * args.antithetical)

  if args.save:
    save_checkpoint(save_checkpoint_name,
                    round_index=checkpoint["round_index"],
                    shapley_value_gt=shapley_value_gt,
                    shapley_ranking_gt=shapley_ranking_gt,
                    shapley_value=shapley_value,
                    shapley_rank=shapley_rank,
                    sample_num=args.sample_num,
                    total_time=total_time)
