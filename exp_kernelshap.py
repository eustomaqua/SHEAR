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
from shapreg import removal, games, shapley

sys.path.append("./")
sys.path.append("./adult_dataset")
sys.path.append("./credit_dataset")


@torch.no_grad()
def shapreg_shapley(imputer, data_loader):
  shapley_value_buf = []
  shapley_rank_buf = []
  MSE_buf = []
  mAP_buf = []

  for index, (x, y, z, sh_gt, rk_gt) in enumerate(data_loader):
    x = x.numpy()
    y = y.numpy()
    z = z.numpy()
    shapley_gt = sh_gt.numpy()
    ranking_gt = rk_gt.numpy()
    feat_num = z.shape[-1]

    attr_buf = []
    for idx in range(args.circ_num * feat_num):
      game = games.PredictionGame(imputer, z[0])
      # Estimate Shapley values
      attr = shapley.ShapleyRegression(
          game, paired_sampling=False, detect_convergence=False,
          n_samples=args.sample_num, batch_size=args.sample_num,
          bar=False)

      attr_buf.append(attr.values.reshape(1, -1))

    attr_buf = np.concatenate(attr_buf, axis=0)
    attr = attr_buf.mean(axis=0).reshape(1, -1)

    ranking = attr.argsort(axis=1)
    shapley_value_buf.append(attr)
    shapley_rank_buf.append(ranking)

    # rank_mAP = ((ranking == ranking_gt).astype(np.float).sum(
    rank_mAP = ((ranking == ranking_gt).astype(float).sum(
        axis=1) / ranking_gt.shape[-1]).mean(axis=0)
    mAP_buf.append(rank_mAP)

    MSE = np.square(attr - shapley_gt).mean(axis=0)
    MSE_sum = np.square(attr - shapley_gt).sum(axis=1).mean(axis=0)
    MSE_buf.append(MSE_sum)
    MMSE = np.array(MSE_buf).mean(axis=0)
    mAP = np.array(mAP_buf).mean(axis=0)

    print("Index: {}, Rank Precision: {}, mAP: {}".format(
        index, rank_mAP, mAP))

  shapley_value_buf = np.concatenate(shapley_value_buf, axis=0)
  shapley_rank_buf = np.concatenate(shapley_rank_buf, axis=0)

  shapley_value_buf = torch.from_numpy(shapley_value_buf).type(torch.float)
  shapley_rank_buf = torch.from_numpy(shapley_rank_buf).type(torch.int)

  return shapley_value_buf, shapley_rank_buf


parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int,
                    help="number of samples",
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

parser.add_argument('-data', '--dataset', type=str, default='adult')
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

  # if args.dataset == 'adult':
  #   data_loader = DataLoader(TensorDataset(
  #       x_test, y_test, z_test, shapley_value_gt,
  #       shapley_ranking_gt), batch_size=1, shuffle=False,
  #       drop_last=False, pin_memory=True)
  # elif args.dataset == 'credit':
  N = shapley_value_gt.shape[0]
  data_loader = DataLoader(TensorDataset(
      x_test[0: N], y_test[0: N], z_test[0: N], shapley_value_gt,
      shapley_ranking_gt), batch_size=1, shuffle=False,
      drop_last=False, pin_memory=True)

  reference_dense = x_test[:, dense_feat_index].mean(dim=0)
  reference_sparse = -torch.ones_like(sparse_feat_index).type(torch.long)
  reference = torch.cat((reference_dense, reference_sparse),
                        dim=0).unsqueeze(dim=0).numpy()

  if args.softmax:
    imputer = removal.MarginalExtension(
        reference, model_for_shap.forward_softmax_1_np)
    '''
    save_checkpoint_name = (
        "./{}_dataset/softmax/kernelshap_{}_m_1_s_".format(
            args.dataset, args.dataset) + str(args.sample_num) +
        "_r_" + str(checkpoint["round_index"]) + "_c_" +
        str(args.circ_num) + ".pth.tar")
    '''

    save_checkpoint_name = (
        "./ckpts/softmax/kernelshap_{}_m_1_s_".format(args.dataset))

  else:
    imputer = removal.MarginalExtension(
        reference, model_for_shap.forward_1_np)
    '''
    save_checkpoint_name = (
        "./{}_dataset/wo_softmax/kernelshap_{}_m_1_s_".format(
            args.dataset, args.dataset) + str(args.sample_num) +
        "_r_" + str(checkpoint["round_index"]) + "_c_" +
        str(args.circ_num) + ".pth.tar")
    '''

    save_checkpoint_name = (
        "./ckpts/wo_softmax/kernelshap_{}_m_1_s_".format(
            args.dataset))
  save_checkpoint_name += (str(args.sample_num) + "_r_" +
                           str(checkpoint["round_index"]) + "_c_" +
                           str(args.circ_num) + ".pth.tar")

  shapley_value, shapley_rank = shapreg_shapley(imputer, data_loader)

  if args.save:
    save_checkpoint(save_checkpoint_name,
                    round_index=checkpoint["round_index"],
                    shapley_value_gt=shapley_value_gt,
                    shapley_ranking_gt=shapley_ranking_gt,
                    shapley_value=shapley_value,
                    shapley_rank=shapley_rank,
                    sample_num=args.sample_num)
