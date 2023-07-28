# coding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
from evaluation import pearsonr_corr, pearsonr_evaluate
import seaborn as sns

from reproduce.parameters import checkpoint_save_locat
import argparse
sys.path.append("./")
sys.path.append("./adult_dataset")
sys.path.append("./credit_dataset")


def errorbar_plot(ax, x, data, **kw):
  est = np.mean(data, axis=0)
  sd = np.std(data, axis=0)
  cis = (est - sd, est + sd)
  ax.fill_between(x, cis[0], cis[1], **kw)
  ax.margins(x=0)


algorithm_buf = [
    "KernelShap", "KS-WF", "KS-Pair", "PS", "APS", "SHEAR"
]  # , "Sage", "Soft-KS", "Unbiased-KS"
marker_buf = ['^', '>', '<', 'v', 's', 'o', 'p', 'h']
color_buf = ["blue", "orange", "black", "magenta", "green",
             "red", "#a87900", "#5d1451"]

AE_buf_buf = []
AE_std_buf_buf = []
AE_sample_buf_buf = []
kl_dis_buf_buf = []
rank_mAP_buf_buf = []
mAP_std_buf_buf = []
mAP_sample_buf_buf = []
n_sample_buf_buf = []
time_buf_buf = []
corr_value_buf_buf = []


parser = argparse.ArgumentParser()
parser.add_argument("-data", "--dataset", type=str, default='adult')
args = parser.parse_args()

ckpt_temporary = [8, 16, 32, 64, 128]
checkpoint_buf_wo_softmax = {
    "SHEAR": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "efficient_shap_"
    ) for s in ckpt_temporary],
    "KernelShap": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "kernelshap_"
    ) for s in [32, 64, 128]],
    "KS-WF": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "soft_kernelshap_"
    ) for s in ckpt_temporary],
    "KS-Pair": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "kernelshap_pair_"
        # ) for s in [32, 36, 40, 48, 56, 64, 128]],
    ) for s in [40, 48, 56, 64, 128]],
    "PS": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "permutation_"
    ) for s in ckpt_temporary],
    "APS": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "permutation_att_"
    ) for s in ckpt_temporary],
    "Unbiased-KS": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "unbiased_kernelshap_pair_"
    ) for s in [256, 512, 1024]],
    "Sage": [checkpoint_save_locat(
        args.dataset, s, 0, 1, False, "sage_"
    ) for s in ckpt_temporary],
}


for alg_index, alg in enumerate(algorithm_buf):
  print(alg)

  fname_buf = checkpoint_buf_wo_softmax[alg]
  # fname_buf = checkpoint_buf_softmax[alg]

  AE_buf = np.zeros((len(fname_buf),))
  AE_std_buf = np.zeros((len(fname_buf),))
  kl_dis_buf = np.zeros((len(fname_buf),))
  rank_mAP_buf = np.zeros((len(fname_buf),))
  mAP_std_buf = np.zeros((len(fname_buf),))
  mAP_max_buf = np.zeros((len(fname_buf),))
  n_sample_buf = np.zeros((len(fname_buf),))
  time_buf = np.zeros((len(fname_buf),))
  corr_value_buf = np.zeros((len(fname_buf),))
  AE_sample_buf = []
  mAP_sample_buf = []

  for checkpoint_index, checkpoint_fname in enumerate(fname_buf):
    checkpoint = torch.load(checkpoint_fname)
    n_sample = checkpoint["sample_num"]
    shapley_value = checkpoint["shapley_value"]
    shapley_rank = shapley_value.argsort(dim=1)

    N = shapley_value.shape[0]
    feature_num = shapley_value.shape[-1]
    shapley_value_gt = checkpoint["shapley_value_gt"][0: N]
    shapley_ranking_gt = checkpoint["shapley_ranking_gt"][0: N]

    absolute_error = torch.abs(
        shapley_value - shapley_value_gt).sum(dim=1)
    corr_value = pearsonr_corr(shapley_value, shapley_value_gt)
    # np.corrcoef(x, y)

    mAP_weight = torch.tensor([[
        1. / (feature_num - x) for x in range(feature_num)]])
    rank_mAP = torch.sum((shapley_rank == shapley_ranking_gt).type(
        torch.float) * mAP_weight, dim=1) / mAP_weight.sum()

    shapley_value_logstd = torch.log(torch.abs(
        shapley_value) / torch.abs(shapley_value).sum(
        dim=1).unsqueeze(dim=1) + 1e-8)
    shapley_value_gt_std = torch.abs(shapley_value_gt) / torch.abs(
        shapley_value_gt).sum(dim=1).unsqueeze(dim=1)
    kl_dis = torch.nn.functional.kl_div(
        shapley_value_logstd, shapley_value_gt_std).detach()

    AE_buf[checkpoint_index] = absolute_error.mean(dim=0)
    AE_std_buf[checkpoint_index] = absolute_error.std(dim=0)
    kl_dis_buf[checkpoint_index] = kl_dis
    rank_mAP_buf[checkpoint_index] = rank_mAP.mean(dim=0)
    mAP_std_buf[checkpoint_index] = rank_mAP.std(dim=0)
    mAP_max_buf[checkpoint_index] = rank_mAP.min(dim=0)[0]  # ?
    n_sample_buf[checkpoint_index] = n_sample
    corr_value_buf[checkpoint_index] = corr_value
    # time_buf[checkpoint_index] = total_time
    AE_sample_buf.append(absolute_error.unsqueeze(dim=0))
    mAP_sample_buf.append(rank_mAP.unsqueeze(dim=0))

  print(alg)
  print(AE_buf)
  # print(AE_std_buf)
  # print(kl_dis_buf)
  print(rank_mAP_buf)
  # print(mAP_std_buf)
  # print(mAP_max_buf)
  # print(corr_value_buf)

  AE_sample_buf = torch.cat(AE_sample_buf, dim=0)
  mAP_sample_buf = torch.cat(mAP_sample_buf, dim=0)
  AE_buf_buf.append(AE_buf)
  AE_std_buf_buf.append(AE_std_buf)
  kl_dis_buf_buf.append(kl_dis_buf)
  rank_mAP_buf_buf.append(rank_mAP_buf)
  mAP_std_buf_buf.append(mAP_std_buf)
  n_sample_buf_buf.append(n_sample_buf)
  # time_buf_buf.append(time_buf)
  corr_value_buf_buf.append(corr_value_buf)
  AE_sample_buf_buf.append(AE_sample_buf)
  mAP_sample_buf_buf.append(mAP_sample_buf)

for alg_index, alg in enumerate(algorithm_buf):
  MSE_buf = AE_buf_buf[alg_index]
  MSE_buf[MSE_buf > 1] = np.nan
  AE_sample_buf = AE_sample_buf_buf[alg_index].T.numpy()
  n_sample_buf = n_sample_buf_buf[alg_index]
  AE_std_buf = AE_std_buf_buf[alg_index]

  if alg == "KS-Pair":
    AE_sample_buf = AE_sample_buf[:, 4:]
    n_sample_buf = n_sample_buf[4:]

  corr_value_buf = corr_value_buf_buf[alg_index]
  corr_value_buf[corr_value_buf < .9] = np.nan

  sns.tsplot(
      # sns.lineplot(
      time=np.log2(n_sample_buf), data=AE_sample_buf,
      marker=marker_buf[alg_index],
      condition=algorithm_buf[alg_index],
      linewidth=0.5, markersize=8,
      color=color_buf[alg_index], ci=[100])

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("AE of Estimated Shapley Value", fontsize=18)
plt.legend(loc='upper right', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(
    int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
if algs.dataset == 'adult':
  tmp = (0, -3)
elif args.dataset == 'credit':
  tmp = (0, -4)
plt.gca().ticklabel_format(style='sci', scilimits=tmp, axis='y')
del tmp

# plt.xticks(fontsize=15)
plt.yticks(fontsize=18)
if args.dataset == 'credit':
  plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(
    left=.15, bottom=.13, top=.97, right=.99, wspace=.01)
plt.savefig("./figures/AE_vs_n_sample_{}.pdf".format(args.dataset))
# plt.savefig("./figures/AE_vs_n_sample_{}.png".format(args.dataset))

# plt.show()
plt.close()

for alg_index, alg in enumerate(algorithm_buf):
  # MSE_buf = MSE_buf_buf[alg_index]
  rank_mAP_buf = rank_mAP_buf_buf[alg_index]
  mAP_std_buf = mAP_std_buf_buf[alg_index]
  n_sample_buf = n_sample_buf_buf[alg_index]
  mAP_sample_buf = mAP_sample_buf_buf[alg_index].T.numpy()

  sns.tsplot(
      # sns.lineplot(
      time=np.log2(n_sample_buf), data=mAP_sample_buf,
      marker=marker_buf[alg_index],
      condition=algorithm_buf[alg_index],
      linewidth=.5, markersize=8,
      color=color_buf[alg_index], ci=[100])

plt.xlabel("Eval. number", fontsize=18)
plt.ylabel("ACC of Feature Importance Ranking", fontsize=18)
plt.legend(loc='lower right', fontsize=18, frameon=True)

plt.xticks(np.log2(n_sample_buf), ["$2^{}$".format(
    int(x)) for x in np.log2(n_sample_buf)], fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([2.96, 7.04])
plt.grid()
plt.subplots_adjust(
    left=0.18, bottom=0.13, top=0.99, right=0.99, wspace=0.01)
plt.savefig("../figure/mAP_vs_n_sample_{}.pdf".format(args.dataset))
# plt.savefig("../figure/mAP_vs_n_sample_{}.png".format(args.dataset))

# plt.show()
plt.close()


"""
refs:
https://stackoverflow.com/questions/60231029/attributeerror-module-seaborn-has-no-attribute-tsplot
https://www.datasciencelearner.com/seaborn-tsplot-implement-python-example/
https://blog.csdn.net/XING_Gou/article/details/119980661
"""
