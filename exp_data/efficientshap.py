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

sys.path.append("../adult_dataset")
sys.path.append("../credit_dataset")
sys.path.append("../")


def get_second_order_grad(model, x, device=None):
  with torch.set_grad_enabled(True):

    if x.nelement() < 2:
      return np.array([])
