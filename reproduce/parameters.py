# coding: utf-8
# params.py

import argparse


# train_data.py

def raw_data_path(dataset_name):
  if dataset_name == 'adult':
    path = './adult_dataset/adult.csv'
  elif dataset_name == 'credit':
    path = './credit_dataset/bank-full-dataset.csv'
  return path


# def train_path(dataset_name, round_num=0):
#   ckpt = "./ckpts/model_{}_m_1_r_".format(dataset_name)
#   return ckpt + str(round_num) + ".pth.tar"
def train_save_path(dataset_name, round_num=0, softmax=False):
  if softmax:
    ckpt = "./ckpts/model_softmax_{}_m_1_r_".format(dataset_name)
  else:
    ckpt = "./ckpts/model_{}_m_1_r_".format(dataset_name)
  return ckpt + str(round_num) + ".pth.tar"


# benchmark_shap.py


# grad_benchmark.py


# exp_efficientshap.py
# exp_kernelshap.py

def empirical_params():
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

  parser.add_argument('-data', '--dataset', type=str,
                      default='adult')
  args = parser.parse_args()
  return args


def checkpoint_save_locat(dataset_name,
                          sample_num, round_index, circ_num,
                          softmax=False, exp_name=''):
  if exp_name == '':
    exp_name = 'efficient_shap_'

  if softmax:
    save_checkpoint_name = ("./ckpts/softmax/{}{}_m_1_s_"
                            "".format(exp_name, dataset_name))

  else:
    save_checkpoint_name = ("./ckpts/wo_softmax/{}{}_m_1_s_"
                            "".format(exp_name, dataset_name))

  save_checkpoint_name += (str(sample_num) + "_r_" +
                           str(round_index) + "_c_" +
                           str(circ_num) + ".pth.tar")
  return save_checkpoint_name
