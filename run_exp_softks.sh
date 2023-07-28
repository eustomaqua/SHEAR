#!/bin/sh

for s in 8 16 32 64 128
do
  python exp_soft_kernelshap.py --dataset adult --sample_num $s --save
  # python exp_soft_kernelshap.py --dataset credit --sample_num $s --save
done
