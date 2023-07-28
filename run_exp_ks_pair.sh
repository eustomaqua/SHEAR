#!/bin/sh

# for s in 32 36 40 48 56 64 128
for s in 40 48 56 64 128
do
  python exp_kernelshap_pair.py --dataset adult --sample_num $s --save
  # python exp_kernelshap_pair.py --dataset credit --sample_num $s --save
done
