#!/bin/sh

for s in 8 16 32 64 128
do
  python exp_permutation.py --dataset adult --sample_num $s --antithetical --save
  python exp_permutation.py --dataset adult --sample_num $s --save
done
