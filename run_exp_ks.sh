#!/bin/sh

# python exp_kernelshap.py --dataset adult --sample_num 8 --save
# python exp_kernelshap.py --dataset adult --sample_num 16 --save
# python exp_kernelshap.py --dataset adult --sample_num 32 --save
# python exp_kernelshap.py --dataset adult --sample_num 64 --save
# python exp_kernelshap.py --dataset adult --sample_num 128 --save


for s in 8 16 32 64 128
do
  python exp_kernelshap.py --dataset adult --sample_num $s --save
  # python exp_kernelshap.py --dataset credit --sample_num $s --save
done
