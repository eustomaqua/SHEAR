#!/bin/sh
# chmod +x ?.sh
# ./?.sh

python exp_efficientshap.py --dataset adult --sample_num 8 --save
python exp_efficientshap.py --dataset adult --sample_num 16 --save
python exp_efficientshap.py --dataset adult --sample_num 32 --save
python exp_efficientshap.py --dataset adult --sample_num 64 --save
python exp_efficientshap.py --dataset adult --sample_num 128 --save
