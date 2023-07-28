## Raw


## Reproduction

### Dependency

### At the beginning

Train a classifier on the Adult dataset
```shell
python train_data.py # default adult
python train_data.py -data credit
```

Benchmark the GT-Shapley value for the Adult dataset
```shell
python benchmark_shap.py
```

Benchmark the feature cross-contribution for SHEAR
```shell
python grad_benchmark.py
```

### Run SHEAR and baseline methods

```shell
# chmod +x ?.sh
./run_exp_eff_shap.sh    # SHEAR
./run_exp_ks.sh          # Kernel-SHAR
```

### Evaluating SHEAR
