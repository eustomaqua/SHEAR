## Accelerating Shapley Explanation via Contributive Cooperator Selection 

### Research Motivation

Explaining the behavior of deep neural networks (DNNs) is a significant problem according to not only the practical requirement and but also the regulations in different domains.
Among the state-of-art techniques of DNN interpretation, the Shapley value provides a natural and effective explanation from the perspective of cooperative game theory. 
However, the calculation of Shapley value is known to be an NP-hard problem with extremely high computational complexity. 
To solve this problem, we propose <b>SHEAR</b> for efficient Shapley value estimation in this repo.
           
### Research Challenges  

The brute-force algorithm to calculate exact Shapley values requires the enumeration of all possible feature coalitions, where the complexity grows exponentially with the feature number.
To address this issue, SHEAR only involves few feature coalitions for the estimation.
In such a manner, the enumeration throughout the whole feature space can be avoided such that the computational efficiency can be significantly improved. 

### SHEAR Framework

As shown in the following figure, given a DNN model <i>f</i> and feature value <b>x</b> = [x<sub>1</sub>,··· ,x<sub>M</sub>], SHEAR estimates the contribution of each feature independently.
Specifically, for each feature, SHEAR first calculates its cross-contribution with other features; 
then greedily selects the contributive cooperators to maximize the cumulative cross-contribution; 
finally estimates the feature contribution throughout the coalitions of contributive cooperators.
<div align=center>
<img width="1000" height="200" src="https://github.com/guanchuwang/SHEAR/blob/main/figure/figure/eff_shap.png">
</div>

### Time Complexity of SHEAR

SHEAR has one model backward process to calculate the gradient for cross-contribution estimation, and <i>N</i> model forward process for feature contribution estimation. 
Hence, SHEAR has the time consumption given by <i>T<sub>SHEAR</sub> ≈ t<sub>backward</sub> + N t<sub>forward</sub></i> for single feature
contribution estimation.
Considering to interpretation of a model <i>f</i> having <i>M</i> features, we process the <i>M</i> features consecutively in this repo, where the overall time-cost increases to <i>M</i> times of single feature.
The strucuture can be improved via parallel processing of the <i>M</i> features.


### Dependency
````angular2html
numpy >= 1.19.5
torch >= 1.10.0
pandas >= 1.1.5
scipy >= 1.5.4
````

### Train a classifier on the Adult dataset
````angular2html
python train_adult.py
````


### Benchmark the GT-Shapley value for the Adult dataset
````angular2html
python benchmark_shap_adult.py
````

### Benchmark the feature cross-contribution for SHEAR
````angular2html
python grad_benchmark_adult.py
````  
 

### Run SHEAR and Baseline Methods:
````angular2html
cd exp_adult
bash exp_adult/run_exp_eff_shap.sh     # SHEAR
bash exp_adult/run_exp_ks.sh           # Kernel-SHAP
bash exp_adult/run_exp_softks.sh       # Kernel-SHAP with Welford algorithm 
bash exp_adult/run_exp_ks_pair.sh      # Kernel-SHAP with Pair Sampling
bash exp_adult/run_exp_permutation.sh  # (Antithetical) Permutation Sampling
cd ../
````

### Evaluate SHEAR and Baseline Methods:
````angular2html
cd exp_adult
python err_plot.py
python monotonicity_plot.py
python run_time_plot.py
cd ../
````

### Illustrate the feature contribution generated by SHEAR
````angular2html
cd exp_adult
python interpret_plot.py
cd ../
````

### Reproduce our experiment results on the Adult dataset:

#### Absolute error of estimated Shapley value, Accuracy of feature importance ranking and Algorithmic throughput of SHEAR and baseline methods:

<div align=center>
<img width="350" height="250" src="https://github.com/guanchuwang/SHEAR/blob/main/figure/figure/AE_vs_n_sample_adult.png">
<img width="350" height="250" src="https://github.com/guanchuwang/SHEAR/blob/main/figure/figure/mAP_vs_n_sample_adult.png">
<img width="350" height="250" src="https://github.com/guanchuwang/SHEAR/blob/main/figure/figure/Throughput_vs_ACC_adult.png">
</div>


#### Illustration of Model Interpretation:
<div align=left>
<img width="400" height="300" src="https://github.com/guanchuwang/SHEAR/blob/main/figure/figure/Interpretation_adult.png">
</div>
