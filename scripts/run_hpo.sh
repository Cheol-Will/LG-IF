#!/usr/bin/env bash
set -euo pipefail

# ml_model_list=(OCSVM IForest KNN LOF PCA) # PCA might cause some error if # features is small.
ml_model_list=(OCSVM IForest KNN LOF)
data_list=(automobile backdoor campaign cardiotocography census churn cirrhosis covertype credit equip gallstone glass glioma quasar seismic stroke vertebral wbc wine yeast)

for data in "${data_list[@]}"; do
    for model in "${ml_model_list[@]}"; do
        cfg_file="configs/hpo/ml/${model}.yaml"
        python run_hpo.py --data_name "$data" --model_name "$model" --cfg_file "$cfg_file" --seeds 0 1 2 3 4
    done
done