#!/usr/bin/env bash
set -euo pipefail

# ml_model_list=(PCA)
# data_list=(equip)

ml_model_list=(OCSVM IForest KNN LOF)
data_list=(automobile backdoor campaign cardiotocography census churn cirrhosis covertype credit gallstone glioma seismic wbc wine yeast equip glass quasar stroke vertebral)

for data in "${data_list[@]}"; do
    for model in "${ml_model_list[@]}"; do
        cfg_file="configs/default/ml/${model}.yaml"
        echo "$model on $data with cfg_file=$cfg_file"
        python run_default.py --data_name "$data" --model_name "$model" --cfg_file "$cfg_file" --seed 42
    done
done