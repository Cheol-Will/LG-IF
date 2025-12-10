#!/usr/bin/env bash
set -euo pipefail

deep_model_list=(
    #
    # DeepIsolationForest
    # DeepSVDD
    # REPEN
    # RDP
    #
    # RCA
    # GOAD
    # NeuTraL
    #
    SLAD
    MCM
    DRL
)

data_list=(automobile backdoor campaign cardiotocography census churn cirrhosis covertype credit equip gallstone glass glioma quasar seismic stroke vertebral wbc wine yeast)

for data in "${data_list[@]}"; do
    for model in "${deep_model_list[@]}"; do
        cfg_file="configs/hpo/dl/${model}.yaml"
        python run_hpo.py --data_name "$data" --model_name "$model" --cfg_file "$cfg_file" --seeds 0 1 2 3 4
    done
done
