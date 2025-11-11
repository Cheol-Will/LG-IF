#!/usr/bin/env bash
set -euo pipefail

model='LLMGuidedIForest'
data_list=(automobile)

for data in "${data_list[@]}"; do
    cfg_file="configs/default/lgif/${model}.yaml"
    echo "$model on $data with cfg_file=$cfg_file"
    python run_default.py --data_name "$data" --model_name "$model" --cfg_file "$cfg_file" --seed 42
done