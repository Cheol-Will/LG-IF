import os
import json
import pandas as pd
from glob import glob

def collect_results(root_dir="results/default"):
    """
    Collects all model performance JSONs into three DataFrames: f1_df, auroc_df, auprc_df.
    Each DataFrame has model_name x dataset structure with an 'Average' column.
    """

    records = {}  # {model_name: {dataset: {f1, auroc, auprc}}}

    # find every directory (e.g., results/default/wine/)
    for dataset_path in sorted(glob(os.path.join(root_dir, "*"))):
        dataset_name = os.path.basename(dataset_path)
        if not os.path.isdir(dataset_path):
            continue

        # find models (e.g., results/default/wine/IForest)
        for model_path in sorted(glob(os.path.join(dataset_path, "*"))):
            model_name = os.path.basename(model_path)
            if not os.path.isdir(model_path):
                continue

            json_files = glob(os.path.join(model_path, "*.json"))
            if not json_files:
                continue

            json_path = json_files[0]
            with open(json_path, "r") as f:
                metrics = json.load(f)

            if model_name not in records:
                records[model_name] = {}
            records[model_name][dataset_name] = metrics

    # get DataFrame for each metric
    def build_metric_df(metric_key):
        data = {}
        for model, ds_results in records.items():
            row = {ds: ds_results[ds][metric_key] for ds in ds_results if metric_key in ds_results[ds]}
            if row:
                row["Average"] = sum(row.values()) / len(row)
                data[model] = row
        df = pd.DataFrame(data).T
        df.index.name = "model_name"
        return df

    f1_df = build_metric_df("f1")
    auroc_df = build_metric_df("auroc")
    auprc_df = build_metric_df("auprc")

    return f1_df, auroc_df, auprc_df


def main():
    f1_df, auroc_df, auprc_df = collect_results()
    print(auroc_df)

if __name__ == "__main__":
    main()