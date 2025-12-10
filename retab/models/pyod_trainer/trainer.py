import os
import json
import torch
from joblib import dump, load

from addict import Dict
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.pca import PCA

from retab.utils import get_summary_metrics
from retab.datasets import Preprocessor
from retab.models import BaseTrainer


PYOD_MODELS = {
    "IForest": IForest,
    "KNN": KNN,
    "LOF": LOF,
    "OCSVM": OCSVM,
    "PCA": PCA,
}


class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        self.model_name = meta_info.model_name

        model_obj = PYOD_MODELS[self.model_name]
        print(model_params)
        print(data_params)
        if hasattr(model_obj, "random_state"):
            model_params.random_state = meta_info.seed
            self.model = model_obj(**model_params)
        else:
            self.model = model_obj(**model_params)
        self.ckpt_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.pth')
        self.json_path = os.path.join(meta_info.checkpoint_path, meta_info.data_name, meta_info.model_name, meta_info.exp_id, f'{meta_info.seed}.json')
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        self.model.fit(X=self.X_train_cont)
        # self.save()

    @torch.no_grad()
    def evaluate(self):
        # self.load()
        ascs = self.model.decision_function(self.X_test_cont)
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=ascs)
        with open(self.json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        return metrics
    
    def save(self):
        dump(self.model, self.ckpt_path)

    def load(self):
        self.model = load(self.ckpt_path)