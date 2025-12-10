import os
import sys
import torch
import pandas as pd
from joblib import dump, load

from addict import Dict
from retab.datasets import Preprocessor
from retab.models import BaseTrainer
from retab.utils import get_summary_metrics

from retab.models.llm_guided_iforest.llm_guided_iforest import LLMGuidedIForest
from .llm_guided_iforest.llm_utils import llm_call
from .llm_guided_iforest.prompt_generator import LGIFPromptGenerator

BASE_DIRS = {
    'iforest': 'result/iforest',
    'refined_iforest': 'result/iforest_refined',
    'prompt': 'result/prompts',
    'answer': 'result/answers'
}

EXTENSIONS = {
    'iforest': 'pkl',
    'refined_iforest': 'pkl',
    'prompt': 'txt',
    'answer': 'txt'
}

class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)

        self.model_params = model_params
        self.model = LLMGuidedIForest(
            tree_params=getattr(model_params, 'tree_params', ''), # Note that we need tree parameters.
            model_name=getattr(model_params, 'llm_model', 'gemini-2.5-pro'),
            # batch_size=getattr(model_params, 'inference_batch_size', data_params.batch_size),
            max_retry=getattr(model_params, 'max_retry', 3),
        )

        if llm_call is not None:
            self.model.set_llm_call_func(llm_call)
        else:
            print("Warning: LLM call function not available. Model will not work properly.")

        # note that we need to save LLM response and the refiend tree.
        # so that we can debug if some parsing error occurs. 
        self._setup_paths(meta_info)
        self._create_directories()

        # train_normal_df = pd.DataFrame(self.X_train, columns=self.column_names)
        # Calculate statistics for each numerical column
        stats = {}
        # for col in self.column_names:
        #     col_data = train_normal_df[col].dropna()
        #     if len(col_data) > 0:
        #         stats[col] = {
        #             'count': len(col_data),
        #             'mean': float(col_data.mean()),
        #             'std': float(col_data.std()),
        #             'min': float(col_data.min()),
        #             'max': float(col_data.max()),
        #             'q5': float(col_data.quantile(0.05)),
        #             'q25': float(col_data.quantile(0.25)),
        #             'q50': float(col_data.quantile(0.50)),
        #             'q75': float(col_data.quantile(0.75)),
        #             'q95': float(col_data.quantile(0.95))
        #         }    
        metadata = self.preprocessor.textmeta
        dataset_info = {
            'metadata': metadata,
            'stats': stats  # Calculated statistics from normal samples
        }        
        self.prompt_generator = LGIFPromptGenerator(dataset_info, self.iforest_path, self.prompt_path) # tree, meta_data

    
    def _setup_paths(self, meta_info: Dict):
        """Set up LLM-related paths and checkpoint path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(meta_info.data_name, meta_info.model_name, meta_info.exp_id)

        # current_dir / * / data / model / exp_id / 
        for key, base_dir in BASE_DIRS.items():
            ext = EXTENSIONS[key]
            path = os.path.abspath(os.path.join(current_dir, base_dir, base_path, f'{meta_info.seed}.{ext}'))
            setattr(self, f'{key}_path', path)
        
        self.ckpt_path = os.path.join(
            meta_info.checkpoint_path, 
            base_path, 
            f'{meta_info.seed}.pth'
        )

    def _create_directories(self):
        """Create necessary directories."""
        for key in BASE_DIRS:
            path = getattr(self, f'{key}_path')
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

    def train(self):
        self.model.train_iforest(X=self.X_train_cont, path=self.iforest_path)
        self.prompt_generator.save_prompt(self.iforest_path, self.prompt_path)
        # self.model.query_llm(self.prompt_path, self.answer_path)
        
        # parse llm response into IForest shit.


    @torch.no_grad()
    def evaluate(self):
        # self.load()
        ascs = self.model.iforest.decision_function(self.X_test_cont) # Need to fix evaluate logic
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=ascs)
        return metrics
    
    def save(self):
        dump(self.model, self.ckpt_path)

    def load(self):
        self.model = load(self.ckpt_path)