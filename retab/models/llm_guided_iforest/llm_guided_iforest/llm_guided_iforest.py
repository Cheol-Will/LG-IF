import os
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from pyod.models.iforest import IForest
import joblib

class LLMGuidedIForest:
    """
    Refined IForest for anomaly detection using LLM
    """
    
    def __init__(
        self,
        tree_params: dict, 
        model_name: str = "gpt-4o", # default in DELTA
        max_retry: int = 3,
    ):
        self.iforest = IForest(**tree_params)
        self.iforest_refined = None # Will be filled via LLM
        self.LLM = None # ...

        self.tree_params = tree_params
        self.model_name = model_name
        self.max_retry = max_retry
        self.llm_call_func = None  # Will be set during training
        self.client = None

    def train_iforest(self, X, path):
        """Fit IForest and save."""
        self.iforest.fit(X=X)
        joblib.dump(self.iforest, path)

    def set_llm_call_func(self, llm_call_func):
        """Set the LLM call function"""
        self.llm_call_func = llm_call_func

    def query_llm(self, prompt_path, answer_path, num_queries=10, max_retries=5, retry_delay=5):
        """
        Query LLM with content from a prompt file
        
        Parameters:
            file_path: Path to the file containing prompt content
            num_queries: Number of queries to perform
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        results = []
        
        for i in range(num_queries):
            retries = 0
            print(f"Query {i + 1}", '-'*30)
            while retries < max_retries:
                try:
                    # Read prompt content
                    with open(prompt_path, 'r', encoding='utf-8') as file:
                        prompt_content = file.read().strip()

                    # Call API
                    response = self.llm_call_func(prompt_content, self.model_name)
                    results.append(response.choices[0].message.content)
                    print(f"Query {i + 1} succesfully got response.")
                    break
                    
                # except (openai.InternalServerError, Exception) as e:
                except (Exception) as e:
                    print(f"Error: {e}, retrying {retries + 1}/{max_retries}...")
                    retries += 1
                    time.sleep(retry_delay)
            
            if retries == max_retries:
                print(f"Query {i + 1} exceeded maximum retries, skipping...")
        
           
        # Save results
        for idx, result in enumerate(results):
            # Construct output filename
            file_name = f"{idx}.txt"
            output_file = os.path.join(answer_path, file_name)
            
            # Write result to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"Result {idx} saved to: {output_file}")
        

    def parse_answer_to_tree(self, result):
        
        pass

    def refine_iforest_with_LLM(self, ):
        
        return