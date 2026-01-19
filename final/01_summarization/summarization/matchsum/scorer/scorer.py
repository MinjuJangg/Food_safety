import os
import pandas as pd
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

from .src import *

class BertExtScorer:
    def __init__(self, config: dict = None):
        self.hydra_config_name = config.get("config_name")
        self.hydra_config_path = config.get("config_path", "./config/")
        
        if not self.hydra_config_name:
            raise ValueError("BertExtScorer requires 'config_name' in its configuration.")
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path=self.hydra_config_path, version_base=None)
        self.cfg = hydra.compose(config_name=self.hydra_config_name)
        
        self.load_model()
    
    def load_model(self):
        print(f"Initializing BertExtScorer (Hydra config: {self.hydra_config_name})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_checkpoint)
        self.model = BertExt(**self.cfg.model)
        
        engine_cfg = {k: v for k, v in self.cfg.engine.items() if k != 'test_df'}
        self.engine = BertExt_Engine(self.model, **engine_cfg)
        
        cfg_trainer = Config_Trainer(self.cfg.trainer)()
        self.trainer = pl.Trainer(**cfg_trainer, logger=False)
        
        if 'test_checkpoint' in self.cfg:
            self.ckpt_path = self.cfg.test_checkpoint
        else:
            raise RuntimeError("Hydra config for BertExtScorer is missing 'test_checkpoint'.")
        
        print(f"BertExtScorer components loaded. Checkpoint: {self.ckpt_path}")
    
    def score(self, text_content: str) -> tuple[list[str], list[float]]:
        print(f"[Scorer] Running BertExtScorer logic (via pl.Trainer.predict)...")
        
        temp_df = pd.DataFrame([{'text': text_content}])
        inference_dataset = BertExt_Dataset(temp_df, self.tokenizer, self.cfg.max_seq_len)
        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        predictions = self.trainer.predict(self.engine, inference_loader, ckpt_path=self.ckpt_path)
        
        if not predictions:
            raise RuntimeError("BertExtScorer: Prediction returned no output.")
        
        sentences, scores = predictions[0]
        return sentences, scores