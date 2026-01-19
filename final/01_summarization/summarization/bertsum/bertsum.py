from ..base import BaseSummarizer
from .src import *
import hydra
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import AutoTokenizer

class BertSumSummarizer(BaseSummarizer):
    def __init__(self, config: dict=None):
        self.hydra_config_name = config.get("config_name")
        self.hydra_config_path = config.get("config_path", "./config/")
        
        if not self.hydra_config_name:
            raise ValueError("BertSumSummarizer requires 'config_name' in its configuration")
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path = self.hydra_config_path, version_base=None)
        
        self.cfg = hydra.compose(config_name=self.hydra_config_name)
        
        super().__init__(config=OmegaConf.to_object(self.cfg))
    
    def load_model(self):
        print(f"Initializing BERTSUM Summarizer (Hydra config: {self.hydra_config_name})...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_checkpoint)
        self.model = BertSum_Ext(**self.cfg.model)
        
        self.engine = ExtSum_Engine(self.model, **self.cfg.engine)
        
        cfg_trainer = Config_Trainer(self.cfg.trainer)()
        self.trainer = pl.Trainer(**cfg_trainer, logger=False)
        
        if 'test_checkpoint' in self.cfg:
            self.ckpt_path = self.cfg.test_checkpoint
        else:
            raise RuntimeError("Hydra config for BERTSUM is missing 'test_checkpoint'.")
        
        print(f"BERTSUM components loaded. Checkpoint: {self.ckpt_path}")
    
    def summarize(self, text_content: str) -> str:
        print(f"[Summarizer] Running BERTSUM logic (via pl.Trainer.predict)...")
        
        temp_df = pd.DataFrame([
            {'id': 0, 'text': text_content}
        ])
        
        try:
            inference_dataset = ExtSum_Dataset(temp_df, self.tokenizer, self.cfg.max_seq_len)
        except:
            raise KeyError("Failed to create ExtSum_Dataset. Does it expect a 'text' column? Please check your ExtSum_Dataset implementation.")
    
        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0) 
        
        predictions = self.trainer.predict(self.engine, inference_loader, ckpt_path=self.ckpt_path)
        
        if not predictions or len(predictions) == 0:
            raise RuntimeError("BERTSUM prediction returned no output.")
        
        try:
            summary = predictions[0][0]
        except (TypeError, IndexError):
            try:
                summary = predictions[0]
            except Exception as e:
                print(f"Failed to parse prediction output: {predictions}")
                print("Hint: Check the return value of your ExtSum_Engine's predict_step method.")
                raise e
        
        if not isinstance(summary, str):
            raise TypeError(f"Expected summary to be a string, but got {type(summary)}. Output: {summary}")
    
        return summary