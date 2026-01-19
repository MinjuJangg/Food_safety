import os
import json
from omegaconf import OmegaConf, DictConfig
from typing import Type

from extraction.base import BaseExtractor
from extraction.cetd.cetd import CETDExtractor
from extraction.method1.method1 import Method1Extractor
from extraction.method2.method2 import Method2Extractor

from summarization.base import BaseSummarizer
from summarization.bertsum.bertsum import BertSumSummarizer
from summarization.matchsum.matchsum import MatchSumSummarizer
from summarization.exaone.exaone import ExaoneSummarizer

CONFIG_PATH = "./config/main_config.yaml"

EXTRACTOR_REGISTRY: dict[str, Type[BaseExtractor]] = {
    "cetd": CETDExtractor,
    "method1": Method1Extractor,
    "method2": Method2Extractor,
}

SUMMARIZER_REGISTRY: dict[str, Type[BaseSummarizer]] = {
    "bertsum": BertSumSummarizer,
    "matchsum": MatchSumSummarizer,
    "exaone": ExaoneSummarizer,
}

def get_extractor(name: str, config: dict = None) -> BaseExtractor:
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}. Available: {list(EXTRACTOR_REGISTRY.keys())}")
    extractor_class = EXTRACTOR_REGISTRY[name]
    return extractor_class(config=config or {})

def get_summarizer(name: str, config: dict = None) -> BaseSummarizer:
    if name not in SUMMARIZER_REGISTRY:
        raise ValueError(f"Unknown summarizer: {name}. Available: {list(SUMMARIZER_REGISTRY.keys())}")
    summarizer_class = SUMMARIZER_REGISTRY[name]
    return summarizer_class(config=config or {})

def main():
    try:
        cfg = OmegaConf.load(CONFIG_PATH)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {CONFIG_PATH}")
        return
    
    print(f"Starting pipeline with configuration from '{CONFIG_PATH}':")
    print(OmegaConf.to_yaml(cfg.pipeline))
    
    try:
        ext_name = cfg.pipeline.extraction
        sum_name = cfg.pipeline.summarization
        
        ext_config = OmegaConf.to_object(cfg.extractors.get(ext_name, {}))
        sum_config = OmegaConf.to_object(cfg.summarizers.get(sum_name, {}))
        
        print("\n[Pipeline] Initializing modules...")
        extractor = get_extractor(ext_name, config=ext_config)
        summarizer = get_summarizer(sum_name, config=sum_config)
        
        input_file = cfg.pipeline.get("input_file", "sample.html")
        print(f"\n[Pipeline] Loading data from {input_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            print(f"Loaded {len(html_content)} bytes of HTML.")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")
            return

        print("\n[Pipeline] Running Stage 1: Extraction...")
        extracted_text = extractor.extract(html_content)
        print(f"  > Extracted Text: '{extracted_text[:100]}...'")
            
        print("\n[Pipeline] Running Stage 2: Summarization...")
        summary = summarizer.summarize(extracted_text)
        print(f"  > Final Summary: '{summary}'")
        
        print("\nPipeline finished successfully.")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    main()