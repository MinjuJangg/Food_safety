import torch
from transformers import BertTokenizer

from ..base import BaseSummarizer

from .src.utils import *
from .scorer.scorer import *

class MatchSumSummarizer(BaseSummarizer):
    def __init__(self, config: dict = None):
        if 'checkpoint_path' not in config:
            raise ValueError("MatchsumSummarizer config requires 'checkpoint_path'")
        if 'scorer_config' not in config:
            raise ValueError("MatchsumSummarizer config requires 'scorer_config'")
        
        scorer_config = config['scorer_config']
            
        self.ckpt_path = config['checkpoint_path']
        self.bert_name = config.get('bert_model_name', 'bert-base-uncased')
        self.device = torch.device(config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ext = config.get('ext_param', 12)
        self.sel = config.get('sel_param', 10)

        print("[MatchSum] Initializing dependent Scorer (BertExtScorer)...")
        self.scorer = BertExtScorer(config=scorer_config)

        super().__init__(config)
    
    def load_model(self):
        print(f"Initializing MatchSUM Summarizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        self.model = load_matchsum_model(self.ckpt_path, self.bert_name, self.device)

    def summarize(self, text_content: str) -> str:
        print("[Summarizer] Running MatchSUM logic...")
        
        print("[MatchSum] Running dependent Scorer...")
        sentences, scores = self.scorer.score(text_content)
        print(f"  > Scorer returned {len(sentences)} sentences.")
            
        # 2. 데이터 처리
        document_text_str = str(sentences)
        salient_scores_str = str(list(scores))
        
        document_text_list = str_preprocesse(document_text_str)
        document_full_text = ' '.join(document_text_list)
        salient_scores = str_to_float(salient_scores_str)
        
        # 3. 후보 생성
        candidates = generate_candidates_with_scores(
            document_text_list, salient_scores, self.ext, self.sel
        )
        
        # 4. 스코어링 및 선택
        best_summary = find_best_summary(
            document_full_text, candidates, self.model, self.tokenizer, self.device
        )
        
        return best_summary