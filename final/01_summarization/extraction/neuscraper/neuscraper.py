import os
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
import requests
import torch
import warnings
import argparse

from .src.builder import build
from .src.extractor import ContentExtractionDeepModel, inference, save_predictions, get_text_spans_from_nodes
from .src.arguments import create_parser

from ..base import BaseExtractor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`resume_download` is deprecated"
)

def init_model(device: str):
    """
    NeuScraper 모델 초기화
    """

    parser = create_parser()
    args, _ = parser.parse_known_args([])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_path = os.path.join(current_dir, "neuscraper-v1-clueweb", "training_state_checkpoint.tar")

    if not os.path.exists(args.model_path):
        print(f"[NeuScraper Error] 모델 파일을 찾을 수 없습니다: {args.model_path}")
        print("  > 'neuscraper-v1-clueweb' 폴더가 'extraction' 폴더 내에 있는지 확인하세요.")
        raise FileNotFoundError(args.model_path)

    args.device = device if torch.cuda.is_available() else 'cpu'
    print(f"[NeuScraper] 모델을 {args.device} 장치로 로드합니다...")
    model = ContentExtractionDeepModel(args)
    return model, args

def neuscraper_extraction(html, url, threshold, model, args):
    """
    NeuScraper 추론 로직
    """
    if html in ["Fail: no body tag found", "Fail: something is wrong"]:
        return [html]

    try:
        # original: threshold별 콘텐츠 추출
        html_content = html.encode('utf-8')
        text_nodes_df, data = build(url, html_content)
    
        pred_nodes = inference(args, model, data)
        pred_nodes_df = save_predictions(pred_nodes, threshold)

        pred_df = get_text_spans_from_nodes(text_nodes_df, pred_nodes_df).dropna().sort_values(['TextNodeId'], ascending=[False])
        pred_df = pred_df.groupby(['Url', 'Task'], as_index=False).agg({'Text': list})

        text_list = pred_df['Text'].iloc[0]

        if len(text_list) == 0:
            return ["Check: no content extracted"]
        else:
            return text_list
        
    except Exception as e:
        print(f"[NeuScraper Error] Extraction failed during process: {e}")
        import traceback
        traceback.print_exc()
        return ["Fail: something is wrong"]

class NeuScraperExtractor(BaseExtractor):
    def __init__(self, config: dict=None):
        super().__init__(config)
        self.device = self.config.get("device", "auto")
        self.threshold = self.config.get("threshold", 0.5)
        self.url_placeholder = self.config.get("url_placeholder", "http://example.com")

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """
        NeuScraper 모델 로드
        init_model 헬퍼 함수 호출
        """

        print("Initializing NeuScraper (Deep Learning Extractor)...")
        self.model, self.args = init_model(self.device)

    def _run_neuscraper_extraction(self, html_content: str) -> list[str]:
        """
        NeuScraper 추론을 실행하고 '텍스트 리스트'를 반환
        Method 1이 이 메서드 호출
        """

        print("[NeuScraper] Running core inference...")
        return neuscraper_extraction( # 헬퍼 함수
            html=html_content,
            url=self.url_placeholder,
            threshold=self.threshold,
            model=self.model,
            args=self.args
        )
    
    def extract(self, html_content: str) -> str:
        text_list = self._run_neuscraper_extraction(html_content)

        return "\n\n".join(text_list)