import itertools
from ..base import BaseExtractor
from ..cetd.cetd import CETDExtractor
from ..neuscraper.neuscraper import NeuScraperExtractor
from ..utils import *

class Method1Extractor(BaseExtractor):
    def __init__(self, config: dict = None):
        print("[Method 1] Initializing Method 1 Extractor...")

        if config is None:
            config = {}

        self.cetd_config = self.config.get("cetd_config", {})
        self.neu_config = self.config.get("neuscraper_config", {})

        self.neu_threshold = self.config.get("neu_threshold", 0.5)
        self.neu_config["threshold"] = self.neu_threshold
        
        super().__init__(config)

    def load_model(self):
        print("[Method 1] Loading child models (CETD & NeuScraper)...")
        self.cetd_extractor = CETDExtractor(config=self.cetd_config)
        self.neu_extractor = NeuScraperExtractor(config=self.neu_config)

    def extract(self, html_content: str) -> str:
        print("[Extractor] Running Method 1 (Parallel) logic...")
        
        # 1. CETD로 content 추출(soup의 문자열 형태로 반환)
        cetd_soup = self.cetd_extractor._run_cetd_algorithm(html_content)
        if not isinstance(cetd_soup, Tag):
            return "[Method 1 Fail: CETD Fail]"

        # 2. CETD 블럭 추출
        cetd_blocks = extract_block(cetd_soup)

        # 3. 각 CETD 블럭에서 텍스트 추출
        cetd_blocks_contents = extract_text_for_blocks(cetd_blocks)

        # 4. Neuscraper로 중요 리프 노드 추출
        neu_text_list = self.neu_extractor._run_neuscraper_extraction(html_content)

        # 5. 병렬 검증
        method1_blocks = parallel_cross_validation(cetd_blocks, cetd_blocks_contents, neu_text_list)

        # 6. 각 블럭의 table 태그를 텍스트로 반환
        method1_blocks_table2text = table2text_for_blocks(method1_blocks)

        # 7. 각 블럭에서 텍스트 추출
        method1_contents = list(itertools.chain.from_iterable(extract_text_for_blocks(method1_blocks_table2text)) if method1_blocks_table2text != [] else [])

        # 8. 전처리
        method1_texts = leaf_text_preprocessing(method1_contents)

        return "\n\n".join(method1_texts)
