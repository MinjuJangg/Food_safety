from ..base import BaseExtractor
from ..cetd.cetd import CETDExtractor
from ..neuscraper.neuscraper import NeuScraperExtractor
from ..utils import *

class Method2Extractor(BaseExtractor):
    def __init__(self, config: dict = None):
        print("[Method 2] Initializing Method 2 Extractor...")

        if config is None:
            config = {}

        self.cetd_config = self.config.get("cetd_config", {})
        self.neu_config = self.config.get("neuscraper_config", {})

        self.neu_threshold = self.config.get("neu_threshold", 0.5)
        self.neu_config["threshold"] = self.neu_threshold

        super().__init__(config)
    
    def load_model(self):
        print("[Method 2] Loading child models (CETD & NeuScraper)...")
        self.cetd_extractor = CETDExtractor(config=self.cetd_config)
        self.neu_extractor = NeuScraperExtractor(config=self.neu_config)
    
    def extract(self, html_content: str) -> str:
        print("[Extractor] Running Method 2 (Serial) logic...")

        cetd_soup = self.cetd_extractor._run_cetd_algorithm(html_content)
        if not isinstance(cetd_soup, Tag):
            return "[Method 2 Fail: CETD Failed]"

        cetd_html_str = str(cetd_soup)
        cetd_soup_table2text = table2text(cetd_html_str)

        neu_text_list = self.neu_extractor._run_neuscraper_extraction(cetd_soup_table2text)

        method2_texts = leaf_text_preprocessing(neu_text_list)

        return "\n\n".join(method2_texts)