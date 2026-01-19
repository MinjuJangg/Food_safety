from .utils.clean import *
from .utils.content_extraction import *
from .utils.strip_non_content_tags import *
from ..base import BaseExtractor

class CETDExtractor(BaseExtractor):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.ratio = self.config.get("threshold", 0.5)
        self.final_content = []
    
    def load_model(self):
        print("Initializing CETD (Text-Density based Extractor)...")
        pass
    
    def _run_cetd_algorithm(self, html_content: str) -> Union[Tag, str]:
        """
        CETD 알고리즘을 실행하고 마킹된 'root' soup 객체를 반환
        Method 1이 이 메서드 호출
        """
        print("[CETD] Running core algorithm...")
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            stripNonContentTags(soup) # 헬퍼 함수
            
            body = soup.find("body")
            root = body if body else soup
            
            countChar(root)
            countTag(root)
            countLinkChar(root)
            countLinkTag(root)
            
            char_num = float(root.get('cetd_char_num', 0))
            if char_num == 0:
                return "[CETD Fail] No characters"
                
            linkchar_num = float(root.get('cetd_linkchar_num', 0))
            ratio = linkchar_num / char_num
            
            computeTextDensity(root, ratio)
            computeDensitySum(root, ratio)
            
            max_density_sum = findMaxDensitySum(root)
            if max_density_sum <= eps:
                return "[CETD Fail] Max density sum is zero"
                
            setMark(root, 0)
            threshold = getThreshold(root, max_density_sum)
            markContent(root, threshold)
            
            return root
            
        except Exception as e:
            print(f"[CETD Error] Algorithm failed: {e}")
            return f"[CETD Fail: {e}]"

    def _clean_and_get_text(self, root: Tag) -> str:
        """
        마킹된 'root'를 받아 최종 텍스트로 변환
        """
        if not isinstance(root, Tag):
            return str(root) # 오류 메시지 반환
            
        cleanTreeByMark(root) # 헬퍼 함수
        return root.get_text(separator='\n\n', strip=False)
    
    def extract(self, html_content: str) -> str:
        root = self._run_cetd_algorithm(html_content)
        return self._clean_and_get_text(root)