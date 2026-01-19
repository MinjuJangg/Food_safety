from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
    모든 Extractor의 기본 인터페이스 역할을 하는 추상 클래스
    """
    def __init__(self, config: dict=None):
        self.config = config or {}
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        """
        모델 로딩을 위한 추상 메소드
        """
        pass


    @abstractmethod
    def extract(self, html_content: str) -> str:
        """
        입력 html 텍스트에서 주요 텍스트를 추출하는 추상 메소드
        
        Args:
            html_content(str): 입력된 html 문자열
        
        return(str): 추출된 텍스트
        """
        pass

    