from abc import ABC, abstractmethod

class BaseSummarizer(ABC):
    """
    모든 Summarizer의 기본 인터페이스 역할을 하는 추상 클래스
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
    def summarize(self, text_content: str) -> str:
        """
        입력 텍스트를 요약하는 추상 메소드
        
        Args:
            text_content(str): 추출된 텍스트
        
        return(str): 요약된 텍스트
        """
        pass
    