from ..base import BaseSummarizer
from .src.utils import *
import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


class ExaoneSummarizer(BaseSummarizer):
    def __init__(self, config: dict=None):
        self.hydra_config_name = config.get("bertsum_config_name")
        self.hydra_config_path = config.get("bertsum_config_path", "./config/")
        
        if not self.hydra_config_name:
            raise ValueError("BertSumSummarizer requires 'bertsum_config_name' in its configuration")
        
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialze(config_path = self.hydra_config_path, version_base=None)
        
        self.cfg = hydra.compose(config_name=self.hydra_config_name)
        
        super().__init__(config=OmegaConf.to_object(self.cfg))
    
    def load_model(self):
        print(f"Initializing EXAONE Summarizer (Hydra config: {self.hydra_config_name})...")
        
        model_name = 'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct'
        device = "cuda:0"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remore_code=True
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        self.prompt = """
        너는 식품안전정보 웹사이트의 내용을 읽고, 위험 요소를 중심으로 핵심 내용을 요약하는 전문 어시스턴트다.
        사용자가 웹사이트 HTML 코드의 중요 콘텐츠 블럭의 리프 노드 텍스트 리스트를 제공하면, 너는 다음 규칙을 반드시 준수하여 결과를 하나의 단락으로 요약해야 한다.

        1. 목록이나 제목을 절대 사용하지 않는다.
        2. 모든 문장은 완전한 형태로 작성한다.
        3. 요약 내용에는 **핵심 정보, 위험 요소, 필요한 조치**를 반드시 포함하며 간결함을 유지한다.
        4. 식품안전정보와 관련이 없는 불필요한 내용은 포함하지 않는다.
        5. 반드시 주어진 리프 노드 텍스트의 내용을 기반으로 요약하며, 배경지식을 사용하지 않는다. **만약 제공된 리프 노드 텍스트 리스트만으로는 충분한 정보를 추출하여 상세한 요약을 작성하기 어려운 경우, '요약 불가' 텍스트만을 출력**한다.
        6. 결과는 항상 한국어로 작성한다.
        7. "요약해 드리겠습니다."와 같은 도입부나 부가적인 설명 없이, 오직 최종 요약문 단락만 출력한다.

        예시 요약문은 다음과 같다.
        예시:
        미국 일간지 휴스턴 크로니클(Houston Chronicle)이 텍사스에 본사를 둔 아이스크림 제조업체 ‘블루벨(Blue Bell)’이 연방 당국에 리스테리아 식중독 사건으로 적용 중인 예방조치를 완화하고 경쟁사들이 따르는 일반 절차로의 복원을 요청하고 있다고 보도함.연방 정보공개요청에 입수한 서류를 검토한 해당 일간보도에 따르면, 블루벨은 연방 식품의약품청 요건을 충족하고 향후 식중독 예방은 물론 수익개선에 이바지하기 위한 시험법 구축을 위해 실험실과 여러 달 협조해왔다고 함.블루벨은 지난해 3명의 사망자가 발생한 캔자스 주를 포함, 4개 주에서 발생한 리스테리아 감염사례 10건과 관련해 진행한 회수 이후, 몇 개월간 영업을 중단했던 바 있음.
        """.strip()
    
    def summarize(self, text_content: str) -> str:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_prompt_maker(str(text_content))}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        input_ids = self.tokenizer([input_text], return_tensors="pt", max_length=3072, truncation=True).to(self.model.device)

        output_ids = self.model.generate(
            **input_ids,
            eos_token_ids=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False
        )        
        
        output=[output_id[len(input_id):] for input_id, output_id in zip(input_ids,input_ids, output_ids)]
        
        response = self.tokenizer.batch_decode(output_skip_special_tokens=True)[0]
        
        return response.strip()