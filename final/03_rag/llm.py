from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import os
import random
import numpy as np

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_llm(model_path):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    generate_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype="auto"
    )
    
    print("모델 및 토크나이저 로드 완료!")
    return generate_pipeline, tokenizer

def prompting_answer(question: str, context: str, rag: bool = True) -> list:
    """
    RAG 기반 또는 LLM 단독 평가용 메시지 생성.
    """
    if rag:
        # RAG 기반 정답 생성 평가용 프롬프트
        messages = [
            {"role": "system", "content": "당신은 유용한 AI 도우미입니다. 당신은 항상 한글로 대답해야 하며, 고유명사 외에는 외국어를 사용해선 안됩니다."},
            {"role": "user", "content": "안녕하세요, 당신의 역할은 내 질문에 대해 주어진 정보를 바탕으로 한글로 대답하는 것입니다. 답변할 때에는 주어진 조건을 충족해야 합니다."},
            {"role": "assistant", "content": "알겠습니다. 질문과 정보를 알려주시면 최선을 다해 한글로 답변드리겠습니다."},
            {"role": "user", "content": f'''
    **정보**:
    {context}

    **질문**:
    {question}

    **조건**:
    1. 반드시 위 문서에서 확인 가능한 내용만으로 답변할 것.
    2. 문서에 명확한 근거가 없으면 "해당 문서에 없음"으로 답변할 것.
    3. 질문에 정확히 일치하는 정보만 간결하게 답변할 것.
    4. 불필요한 배경 설명 없이, 답변만 한 문장으로 작성할 것.
    5. 표현은 사람스러운 자연어로 하되, 문장 끝맺음은 반드시 마침표로 할 것.
    '''}
        ]
    else:
        # 일반 LLM 질문-답변 프롬프트 (컨텍스트 없이)
        messages = [
            {"role": "system", "content": "당신은 유용한 AI 도우미입니다. 당신은 항상 한글로 대답해야 하며, 고유명사 외에는 외국어를 사용해선 안됩니다."},
            {"role": "user", "content": "안녕하세요, 당신의 역할은 내 질문에 대해 한글로 대답하는 것입니다. 답변할 때에는 주어진 조건을 충족해야 합니다."},
            {"role": "assistant", "content": "알겠습니다. 질문을 알려주시면 최선을 다해 한글로 답변드리겠습니다."},
            {"role": "user", "content": f'''
            **질문**:
            {question}

            **조건**:
            1. 가능한 정보에 근거하여 간결하고 정확하게 답변할 것.
            2. 질문에 완전히 부합하는 정보만 포함할 것.
            3. 불필요한 설명 없이 한 문장으로 응답할 것.
            '''}
        ]
    
    return messages
