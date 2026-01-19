def user_prompt_maker(text):
    prompt = f"""
다음 리프 노드 텍스트 리스트를 보고, 핵심 내용을 요약해라.
리프 노드 텍스트 리스트:
{text}
요약문:
    """.strip()
    return prompt