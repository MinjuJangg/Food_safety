from openai import OpenAI
import pandas as pd
import json

df = pd.read_csv("~~~~~~~~~~.csv")       #########연도별 파일 입력##########
df = df[["제목", "내용"]].dropna().reset_index(drop=True)

client = OpenAI(api_key="API_KEY_HERE") 


def build_prompt(title, content):
    return f"""
아래 문서를 바탕으로 사람이 가장 먼저 궁금해할 질문 하나와 그에 대한 답변 하나를 생성하라.

문서:
제목: {title}
내용: {content}

출력 규칙:
1) 질문은 한 문장, 핵심만 묻기.
2) 답변은 반드시 문서에서 직접 확인 가능한 정보만 사용할 것.
3) 정보가 문서에 없으면 답변: "해당 문서에 없음"
4) '이 제품', '해당 제품' 같은 모호한 표현 금지. 반드시 대상 명시.
5) 출력은 정확히 아래 형식의 JSON 한 개만:

[
  {{"query": "질문", "answer": "답변"}}
]
"""

def make_qa(title, content, model="gpt-4o"):
    prompt = build_prompt(title, content)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "너는 문서 기반 QA 생성 모델이다."},
                {"role": "user", "content": prompt}
            ]
        )
        text = response.choices[0].message.content.strip()

        data = json.loads(text)

        # 유효성
        if isinstance(data, list) and len(data) == 1 and "query" in data[0] and "answer" in data[0]:
            return data[0]["query"], data[0]["answer"]
        else:
            return "생성 실패", "생성 실패"

    except:
        return "생성 실패", "생성 실패"

results = []
for idx, row in df.iterrows():
    q, a = make_qa(row["제목"], row["내용"])
    results.append({
        "id": idx,
        "title": row["제목"],
        "content": row["내용"],
        "query": q,
        "answer": a
    })

output_path = "qa_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[DONE] 생성 완료 → {output_path}")
