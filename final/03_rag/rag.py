import os, random, pickle, re, json
import numpy as np
import pandas as pd
import torch, faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)


try:
    from llm import prompting_answer
except:
    prompting_answer = None


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


file_path = "/home/food/people/minju/data/식품안전정보DB-url 추가(2014~2024).xls"
xls = pd.ExcelFile(file_path)

dfs = []
for sheet in sorted(xls.sheet_names, key=int):
    df_tmp = pd.read_excel(file_path, sheet_name=sheet, usecols=["제목", "내용"])
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df = df[df['내용'].notna()].reset_index(drop=True)

def compose(title, content):
    return f"{str(title)} {str(content)}".strip()

df["검색텍스트"] = df.apply(lambda r: compose(r["제목"], r["내용"]), axis=1)
DATA = df["검색텍스트"].tolist()
print(f"[INFO] 문서 수: {len(DATA)}")




@dataclass
class SearchConfig:
    mode: str = "embedding"      # "embedding" / "bm25" / "hybrid_sum" / "hybrid_max"
    k: int = 10
    k_bm25: int = 5
    k_embed: int = 5
    bm25_weight: float = 0.6
    embed_weight: float = 0.4
    use_reranker: bool = True
    reranker_top_k: int = 10
    enable_llm: bool = False

CFG = SearchConfig()



try:
    with open("/home/food/final/food/00_data/03_rag/index/bm25.pkl","rb") as f:
        kiwi = pickle.load(f)
    print("[INFO] BM25 로드 완료")
except:
    kiwi = None
    print("[WARN] BM25 사용 불가")

# Embedding + FAISS
with open("/home/food/final/food/00_data/03_rag/index/labse.pkl","rb") as f:
    EMB = pickle.load(f)

faiss.normalize_L2(EMB)
index = faiss.IndexFlatIP(EMB.shape[1])
index.add(EMB)
embed_model = SentenceTransformer("sentence-transformers/LaBSE")
print("[INFO] FAISS + LaBSE 준비 완료")

# Re-ranker
try:
    tok_rerank = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
    model_rerank = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker").eval()
    print("[INFO] Re-ranker 로드 완료")
except:
    model_rerank = None
    print("[WARN] Re-ranker 사용 불가")

# LLM 
GEN = None
if CFG.enable_llm:
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    tokenizer_exa = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model_exa = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb, trust_remote_code=True)
    GEN = pipeline("text-generation", model=model_exa, tokenizer=tokenizer_exa, max_new_tokens=300)



def retrieve_embedding(query, k):
    q = embed_model.encode([query], normalize_embeddings=True)
    D,I = index.search(q, k)
    return [(DATA[i], float(D[0][r])) for r,i in enumerate(I[0])]

def retrieve_bm25(query, k):
    if kiwi is None: return []
    kiwi.k = k
    docs = kiwi.get_relevant_documents(query)
    return [(d.page_content, (k-r)) for r,d in enumerate(docs,1)]

def merge_sum(b, e, wb, we, k):
    score = {}
    for doc,s in b: score[doc]=score.get(doc,0)+wb*s
    for doc,s in e: score[doc]=score.get(doc,0)+we*s
    return sorted(score.items(), key=lambda x:x[1], reverse=True)[:k]

def merge_max(b, e, wb, we, k):
    score = {}
    for doc,s in b: score[doc]=max(score.get(doc,-1e9), wb*s)
    for doc,s in e: score[doc]=max(score.get(doc,-1e9), we*s)
    return sorted(score.items(), key=lambda x:x[1], reverse=True)[:k]

def rerank(query, docs, topk):
    if model_rerank is None: return docs
    pairs = [[query, d[0]] for d in docs]
    inputs = tok_rerank(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        scores = model_rerank(**inputs).logits.squeeze().tolist()
    reranked = sorted(zip([d[0] for d in docs], scores), key=lambda x:x[1], reverse=True)
    return reranked[:topk]

def retrieve(query, cfg):
    if cfg.mode=="embedding": docs = retrieve_embedding(query, cfg.k)
    elif cfg.mode=="bm25": docs = retrieve_bm25(query, cfg.k)
    elif cfg.mode=="hybrid_sum":
        docs = merge_sum(retrieve_bm25(query,cfg.k_bm25), retrieve_embedding(query,cfg.k_embed), cfg.bm25_weight, cfg.embed_weight, cfg.k)
    elif cfg.mode=="hybrid_max":
        docs = merge_max(retrieve_bm25(query,cfg.k_bm25), retrieve_embedding(query,cfg.k_embed), cfg.bm25_weight, cfg.embed_weight, cfg.k)
    else:
        raise ValueError(cfg.mode)

    if cfg.use_reranker: docs = rerank(query, docs, cfg.reranker_top_k)
    return docs



def run_and_save_from_json(input_json, cfg, out_csv):
    qa = json.load(open(input_json,encoding="utf-8"))
    rows = []
    for item in qa:
        q = item["query"]
        gold = item.get("answer","")
        docs = retrieve(q, cfg)
        top = docs[0][0] if docs else ""
        rows.append([item.get("id",""), q, gold, top])
    pd.DataFrame(rows, columns=["id","질문","정답","top1 문서"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[DONE]", out_csv)



# MAIN

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Default GPU 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--mode", type=str, default="embedding",
                        choices=["embedding", "bm25", "hybrid_sum", "hybrid_max"])

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--reranker_top_k", type=int, default=10)

    parser.add_argument("--use_reranker", type=str, default="True")
    parser.add_argument("--enable_llm", type=str, default="False")

    parser.add_argument("--gpus", type=str, default="0",
                        help='사용할 GPU (예: "0" 또는 "0,1")')

    args = parser.parse_args()

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    CFG.mode = args.mode
    CFG.k = args.k
    CFG.reranker_top_k = args.reranker_top_k

    CFG.use_reranker = (args.use_reranker.lower() == "true")
    CFG.enable_llm = (args.enable_llm.lower() == "true")

    run_and_save_from_json(args.input_json, CFG, args.output_csv)

