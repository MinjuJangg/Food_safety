import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
from konlpy.tag import Okt

#######결과 파일 입력#####
csv_path = "~~~~~~~~~~~~~~.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")


preds = df["예상답변"].astype(str).tolist()
refs = df["정답"].astype(str).tolist()

# Retriever 평가 (MRR, Top-k)
target_found = (df['target_found'] == True).sum()
total = len(df)
mean_rank = df['target_rank'].mean()
topk_accuracy = target_found / total
mrr = (1 / df['target_rank']).fillna(0).mean()

print(f" Retriever 평가, meanrank")
print(mean_rank)
print(f"Top-k Accuracy: {topk_accuracy:.4f} ({target_found}/{total})")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

##########################################################################################

# Generator 평가 (QA 응답)
# Exact Match
def compute_exact_match(preds, refs):
    return np.mean([int(p.strip() == r.strip()) for p, r in zip(preds, refs)])

# Token-level F1 (형태소 기반)
okt = Okt()
def compute_token_f1(preds, refs):
    f1s = []
    for p, r in zip(preds, refs):
        p_tokens = okt.morphs(p)
        r_tokens = okt.morphs(r)
        common = set(p_tokens) & set(r_tokens)
        if not common:
            f1s.append(0.0)
            continue
        precision = len(common) / len(p_tokens)
        recall = len(common) / len(r_tokens)
        f1s.append(2 * precision * recall / (precision + recall))
    return np.mean(f1s)

# ROUGE-L
def compute_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
    return np.mean(scores)

# BLEU 
def compute_bleu(preds, refs):
    smooth = SmoothingFunction().method1
    scores = [sentence_bleu([r.split()], p.split(), weights=(1, 0, 0, 0), smoothing_function=smooth) for p, r in zip(preds, refs)]
    return np.mean(scores)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu4(preds, refs):
    smooth = SmoothingFunction().method1
    scores = [
        sentence_bleu(
            [r.split()], p.split(),
            weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4
            smoothing_function=smooth
        )
        for p, r in zip(preds, refs)
    ]
    return np.mean(scores)

from bert_score import score  # 

# BERTScore 
def compute_bertscore(preds, refs, model_name="xlm-roberta-base"):
    P, R, F1 = score(preds, refs, lang="ko", model_type=model_name, verbose=False)
    return F1.mean().item()


from nltk.translate.meteor_score import meteor_score

def compute_meteor(preds, refs):
    scores = [
        meteor_score(
            [okt.morphs(r)],   # ref는 list of token list
            okt.morphs(p)      # pred는 token list
        )
        for p, r in zip(preds, refs)
    ]
    return np.mean(scores)

# 결과 출력
print("\nGenerator 평가 (QA 응답)")
print(f"- Exact Match Score       : {compute_exact_match(preds, refs):.4f}")
print(f"- Token-level F1 Score    : {compute_token_f1(preds, refs):.4f}")
print(f"- ROUGE-L Score           : {compute_rouge_l(preds, refs):.4f}")
print(f"- BLEU (1-gram) Score     : {compute_bleu(preds, refs):.4f}")
print(f"- BLEU (4-gram) Score     : {compute_bleu4(preds, refs):.4f}")
print(f"- BERTScore (F1)          : {compute_bertscore(preds, refs):.4f}")
print(f"- METEOR      : {compute_meteor(preds, refs):.4f}")
