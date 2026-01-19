import os
import sys
import json
import math
import time
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# ========== 상대경로 설정 ==========
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[3]
DATA_DIR = ROOT / "00_data" / "02_classification_0919"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "valid.csv"
RUN_DIR = ROOT / "02_classification" / "run"
SUB_CKPT_DIR = RUN_DIR / "checkpoints" / "2_sub_pth"
LOG_DIR = RUN_DIR / "logs" / "2_sub_result"
SUB_CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 고정 라벨 스키마==========
MAIN_LABEL2ID: Dict[str, int] = {
    "물리적 위해요소": 0,
    "생물학적 위해요소": 1,
    "신규 위해요소": 2,
    "안전위생": 3,
    "영양건강": 4,
    "표시광고": 5,
    "화학적 위해요소": 6,
}
MAIN_ID2LABEL = {v: k for k, v in MAIN_LABEL2ID.items()}

SUB_LABELS: Dict[str, List[str]] = {
    "물리적 위해요소": [
        "기타", "방사선조사", "성상", "이물질"
    ],
    "생물학적 위해요소": [
        "곰팡이독소", "기생충", "기타", "동식물질병", "미생물", "생물독소"
    ],
    "신규 위해요소": [
        "GMO(LMO)", "기능성소재", "기타", "나노", "멜라민", "방사능", "복제동물", "신규식품"
    ],
    "안전위생": [
        "기타", "안전", "위생"
    ],
    "영양건강": [
        "건강", "영양", "영양성분"
    ],
    "표시광고": [
        "GMO", "광고(허위, 과대 등)", "기간(유통기한, 제조일자 등)", "기타",
        "안내(경고, 주의사항 등)", "알레르기", "영양성분", "원료·성분·함량",
        "원산지", "유기식품", "제품명"
    ],
    "화학적 위해요소": [
        "기구용기포장유래물질", "기타", "동물용의약품", "식품첨가물",
        "의약품성분", "잔류농약", "중금속"
    ],
}

def get_sub_label2id(main_label: str) -> Dict[str, int]:
    names = SUB_LABELS[main_label]
    return {name: i for i, name in enumerate(names)}

def get_sub_id2label(main_label: str) -> Dict[int, str]:
    names = SUB_LABELS[main_label]
    return {i: name for i, name in enumerate(names)}

# ========== 설정값 ==========
TEXT_COL_CANDIDATES = ["제목_내용", "text"]
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 10
LR = 2e-5
PATIENCE = 3
DEVICE = "cuda:0"

# ========== 유틸 ==========
def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"텍스트 컬럼을 찾지 못했습니다. 후보: {TEXT_COL_CANDIDATES}")

class TextDS(Dataset):
    def __init__(self, xs, ys, tok, maxlen: int):
        self.xs = list(xs)
        self.ys = list(ys)
        self.tok = tok
        self.maxlen = maxlen

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i: int):
        enc = self.tok(
            str(self.xs[i]),
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.ys[i]), dtype=torch.long),
        }
        return item

@torch.no_grad()
def _eval_epoch(model, dl, device):
    model.eval()
    tot_loss, tot_acc, n, steps = 0.0, 0, 0, 0
    for b in dl:
        ids = b["input_ids"].to(device)
        att = b["attention_mask"].to(device)
        y = b["labels"].to(device)
        out = model(ids, attention_mask=att, labels=y)
        loss, logits = out.loss, out.logits
        pred = logits.argmax(1)
        tot_loss += float(loss.item())
        tot_acc += int((pred == y).sum().item())
        n += y.size(0)
        steps += 1
    return (tot_loss / max(1, steps), tot_acc / max(1, n))

def _train_epoch(model, dl, device, opt):
    model.train()
    tot_loss, tot_acc, n, steps = 0.0, 0, 0, 0
    for b in dl:
        opt.zero_grad()
        ids = b["input_ids"].to(device)
        att = b["attention_mask"].to(device)
        y = b["labels"].to(device)
        out = model(ids, attention_mask=att, labels=y)
        loss, logits = out.loss, out.logits
        loss.backward()
        opt.step()

        pred = logits.argmax(1)
        tot_loss += float(loss.item())
        tot_acc += int((pred == y).sum().item())
        n += y.size(0)
        steps += 1
    return (tot_loss / max(1, steps), tot_acc / max(1, n))

def _safe_slug(s: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in s])

def train_one_sub(main_label: str):
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    text_col = pick_text_col(train_df)

    train_df = train_df[train_df["대분류"] == main_label].reset_index(drop=True)
    val_df = val_df[val_df["대분류"] == main_label].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        print(f"[WARN] '{main_label}' 학습 데이터가 부족합니다. (train={len(train_df)}, val={len(val_df)})")
        return

    label2id = get_sub_label2id(main_label)
    id2label = get_sub_id2label(main_label)

    # 라벨 매핑 검증
    y_train = train_df["중분류"].map(label2id)
    y_val = val_df["중분류"].map(label2id)
    if y_train.isna().any() or y_val.isna().any():
        unknown = set(train_df["중분류"][y_train.isna()].unique().tolist() +
                      val_df["중분류"][y_val.isna()].unique().tolist())
        raise KeyError(f"[{main_label}] 정의되지 않은 중분류 라벨 존재: {unknown}")

    X_train = train_df[text_col].astype(str).tolist()
    X_val = val_df[text_col].astype(str).tolist()
    y_train = y_train.astype(int).tolist()
    y_val = y_val.astype(int).tolist()

    tok = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    tr_loader = DataLoader(TextDS(X_train, y_train, tok, MAX_LENGTH),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TextDS(X_val, y_val, tok, MAX_LENGTH),
                           batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id={k: int(v) for k, v in label2id.items()},
    ).to(device)

    opt = AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience = 0

    tag = f"sub{MAIN_LABEL2ID[main_label]+1}_{_safe_slug(main_label)}"
    pth_path = SUB_CKPT_DIR / f"{tag}_best_model_xlm_roberta.pth"
    json_path = SUB_CKPT_DIR / f"{tag}_labels.json"

    print(f"[TRAIN] {main_label} ({tag}) | train={len(X_train)} val={len(X_val)}")
    for ep in range(1, EPOCHS + 1):
        tr_l, tr_a = _train_epoch(model, tr_loader, device, opt)
        va_l, va_a = _eval_epoch(model, va_loader, device)
        print(f"[{tag}] Epoch {ep:02d} | train {tr_l:.4f}/{tr_a:.4f} | val {va_l:.4f}/{va_a:.4f}")

        if va_l < best_val_loss:
            best_val_loss = va_l
            patience = 0
            torch.save(model.state_dict(), pth_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[{tag}] Early stopping.")
                break

    # 라벨 메타 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "main_label": main_label,
                "label2id": {k: int(v) for k, v in label2id.items()},
                "id2label": {int(k): v for k, v in id2label.items()},
                "text_col": text_col,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SAVE] model: {pth_path}")
    print(f"[SAVE] labels: {json_path}")

def main():
    import argparse    
    global DEVICE, EPOCHS, BATCH_SIZE, LR

    parser = argparse.ArgumentParser(description="중분류 학습(공용 스크립트)")
    parser.add_argument("--only_main", nargs="*", default=None,
                        help="지정 시 해당 대분류만 학습. 예: --only_main 표시광고 화학적 위해요소")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    DEVICE = args.device
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    set_seed(args.seed)

    mains = list(MAIN_LABEL2ID.keys())
    if args.only_main:
        mains = args.only_main

    t0 = time.time()
    for m in mains:
        if m not in MAIN_LABEL2ID:
            print(f"[WARN] 알 수 없는 대분류: {m} (건너뜀)")
            continue
        train_one_sub(m)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"[DONE] 총 소요: {elapsed:.2f}s ({elapsed/60:.2f}분)")

if __name__ == "__main__":
    main()