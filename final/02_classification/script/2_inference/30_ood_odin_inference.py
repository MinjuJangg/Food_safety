# (ODIN 방법론은 별도의 추가 학습이 필요 없어서, Train 코드가 없음)

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
from pathlib import Path
HERE = Path(__file__).resolve().parent
def R(p: str) -> str:    
    return str((HERE / p).resolve())

# --------------------
# Configs
# --------------------
TXT_COL_CANDIDATES = ["제목_내용", "text", "content", "제목", "내용"]
LBL_COL_CANDIDATES = ["대분류", "label", "labels"]

ID_TO_TEXT = {
    0: "물리적 위해요소",
    1: "생물학적 위해요소",
    2: "신규 위해요소",
    3: "안전위생",
    4: "영양건강",
    5: "표시광고",
    6: "화학적 위해요소",
}
TEXT_TO_ID = {v: k for k, v in ID_TO_TEXT.items()}

# --------------------
# Utils
# --------------------
def _pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    df["__row_id__"] = np.arange(len(df))
    return df

def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _normalize_label_id(val) -> int:    
    if isinstance(val, (np.integer, int)):
        return int(val)
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    if s in TEXT_TO_ID:
        return TEXT_TO_ID[s]
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"라벨을 정수 id로 변환할 수 없습니다: {val!r}")

def _id_to_text(x: int) -> str:
    return ID_TO_TEXT.get(int(x), str(x))

# --------------------
# Dataset
# --------------------
class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok, txt_col: str, lbl_col: Optional[str], max_len: int = 512):
        self.df = df
        self.tok = tok
        self.txt_col = txt_col
        self.lbl_col = lbl_col
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = str(row[self.txt_col])
        enc = self.tok(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["row_id"] = int(row["__row_id__"])
        item["raw_text"] = text
        if self.lbl_col is not None:
            item["labels"] = torch.tensor(_normalize_label_id(row[self.lbl_col]), dtype=torch.long)
        return item

# --------------------
# ODIN Score
# --------------------
@torch.no_grad()
def _softmax_max(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values

def odin_scores_for_batch(
    model: AutoModelForSequenceClassification,
    batch: dict,
    temperature: float,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      base_msp: [B] baseline MSP
      odin_msp: [B] MSP after T-scaling + embedding perturbation
      pred_ids: [B] predicted IDs
    """
    model.eval()
    outputs = model(**{k: v for k, v in batch.items() if k not in ("labels","row_id","raw_text")})
    logits = outputs.logits
    base_msp = _softmax_max(logits)
    pseudo = logits.argmax(dim=-1)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(input_ids).detach()
    emb.requires_grad_(True)
    out_T = model(inputs_embeds=emb, attention_mask=attention_mask)
    logits_T = out_T.logits / temperature

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.detach_(); p.grad.zero_()

    loss = F.cross_entropy(logits_T, pseudo)
    loss.backward()
    grad = emb.grad.detach()
    pert = -epsilon * torch.sign(grad)
    emb_pert = emb + pert
    out_adv = model(inputs_embeds=emb_pert, attention_mask=attention_mask)
    logits_adv = out_adv.logits / temperature
    odin_msp = _softmax_max(logits_adv)
    pred_ids = logits_adv.argmax(dim=-1)

    return base_msp.detach(), odin_msp.detach(), pred_ids.detach()

def collect_scores_and_preds(
    loader: DataLoader,
    model,
    device,
    T: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
    base_all, odin_all, pred_all = [], [], []
    row_ids, texts = [], []
    model.eval()
    for batch in loader:
        row_ids.extend(batch["row_id"].tolist())
        texts.extend(batch["raw_text"])
        batch = _to_device(batch, device)
        b, o, p = odin_scores_for_batch(model, batch, T, eps)
        base_all.append(b.cpu().numpy())
        odin_all.append(o.cpu().numpy())
        pred_all.append(p.cpu().numpy())
    return (np.array(row_ids),
            texts,
            np.concatenate(base_all),
            np.concatenate(odin_all),
            np.concatenate(pred_all))

# ------------------------------
# Metrics & Thresholding
# ------------------------------
def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))

def thr_at_tpr(id_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    q = 1.0 - target_tpr
    return float(np.quantile(id_scores, q))

def fpr_at_thr(ood_scores: np.ndarray, thr: float) -> float:
    return float((ood_scores >= thr).mean())

def tpr_at_thr(id_scores: np.ndarray, thr: float) -> float:
    return float((id_scores >= thr).mean())

# ------------------------------
# CSV Writers for Misdetections
# ------------------------------
def save_misdetections_csv(
    out_dir: str,
    tag: str,
    thr: float,
    T: float,
    eps: float,
    
    # ID
    id_row_ids: np.ndarray,
    id_texts: List[str],
    id_scores: np.ndarray,
    id_true_lbls: Optional[np.ndarray],
    id_pred_ids: np.ndarray,
    
    # OOD
    ood_row_ids: np.ndarray,
    ood_texts: List[str],
    ood_scores: np.ndarray,
    ood_pred_ids: np.ndarray,
    lbl_col_present: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    # OOD → ID (False Positive)
    fp_mask = ood_scores >= thr
    if fp_mask.any():
        df_fp = pd.DataFrame({
            "row_id": ood_row_ids[fp_mask],
            "odin_score": ood_scores[fp_mask],
            "thr95": thr,
            "T": T,
            "eps": eps,
            "실제_대분류": ["OOD"] * int(fp_mask.sum()),
            "예측한_대분류": [_id_to_text(int(x)) for x in ood_pred_ids[fp_mask]],
            "제목_내용": [ood_texts[i] for i, m in enumerate(fp_mask) if m],
        })
    else:
        df_fp = pd.DataFrame(columns=["row_id","odin_score","thr95","T","eps","실제_대분류","예측한_대분류","제목_내용"])
    df_fp.sort_values("odin_score", ascending=False, inplace=True)
    df_fp.to_csv(os.path.join(out_dir, f"mispredict_ood_to_id_{tag}.csv"), index=False)

    # ID → OOD (False Negative)
    fn_mask = id_scores < thr
    if fn_mask.any():
        if lbl_col_present and id_true_lbls is not None:
            true_lbl_texts = [_id_to_text(int(v)) for v in id_true_lbls[fn_mask]]
        else:
            true_lbl_texts = ["(ID)"] * int(fn_mask.sum())
        df_fn = pd.DataFrame({
            "row_id": id_row_ids[fn_mask],
            "odin_score": id_scores[fn_mask],
            "thr95": thr,
            "T": T,
            "eps": eps,
            "실제_대분류": true_lbl_texts,
            "예측한_대분류": ["OOD"] * int(fn_mask.sum()),
            "제목_내용": [id_texts[i] for i, m in enumerate(fn_mask) if m],
        })
    else:
        df_fn = pd.DataFrame(columns=["row_id","odin_score","thr95","T","eps","실제_대분류","예측한_대분류","제목_내용"])
    df_fn.sort_values("odin_score", ascending=True, inplace=True)
    df_fn.to_csv(os.path.join(out_dir, f"mispredict_id_to_ood_{tag}.csv"), index=False)

# --------------------
# Main
# --------------------
def main():
    t_all = time.time()
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--id_train", default="../../../00_data/02_classification_0808/in_distribution_train.csv")
    ap.add_argument("--id_val",   default="../../../00_data/02_classification_0808/in_distribution_valid.csv")
    ap.add_argument("--id_test",  default="../../../00_data/02_classification_0808/in_distribution_test.csv")
    ap.add_argument("--ood_val",  default="../../../00_data/02_classification_0808/out_of_distribution_valid.csv")
    ap.add_argument("--ood_test", default="../../../00_data/02_classification_0808/out_of_distribution_test.csv")
    ap.add_argument("--pretrained", default="xlm-roberta-base")
    ap.add_argument("--ckpt",       default="../../run/checkpoints/1_main_pth/main_best.pth")    
    ap.add_argument("--num_labels", type=int, default=7)
    ap.add_argument("--max_len",    type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--T_list",   default="100,500,1000")
    ap.add_argument("--eps_list", default="0.0005,0.0015,0.003")
    ap.add_argument("--results_csv",     default="../../run/logs/3_ood_result/1_odin/result_odin.csv")
    ap.add_argument("--mistakes_outdir", default="../../run/logs/3_ood_result/1_odin")
    ap.add_argument("--fp16", action="store_true", help="use half precision to reduce GPU memory")

    args = ap.parse_args()
    device = torch.device(args.device)

    # ---------- Load CSVs ----------
    id_val_df   = _load_csv(R(args.id_val))
    id_test_df  = _load_csv(R(args.id_test))
    ood_val_df  = _load_csv(R(args.ood_val))
    ood_test_df = _load_csv(R(args.ood_test))

    txt_col = _pick_col(id_val_df, TXT_COL_CANDIDATES)
    if txt_col is None:
        raise ValueError(f"텍스트 컬럼을 찾을 수 없습니다. candidates={TXT_COL_CANDIDATES}")
    lbl_col = _pick_col(id_val_df, LBL_COL_CANDIDATES)

    tok = AutoTokenizer.from_pretrained(args.pretrained)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=args.num_labels
    )

    # ---------- Load finetuned weights ----------
    ckpt_path = R(args.ckpt)
    if args.ckpt and os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            new_sd[nk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"[load ckpt] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"[warn] ckpt not found at {ckpt_path}. Using base {args.pretrained} weights.")

    if args.fp16 and device.type == "cuda":
        model.half()
    model.to(device).eval()

    # ---------- DataLoaders ----------
    def make_loader(df, with_label: bool):
        ds = TextClsDataset(
            df, tok, txt_col,
            (lbl_col if with_label and (lbl_col in df.columns) else None),
            args.max_len
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    id_val_loader   = make_loader(id_val_df, True)
    id_test_loader  = make_loader(id_test_df, True)
    ood_val_loader  = make_loader(ood_val_df, False)
    ood_test_loader = make_loader(ood_test_df, False)

    # ---------- Baseline MSP (no T, no eps) ----------
    def collect_msp(loader):
        all_scores = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, device)
                outputs = model(**{k: v for k, v in batch.items() if k not in ("labels","row_id","raw_text")})
                logits = outputs.logits
                all_scores.append(_softmax_max(logits).cpu().numpy())
        return np.concatenate(all_scores)

    base_id_val   = collect_msp(id_val_loader)
    base_ood_val  = collect_msp(ood_val_loader)
    base_thr95    = thr_at_tpr(base_id_val, 0.95)
    base_fpr_val  = fpr_at_thr(base_ood_val, base_thr95)
    base_auroc_val = auroc(base_id_val, base_ood_val)
    base_id_test   = collect_msp(id_test_loader)
    base_ood_test  = collect_msp(ood_test_loader)
    base_fpr_test  = fpr_at_thr(base_ood_test, base_thr95)
    base_auroc_test = auroc(base_id_test, base_ood_test)

    torch.cuda.empty_cache()

    # ---------- ODIN Grid (VAL tuning) ----------
    T_list   = _parse_float_list(args.T_list)
    eps_list = _parse_float_list(args.eps_list)
    rows = []
    cache_scores_val = {}
    cache_scores_test = {}

    for T in T_list:
        for eps in eps_list:
            # VAL
            id_row_ids, id_texts, base_in, odin_in, pred_in = collect_scores_and_preds(id_val_loader, model, device, T, eps)
            o_row_ids, o_texts, base_out, odin_out, pred_out = collect_scores_and_preds(ood_val_loader, model, device, T, eps)
            cache_scores_val[(T,eps)] = (odin_in, odin_out)
            thr95    = thr_at_tpr(odin_in, 0.95)
            fpr_val  = fpr_at_thr(odin_out, thr95)
            tpr_val  = tpr_at_thr(odin_in, thr95)
            auroc_val = auroc(odin_in, odin_out)

            # TEST (same δ from VAL)
            if (T,eps) not in cache_scores_test:
                id_row_ids_t, id_texts_t, base_in_t, odin_in_t, pred_in_t = collect_scores_and_preds(id_test_loader, model, device, T, eps)
                o_row_ids_t, o_texts_t, base_out_t, odin_out_t, pred_out_t = collect_scores_and_preds(ood_test_loader, model, device, T, eps)
                cache_scores_test[(T,eps)] = (odin_in_t, odin_out_t, pred_in_t, pred_out_t,
                                              id_row_ids_t, id_texts_t, o_row_ids_t, o_texts_t)

            odin_in_t, odin_out_t, pred_in_t, pred_out_t, *_ = cache_scores_test[(T,eps)]
            fpr_test  = fpr_at_thr(odin_out_t, thr95)
            tpr_test  = tpr_at_thr(odin_in_t, thr95)
            auroc_test = auroc(odin_in_t, odin_out_t)

            rows.append({
                "T": T, "eps": eps, "Thr95(VAL)": thr95,
                "VAL_AUROC": auroc_val, "VAL_TPR@Thr95": tpr_val, "VAL_FPR@TPR95": fpr_val,
                "TEST_AUROC": auroc_test, "TEST_TPR@Thr95": tpr_test, "TEST_FPR@TPR95": fpr_test,
            })

    grid_df = pd.DataFrame(rows).sort_values(["VAL_FPR@TPR95", "VAL_AUROC"], ascending=[True, False])
    results_csv_path = R(args.results_csv)
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    grid_df.to_csv(results_csv_path, index=False)

    # ---------- Pick best by VAL FPR ----------
    best = grid_df.iloc[0]
    T_best   = float(best["T"])
    eps_best = float(best["eps"])
    thr_best = float(best["Thr95(VAL)"])
    (odin_in_t, odin_out_t, pred_in_t, pred_out_t,
     id_row_ids_t, id_texts_t, o_row_ids_t, o_texts_t) = cache_scores_test[(T_best, eps_best)]

    if ("대분류" in id_test_df.columns) or (lbl_col and lbl_col in id_test_df.columns):
        col = "대분류" if "대분류" in id_test_df.columns else lbl_col
        id_true_lbls = np.array([_normalize_label_id(v) for v in id_test_df[col].values], dtype=np.int64)
    else:
        id_true_lbls = None

    mistakes_dir = R(args.mistakes_outdir)
    save_misdetections_csv(
        out_dir=mistakes_dir,
        tag=f"T{T_best:g}_eps{eps_best:g}",
        thr=thr_best, T=T_best, eps=eps_best,
        id_row_ids=id_row_ids_t, id_texts=id_texts_t,
        id_scores=odin_in_t, id_true_lbls=id_true_lbls, id_pred_ids=pred_in_t,
        ood_row_ids=o_row_ids_t, ood_texts=o_texts_t,
        ood_scores=odin_out_t, ood_pred_ids=pred_out_t,
        lbl_col_present=(id_true_lbls is not None)
    )

    
    def pct(x): return f"{100*x:.2f}%"
    print("\n== Summary ==")
    print(f"Baseline (δ from ID_VAL) | VAL: AUROC={base_auroc_val:.4f}, FPR@TPR95={pct(base_fpr_val)}, Thr95={base_thr95:.6f}")
    print(f"                         | TEST: AUROC={base_auroc_test:.4f}, FPR@TPR95={pct(base_fpr_test)}")
    print("\nTop-5 ODIN settings (sorted by VAL FPR@TPR95, tie→VAL AUROC desc):")
    for _, r in grid_df.head(5).iterrows():
        print(f"  T={r['T']:g}, eps={r['eps']:g} | "
              f"VAL: AUROC={r['VAL_AUROC']:.4f}, FPR@TPR95={pct(r['VAL_FPR@TPR95'])}, Thr95={r['Thr95(VAL)']:.6f} | "
              f"TEST: AUROC={r['TEST_AUROC']:.4f}, FPR@TPR95={pct(r['TEST_FPR@TPR95'])}")
    print("\n== Best by VAL FPR@TPR95 (applied to TEST) ==")
    print(f"T={T_best:g}, eps={eps_best:g} | Thr95={thr_best:.6f} | "
          f"TEST: AUROC={best['TEST_AUROC']:.4f}, FPR@TPR95={pct(best['TEST_FPR@TPR95'])}")
    print(f"\nFull grid saved to: {results_csv_path}")
    print(f"Misdetections saved under: {mistakes_dir}/ "
          f"(mispredict_ood_to_id_T{T_best:g}_eps{eps_best:g}.csv, mispredict_id_to_ood_T{T_best:g}_eps{eps_best:g}.csv)")

    total_sec = time.time() - t_all
    print(f"\n[Time] Total elapsed: {total_sec:.2f} sec")

if __name__ == "__main__":
    main()