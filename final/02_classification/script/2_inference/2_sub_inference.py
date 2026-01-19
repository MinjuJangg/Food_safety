import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 상대경로 설정 ==========
THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent

def _find_food_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "00_data" / "02_classification_0919").exists() and (p / "02_classification").exists():
            return p
    for p in [start, *start.parents]:
        if p.name == "food":
            return p
    raise RuntimeError(
        "FOOD_ROOT를 찾지 못했습니다. 상위 폴더에 '00_data/02_classification_0919'와 '02_classification'이 공존하도록 구성해 주세요."
    )

FOOD_ROOT = _find_food_root(SCRIPT_DIR)
PROJECT_ROOT = FOOD_ROOT / "02_classification"
DATA_DIR = FOOD_ROOT / "00_data" / "02_classification_0919"
FONTS_DIR = FOOD_ROOT / "00_data" / "fonts"
FONT_PATH = FOOD_ROOT / "00_data" / "fonts" / "NANUMBARUNGOTHIC.TTF"
TEST_CSV_DEFAULT = DATA_DIR / "test.csv"
RUN_DIR = PROJECT_ROOT / "run"
MAIN_CKPT_DIR = RUN_DIR / "checkpoints" / "1_main_pth"
SUB_CKPT_DIR  = RUN_DIR / "checkpoints" / "2_sub_pth"
OUT_DIR = RUN_DIR / "logs" / "2_sub_result"
CM_DIR = OUT_DIR / "cm_plots"

# ========== 고정 라벨 ==========
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
TEXT_COL_CANDIDATES = ["제목_내용", "text"]
OUT_OF_SET = "OUT_OF_SET(대분류불일치)"

# ========== 서브모델(대분류별) 파일 고정 매핑 ==========
SUB_FILE_MAP: Dict[str, Tuple[Path, Path]] = {
    "물리적 위해요소": (SUB_CKPT_DIR / "sub1_물리적_위해요소_best_model_xlm_roberta.pth",
                     SUB_CKPT_DIR / "sub1_물리적_위해요소_labels.json"),
    "생물학적 위해요소": (SUB_CKPT_DIR / "sub2_생물학적_위해요소_best_model_xlm_roberta.pth",
                       SUB_CKPT_DIR / "sub2_생물학적_위해요소_labels.json"),
    "신규 위해요소": (SUB_CKPT_DIR / "sub3_신규_위해요소_best_model_xlm_roberta.pth",
                   SUB_CKPT_DIR / "sub3_신규_위해요소_labels.json"),
    "안전위생": (SUB_CKPT_DIR / "sub4_안전위생_best_model_xlm_roberta.pth",
              SUB_CKPT_DIR / "sub4_안전위생_labels.json"),
    "영양건강": (SUB_CKPT_DIR / "sub5_영양건강_best_model_xlm_roberta.pth",
              SUB_CKPT_DIR / "sub5_영양건강_labels.json"),
    "표시광고": (SUB_CKPT_DIR / "sub6_표시광고_best_model_xlm_roberta.pth",
              SUB_CKPT_DIR / "sub6_표시광고_labels.json"),
    "화학적 위해요소": (SUB_CKPT_DIR / "sub7_화학적_위해요소_best_model_xlm_roberta.pth",
                     SUB_CKPT_DIR / "sub7_화학적_위해요소_labels.json"),
}

# ========== 유틸 ==========
def _ensure_korean_font():
    try:
        from matplotlib import font_manager as fm
        if FONT_PATH.exists():
            fm.fontManager.addfont(str(FONT_PATH))
            matplotlib.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def pick_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"텍스트 컬럼을 찾지 못했습니다. 후보: {TEXT_COL_CANDIDATES}")

def _find_main_ckpt(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"대분류 가중치(.pth)를 찾지 못했습니다: {dir_path}")
    best = [p for p in candidates if p.name.lower().startswith("best") or "best" in p.name.lower()]
    return best[0] if best else candidates[-1]

def _load_labels_from_json(json_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    label2id = {k: int(v) for k, v in meta["label2id"].items()}
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    return label2id, id2label

def _normalize(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()

def _load_valid_subnames(json_path: Path) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    names: List[str] = []
    if isinstance(obj, dict):
        if "id2label" in obj and isinstance(obj["id2label"], dict):
            names = [str(v).strip() for v in obj["id2label"].values()]
        if not names and "label2id" in obj and isinstance(obj["label2id"], dict):
            names = [str(k).strip() for k in obj["label2id"].keys()]
    names = sorted(list({n for n in names if n}))
    if not names:
        raise ValueError(f"서브라벨을 찾을 수 없습니다: {json_path}")
    return names

def _build_valid_subsets(main_to_json: Dict[str, Path]) -> Dict[str, Set[str]]:
    return {main_name: set(_load_valid_subnames(path)) for main_name, path in main_to_json.items()}

def _fmt_num(x):
    if x == "" or pd.isna(x):
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def _render_text_table(df: pd.DataFrame) -> str:
    df = df.copy()
    df["precision"] = df["precision"].apply(_fmt_num)
    df["recall"] = df["recall"].apply(_fmt_num)
    df["f1-score"] = df["f1-score"].apply(_fmt_num)
    df["support"] = df["support"].apply(lambda v: f"{int(v):d}" if str(v).strip() != "" else "")
    widths = {
        "class": max(7, max(len(str(s)) for s in df["class"])),
        "precision": max(9, max(len(str(s)) for s in df["precision"])),
        "recall": max(6, max(len(str(s)) for s in df["recall"])),
        "f1-score": max(8, max(len(str(s)) for s in df["f1-score"])),
        "support": max(7, max(len(str(s)) for s in df["support"])),
    }
    header = (
        f'{"":<{widths["class"]}}  '
        f'{"precision":>{widths["precision"]}}  '
        f'{"recall":>{widths["recall"]}}  '
        f'{"f1-score":>{widths["f1-score"]}}  '
        f'{"support":>{widths["support"]}}'
    )
    lines = [" " + header]
    for _, r in df.iterrows():
        line = (
            f'{str(r["class"]):<{widths["class"]}}  '
            f'{str(r["precision"]):>{widths["precision"]}}  '
            f'{str(r["recall"]):>{widths["recall"]}}  '
            f'{str(r["f1-score"]):>{widths["f1-score"]}}  '
            f'{str(r["support"]):>{widths["support"]}}'
        )
        lines.append(" " + line)
    return "\n".join(lines)

def _evaluate_one_main(
    df_group: pd.DataFrame, valid_subs: Set[str],
    true_sub_col: str, pred_sub_col: str
):
    y_true_raw = _normalize(df_group[true_sub_col])
    y_pred_raw = _normalize(df_group[pred_sub_col])

    y_true = [t if t in valid_subs else OUT_OF_SET for t in y_true_raw]
    y_pred = [p for p in y_pred_raw]

    label_order = sorted(valid_subs)
    if OUT_OF_SET in y_true:
        label_order.append(OUT_OF_SET)

    rep = classification_report(
        y_true, y_pred, labels=label_order, output_dict=True, zero_division=0
    )

    rows = []
    for lbl in label_order:
        st = rep.get(lbl, {})
        rows.append({
            "class": lbl,
            "precision": st.get("precision", 0.0),
            "recall": st.get("recall", 0.0),
            "f1": st.get("f1-score", 0.0),
            "support": int(st.get("support", 0)),
        })
    per_df = pd.DataFrame(rows)
    macro = rep.get("macro avg", {})
    weighted = rep.get("weighted avg", {})
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0

    summary = {
        "macro_precision": macro.get("precision", 0.0),
        "macro_recall": macro.get("recall", 0.0),
        "macro_f1": macro.get("f1-score", 0.0),
        "weighted_precision": weighted.get("precision", 0.0),
        "weighted_recall": weighted.get("recall", 0.0),
        "weighted_f1": weighted.get("f1-score", 0.0),
        "accuracy": acc,
        "support_total": int(per_df["support"].sum()),
    }
    per_df[["precision", "recall", "f1"]] = per_df[["precision", "recall", "f1"]].astype(float).round(3)
    return per_df, summary

def _plot_and_save_cm(df_counts: pd.DataFrame, png_path: Path, title: str):
    _ensure_korean_font()
    plt.figure(figsize=(min(40, 0.35*len(df_counts.columns)+6), min(40, 0.35*len(df_counts.index)+6)))
    sns.heatmap(df_counts, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

# ========== Main ==========
def main():
    import argparse
    parser = argparse.ArgumentParser(description="대분류→중분류 추론 및 평가")
    parser.add_argument("--test_csv", type=str, default=str(TEST_CSV_DEFAULT))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    df = pd.read_csv(args.test_csv)
    text_col = pick_text_col(df)
    df["text"] = df[text_col].astype(str)

    main_pth = _find_main_ckpt(MAIN_CKPT_DIR)
    main_model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(MAIN_LABEL2ID),
        id2label={i: n for i, n in MAIN_ID2LABEL.items()},
        label2id=MAIN_LABEL2ID,
    )
    state_dict = torch.load(main_pth, map_location="cpu")
    missing, unexpected = main_model.load_state_dict(state_dict, strict=False)
    head = getattr(main_model.classifier, "out_proj", main_model.classifier)
    out_dim = head.weight.shape[0]
    if out_dim != len(MAIN_LABEL2ID):
        raise RuntimeError(f"대분류 헤드 출력차원 {out_dim} != {len(MAIN_LABEL2ID)} (가중치/라벨 매핑 불일치)")
    main_model.to(device).eval()
    print(f"[LOAD] main checkpoint: {main_pth.name}")

    # ===== 1) 대분류 예측 =====
    main_preds: List[int] = []
    main_names: List[str] = []

    bs = args.batch_size
    maxlen = args.max_length
    with torch.no_grad():
        for i in tqdm(range(0, len(df), bs), desc="Main inference"):
            batch_texts = df["text"].iloc[i:i+bs].tolist()
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                            padding=True, max_length=maxlen)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = main_model(**enc).logits
            preds = torch.argmax(logits, dim=1).tolist()
            main_preds.extend(preds)
            main_names.extend([MAIN_ID2LABEL[p] for p in preds])

    df["main_predicted_label"] = main_preds
    df["main_predicted_label_name"] = main_names
    main_csv_path = OUT_DIR / "test_main_predictions.csv"
    df.to_csv(main_csv_path, index=False)
    print(f"[SAVE] {main_csv_path}")

    # ===== 2) 중분류 예측 =====
    for m, (pth, js) in SUB_FILE_MAP.items():
        if not pth.exists() or not js.exists():
            raise FileNotFoundError(f"[SUB MODEL MISSING] {m}: {pth.name} / {js.name}")

    df["sub_predicted_label"] = pd.Series([None] * len(df), dtype=object)
    df["sub_predicted_label_name"] = pd.Series([None] * len(df), dtype=object)

    for main_id in sorted(df["main_predicted_label"].dropna().astype(int).unique().tolist()):
        main_name = MAIN_ID2LABEL[main_id]
        if main_name not in SUB_FILE_MAP:
            print(f"[WARN] '{main_name}' 서브모델 파일 쌍 미존재. 건너뜀.")
            continue

        pth_path, json_path = SUB_FILE_MAP[main_name]
        label2id, id2label = _load_labels_from_json(json_path)

        sub_model = XLMRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(label2id),
            id2label={int(k): v for k, v in id2label.items()},
            label2id={k: int(v) for k, v in label2id.items()},
        )
        state = torch.load(pth_path, map_location="cpu")
        sub_model.load_state_dict(state, strict=False)
        sub_model.to(device).eval()

        idxs = df.index[df["main_predicted_label"] == main_id].tolist()
        if not idxs:
            continue

        texts = df.loc[idxs, "text"].tolist()
        preds_all: List[int] = []
        names_all: List[str] = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs), desc=f"Sub inference [{main_name}]"):
                enc = tokenizer(texts[i:i+bs], return_tensors="pt", truncation=True,
                                padding=True, max_length=maxlen)
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = sub_model(**enc).logits
                preds = torch.argmax(logits, dim=1).tolist()
                preds_all.extend(preds)
                names_all.extend([id2label[int(p)] for p in preds])

        df.loc[idxs, "sub_predicted_label"] = preds_all
        df.loc[idxs, "sub_predicted_label_name"] = names_all

    final_csv = OUT_DIR / "test_main_sub_predictions.csv"
    df.to_csv(final_csv, index=False)
    print(f"[SAVE] {final_csv}")

    # ===== 3) 평가 =====
    main_to_json: Dict[str, Path] = {m: SUB_FILE_MAP[m][1] for m in SUB_FILE_MAP}

    need_cols = ["대분류", "중분류", "main_predicted_label_name", "sub_predicted_label_name"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"입력 CSV에 '{c}' 열이 없습니다: {final_csv}")

    for c in need_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    valid_map = _build_valid_subsets(main_to_json)
    pred_mains = sorted([m for m in df["main_predicted_label_name"].dropna().unique().tolist() if m])

    summary_rows = []
    per_main_tables: Dict[str, pd.DataFrame] = {}

    for main_name in pred_mains:
        valid_subs = valid_map.get(main_name, set())
        group = df[df["main_predicted_label_name"] == main_name].copy()
        per_df, summ = _evaluate_one_main(group, valid_subs, "중분류", "sub_predicted_label_name")

        tbl = per_df[["class", "precision", "recall", "f1"]].rename(columns={"f1": "f1-score"}).copy()
        tbl["support"] = per_df["support"].astype(int)

        acc_row = pd.DataFrame([{
            "class": "accuracy",
            "precision": round(float(summ["accuracy"]), 3),
            "recall": "",
            "f1-score": "",
            "support": int(summ["support_total"]),
        }])
        macro_row = pd.DataFrame([{
            "class": "macro avg",
            "precision": round(float(summ["macro_precision"]), 3),
            "recall": round(float(summ["macro_recall"]), 3),
            "f1-score": round(float(summ["macro_f1"]), 3),
            "support": int(summ["support_total"]),
        }])
        weighted_row = pd.DataFrame([{
            "class": "weighted avg",
            "precision": round(float(summ["weighted_precision"]), 3),
            "recall": round(float(summ["weighted_recall"]), 3),
            "f1-score": round(float(summ["weighted_f1"]), 3),
            "support": int(summ["support_total"]),
        }])

        final_tbl = pd.concat([tbl, acc_row, macro_row, weighted_row], ignore_index=True)
        per_main_tables[main_name] = final_tbl.copy()

        txt = _render_text_table(final_tbl)
        out_txt = OUT_DIR / f"classification_report_{main_name}.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[SAVE] {out_txt}")

        row = {"pred_main": main_name}
        row.update(summ)
        summary_rows.append(row)

    # 대분류별 요약
    summary_df = pd.DataFrame(summary_rows)
    sum_cols = ["macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1", "accuracy"]
    if not summary_df.empty:
        summary_df[sum_cols] = summary_df[sum_cols].astype(float).round(3)

        sum_tbl = summary_df.rename(columns={
            "pred_main": "class",
            "macro_precision": "precision(macro)",
            "macro_recall": "recall(macro)",
            "macro_f1": "f1(macro)",
            "weighted_precision": "precision(weighted)",
            "weighted_recall": "recall(weighted)",
            "weighted_f1": "f1(weighted)",
            "accuracy": "accuracy",
            "support_total": "support",
        })[["class", "precision(macro)", "recall(macro)", "f1(macro)",
            "precision(weighted)", "recall(weighted)", "f1(weighted)", "accuracy", "support"]]

        def _render_summary(df: pd.DataFrame) -> str:
            df = df.copy()
            for c in ["precision(macro)", "recall(macro)", "f1(macro)",
                      "precision(weighted)", "recall(weighted)", "f1(weighted)", "accuracy"]:
                df[c] = df[c].apply(lambda x: f"{float(x):.3f}" if str(x) != "" else "")
            df["support"] = df["support"].astype(int)
            w_class = max(6, max(len(str(x)) for x in df["class"]))
            header = (f'{"class":<{w_class}}  '
                      f'{"precision(macro)":>16}  {"recall(macro)":>13}  {"f1(macro)":>10}  '
                      f'{"precision(weighted)":>20}  {"recall(weighted)":>17}  {"f1(weighted)":>14}  '
                      f'{"accuracy":>8}  {"support":>7}')
            lines = [" " + header]
            for _, r in df.iterrows():
                line = (f'{str(r["class"]):<{w_class}}  '
                        f'{r["precision(macro)"]:>16}  {r["recall(macro)"]:>13}  {r["f1(macro)"]:>10}  '
                        f'{r["precision(weighted)"]:>20}  {r["recall(weighted)"]:>17}  {r["f1(weighted)"]:>14}  '
                        f'{r["accuracy"]:>8}  {int(r["support"]):>7}')
                lines.append(" " + line)
            return "\n".join(lines)

        txt_summary = _render_summary(sum_tbl)
        sum_txt_path = OUT_DIR / "per_main_summary.txt"
        with open(sum_txt_path, "w", encoding="utf-8") as f:
            f.write(txt_summary)
        print(f"[SAVE] {sum_txt_path}")

    # 전체 요약
    all_valid_subs = set()
    for p in main_to_json.values():
        all_valid_subs |= set(_load_valid_subnames(p))
    if all_valid_subs:
        _, all_summary = _evaluate_one_main(df, all_valid_subs, "중분류", "sub_predicted_label_name")
        overall_rows = [{
            "class": "전체",
            "precision(macro)": all_summary["macro_precision"],
            "recall(macro)": all_summary["macro_recall"],
            "f1(macro)": all_summary["macro_f1"],
            "precision(weighted)": all_summary["weighted_precision"],
            "recall(weighted)": all_summary["weighted_recall"],
            "f1(weighted)": all_summary["weighted_f1"],
            "accuracy": all_summary["accuracy"],
            "support": all_summary["support_total"],
        }]
        ov_df = pd.DataFrame(overall_rows)
        for c in ["precision(macro)", "recall(macro)", "f1(macro)",
                  "precision(weighted)", "recall(weighted)", "f1(weighted)", "accuracy"]:
            ov_df[c] = ov_df[c].astype(float).round(3)

        def _render_summary(df: pd.DataFrame) -> str:
            df = df.copy()
            for c in ["precision(macro)", "recall(macro)", "f1(macro)",
                      "precision(weighted)", "recall(weighted)", "f1(weighted)", "accuracy"]:
                df[c] = df[c].apply(lambda x: f"{float(x):.3f}" if str(x) != "" else "")
            df["support"] = df["support"].astype(int)
            w_class = max(6, max(len(str(x)) for x in df["class"]))
            header = (f'{"class":<{w_class}}  '
                      f'{"precision(macro)":>16}  {"recall(macro)":>13}  {"f1(macro)":>10}  '
                      f'{"precision(weighted)":>20}  {"recall(weighted)":>17}  {"f1(weighted)":>14}  '
                      f'{"accuracy":>8}  {"support":>7}')
            lines = [" " + header]
            for _, r in df.iterrows():
                line = (f'{str(r["class"]):<{w_class}}  '
                        f'{r["precision(macro)"]:>16}  {r["recall(macro)"]:>13}  {r["f1(macro)"]:>10}  '
                        f'{r["precision(weighted)"]:>20}  {r["recall(weighted)"]:>17}  {r["f1(weighted)"]:>14}  '
                        f'{r["accuracy"]:>8}  {int(r["support"]):>7}')
                lines.append(" " + line)
            return "\n".join(lines)

        overall_txt = _render_summary(ov_df)
        overall_txt_path = OUT_DIR / "overall_summary.txt"
        with open(overall_txt_path, "w", encoding="utf-8") as f:
            f.write(overall_txt)
        print(f"[SAVE] {overall_txt_path}")

    # ===== 4) 전체 서브라벨 기준 혼동행렬 =====
    df_glob = df.copy()
    df_glob["true_global_sub"] = df_glob["대분류"].astype(str).str.strip() + "/" + df_glob["중분류"].astype(str).str.strip()
    df_glob["pred_global_sub"] = df_glob["main_predicted_label_name"].astype(str).str.strip() + "/" + df_glob["sub_predicted_label_name"].astype(str).str.strip()

    df_glob["true_global_sub"].replace({"//": "N/A/N/A"}, inplace=True)
    df_glob["pred_global_sub"].replace({"//": "N/A/N/A"}, inplace=True)

    true_labels = sorted(df_glob["true_global_sub"].unique().tolist())
    pred_labels = sorted(df_glob["pred_global_sub"].unique().tolist())

    counts = pd.crosstab(index=pd.Categorical(df_glob["true_global_sub"], categories=true_labels, ordered=True),
                         columns=pd.Categorical(df_glob["pred_global_sub"], categories=pred_labels, ordered=True))
    counts_csv = CM_DIR / "cm_sub_all_counts.csv"
    counts.to_csv(counts_csv, encoding="utf-8-sig")
    print(f"[SAVE] {counts_csv}")

    row_norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    row_norm_csv = CM_DIR / "cm_sub_all_row_norm.csv"
    row_norm.to_csv(row_norm_csv, encoding="utf-8-sig")
    print(f"[SAVE] {row_norm_csv}")

    _plot_and_save_cm(counts, CM_DIR / "cm_sub_all_counts.png", "Confusion Matrix (Counts) - Global Sub Labels")
    _plot_and_save_cm(row_norm, CM_DIR / "cm_sub_all_row_norm.png", "Confusion Matrix (Row-Normalized) - Global Sub Labels")
    print(f"[SAVE] {CM_DIR / 'cm_sub_all_counts.png'}")
    print(f"[SAVE] {CM_DIR / 'cm_sub_all_row_norm.png'}")

    xlsx_path = OUT_DIR / "all_reports.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="per_main_summary", index=False)
        if all_valid_subs:
            ov_df.to_excel(writer, sheet_name="overall_summary", index=False)
        for main_name, tbl in per_main_tables.items():
            sheet_name = main_name[:31]
            tbl.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"[SAVE] {xlsx_path}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("[DONE] 추론 및 평가 완료")
    print(f"- Main-only CSV: {main_csv_path}")
    print(f"- Main+Sub CSV:  {final_csv}")
    print(f"- TXT 리포트     : classification_report_<대분류>.txt, per_main_summary.txt, overall_summary.txt")
    print(f"- CM CSV/PNG     : {counts_csv.name}, {row_norm_csv.name} 및 PNG 2종 (cm_plots/)")
    print(f"- Excel          : all_reports.xlsx")

if __name__ == "__main__":
    main()