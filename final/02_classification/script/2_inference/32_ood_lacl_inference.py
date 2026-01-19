import os, time, argparse, warnings, csv, json, random
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, XLMRobertaPreTrainedModel, XLMRobertaModel
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ========== Utils ==========
PAD_ID = 1
def set_pad_token_id(pid: int):
    global PAD_ID
    PAD_ID = int(pid)

task_to_keys = {'food': ("제목_내용", None)}

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path, encoding="utf-8", dtype=str, quoting=csv.QUOTE_MINIMAL, low_memory=False)

def _ensure_label_integer(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise KeyError(f"[utils] 라벨 컬럼 '{label_col}' 이(가) 없습니다. 컬럼 목록: {list(df.columns)}")
    df[label_col] = df[label_col].astype(str).str.strip()
    uniq = sorted(df[label_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}
    df['_label_id'] = df[label_col].map(label2id).astype(int)
    return df, label2id, id2label

def _to_examples(df: pd.DataFrame, text_col: str, label_col_id: str):
    exs = []
    for _, row in df.iterrows():
        text = str(row[text_col]) if text_col in row and pd.notnull(row[text_col]) else ""
        label = int(row[label_col_id]) if label_col_id in row and pd.notnull(row[label_col_id]) else 0
        exs.append({text_col: text, "label": label})
    return exs

def _preprocess(tokenizer, example_tuple, text_key, max_len,
                mask_token=None, span_mask_len=5):
    idx, ex = example_tuple
    text = ex[text_key]
    result = tokenizer(text, max_length=max_len, truncation=True)
    result["labels"] = ex.get("label", 0)
    result["indices"] = idx
    return result

def LACL_load_for_eval(args, tokenizer):
    if args.task_name not in task_to_keys:
        raise ValueError(f"[utils] 지원하지 않는 task_name: {args.task_name}")
    text_key, _ = task_to_keys[args.task_name]

    p_train = os.path.join(args.data_dir, "in_distribution_train.csv")
    p_ind   = os.path.join(args.data_dir, "in_distribution_test.csv")
    p_val   = os.path.join(args.data_dir, "in_distribution_valid.csv")
    p_ood_val  = os.path.join(args.data_dir, "out_of_distribution_valid.csv")
    p_ood_test = os.path.join(args.data_dir, "out_of_distribution_test.csv")
    df_train = _read_csv(p_train)
    df_ind   = _read_csv(p_ind if args.eval_split == "test" else p_val)
    df_ind_name = "test" if args.eval_split == "test" else "valid"

    for need in [text_key, "대분류"]:
        if need not in df_train.columns:
            raise KeyError(f"[utils] '{need}' 컬럼이 train CSV에 없습니다.")
    df_train, label2id, id2label = _ensure_label_integer(df_train, "대분류")

    def _apply_map(df):
        df = df.copy()
        if "대분류" not in df.columns:
            raise KeyError("[utils] ID CSV에도 '대분류' 컬럼이 필요합니다.")
        df["대분류"] = df["대분류"].astype(str).str.strip()
        if not set(df["대분류"].unique()).issubset(set(label2id.keys())):
            unknowns = sorted(set(df["대분류"].unique()) - set(label2id.keys()))
            raise ValueError(f"[utils] ID split에 train에 없던 라벨 존재: {unknowns[:10]} ...")
        df["_label_id"] = df["대분류"].map(label2id).astype(int)
        return df

    df_ind = _apply_map(df_ind)

    ood_df = _read_csv(p_ood_val if args.eval_split == "valid" else p_ood_test)
    if task_to_keys[args.task_name][0] not in ood_df.columns:
        raise KeyError(f"[utils] OOD CSV에 '{text_key}' 컬럼이 없습니다.")
    if "대분류" in ood_df.columns:
        ood_df["_label_id"] = ood_df["대분류"].map(label2id).fillna(0).astype(int)
    else:
        ood_df["_label_id"] = 0

    id_list   = _to_examples(df_ind, text_key, "_label_id")
    ood_list  = _to_examples(ood_df, text_key, "_label_id")
    train_raw = _to_examples(df_train, text_key, "_label_id")

    def preprocess_fn(ex):
        return _preprocess(tokenizer, ex, text_key, args.max_seq_length,
                           mask_token=tokenizer.mask_token, span_mask_len=args.span_mask_len)
    id_ds   = list(map(preprocess_fn, enumerate(id_list)))
    ood_ds  = list(map(preprocess_fn, enumerate(ood_list)))
    prep_ds = list(map(preprocess_fn, enumerate(train_raw)))

    id2label_map = id2label.copy()
    ind_meta = [(ex[text_key], ex["label"], id2label_map.get(ex["label"], str(ex["label"]))) for ex in id_list]
    ood_meta = [(ex[text_key], ex["label"], id2label_map.get(ex["label"], str(ex["label"]))) for ex in ood_list]
    meta = {"text_key": text_key, "id2label": id2label_map, "ind_meta": ind_meta, "ood_meta": ood_meta}

    print(f"[utils] eval_split={df_ind_name} | #ID={len(id_ds)} | #OOD={len(ood_ds)} | #prep_bank={len(prep_ds)}")
    return id_ds, ood_ds, prep_ds, meta

class SimpleDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset
    def __getitem__(self, i): return self.dset[i]
    def __len__(self): return len(self.dset)

def collate_fn(batch):
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids = [f["input_ids"] + [PAD_ID] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(input_mask, dtype=torch.float),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if 'indices' in batch[0]:
        out["indices"] = torch.LongTensor([f["indices"] for f in batch])
    return out

def write_log2(args, dic):
    ddir = args.log_dir
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, 'evaluation_log.csv')

    def as_pct(k):
        return k.endswith(("_auroc","_tpr95","_fpr95","_acc")) or (k == "id_accuracy")
    def is_thr(k): return k.endswith("_thr")

    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(",".join(dic.keys()) + "\n")
        row = []
        for k in dic.keys():
            v = dic[k]
            try:
                v = float(v)
                if is_thr(k):
                    row.append(f"{v:.6f}")
                elif as_pct(k):
                    row.append(f"{v*100:.3f}")
                else:
                    row.append(f"{v:.6f}")
            except Exception:
                row.append(str(v))
        f.write(",".join(row) + "\n")

def cal_num_labels(args):
    df = _read_csv(os.path.join(args.data_dir, "in_distribution_train.csv"))
    if "대분류" not in df.columns:
        raise KeyError("[utils] '대분류' 컬럼이 train CSV에 없습니다.")
    uniq = sorted(pd.Series(df["대분류"].astype(str).str.strip()).unique().tolist())
    return len(uniq)

# ========== Model ==========
def mean_pooling(output, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(output)
    masked = output * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

class LACL(XLMRobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.lacl_config = args
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config)
        self.num_layers = config.num_hidden_layers
        hidden = config.hidden_size
        gp_layers = self.lacl_config.gp_layers
        self.prj_dim = int(self.lacl_config.projection_dim / len(gp_layers))
        self.global_projector = nn.Sequential(
            nn.Linear(hidden, self.lacl_config.encoder_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.lacl_config.encoder_dim, self.prj_dim)
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, indices=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        all_hidden = outputs.hidden_states
        lw_compact = []
        for i in range(1, 1 + self.num_layers):
            if i in self.lacl_config.gp_layers:
                token_vec = mean_pooling(all_hidden[i], attention_mask) if self.lacl_config.gp_location == 'token' \
                            else all_hidden[i][:, 0, :]
                lw_compact.append(self.global_projector(token_vec))
        global_projection = torch.cat(lw_compact, dim=1)
        return {'all_hidden': all_hidden, 'lw_mean_embedding': lw_compact, 'global_projection': global_projection}

    @torch.no_grad()
    def compute_ood(self, input_ids=None, attention_mask=None, labels=None, indices=None, ind=False):
        b = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        pooled = b['global_projection']
        maha_score = []
        for c in self.all_classes:
            centered = pooled - self.class_mean[c].unsqueeze(0)
            m = torch.diag(centered @ self.class_var @ centered.t())
            maha_score.append(m)
        maha_score = torch.stack(maha_score, dim=-1).min(-1).values
        maha_score = -maha_score
        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_mat = norm_pooled @ self.norm_bank.t()
        topk_scores, topk_idx = cosine_mat.topk(k=self.lacl_config.cosine_top_k, dim=-1)
        weights = torch.tensor([1.0/(i+1) for i in range(self.lacl_config.cosine_top_k)],
                               device=topk_scores.device, dtype=topk_scores.dtype)
        weights = weights / weights.sum()
        cosine_score = (topk_scores * weights).sum(dim=-1)
        pred_top1 = self.label_bank[topk_idx[:, 0]]
        cosine_correct = 0
        if ind and labels is not None:
            cosine_correct = (pred_top1 == labels).sum().item()
        use_cos_as_maha = getattr(self.lacl_config, 'maha_via_cosine', False)
        return {
            'maha': (cosine_score if use_cos_as_maha else maha_score).detach().cpu().tolist(),
            'cosine': cosine_score.detach().cpu().tolist(),
            'pred': pred_top1.detach().cpu().tolist(),
            'labels': labels.detach().cpu().tolist() if labels is not None else [],
            'indices': indices.detach().cpu().tolist() if indices is not None else [],
            'cosine_correct': cosine_correct
        }

    @torch.no_grad()
    def prepare_ood(self, dataloader=None, device=None):
        self.bank, self.label_bank = None, None
        device = device or next(self.parameters()).device
        for batch in dataloader:
            self.eval()
            batch = {k: v.to(device) for k, v in batch.items()}
            b = self.forward(**batch)
            pooled = b['global_projection'].detach()
            labels = batch['labels'].detach()
            self.bank = pooled if self.bank is None else torch.cat([self.bank, pooled], dim=0)
            self.label_bank = labels if self.label_bank is None else torch.cat([self.label_bank, labels], dim=0)
        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        self.all_classes = sorted(list(set(self.label_bank.tolist())))
        self.class_mean = torch.zeros(max(self.all_classes)+1, d, device=self.bank.device)
        for c in self.all_classes:
            self.class_mean[c] = self.bank[self.label_bank == c].mean(0)
        centered = (self.bank - self.class_mean[self.label_bank]).cpu().numpy()
        precision = LedoitWolf().fit(centered).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).to(self.bank.device)

# ========== Evaluation ==========
def merge_keys(lst, keys):
    out = {}
    for k in keys:
        col = []
        for d in lst:
            if k in d:
                v = d[k]
                if isinstance(v, list):
                    col += v
                else:
                    col.append(v)
        out[k] = col
    return out

def _thr_for_tpr(scores_pos, target_tpr=0.95):
    s = np.asarray(scores_pos, dtype=np.float64)
    if s.size == 0:
        return float('nan'), 0.0
    s_sorted = np.sort(s)
    q = 1.0 - float(target_tpr)
    k = int(np.floor(q * len(s_sorted)))
    k = np.clip(k, 0, len(s_sorted)-1)
    thr = s_sorted[k]
    tpr = float((s >= thr).sum()) / float(len(s))
    return float(thr), float(tpr)

def fpr_at_thr(ood_scores, thr):
    o = np.asarray(ood_scores, dtype=np.float64)
    if o.size == 0: return 0.0
    return float((o >= thr).sum()) / float(len(o))

def _save_fp_csv(args, tag, method, thr, out_scores, meta, save_name=""):
    import pandas as pd
    o_scores  = np.asarray(out_scores[method], dtype=np.float64)
    o_pred_id = np.asarray(out_scores['pred'], dtype=np.int64)
    o_indices = np.asarray(out_scores['indices'], dtype=np.int64)
    fp_mask = (o_scores >= float(thr))
    if fp_mask.sum() == 0:
        return None
    rows = []
    id2label = meta["id2label"]
    ood_meta = meta["ood_meta"]
    for idx in np.where(fp_mask)[0]:
        ex_idx = int(o_indices[idx]) if o_indices.size == o_scores.size else idx
        text, y_id, y_name = ood_meta[ex_idx] if ex_idx < len(ood_meta) else ("", 0, "UNK_OOD")
        pred_name = id2label.get(int(o_pred_id[idx]), str(int(o_pred_id[idx])))
        rows.append({
            "split": tag, "method": method, "score": float(o_scores[idx]), "tpr95_thr": float(thr),
            "실제_대분류": y_name, "예측_대분류": pred_name, "제목_내용": text
        })
    os.makedirs(args.save_dir, exist_ok=True)
    fname = save_name or f"fp_{tag}_{method}.csv"
    out_path = os.path.join(args.save_dir, fname)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

@torch.inference_mode()
def evaluate_ood(args, model, features, ood, tag, meta):
    keys = ['maha', 'cosine', 'pred', 'labels', 'indices']
    in_dl = DataLoader(features, batch_size=args.batch_size, collate_fn=collate_fn)
    in_scores, total_correct = [], 0
    total_len = 0
    for batch in in_dl:
        model.eval()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        g = model.compute_ood(**batch, ind=True)
        in_scores.append(g)
        total_correct += g['cosine_correct']
        total_len += len(batch['labels'])
    in_scores = merge_keys(in_scores, keys)
    id_accuracy = float(total_correct / max(1, total_len))

    out_dl = DataLoader(ood, batch_size=args.batch_size, collate_fn=collate_fn)
    out_scores = []
    for batch in out_dl:
        model.eval()
        batch = {k: v.to(args.device) for k, v in batch.items()}
        out_scores.append(model.compute_ood(**batch))
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    best = {"method": None, "fpr95": 1.0, "auroc": 0.0, "thr": None, "tpr95": None}

    for method in ['cosine', 'maha']:
        ins  = np.asarray(in_scores[method],  dtype=np.float64)
        outs = np.asarray(out_scores[method], dtype=np.float64)
        labels = np.concatenate([np.ones_like(ins, np.int64), np.zeros_like(outs, np.int64)], axis=0)
        scores = np.concatenate([ins, outs], axis=0)
        auroc = roc_auc_score(labels, scores)
        thr, tpr = _thr_for_tpr(ins, target_tpr=0.95)
        fpr95 = fpr_at_thr(outs, thr)
        outputs[f"{tag}_{method}_auroc"] = float(auroc)
        outputs[f"{tag}_{method}_tpr95_thr"] = float(thr)
        outputs[f"{tag}_{method}_tpr95"] = float(tpr)
        outputs[f"{tag}_{method}_fpr95"] = float(fpr95)
        if (fpr95 < best["fpr95"]) or (abs(fpr95 - best["fpr95"]) < 1e-12 and auroc > best["auroc"]):
            best.update(dict(method=method, fpr95=float(fpr95), auroc=float(auroc), thr=float(thr), tpr95=float(tpr)))

    outputs["id_accuracy"] = id_accuracy
    outputs[f"{tag}_best_method"] = best["method"]
    outputs[f"{tag}_best_fpr95"] = best["fpr95"]
    outputs[f"{tag}_best_auroc"] = best["auroc"]
    outputs[f"{tag}_best_tpr95_thr"] = best["thr"]
    outputs[f"{tag}_best_tpr95"] = best["tpr95"]

    print(f"[BEST] {tag}_{best['method']} | FPR@TPR95={best['fpr95']:.4f}, AUROC={best['auroc']:.4f}, thr={best['thr']:.6f}")
    _save_fp_csv(args, tag, best["method"], best["thr"], out_scores, meta, save_name=f"fp_{tag}_best.csv")
    for m in ['cosine', 'maha']:
        if m != best['method']:
            _save_fp_csv(args, tag, m, outputs[f"{tag}_{m}_tpr95_thr"], out_scores, meta)
    return outputs

# ========== CLI ==========
def _pick_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg.lower() == "cuda":
        return torch.device("cuda:0")
    if device_arg.lower().startswith("cuda:"):
        idx = int(device_arg.split(":")[1]); 
        return torch.device(f"cuda:{idx}") if idx < torch.cuda.device_count() else torch.device("cuda:0")
    if device_arg.lower() == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            free_b, _ = torch.cuda.mem_get_info()
            if free_b > best_free:
                best_free = free_b; best_idx = i
        return torch.device(f"cuda:{best_idx}")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def _parse_layers(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description="LaCL inference+evaluation")
    p.add_argument("--data_dir", type=str, default="../../../00_data/02_classification_0808")
    p.add_argument("--save_dir", type=str, default="../../run/logs/3_ood_result/3_lacl")
    p.add_argument("--log_dir",  type=str, default="../../run/logs/3_ood_result/3_lacl")
    p.add_argument("--model_name_or_path", type=str, default="xlm-roberta-base")
    p.add_argument("--init_ckpt", type=str, default="../../run/logs/3_ood_result/3_lacl/best_model.pth",
                   help="학습된 가중치(.pth) 경로")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--span_mask_len", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--task_name", type=str, default="food")
    p.add_argument("--gp_location", type=str, default="token", choices=["token", "cls"])
    p.add_argument("--gp_layers", type=_parse_layers, default=list(range(1,13)))
    p.add_argument("--encoder_dim", type=int, default=1024)
    p.add_argument("--projection_dim", type=int, default=768)
    p.add_argument("--cosine_top_k", type=int, default=1)
    p.add_argument("--maha_via_cosine", action="store_true", default=False)
    p.add_argument("--eval_split", type=str, default="test", choices=["valid","test"],
                   help="ID=valid 또는 ID=test에 대해 평가; OOD는 동일 split 대응 사용")
    return p

def main(args):
    start = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.n_gpu  = torch.cuda.device_count()
    args.device = _pick_device(args.device)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=cal_num_labels(args))
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    set_pad_token_id(tokenizer.pad_token_id)

    model = LACL.from_pretrained(args.model_name_or_path, config=config, args=args).to(args.device)

    if args.init_ckpt and os.path.exists(args.init_ckpt):
        sd = torch.load(args.init_ckpt, map_location=args.device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load ckpt] missing={len(missing) if isinstance(missing, list) else 0}, "
              f"unexpected={len(unexpected) if isinstance(unexpected, list) else 0}")
    else:
        print("[WARN] 학습 가중치를 찾지 못해 사전학습 가중치로 평가합니다.")

    id_ds, ood_ds, prep_ds, meta = LACL_load_for_eval(args, tokenizer)
    prep_dl = DataLoader(prep_ds, batch_size=min(100, args.batch_size), collate_fn=collate_fn, shuffle=False)
    model.prepare_ood(prep_dl, device=args.device)

    results = evaluate_ood(args, model, id_ds, ood_ds, f"ood_{args.task_name}", meta)
    print(results)
    write_log2(args, results)

    elapsed = time.time() - start
    print(f"[총 실행 시간] {elapsed/60:.2f} 분 ({elapsed:.1f} 초)")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)