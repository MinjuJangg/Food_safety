import os, time, math, argparse, warnings, csv, json, random
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

class TraceLogger:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.f = None
        if path:
            d = os.path.dirname(path)
            if d: os.makedirs(d, exist_ok=True)
            self.f = open(path, 'a', encoding='utf-8')
    def log(self, payload: dict):
        if self.f is None: return
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        self.f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        if self.f:
            self.f.close()
            self.f = None

def get_lr(optimizer) -> float:
    if len(optimizer.param_groups) == 0: return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))

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
                mask_token=None, span_mask_len=5, do_span=False,
                augment_shuffle=False, augment_cutoff=False, cutoff_ratio=0.15):
    idx, ex = example_tuple
    text = ex[text_key]
    if do_span and mask_token is not None and len(text) > max(30, span_mask_len + 1):
        ind = np.random.randint(len(text) - span_mask_len)
        left, right = text[:ind], text[ind + span_mask_len:]
        text = f"{left}{mask_token}{right}"
    result = tokenizer(text, max_length=max_len, truncation=True)
    result["labels"] = ex.get("label", 0)
    result["indices"] = idx
    if augment_shuffle and "input_ids" in result:
        ids = result["input_ids"]; np.random.shuffle(ids)
    if augment_cutoff and "input_ids" in result:
        ids = result["input_ids"]
        cutoff_size = int(len(ids) * cutoff_ratio)
        for _ in range(cutoff_size):
            if len(ids) <= 2: break
            cutoff_idx = np.random.randint(1, len(ids) - 1)
            ids.pop(cutoff_idx)
            for k in ["attention_mask", "token_type_ids"]:
                if k in result and isinstance(result[k], list) and len(result[k]) > cutoff_idx:
                    result[k].pop(cutoff_idx)
    return result

def LACL_load_aug(args, tokenizer):
    if args.task_name not in task_to_keys:
        raise ValueError(f"[utils] 지원하지 않는 task_name: {args.task_name}")
    text_key, _ = task_to_keys[args.task_name]
    print(f"[utils] Loading task={args.task_name} from {args.data_dir}")

    p_train = os.path.join(args.data_dir, "in_distribution_train.csv")
    p_val   = os.path.join(args.data_dir, "in_distribution_valid.csv")
    p_ind   = os.path.join(args.data_dir, "in_distribution_test.csv")
    p_ood_val  = os.path.join(args.data_dir, "out_of_distribution_valid.csv")
    p_ood_test = os.path.join(args.data_dir, "out_of_distribution_test.csv")
    df_train = _read_csv(p_train)
    df_ind   = _read_csv(p_ind)
    df_val   = _read_csv(p_val) if os.path.exists(p_val) else None

    for need in [text_key, "대분류"]:
        if need not in df_train.columns:
            raise KeyError(f"[utils] '{need}' 컬럼이 train CSV에 없습니다. 컬럼: {list(df_train.columns)}")
    df_train, label2id, id2label = _ensure_label_integer(df_train, "대분류")

    def _apply_map(df):
        if df is None: return None
        df = df.copy()
        if "대분류" not in df.columns:
            raise KeyError("[utils] 테스트/밸리드 CSV에도 '대분류' 컬럼이 필요합니다.")
        df["대분류"] = df["대분류"].astype(str).str.strip()
        if not set(df["대분류"].unique()).issubset(set(label2id.keys())):
            unknowns = sorted(set(df["대분류"].unique()) - set(label2id.keys()))
            raise ValueError(f"[utils] 테스트/밸리드에 train에 없던 라벨 존재: {unknowns[:10]} ...")
        df["_label_id"] = df["대분류"].map(label2id).astype(int)
        return df

    df_ind = _apply_map(df_ind)
    if df_val is not None: df_val = _apply_map(df_val)

    use_ood_val = bool(getattr(args, "use_ood_val", False))
    ood_eval_df = _read_csv(p_ood_val if use_ood_val and args.eval_split == "valid" else p_ood_test)
    if task_to_keys[args.task_name][0] not in ood_eval_df.columns:
        raise KeyError(f"[utils] OOD CSV에 '{text_key}' 컬럼이 없습니다.")
    if "대분류" in ood_eval_df.columns:
        ood_eval_df["_label_id"] = ood_eval_df["대분류"].map(label2id).fillna(0).astype(int)
    else:
        ood_eval_df["_label_id"] = 0

    train_raw_list = _to_examples(df_train, text_key, "_label_id")
    test_ind_list  = _to_examples(df_ind,   text_key, "_label_id")
    ood_list       = _to_examples(ood_eval_df, text_key, "_label_id")

    def preprocess_fn(ex):
        return _preprocess(tokenizer, ex, text_key, args.max_seq_length,
                           mask_token=tokenizer.mask_token, span_mask_len=args.span_mask_len, do_span=False)
    def preprocess_fn_aug(ex):
        do_span = ('span' in str(args.augment_method)) or (str(args.augment_both) == 'span')
        return _preprocess(tokenizer, ex, text_key, args.max_seq_length,
                           mask_token=tokenizer.mask_token, span_mask_len=args.span_mask_len, do_span=do_span,
                           augment_shuffle=('shuffle' in str(args.augment_method)),
                           augment_cutoff=('token_cutoff' in str(args.augment_method)),
                           cutoff_ratio=getattr(args, "cutoff_ratio", 0.15))

    aug1 = list(map(preprocess_fn_aug if str(args.augment_both) != 'None' else preprocess_fn,
                    enumerate(train_raw_list)))
    aug2 = list(map(preprocess_fn_aug, enumerate(train_raw_list)))
    test_ind_dataset = list(map(preprocess_fn, enumerate(test_ind_list)))
    test_ood_dataset = list(map(preprocess_fn, enumerate(ood_list)))
    train_raw_dataset = list(map(preprocess_fn, enumerate(train_raw_list)))

    l_ind = [d['labels'] for d in test_ind_dataset]
    l_train = [d['labels'] for d in train_raw_dataset]
    print(f"[utils] #train={len(train_raw_dataset)} | #ind_test={len(test_ind_dataset)} | #ood_eval={len(test_ood_dataset)}")
    print(f"[utils] train labels(unique)={len(set(l_train))} | ind_test labels(unique)={len(set(l_ind))}")

    id2label_map = id2label.copy()
    ind_meta = [(ex[text_key], ex["label"], id2label_map.get(ex["label"], str(ex["label"]))) for ex in test_ind_list]
    ood_meta = [(ex[text_key], ex["label"], id2label_map.get(ex["label"], str(ex["label"]))) for ex in ood_list]
    meta = {
        "text_key": text_key,
        "id2label": id2label_map,
        "ind_meta": ind_meta,
        "ood_meta": ood_meta
    }

    return aug1, aug2, test_ind_dataset, test_ood_dataset, train_raw_dataset, meta

class PairDataset(Dataset):
    def __init__(self, raw_dset, aug_dset):
        self.raw_dset = raw_dset
        self.aug_dset = aug_dset
        assert len(self.raw_dset) == len(self.aug_dset), "[utils] raw/aug 길이가 다릅니다."
    def __getitem__(self, index):
        return [self.raw_dset[index], self.aug_dset[index]]
    def __len__(self):
        return len(self.raw_dset)

def set_seed(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if getattr(args, "n_gpu", 0) > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

def collate_fn2(batch):
    b1 = [i[0] for i in batch]; b2 = [i[1] for i in batch]
    max_len = max(max(len(f["input_ids"]) for f in b1), max(len(f["input_ids"]) for f in b2))
    def _pad(bb):
        ids  = [f["input_ids"] + [PAD_ID] * (max_len - len(f["input_ids"])) for f in bb]
        mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in bb]
        labs = [f["labels"] for f in bb]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float), torch.tensor(labs, dtype=torch.long)
    ids1, m1, y1 = _pad(b1); ids2, m2, y2 = _pad(b2)
    return [{"input_ids": ids1, "attention_mask": m1, "labels": y1},
            {"input_ids": ids2, "attention_mask": m2, "labels": y2}]

def write_log2(args, model_name, dic):
    ddir = args.log_dir
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, 'evaluation_log.csv')

    def as_pct(k):
        return k.endswith(("_auroc","_tpr95","_fpr95","_acc")) or (k == "id_accuracy")
    def is_thr(k):
        return k.endswith("_thr")

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

# ========== Loss ==========
def _reg_loss(x1, x2, margin=0.5):
    x1n = F.normalize(x1, dim=0)
    x2n = F.normalize(x2, dim=0)
    cor_mat = (x1n.t() @ x2n).clamp(min=1e-7)
    diag = torch.diagonal(cor_mat)
    mask = (diag > margin)
    if mask.any():
        return diag[mask].mean()
    return diag.new_tensor(0.0)

def reg_loss(tok_embeddings, margin=0.5):
    total = 0.0
    raw_tok, aug_tok = tok_embeddings
    for i in range(len(raw_tok) - 1):
        total += (_reg_loss(raw_tok[i], raw_tok[i+1], margin) +
                  _reg_loss(aug_tok[i], aug_tok[i+1], margin)) * 0.5
    return total

def get_sim_mat(x):
    x = F.normalize(x, dim=1)
    return (x @ x.t()).clamp(min=1e-7)

def Supervised_NT_xent(sim_matrix, labels, temperature=0.2, chunk=2, eps=1e-8):
    device = sim_matrix.device
    labels = labels.repeat(2)
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()
    B = sim_matrix.size(0) // chunk
    eye = torch.eye(B * chunk, device=device)
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    loss_mat = -torch.log(sim_matrix / (denom + eps) + eps)
    labels = labels.view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)
    loss = torch.sum(Mask * loss_mat) / (2 * B)
    return loss

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

# ========== Evaluation (AUROC/FPR95) ==========
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
    in_dl = DataLoader(features, batch_size=args.train_batch, collate_fn=collate_fn)
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

    out_dl = DataLoader(ood, batch_size=args.train_batch, collate_fn=collate_fn)
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

# ========== 학습 Loop ==========
def _pick_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg.lower() == "cuda":
        return torch.device("cuda:2")
    if device_arg.lower().startswith("cuda:"):
        idx = int(device_arg.split(":")[1]); 
        return torch.device(f"cuda:{idx}") if idx < torch.cuda.device_count() else torch.device("cuda:2")
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
    return torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

def _build_loader_pair(args, tokenizer):
    aug1, aug2, test_ind, test_ood, train_raw, meta = LACL_load_aug(args, tokenizer)
    comb_dset = PairDataset(aug1, aug2)
    comb_train_dl = torch.utils.data.DataLoader(
        comb_dset, shuffle=True, drop_last=False, collate_fn=collate_fn2,
        batch_size=args.train_batch, num_workers=args.num_workers
    )
    prep_dl = torch.utils.data.DataLoader(
        train_raw, batch_size=min(100, args.train_batch), collate_fn=collate_fn,
        shuffle=True, drop_last=False
    )
    return comb_train_dl, prep_dl, test_ind, test_ood, meta

def _grad_l2_norm(model) -> float:
    sqsum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sqsum += float(g.pow(2).sum().item())
    return float(math.sqrt(max(sqsum, 0.0)))

def train(args, model, tokenizer):    
    trace_path = args.debug_trace_path
    tracer = TraceLogger(trace_path if args.debug_enabled else None)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    comb_train_dl, prep_dl, test_ind, test_ood, meta = _build_loader_pair(args, tokenizer)
    total_steps = len(comb_train_dl) * int(args.num_train_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate * args.eta_min)

    model = model.to(args.device)
    global_step = 0

    best_score = -float("inf") if args.monitor_mode == "max" else float("inf")
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
    saved_any = False

    try:
        for epoch in range(int(args.num_train_epochs)):
            if args.refresh_aug_each_epoch:
                comb_train_dl, prep_dl, test_ind, test_ood, meta = _build_loader_pair(args, tokenizer)

            model.train()
            train_bar = tqdm(comb_train_dl, desc=f"E{epoch}")
            total_cl_loss = total_reg_loss = 0.0

            for idx, (batch1, batch2) in enumerate(train_bar):
                batch1 = {k: v.to(args.device) for k, v in batch1.items()}
                batch2 = {k: v.to(args.device) for k, v in batch2.items()}

                b1 = model(**batch1)
                b2 = model(**batch2)

                label = batch1['labels']
                cont = torch.cat([b1['global_projection'], b2['global_projection']], dim=0)
                sim_mat = get_sim_mat(cont)
                loss_cl = Supervised_NT_xent(sim_mat, labels=label, temperature=args.temperature)
                tok_embeddings = [b1['lw_mean_embedding'], b2['lw_mean_embedding']]
                loss_cr = reg_loss(tok_embeddings, margin=args.cr_margin)

                total_loss = loss_cl + args.reg_loss_weight * loss_cr
                total_cl_loss += float(loss_cl.item())
                total_reg_loss += float(loss_cr.item())

                optimizer.zero_grad()
                total_loss.backward()

                if tracer.f and (global_step % max(1, args.debug_every) == 0):
                    with torch.no_grad():
                        z1 = F.normalize(b1['global_projection'], dim=-1)
                        z2 = F.normalize(b2['global_projection'], dim=-1)
                        pair_cos = torch.sum(z1 * z2, dim=-1)
                        pair_cos_mean = float(pair_cos.mean().item())
                        grad_norm = _grad_l2_norm(model)
                        lr_now = get_lr(optimizer)
                        mem_mb = float(torch.cuda.memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                        tracer.log({
                            "type": "train", "step": int(global_step), "epoch": int(epoch),
                            "loss_total": float(total_loss.item()),
                            "loss_cl": float(loss_cl.item()),
                            "loss_cr": float(loss_cr.item()),
                            "pair_cos_mean": pair_cos_mean,
                            "lr": lr_now, "grad_norm_l2": grad_norm, "gpu_mem_mb": mem_mb,
                        })

                optimizer.step(); scheduler.step(); global_step += 1
                train_bar.set_description(f"E{epoch} | CL {total_cl_loss/(idx+1):.3f} | CR {total_reg_loss/(idx+1):.3f}")

            if epoch % args.log_interval == 0:
                model.prepare_ood(prep_dl, device=args.device)
                base_results = evaluate_ood(args, model, test_ind, test_ood, f"ood_{args.task_name}", meta)
                print(base_results)
                write_log2(args, args.model_name, base_results)

                cand_key = f"ood_{args.task_name}_{args.monitor}"
                mon_key = cand_key if cand_key in base_results else None
                if mon_key is None:
                    for k in base_results.keys():
                        if k.endswith(args.monitor):
                            mon_key = k; break

                if mon_key is None:
                    print(f"[WARN] monitor key not found (suffix='{args.monitor}'); skip ES/CKPT.")
                else:
                    score = float(base_results[mon_key])
                    improved = (score > best_score + args.min_delta) if args.monitor_mode == "max" \
                               else (score < best_score - args.min_delta)
                    if improved:
                        best_score = score; epochs_no_improve = 0
                        os.makedirs(args.save_dir, exist_ok=True)
                        to_save = model.module if hasattr(model, "module") else model
                        torch.save(to_save.state_dict(), best_ckpt_path)
                        saved_any = True
                        print(f"[CKPT] New best {mon_key}={best_score:.6f} → {best_ckpt_path}")
                    else:
                        epochs_no_improve += 1
                        print(f"[ES] No improvement on {mon_key} ({epochs_no_improve}/{args.patience})")
                        if args.patience and epochs_no_improve >= args.patience:
                            print(f"[EARLY STOP] Patience reached on {mon_key}. Best={best_score:.6f}")
                            return

        if not saved_any:
            os.makedirs(args.save_dir, exist_ok=True)
            to_save = model.module if hasattr(model, "module") else model
            torch.save(to_save.state_dict(), best_ckpt_path)
        print(f"[SAVE] Best checkpoint at: {best_ckpt_path}")

    finally:
        tracer.close()

def _parse_layers(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description="LaCL training")
    p.add_argument("--data_dir", type=str, default="../../../00_data/02_classification_0808")
    p.add_argument("--save_dir", type=str, default="../../run/logs/3_ood_result/3_lacl")
    p.add_argument("--log_dir",  type=str, default="../../run/logs/3_ood_result/3_lacl")
    p.add_argument("--model_name_or_path", type=str, default="xlm-roberta-base")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--device", type=str, default="auto", help="'cpu' | 'cuda' | 'cuda:IDX' | 'auto'")
    p.add_argument("--model_name", type=str, default="lacl_xlmr")
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--train_batch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--refresh_aug_each_epoch", action="store_true", default=True)
    p.add_argument("--no_refresh_aug_each_epoch", dest="refresh_aug_each_epoch", action="store_false")
    p.add_argument("--no_dropout", action="store_true", default=False)
    p.add_argument("--task_name", type=str, default="food")
    p.add_argument("--split", action="store_true", default=False)
    p.add_argument("--split_ratio", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--eta_min", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_epsilon", type=float, default=1e-6)
    p.add_argument("--augment_both", type=str, default="None")
    p.add_argument("--augment_method", type=str, default="None")
    p.add_argument("--span_mask_len", type=int, default=5)
    p.add_argument("--cutoff_ratio", type=float, default=0.15)
    p.add_argument("--gp_location", type=str, default="token", choices=["token", "cls"])
    p.add_argument("--gp_layers", type=_parse_layers, default=list(range(1,13)))
    p.add_argument("--encoder_dim", type=int, default=1024)
    p.add_argument("--projection_dim", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--reg_loss_weight", type=float, default=0.1)
    p.add_argument("--cr_margin", type=float, default=0.5)
    p.add_argument("--cosine_top_k", type=int, default=1)
    p.add_argument("--maha_via_cosine", action="store_true", default=False)
    p.add_argument("--use_ood_val", action="store_true", default=False,
                   help="out_of_distribution_valid.csv / out_of_distribution_test.csv 사용")
    p.add_argument("--eval_split", type=str, default="valid", choices=["valid","test"],
                   help="--use_ood_val일 때 어떤 OOD split으로 평가/FP 저장할지")
    p.add_argument("--debug_enabled", action="store_true", default=True)
    p.add_argument("--debug_every", type=int, default=50)
    p.add_argument("--debug_trace_path", type=str, default=None)
    p.add_argument("--monitor", type=str, default="cosine_auroc",
                   help="감시 지표 접미사(예: cosine_auroc, maha_auroc, cosine_fpr95)")
    p.add_argument("--monitor_mode", type=str, default="max", choices=["max", "min"])
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min_delta", type=float, default=1e-4)
    return p

def main(args):
    start_time = time.time()
    args.n_gpu  = torch.cuda.device_count()
    args.device = _pick_device(args.device)
    set_seed(args)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=cal_num_labels(args))
    if args.no_dropout:
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
    config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    set_pad_token_id(tokenizer.pad_token_id)

    model = LACL.from_pretrained(args.model_name_or_path, config=config, args=args).to(args.device)

    print(vars(args))
    train(args, model, tokenizer)

    elapsed = time.time() - start_time
    print(f"[총 실행 시간] {elapsed/60:.2f} 분 ({elapsed:.1f} 초)")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "log_dir", None):
        args.log_dir = args.save_dir
    if not getattr(args, "debug_trace_path", None):
        args.debug_trace_path = os.path.join(args.save_dir, "train_debug.jsonl")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
