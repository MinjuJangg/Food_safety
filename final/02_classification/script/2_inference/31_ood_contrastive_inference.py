import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from tqdm import tqdm

# ======================= Utils =======================
PAD_ID = 0

def set_pad_id(pad_id: int):
    global PAD_ID
    PAD_ID = int(pad_id) if pad_id is not None else 0

def collate_fn(batch):
    pad_id = PAD_ID
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids = [f["input_ids"] + [pad_id] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1] * len(f["input_ids"]) + [0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    texts  = [f.get("raw_text", "") for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": input_mask, "labels": labels, "texts": texts}

def _only_tensor_items(d):
    return {k: v for k, v in d.items() if torch.is_tensor(v)}

def _to_device(batch, device):
    tb = _only_tensor_items(batch)
    return {k: v.to(device) for k, v in tb.items()}

def _get_inputs(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask", "labels") if k in batch}

# ======================= Data =======================
FOOD_LABEL_ORDER = [
    "물리적 위해요소", "생물학적 위해요소", "신규 위해요소",
    "안전위생", "영양건강", "표시광고", "화학적 위해요소"
]

def _tokenize_list(examples, tokenizer, max_seq_length):
    def _pp(ex):
        res = tokenizer(ex["text"], max_length=max_seq_length, truncation=True)
        res["labels"] = ex.get("label", 0)
        res["raw_text"] = ex.get("text", "")
        return res
    return list(map(_pp, examples))

def _csv_to_ood_examples(csv_path):
    df = pd.read_csv(csv_path)
    col = "text"
    if "text" not in df.columns and "제목_내용" in df.columns:
        col = "제목_내용"
    exs = [{"text": str(t), "label": 0} for t in df[col].tolist()]
    return {"test": exs}

def load_ood_from_csv(path, tokenizer, max_seq_length=512):
    d = _csv_to_ood_examples(path)
    tag = f"ood_csv_{Path(path).stem}"
    feats = _tokenize_list(d["test"], tokenizer, max_seq_length)
    return tag, feats

def _ensure_label_map(train_df, val_df, test_df):
    uniq = sorted(set(train_df["대분류"]) | set(val_df["대분류"]) | set(test_df["대분류"]))
    base = FOOD_LABEL_ORDER.copy()
    for u in uniq:
        if u not in base:
            base.append(u)
    return {lab: i for i, lab in enumerate(base)}

def _df_to_examples(df, label2id, use_label=True):
    exs = []
    for _, r in df.iterrows():
        text = str(r["제목_내용"])
        if use_label:
            exs.append({"text": text, "label": int(label2id[str(r["대분류"])])})
        else:
            exs.append({"text": text, "label": 0})
    return exs

def load_food_datasets(data_dir, tokenizer, max_len=512):
    tr = pd.read_csv(Path(data_dir) / "in_distribution_train.csv")
    va = pd.read_csv(Path(data_dir) / "in_distribution_valid.csv")
    te = pd.read_csv(Path(data_dir) / "in_distribution_test.csv")
    label2id = _ensure_label_map(tr, va, te)
    id2label = {v: k for k, v in label2id.items()}
    dev  = _tokenize_list(_df_to_examples(va, label2id, True), tokenizer, max_len)
    test = _tokenize_list(_df_to_examples(te, label2id, True), tokenizer, max_len)
    return dev, test, label2id, id2label

# ======================= Model =======================
class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = pooled = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, pooled

class XLMRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)
        output = (logits,) + outputs[2:] + (pooled,)
        return output

    def compute_ood(self, input_ids=None, attention_mask=None, labels=None):
        self.eval()
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)

        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

        maha_score = []
        for c in self.all_classes:
            centered = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered @ self.class_var @ centered.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1).min(-1)[0]
        maha_score = -maha_score

        cosine_score = F.normalize(pooled, dim=-1) @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]
        energy_score = torch.logsumexp(logits, dim=-1)

        return {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist()
        }

    def prepare_ood(self, dataloader=None):
        self.eval()
        device = next(self.parameters()).device
        bank, label_bank = [], []

        with torch.inference_mode():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                logits, pooled = self.classifier(out[0])
                bank.append(pooled.detach().cpu())
                label_bank.append(labels.detach().cpu())

        self.bank = torch.cat(bank, 0).to(device, non_blocking=True)
        self.label_bank = torch.cat(label_bank, 0).to(device, non_blocking=True)
        self.norm_bank = F.normalize(self.bank, dim=-1)

        classes = torch.unique(self.label_bank).tolist()
        self.all_classes = classes
        d = self.bank.size(1)
        self.class_mean = torch.zeros(max(classes)+1, d, device=device)
        for c in classes:
            self.class_mean[c] = self.bank[self.label_bank == c].mean(0)

        centered = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).to(device)

# ======================= OOD Evaluation =======================
def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def _fpr_and_threshold_at_tpr95(y_true, y_score):
    y_true = (y_true == 1)
    id_scores  = y_score[y_true]
    ood_scores = y_score[~y_true]
    thr = np.quantile(id_scores, 0.05)
    fpr = (ood_scores >= thr).mean() if ood_scores.size > 0 else 0.0
    return float(fpr), float(thr)

def evaluate_id_accuracy(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    labels_all, logits_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[EVAL:ID]"):
            labels = batch["labels"].detach().cpu().numpy()
            inputs = _get_inputs(batch); inputs["labels"] = None
            inputs = _to_device(inputs, device)
            logits = model(**inputs)[0].detach().cpu().numpy()
            labels_all.append(labels)
            logits_all.append(logits)
    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    preds = np.argmax(logits_all, axis=1)
    acc = (preds == labels_all).mean().item()
    return float(acc)

def evaluate_ood(model, id_features, ood_features, device, batch_size, tag):
    keys = ['softmax', 'maha', 'cosine', 'energy']

    in_dl = DataLoader(id_features, batch_size=batch_size, collate_fn=collate_fn)
    out_dl = DataLoader(ood_features, batch_size=batch_size, collate_fn=collate_fn)

    in_scores = []
    with torch.no_grad():
        for batch in in_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            in_scores.append(model.compute_ood(input_ids=input_ids, attention_mask=attention_mask))
    in_scores = merge_keys(in_scores, keys)

    out_scores = []
    with torch.no_grad():
        for batch in out_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            out_scores.append(model.compute_ood(input_ids=input_ids, attention_mask=attention_mask))
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:
        ins = np.asarray(in_scores[key], dtype=np.float64)
        outs = np.asarray(out_scores[key], dtype=np.float64)
        labels = np.concatenate([np.ones_like(ins), np.zeros_like(outs)])
        scores = np.concatenate([ins, outs])
        auroc = roc_auc_score(labels, scores)
        fpr95, thr95 = _fpr_and_threshold_at_tpr95(labels, scores)
        outputs[f"{tag}_{key}_auroc"] = float(auroc)
        outputs[f"{tag}_{key}_fpr95"] = float(fpr95)
        outputs[f"{tag}_{key}_thr95"] = float(thr95)
    return outputs

def _pick_best_fpr95(ood_results_all: dict):
    items = [(k, v) for k, v in ood_results_all.items() if k.endswith("_fpr95")]
    best_key, best_val = min(items, key=lambda kv: kv[1])
    return best_key, best_key, best_val

def _compute_scores_and_preds(model, dataset, device, batch_size, score_key):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    scores, preds, labels, texts = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = _get_inputs(batch); inputs["labels"] = None
            inputs = _to_device(inputs, device)
            logits = model(**inputs)[0]
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            texts.extend(batch["texts"])
            s = model.compute_ood(**_to_device(_get_inputs(batch), device))[score_key]
            scores.extend(s)
    return np.array(scores), np.array(preds), np.array(labels), texts

def dump_error_csvs(outdir, model, id_test, ood_feats, id2label, device, batch_size, best_metric_key, thr):
    import csv
    parts = best_metric_key.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected best_metric_key: {best_metric_key}")
    tag_name, score_key, _ = parts

    id_scores, id_preds, id_labels, id_texts = _compute_scores_and_preds(model, id_test, device, batch_size, score_key)
    id_false_reject = id_scores < float(thr)
    with open(Path(outdir) / "id_false_reject_at_tpr95.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","ood_tag","method","threshold","score","true_label","pred_label","text"])
        for sc, y, yhat, tx, flag in zip(id_scores, id_labels, id_preds, id_texts, id_false_reject):
            if flag:
                w.writerow(["ID", tag_name, score_key, f"{thr:.6f}", f"{sc:.6f}",
                           id2label.get(int(y), f"label_{int(y)}"),
                           id2label.get(int(yhat), f"label_{int(yhat)}"),
                           tx])

    ood_scores, ood_preds, _, ood_texts = _compute_scores_and_preds(model, ood_feats, device, batch_size, score_key)
    ood_false_accept = ood_scores >= float(thr)
    with open(Path(outdir) / "ood_false_accept_at_tpr95.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split","ood_tag","method","threshold","score","true_label","pred_label","text"])
        for sc, yhat, tx, flag in zip(ood_scores, ood_preds, ood_texts, ood_false_accept):
            if flag:
                w.writerow(["OOD", tag_name, score_key, f"{thr:.6f}", f"{sc:.6f}",
                           "N/A",
                           id2label.get(int(yhat), f"label_{int(yhat)}"),
                           tx])

# ======================= Main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="xlm-roberta-base", type=str)
    ap.add_argument("--ckpt_path", default="../../run/logs/3_ood_result/2_contrastive/best_model.pth", type=str)
    ap.add_argument("--data_dir", default="../../../00_data/02_classification_0808", type=str)
    ap.add_argument("--output_dir", default="../../run/logs/3_ood_result/2_contrastive", type=str)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--max_seq_length", default=512, type=int)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    set_pad_id(tokenizer.pad_token_id)

    # datasets
    dev_ds, test_ds, label2id, id2label = load_food_datasets(args.data_dir, tokenizer, args.max_seq_length)    
    tag_valid, ood_valid = load_ood_from_csv(str(Path(args.data_dir) / "out_of_distribution_valid.csv"),
                                             tokenizer, args.max_seq_length)
    tag_test,  ood_test  = load_ood_from_csv(str(Path(args.data_dir) / "out_of_distribution_test.csv"),
                                             tokenizer, args.max_seq_length)
    benchmarks = ((tag_valid, ood_valid), (tag_test, ood_test))

    # model
    config = XLMRobertaConfig.from_pretrained(args.model_name_or_path, num_labels=7)
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    # load checkpoint weights
    if args.ckpt_path:
        sd = torch.load(args.ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[load ckpt] missing:", missing)
        print("[load ckpt] unexpected:", unexpected)

    # ID accuracy
    id_acc = evaluate_id_accuracy(model, test_ds, device, args.batch_size)
    print(f"[ID] test_accuracy={id_acc:.4f}")

    # OOD prepare on dev
    model.prepare_ood(DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate_fn))

    all_ood = {}
    for tag, ood_feats in benchmarks:
        r = evaluate_ood(model, test_ds, ood_feats, device, args.batch_size, tag)
        all_ood.update(r)

    # choose best by FPR@TPR95
    best_key, metric_key, best_val = _pick_best_fpr95(all_ood)
    best = {
        "best_metric_key": metric_key,
        "best_fpr95": best_val,
        "best_auroc": all_ood[metric_key.replace("_fpr95","_auroc")],
        "best_thr95": all_ood[metric_key.replace("_fpr95","_thr95")],
        "test_accuracy": id_acc
    }

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(outdir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"test_accuracy": id_acc, **all_ood, "_best": best}, ensure_ascii=False) + "\n")
    print(f"[BEST] {metric_key} | FPR@TPR95={best['best_fpr95']:.4f}, AUROC={best['best_auroc']:.4f}, thr={best['best_thr95']:.6f}")
   
    tag_name = best_key.rsplit("_", 2)[0] 
    chosen_ood = None
    for tag, feats in benchmarks:
        if tag == tag_name:
            chosen_ood = feats; break
    if chosen_ood is not None:
        dump_error_csvs(outdir, model, test_ds, chosen_ood, id2label, device, args.batch_size,
                        best_metric_key=metric_key, thr=best["best_thr95"])

if __name__ == "__main__":
    main()