import argparse, json, os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")

# ======================= Utils =======================
PAD_ID = 0

def set_pad_id(pad_id: int):
    global PAD_ID
    PAD_ID = int(pad_id) if pad_id is not None else 0

def set_seed(seed: int, n_gpu: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def load_food_datasets(data_dir, tokenizer, max_len=512, for_train=True):
    tr = pd.read_csv(Path(data_dir) / "in_distribution_train.csv")
    va = pd.read_csv(Path(data_dir) / "in_distribution_valid.csv")
    te = pd.read_csv(Path(data_dir) / "in_distribution_test.csv")
    label2id = _ensure_label_map(tr, va, te)
    id2label = {v: k for k, v in label2id.items()}

    train = _tokenize_list(_df_to_examples(tr, label2id, True), tokenizer, max_len) if for_train else None
    valid = _tokenize_list(_df_to_examples(va, label2id, True), tokenizer, max_len)
    test  = _tokenize_list(_df_to_examples(te, label2id, True), tokenizer, max_len)
    return train, valid, test, label2id, id2label

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

        loss = None
        if labels is not None:  # contrastive-like loss (margin or scl)            
            if getattr(self.config, "loss", "margin") == 'margin':
                dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                max_dist = (dist * mask).max()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + \
                           (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()
            else:
                norm_pooled = F.normalize(pooled, dim=-1)
                cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
                mask = mask - torch.diag(torch.diag(mask))
                cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
                cos_loss = -torch.log(cos_loss + 1e-5)
                cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

            ce = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce + getattr(self.config, "alpha", 2.0) * cos_loss

        output = (logits,) + outputs[2:] + (pooled,)
        return ((loss, cos_loss) + output) if loss is not None else output

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
        maha_score = -maha_score  # ID-friendly

        cosine_score = F.normalize(pooled, dim=-1) @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]
        energy_score = torch.logsumexp(logits, dim=-1)  # ID-friendly

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

# ======================= OOD Eval =======================
def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def _fpr_and_threshold_at_tpr95(y_true, y_score):
    y_true = (y_true == 1)  # ID=1, OOD=0
    id_scores  = y_score[y_true]
    ood_scores = y_score[~y_true]
    thr = np.quantile(id_scores, 0.05)  # ID-friendly 스코어 기준, 규칙: score>=thr → ID
    fpr = (ood_scores >= thr).mean() if ood_scores.size > 0 else 0.0
    return float(fpr), float(thr)

def evaluate_ood(args, model, id_features, ood_features, tag):
    keys = ['softmax', 'maha', 'cosine', 'energy']

    in_dl = DataLoader(id_features, batch_size=args.batch_size, collate_fn=collate_fn)
    in_scores = []
    for batch in in_dl:
        model.eval()
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        with torch.no_grad():
            in_scores.append(model.compute_ood(input_ids=input_ids, attention_mask=attention_mask))
    in_scores = merge_keys(in_scores, keys)

    out_dl = DataLoader(ood_features, batch_size=args.batch_size, collate_fn=collate_fn)
    out_scores = []
    for batch in out_dl:
        model.eval()
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        with torch.no_grad():
            out_scores.append(model.compute_ood(input_ids=input_ids, attention_mask=attention_mask))
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:
        ins = np.asarray(in_scores[key], dtype=np.float64)
        outs = np.asarray(out_scores[key], dtype=np.float64)
        labels = np.concatenate([np.ones_like(ins), np.zeros_like(outs)])
        scores = np.concatenate([ins, outs])  # ID-friendly

        auroc = roc_auc_score(labels, scores)
        fpr95, thr95 = _fpr_and_threshold_at_tpr95(labels, scores)
        outputs[f"{tag}_{key}_auroc"] = float(auroc)
        outputs[f"{tag}_{key}_fpr95"] = float(fpr95)
        outputs[f"{tag}_{key}_thr95"] = float(thr95)
    return outputs

# ======================= Train/Eval loops =======================
def _only_tensor_items(d):
    return {k: v for k, v in d.items() if torch.is_tensor(v)}

def _to_device(batch, device):
    tb = _only_tensor_items(batch)
    return {k: v.to(device) for k, v in tb.items()}

def _get_inputs(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask", "labels") if k in batch}

def evaluate_id(args, model, eval_dataset, tag="dev"):
    def compute_metrics(logits_np, labels_np):
        preds = np.argmax(logits_np, axis=1)
        acc = (preds == labels_np).mean().item()
        return {"accuracy": acc, "score": acc}

    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    label_list, logit_list = [], []
    model.eval()
    for _, batch in enumerate(tqdm(dataloader, desc=f"[EVAL:{tag}]")):
        labels = batch["labels"].detach().cpu().numpy()
        inputs = _get_inputs(batch); inputs["labels"] = None
        inputs = _to_device(inputs, args.device)
        with torch.no_grad():
            logits = model(**inputs)[0].detach().cpu().numpy()
        label_list.append(labels); logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    return {"{}_{}".format(tag, k): v for k, v in results.items()}

def _pick_best_fpr95(ood_results_all: dict):
    items = [(k, v) for k, v in ood_results_all.items() if k.endswith("_fpr95")]
    best_key, best_val = min(items, key=lambda kv: kv[1])
    return best_key, best_key, best_val

def train(args):    
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    set_pad_id(tokenizer.pad_token_id)

    # datasets
    train_ds, dev_ds, test_ds, _, _ = load_food_datasets(
        args.data_dir, tokenizer, max_len=args.max_seq_length, for_train=True
    )

    # OOD datasets (VALID & TEST)
    tag_valid, ood_valid = load_ood_from_csv(
        str(Path(args.data_dir) / "out_of_distribution_valid.csv"),
        tokenizer, args.max_seq_length
    )
    tag_test,  ood_test  = load_ood_from_csv(
        str(Path(args.data_dir) / "out_of_distribution_test.csv"),
        tokenizer, args.max_seq_length
    )
    benchmarks = ((tag_valid, ood_valid), (tag_test, ood_test))

    # model
    config = XLMRobertaConfig.from_pretrained(args.model_name_or_path, num_labels=7)
    config.gradient_checkpointing = True
    config.alpha = args.alpha
    config.loss = args.loss
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    if getattr(config, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model.to(args.device)

    # optional warm-start
    if args.init_from_checkpoint:
        sd = torch.load(args.init_from_checkpoint, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[init_from_checkpoint] missing:", missing)
        print("[init_from_checkpoint] unexpected:", unexpected)

    # optim/scheduler
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    total_steps = int(len(train_loader) * int(args.num_train_epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # logging dir
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metrics_path = outdir / "metrics.jsonl"

    best = {"dev_score": -1.0, "epoch": 0}
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(int(args.num_train_epochs)):
        model.train()
        pbar = tqdm(DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True),
                    desc=f"[TRAIN] epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            inputs = _get_inputs(batch)
            inputs = _to_device(inputs, args.device)
            outputs = model(**inputs)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            optimizer.step(); scheduler.step(); model.zero_grad()
            global_step += 1
            if (step + 1) % 50 == 0 or (step + 1) == len(pbar):
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "cos": f"{cos_loss.item():.4f}"})

        # epoch-end ID eval
        dev_results  = evaluate_id(args, model, dev_ds, tag="dev")
        test_results = evaluate_id(args, model, test_ds, tag="test")
        dev_score = dev_results["dev_score"]

        # OOD eval (prepare on dev)
        model.prepare_ood(DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate_fn))
        ood_results_all = {}
        for tag, ood_features in benchmarks:
            r = evaluate_ood(args, model, test_ds, ood_features, tag=tag)
            ood_results_all.update(r)

        # summarize best fpr95
        best_key, best_tag, best_val = _pick_best_fpr95(ood_results_all)
        best_row = {
            "epoch": epoch + 1,
            "dev_accuracy": dev_results["dev_accuracy"],
            "test_accuracy": test_results["test_accuracy"],
            "best_fpr95_key": best_key,
            "best_fpr95_tag": best_tag,
            "best_fpr95": best_val,
            "best_auroc": ood_results_all[best_tag.replace("_fpr95","_auroc")],
            "best_thr95": ood_results_all[best_tag.replace("_fpr95","_thr95")],
        }

        row = {"epoch": epoch + 1, "step": global_step, **dev_results, **test_results, **ood_results_all, "_best": best_row}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # early stopping based on dev_score
        if dev_score > best["dev_score"] + args.early_stopping_min_delta:
            best.update({"dev_score": dev_score, "epoch": epoch + 1})
            torch.save(model.state_dict(), outdir / "best_model.pth")
            epochs_no_improve = 0
            print(f"  ↳ Best updated (epoch {epoch+1}) — saved to {outdir/'best_model.pth'}")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement ({epochs_no_improve}/{args.early_stopping_patience})")
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"[EARLY STOP] Stop at epoch {epoch+1}."); break

    print(f"[DONE] Best epoch={best['epoch']} (dev_score={best['dev_score']:.4f})")
    print(f"       Saved best_model.pth under {outdir}")

def parse_args():
    p = argparse.ArgumentParser()    
    p.add_argument("--model_name_or_path", default="xlm-roberta-base", type=str)
    p.add_argument("--max_seq_length", default=512, type=int)
    p.add_argument("--data_dir", default="../../../00_data/02_classification_0808", type=str)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--learning_rate", default=1e-5, type=float)
    p.add_argument("--adam_epsilon", default=1e-6, type=float)
    p.add_argument("--warmup_ratio", default=0.06, type=float)
    p.add_argument("--weight_decay", default=0.01, type=float)
    p.add_argument("--num_train_epochs", default=10.0, type=float)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--loss", type=str, default="margin", choices=["margin", "scl"])
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init_from_checkpoint", type=str, default="")
    p.add_argument("--output_dir", type=str, default="../../run/logs/3_ood_result/2_contrastive")
    args = p.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args

def main():
    args = parse_args()
    set_seed(args.seed, args.n_gpu)
    start = time.time()
    train(args)
    dur = time.time() - start
    Path(args.output_dir, "runtime.txt").write_text(f"[RUNTIME] total {dur:.2f} sec\n", encoding="utf-8")

if __name__ == "__main__":
    main()