import os
import sys
import time
import random
import logging
from datetime import datetime
from tqdm import tqdm
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

csv.field_size_limit(sys.maxsize)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: str, config: dict):
    log_filename = f"{config['TIMESTAMP']}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("로깅 설정 완료")
    logging.info(f"로그 파일: {log_filepath}")

def try_set_korean_font(font_dir: str):  # 폰트 설정
    try:
        if not os.path.isdir(font_dir):
            logging.warning(f"폰트 디렉토리를 찾을 수 없습니다: {font_dir}")
            return
        ttf_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
        for p in ttf_paths:
            try:
                fm.fontManager.addfont(p)
            except Exception as e:
                logging.warning(f"폰트 등록 실패: {p} ({e})")
        candidates = [
            "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR",
            "Pretendard", "AppleGothic", "Malgun Gothic"
        ]
        family_from_file = None
        if ttf_paths:
            try:
                family_from_file = fm.FontProperties(fname=ttf_paths[0]).get_name()
            except Exception:
                pass
        if family_from_file:
            candidates = [family_from_file] + candidates
        available = set(f.name for f in fm.fontManager.ttflist)
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            mpl.rcParams['font.family'] = chosen
            mpl.rcParams['axes.unicode_minus'] = False
            logging.info(f"한글 폰트 설정 완료: {chosen}")
        else:
            logging.warning("한글 폰트를 찾지 못했습니다. 기본 폰트로 진행합니다.")
    except Exception as e:
        logging.warning(f"한글 폰트 설정 중 예외 발생: {e}")

def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, encoding='utf-8', engine='python')
    except Exception as e:
        logging.warning(f"'{file_path}' 로드 중 오류 발생: {e}. 'on_bad_lines' 옵션으로 재시도합니다.")
        try:
            return pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='skip')
        except Exception as e2:
            logging.error(f"'{file_path}' 로드 최종 실패: {e2}")
            raise

def filter_and_convert_labels(texts: pd.Series, labels: pd.Series, label_mapping: dict):
    is_valid = labels.isin(label_mapping.keys())
    valid_labels = labels[is_valid]
    valid_texts = texts[is_valid]
    converted_labels = valid_labels.map(label_mapping).tolist()
    logging.info(f"{len(labels)}개 중 {len(converted_labels)}개의 유효한 라벨을 필터링 및 변환했습니다.")
    return valid_texts.tolist(), converted_labels

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            text = ""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(model, data_loader, optimizer, device, epoch, config, global_step, checkpoint_path):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    loop = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Training Epoch {epoch+1}", leave=False)

    for step, batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        loop.set_postfix(loss=loss.item())
        global_step += 1

        if global_step % config['SAVE_INTERVAL'] == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            logging.info(f"체크포인트 저장 완료 (Step: {global_step})")

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy, global_step

def evaluate_model(model, data_loader, device, mode="validation"):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_predictions, all_labels, all_probabilities = [], [], []
    loop = tqdm(data_loader, desc=f"{mode.capitalize()} Evaluating", leave=True)

    with torch.no_grad():
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities

def main():
    t0 = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    C02_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    FOOD_ROOT = os.path.abspath(os.path.join(C02_DIR, ".."))
    DATA_DIR = os.path.join(FOOD_ROOT, "00_data", "02_classification_0919")
    FONT_DIR = os.path.join(FOOD_ROOT, "00_data", "fonts")
    RUN_DIR = os.path.join(C02_DIR, "run")
    BASE_SAVE_DIR = os.path.join(RUN_DIR, "logs", "1_main_result")
    BEST_MODEL_DIR = os.path.join(RUN_DIR, "checkpoints", "1_main_pth")

    config = {
        "SEED": 42,
        "BASE_MODEL": "xlm-roberta-base",
        "EXPERIMENT_NAME_PREFIX": "id_classification",
        "DATASET_NAME": "data_0919",
        "MAX_LENGTH": 512,
        "BATCH_SIZE": 16,
        "EPOCHS": 10,
        "LEARNING_RATE": 2e-5,
        "EARLY_STOPPING_PATIENCE": 3,
        "SAVE_INTERVAL": 1000,
        "DEVICE_ID": "cuda:0",

        # --- 데이터 경로 ---
        "train_path": os.path.join(DATA_DIR, "train.csv"),
        "val_path":   os.path.join(DATA_DIR, "valid.csv"),
        "test_path":  os.path.join(DATA_DIR, "test.csv"),

        # --- 저장 경로 ---
        "base_save_dir": BASE_SAVE_DIR,

        # --- 라벨 매핑 ---
        "label_mapping": {
            "물리적 위해요소": 0, "생물학적 위해요소": 1, "신규 위해요소": 2,
            "안전위생": 3, "영양건강": 4, "표시광고": 5, "화학적 위해요소": 6
        },

        # --- 최적 모델 저장 폴더 ---
        "best_model_dir": BEST_MODEL_DIR
    }

    set_seed(config['SEED'])

    # ===== 1. 실험 디렉토리 및 로깅 설정 =====
    config['TIMESTAMP'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['TIMESTAMP']}_{config['EXPERIMENT_NAME_PREFIX']}_{config['DATASET_NAME']}"
    experiment_dir = os.path.join(config['base_save_dir'], experiment_name)
    log_dir = os.path.join(experiment_dir, "logs")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    result_dir = os.path.join(experiment_dir, "results")
    best_model_dir = config['best_model_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    setup_logging(log_dir, config)
    try_set_korean_font(FONT_DIR)

    # ===== 2. 데이터 준비 =====
    logging.info("데이터 로드를 시작합니다.")
    train_df = load_data(config['train_path'])
    val_df = load_data(config['val_path'])
    test_df = load_data(config['test_path'])
    logging.info(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

    # 라벨 필터링 및 변환
    X_train, Y_train = filter_and_convert_labels(train_df["제목_내용"], train_df["대분류"], config['label_mapping'])
    X_val, Y_val = filter_and_convert_labels(val_df["제목_내용"], val_df["대분류"], config['label_mapping'])
    X_test, Y_test = filter_and_convert_labels(test_df["제목_내용"], test_df["대분류"], config['label_mapping'])

    # ===== 3. 토크나이저 및 데이터로더 =====
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['BASE_MODEL'])
    train_dataset = CustomDataset(X_train, Y_train, tokenizer, config['MAX_LENGTH'])
    val_dataset = CustomDataset(X_val, Y_val, tokenizer, config['MAX_LENGTH'])
    test_dataset = CustomDataset(X_test, Y_test, tokenizer, config['MAX_LENGTH'])
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'])
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'])

    # ===== 4. 모델 및 최적화 준비 =====
    device = torch.device(config['DEVICE_ID'] if torch.cuda.is_available() else "cpu")
    logging.info(f"사용 디바이스: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    num_labels = len(config['label_mapping'])
    model = XLMRobertaForSequenceClassification.from_pretrained(config['BASE_MODEL'], num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['LEARNING_RATE'])

    # ===== 5. 체크포인트 로드 (재개 기능) =====
    global_step = 0
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        logging.info(f"체크포인트 발견: {checkpoint_path}, 로드합니다...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logging.info(f"Epoch {start_epoch}, Global Step {global_step} 부터 학습을 재개합니다.")

    # ===== 6. 학습 루프 (Early Stopping 포함) =====
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(best_model_dir, "main_best.pth")

    logging.info("모델 학습을 시작합니다.")
    for epoch in range(start_epoch, config['EPOCHS']):
        train_loss, train_acc, global_step = train_model(
            model, train_loader, optimizer, device, epoch, config, global_step, checkpoint_path
        )
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, device, mode="validation")

        logging.info(
            f"Epoch {epoch + 1}/{config['EPOCHS']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Validation Loss 개선 ({best_val_loss:.4f}). 최적 모델 저장: {best_model_path}")
        else:
            patience_counter += 1
            logging.info(f"Early stopping counter: {patience_counter}/{config['EARLY_STOPPING_PATIENCE']}")
            if patience_counter >= config['EARLY_STOPPING_PATIENCE']:
                logging.info("Early stopping 발동. 학습을 조기 종료합니다.")
                break

    # ===== 7. 최종 평가 및 결과 저장 =====
    logging.info("최적 모델을 로드하여 최종 평가를 진행합니다.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    _, test_acc, y_preds, y_labels, y_probs = evaluate_model(model, test_loader, device, mode="test")
    logging.info(f"최종 Test Accuracy: {test_acc:.4f}")

    # --- Classification Report 저장 ---
    reverse_label_mapping = {v: k for k, v in config['label_mapping'].items()}
    class_names = [reverse_label_mapping[i] for i in range(num_labels)]
    report = classification_report(y_labels, y_preds, target_names=class_names, digits=4)
    logging.info("Classification Report:\n" + report)
    report_path = os.path.join(result_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logging.info(f"Classification Report 저장 완료: {report_path}")

    # --- Confusion Matrix 저장 ---
    cm = confusion_matrix(y_labels, y_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion Matrix 저장 완료: {cm_path}")

    # --- 예측 결과 CSV 저장 ---
    df_results = pd.DataFrame({
        "Text": [t for t in X_test],
        "True_Label_ID": y_labels,
        "Predicted_Label_ID": y_preds
    })
    df_results["True_Label"] = df_results["True_Label_ID"].map(reverse_label_mapping)
    df_results["Predicted_Label"] = df_results["Predicted_Label_ID"].map(reverse_label_mapping)
    for i, label_name in enumerate(class_names):
        df_results[f"Prob_{label_name}"] = [p[i] for p in y_probs]
    csv_path = os.path.join(result_dir, "prediction_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"예측 결과 CSV 저장 완료: {csv_path}")
   
    elapsed = time.time() - t0
    logging.info(f"총 소요시간: {elapsed/60:.2f}분 ({elapsed:.1f}초)")
    logging.info(f"모든 실험 결과가 다음 폴더에 저장되었습니다: {experiment_dir}")

if __name__ == '__main__':
    main()