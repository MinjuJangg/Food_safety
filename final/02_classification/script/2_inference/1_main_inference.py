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

def setup_logging(log_dir: str):
    """로깅 설정 함수"""
    log_filename = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        if ttf_paths:
            try:
                fam = fm.FontProperties(fname=ttf_paths[0]).get_name()
                candidates = [fam] + candidates
            except Exception:
                pass
        available = set(f.name for f in fm.fontManager.ttflist)
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            mpl.rcParams['font.family'] = chosen
            mpl.rcParams['axes.unicode_minus'] = False
            mpl.rcParams['pdf.fonttype'] = 42
            mpl.rcParams['ps.fonttype'] = 42
            logging.info(f"한글 폰트 설정 완료: {chosen}")
        else:
            logging.warning("한글 폰트를 찾지 못했습니다. 기본 폰트로 진행합니다.")
    except Exception as e:
        logging.warning(f"한글 폰트 설정 중 예외 발생: {e}")

def load_data(file_path: str) -> pd.DataFrame:
    """CSV 파일을 안전하게 로드하는 함수"""
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
    """유효한 라벨만 필터링하고 숫자로 변환"""
    is_valid = labels.isin(label_mapping.keys())
    valid_labels = labels[is_valid]
    valid_texts = texts[is_valid]
    converted_labels = valid_labels.map(label_mapping).tolist()
    logging.info(f"{len(labels)}개 중 {len(converted_labels)}개의 유효한 라벨을 필터링 및 변환했습니다.")
    return valid_texts.tolist(), converted_labels

class CustomDataset(Dataset):
    """토크나이저를 적용하는 커스텀 데이터셋"""
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

def evaluate_model(model, data_loader, device, mode="Inference"):
    """모델 평가 함수"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_predictions, all_labels, all_probabilities = [], [], []
    loop = tqdm(data_loader, desc=f"{mode}", leave=True)

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

def inference():
    t0 = time.time()

    # 상대경로 기준점
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    FOOD_ROOT    = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

    # 상대경로 설정
    DATA_DIR         = os.path.join(FOOD_ROOT, "00_data", "02_classification_0919")
    FONT_DIR         = os.path.join(FOOD_ROOT, "00_data", "fonts")
    RUN_DIR          = os.path.join(PROJECT_ROOT, "run")
    RESULTS_BASE_DIR = os.path.join(RUN_DIR, "logs", "1_main_result")
    MODEL_DIR        = os.path.join(RUN_DIR, "checkpoints", "1_main_pth")
    
    config = {
        "SEED": 42,
        "BASE_MODEL": "xlm-roberta-base",
        "MODEL_PATH": os.path.join(MODEL_DIR, "main_best.pth"),
        "TEST_DATA_PATH": os.path.join(DATA_DIR, "test.csv"),
        "MAX_LENGTH": 512,
        "BATCH_SIZE": 16,
        "DEVICE_ID": "cuda:0",
        "RESULTS_BASE_DIR": RESULTS_BASE_DIR,
        "FONT_DIR": FONT_DIR,
        "label_mapping": {
            "물리적 위해요소": 0, "생물학적 위해요소": 1, "신규 위해요소": 2,
            "안전위생": 3, "영양건강": 4, "표시광고": 5, "화학적 위해요소": 6
        }
    }

    set_seed(config['SEED'])

    # ===== 1) 결과 디렉토리/로깅/폰트 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(config['RESULTS_BASE_DIR'], f"inference_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    setup_logging(result_dir)
    try_set_korean_font(config["FONT_DIR"])

    # ===== 2) 디바이스 =====
    device = torch.device(config['DEVICE_ID'] if torch.cuda.is_available() else "cpu")
    logging.info(f"사용 디바이스: {device}")

    # ===== 3) 토크나이저/모델 로드 =====
    logging.info(f"토크나이저 로드: {config['BASE_MODEL']}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(config['BASE_MODEL'])

    logging.info("모델 아키텍처를 로드합니다.")
    num_labels = len(config['label_mapping'])
    model = XLMRobertaForSequenceClassification.from_pretrained(config['BASE_MODEL'], num_labels=num_labels)

    logging.info(f"학습된 가중치 로드: {config['MODEL_PATH']}")
    state_dict = torch.load(config['MODEL_PATH'], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # ===== 4) 데이터 =====
    logging.info(f"테스트 데이터 로드: {config['TEST_DATA_PATH']}")
    test_df = load_data(config['TEST_DATA_PATH'])
    logging.info(f"테스트 데이터 크기: {test_df.shape}")

    X_test, Y_test = filter_and_convert_labels(test_df["제목_내용"], test_df["대분류"], config['label_mapping'])

    test_dataset = CustomDataset(X_test, Y_test, tokenizer, config['MAX_LENGTH'])
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'])

    # ===== 5) 평가 =====
    logging.info("모델 평가 시작")
    avg_loss, test_acc, y_preds, y_labels, y_probs = evaluate_model(model, test_loader, device)
    logging.info(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # ===== 6) 결과 저장 =====
    reverse_label_mapping = {v: k for k, v in config['label_mapping'].items()}
    unique_labels = sorted(list(set(y_labels) | set(y_preds)))
    class_names = [reverse_label_mapping.get(i, f"unknown_{i}") for i in unique_labels]
    num_labels_in_data = len(unique_labels)

    # Classification Report
    report = classification_report(
        y_labels, y_preds, labels=unique_labels, target_names=class_names, digits=4, zero_division=0
    )
    logging.info("Classification Report:\n" + report)
    report_path = os.path.join(result_dir, "classification_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    logging.info(f"Classification Report 저장: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_labels, y_preds, labels=unique_labels)
    plt.figure(figsize=(max(10, num_labels_in_data * 1.2), max(8, num_labels_in_data * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Confusion Matrix 저장: {cm_path}")

    # 예측 결과 CSV
    df_results = pd.DataFrame({
        "Text": X_test,
        "True_Label_ID": y_labels,
        "Predicted_Label_ID": y_preds
    })
    df_results["True_Label"] = df_results["True_Label_ID"].map(reverse_label_mapping)
    df_results["Predicted_Label"] = df_results["Predicted_Label_ID"].map(reverse_label_mapping)

    num_total_labels = len(config['label_mapping'])
    prob_columns = {
        f"Prob_{reverse_label_mapping.get(i, f'unknown_{i}')}": [p[i] for p in y_probs]
        for i in range(num_total_labels)
    }
    prob_df = pd.DataFrame(prob_columns)
    df_results = pd.concat([df_results, prob_df], axis=1)

    csv_path = os.path.join(result_dir, "prediction_results.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logging.info(f"예측 결과 CSV 저장: {csv_path}")

    elapsed = time.time() - t0
    logging.info(f"총 소요시간: {elapsed/60:.2f}분 ({elapsed:.1f}초)")
    logging.info(f"모든 추론 결과가 다음 폴더에 저장되었습니다: {result_dir}")

if __name__ == '__main__':
    inference()