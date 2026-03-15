import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

from classification.face.Implementation.xception_optimized.advanced_xception import improved_xception
from classification.face.Implementation.advanced_transforms import DeepfakeDataset, val_test_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audios_videos")
MODEL_PATH = os.path.join(PROJECT_ROOT, "classification", "face", "models", "model_advanced_xception")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "classification", "multimodal", "output", "output_face")

BATCH_SIZE = 64
NUM_WORKERS = 10


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'fake_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'real_detection_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def plot_confusion_matrix(cm, save_path, labels=["real", "fake"]):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_misclassifications(model, test_loader, device):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Analyzing misclassifications', leave=False):
            images = images.to(device)
            labels_np = labels.numpy()

            with autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            preds = (probs >= 0.5).astype(int)
            for idx in range(len(labels)):
                if labels_np[idx] != preds[idx]:
                    misclassified.append({
                        'image_path': paths[idx],
                        'true_label': 'fake' if labels_np[idx] == 1 else 'real',
                        'predicted_label': 'fake' if preds[idx] == 1 else 'real',
                        'confidence': float(probs[idx]) if preds[idx] == 1 else float(1 - probs[idx]),
                        'probability': float(probs[idx])
                    })
    return misclassified


def evaluate_model():
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/labels.csv", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    print(f"Test dataset size: {len(test_dataset):,}")
    print(f"Device being used: {DEVICE}")

    model = improved_xception(num_classes=1, pretrained=None).to(DEVICE)

    print(f"Loading checkpoint from {MODEL_PATH}/final_model.pth ...")
    checkpoint = torch.load(f"{MODEL_PATH}/final_model.pth", map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {params:.2f}M")

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_metrics = 0.0, {}
    y_true, y_pred_prob = [], []

    frame_predictions = []

    print("Starting evaluation...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Evaluating', leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred_prob.extend(probs)

            for i in range(len(paths)):
                sample_id = os.path.normpath(paths[i]).split(os.sep)[-2]
                label_str = "fake" if labels[i].item() == 1 else "real"
                frame_predictions.append({
                    "sample_id": sample_id,
                    "frame_path": paths[i].replace("\\", "/"),
                    "label": label_str,
                    "face_prob": float(probs[i])
                })

    test_loss /= len(test_dataset)
    test_metrics = calculate_metrics(np.array(y_true), np.array(y_pred_prob))

    print("Measuring inference time...")
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        limit_batches = 10
        for i, (images, _, _) in enumerate(tqdm(test_loader, desc='Inference timing', leave=False)):
            if i >= limit_batches: break
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                _ = model(images)
            total_samples += images.size(0)

    total_time = time.time() - start_time
    inference_time_ms = (total_time / total_samples) * 1000
    fps = total_samples / total_time

    misclassified = analyze_misclassifications(model, test_loader, DEVICE)

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"  Inference: {inference_time_ms:.2f} ms/sample ({fps:.1f} FPS)")
    print(f"  Misclassified: {len(misclassified)} samples")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(f"{OUTPUT_PATH}/test_results.txt", 'w') as f:
        f.write("Test Results:\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"MCC: {test_metrics['mcc']:.4f}\n")
        f.write(f"Fake Detection Rate: {test_metrics['fake_detection_rate']:.4f}\n")
        f.write(f"Real Detection Rate: {test_metrics['real_detection_rate']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Real: {test_metrics['true_negatives']} TN, {test_metrics['false_positives']} FP\n")
        f.write(f"Fake: {test_metrics['false_negatives']} FN, {test_metrics['true_positives']} TP\n")

    with open(f"{OUTPUT_PATH}/test_results.json", 'w') as f:
        json.dump({'test_loss': test_loss, **test_metrics, 'inference_time_ms': inference_time_ms, 'fps': fps,
                   'misclassified_count': len(misclassified)}, f, indent=2)

    with open(f"{OUTPUT_PATH}/misclassification_analysis.txt", 'w') as f:
        f.write("MISCLASSIFICATION ANALYSIS - Advanced Xception\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total misclassified samples: {len(misclassified)}\n")
        fake_as_real = [m for m in misclassified if m['true_label'] == 'fake']
        real_as_fake = [m for m in misclassified if m['true_label'] == 'real']
        f.write(f"Fake misclassified as Real: {len(fake_as_real)}\n")
        f.write(f"Real misclassified as Fake: {len(real_as_fake)}\n\n")
        if fake_as_real:
            f.write("TOP 5 FAKE MISCLASSIFIED AS REAL:\n")
            for i, error in enumerate(sorted(fake_as_real, key=lambda x: x['confidence'], reverse=True)[:5], 1):
                f.write(f"{i}. {error['image_path']}\n")
                f.write(f"   Confidence: {error['confidence']:.4f}\n\n")
        if real_as_fake:
            f.write("TOP 5 REAL MISCLASSIFIED AS FAKE:\n")
            for i, error in enumerate(sorted(real_as_fake, key=lambda x: x['confidence'], reverse=True)[:5], 1):
                f.write(f"{i}. {error['image_path']}\n")
                f.write(f"   Confidence: {error['confidence']:.4f}\n\n")
        if misclassified:
            pd.DataFrame(misclassified).to_csv(f"{OUTPUT_PATH}/misclassified_samples.csv", index=False)

    plot_confusion_matrix(test_metrics['confusion_matrix'], os.path.join(OUTPUT_PATH, "confusion_matrix.png"))

    frame_pred_df = pd.DataFrame(frame_predictions)
    frame_pred_df.to_csv(os.path.join(OUTPUT_PATH, "face_frame_predictions.csv"), index=False)
    print(f"Đã ghi {len(frame_predictions)} dòng vào face_frame_predictions.csv")


if __name__ == "__main__":
    evaluate_model()