import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
import json
import os
from tqdm import tqdm
import time
import warnings
import random

warnings.filterwarnings('ignore')

from classification.face.Implementation.advanced_transforms import DeepfakeDataset, train_transform, val_test_transform
from classification.face.Implementation.baseline.baseline_models import BASELINE_MODELS

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[0] * (1 - targets) + self.alpha[1] * targets
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Project root (4 levels up from baseline/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATA_PATH = os.path.join(_PROJECT_ROOT, "data_preprocessing", "ff_labels")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "classification", "face", "models", "model_xception_enhanced")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "classification", "face", "output", "output_xception_enhanced")
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
ALPHA = 0.82
GAMMA = 2.0
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 15

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
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels, _ in progress_bar:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()

        with autocast(enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        current_loss = running_loss / (len(all_preds) * images.size(0) / len(images))
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            current_loss = running_loss / (len(all_preds) * images.size(0) / len(images))
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return epoch_loss, metrics

def save_plots(history, test_metrics, y_true, y_pred_prob, output_path, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(history['val_acc'], 'r-', label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_f1'], 'b-', label='Train')
    axes[1, 0].plot(history['val_f1'], 'r-', label='Val')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_auc'], 'b-', label='Train')
    axes[1, 1].plot(history['val_auc'], 'r-', label='Val')
    axes[1, 1].set_title('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f'{model_name} Training History')
    plt.tight_layout()
    plt.savefig(f"{output_path}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_path}/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_path}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_misclassifications(model, test_loader, device, output_path, model_name):
    model.eval()
    misclassified = []

    progress_bar = tqdm(test_loader, desc='Analyzing misclassifications')
    with torch.no_grad():
        for images, labels, paths in progress_bar:
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

            progress_bar.set_postfix({'misclassified': len(misclassified)})

    with open(f"{output_path}/misclassification_analysis.txt", 'w') as f:
        f.write(f"MISCLASSIFICATION ANALYSIS - {model_name}\n")
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
        pd.DataFrame(misclassified).to_csv(f"{output_path}/misclassified_samples.csv", index=False)

    return misclassified

def train_model(model_key, model_config, train_loader, val_loader, test_loader):
    print(f"\nTraining {model_config['name']}")

    model = model_config['model_fn'](dropout_rate=0.5).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {params:.2f}M")

    model_save_path = os.path.join(MODEL_PATH, f"baseline_{model_key}")
    output_save_path = os.path.join(OUTPUT_PATH, f"baseline_{model_key}")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(output_save_path, exist_ok=True)

    criterion = FocalLoss(alpha=torch.tensor([ALPHA, 1 - ALPHA]), gamma=GAMMA)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda')
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []
    }
    best_val_loss = float('inf')

    # Phase 1: pretrain classifier only
    for name, param in model.named_parameters():
        if model_config['classifier_name'] in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not trainable_params:
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = list(model.parameters())

    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("Phase 1: Pretraining...")
    for epoch in range(PRETRAIN_EPOCHS):
        print(f"  Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        print(
            f"    Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, Train AUC: {train_metrics['auc']:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, f"{model_save_path}/best_model.pth")

    # Phase 2: full finetune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE)

    print("Phase 2: Finetuning...")
    for epoch in range(FINETUNE_EPOCHS):
        print(f"  Epoch {epoch + 1}/{FINETUNE_EPOCHS}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])

        print(
            f"    Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, Train AUC: {train_metrics['auc']:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, f"{model_save_path}/best_model.pth")

    torch.save({'model_state_dict': model.state_dict()}, f"{model_save_path}/final_model.pth")

    checkpoint = torch.load(f"{model_save_path}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, DEVICE)

    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_pred_prob.extend(probs)

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    print("Measuring inference time...")
    model.eval()
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for images, _, _ in tqdm(test_loader, desc='Inference timing', leave=False):
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                _ = model(images)
            total_samples += images.size(0)
    total_time = time.time() - start_time
    inference_time_ms = (total_time / total_samples) * 1000
    fps = total_samples / total_time

    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"  Inference: {inference_time_ms:.2f} ms/sample ({fps:.1f} FPS)")

    misclassified = analyze_misclassifications(model, test_loader, DEVICE, output_save_path, model_config['name'])

    print(f"  Misclassified: {len(misclassified)} samples")
    if misclassified:
        print("  Sample misclassifications:")
        for i, sample in enumerate(misclassified[:5], 1):
            print(
                f"    {i}. True: {sample['true_label']}, Pred: {sample['predicted_label']}, Conf: {sample['confidence']:.3f}")

    save_plots(history, test_metrics, y_true, y_pred_prob, output_save_path, model_config['name'])

    results = {
        'model_name': model_config['name'],
        'test_loss': test_loss,
        'accuracy': test_metrics['accuracy'],
        'f1': test_metrics['f1'],
        'auc': test_metrics['auc'],
        'mcc': test_metrics['mcc'],
        'fake_detection_rate': test_metrics['fake_detection_rate'],
        'real_detection_rate': test_metrics['real_detection_rate'],
        'inference_time_ms': inference_time_ms,
        'fps': fps,
        'parameters_millions': params,
        'total_misclassified': len(misclassified)
    }

    with open(f"{output_save_path}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(f"{output_save_path}/results.txt", 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"F1: {test_metrics['f1']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"MCC: {test_metrics['mcc']:.4f}\n")
        f.write(f"Fake Detection Rate: {test_metrics['fake_detection_rate']:.4f}\n")
        f.write(f"Real Detection Rate: {test_metrics['real_detection_rate']:.4f}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"Real: {test_metrics['true_negatives']} TN, {test_metrics['false_positives']} FP\n")
        f.write(f"Fake: {test_metrics['false_negatives']} FN, {test_metrics['true_positives']} TP\n")

    torch.cuda.empty_cache()
    return results

def main():
    print("Baseline Models Training")
    print("====================\n")

    train_dataset = DeepfakeDataset(f"{DATA_PATH}/train.csv", transform=train_transform)
    val_dataset = DeepfakeDataset(f"{DATA_PATH}/val.csv", transform=val_test_transform)
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test.csv", transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)

    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    all_results = []
    for i, (model_key, model_config) in enumerate(BASELINE_MODELS.items(), 1):
        print(f"\n[{i}/{len(BASELINE_MODELS)}] {model_config['name']}")
        try:
            results = train_model(model_key, model_config, train_loader, val_loader, test_loader)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
            continue

    if all_results:
        comparison_df = pd.DataFrame(all_results)
        comparison_path = os.path.join(OUTPUT_PATH, "baseline_comparison")
        os.makedirs(comparison_path, exist_ok=True)
        comparison_df.to_csv(f"{comparison_path}/results.csv", index=False)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for result in all_results:
            print(f"{result['model_name']:<20} Acc: {result['accuracy']:.4f} F1: {result['f1']:.4f} "
                  f"Params: {result['parameters_millions']:.1f}M FPS: {result['fps']:.1f}")
        print("=" * 50)

if __name__ == "__main__":
    main()