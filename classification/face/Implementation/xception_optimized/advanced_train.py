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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
import json
import os
from tqdm import tqdm
import time
import cv2
import warnings
import random

warnings.filterwarnings('ignore')

from classification.face.Implementation.advanced_transforms import DeepfakeDataset, train_transform, val_test_transform
from advanced_xception import improved_xception

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Project root (5 levels up from xception_optimized/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DATA_PATH = os.path.join(_PROJECT_ROOT, "data_preprocessing", "ff_labels")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "classification", "face", "models", "model_advanced_xception")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "classification", "face", "output", "output_advanced_xception")
BATCH_SIZE = 48
NUM_WORKERS = 8
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
    for images, labels, _ in tqdm(dataloader, desc='Training', leave=False):
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
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, calculate_metrics(np.array(all_labels), np.array(all_preds))

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, calculate_metrics(np.array(all_labels), np.array(all_preds))

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
    sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
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
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Analyzing misclassifications'):
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

def generate_attention_maps(model, dataloader, device, output_path, epoch, num_samples=2):
    model.eval()
    samples_processed = 0
    attention_dir = os.path.join(output_path, f'attention_maps_epoch_{epoch + 1}')
    os.makedirs(attention_dir, exist_ok=True)
    attention_maps = []

    def hook_fn(module, input, output):
        if hasattr(output, 'mean'):
            attention_maps.append(torch.mean(output, dim=1, keepdim=True).cpu().detach().numpy())

    hooks = [module.register_forward_hook(hook_fn) for name, module in model.named_modules() if name in ['attention1', 'attention2', 'attention3']]

    with torch.no_grad():
        for images, _, paths in dataloader:
            if samples_processed >= num_samples:
                break
            images = images.to(device)
            with autocast(enabled=device.type == 'cuda'):
                _ = model(images)
            for i in range(min(images.size(0), num_samples - samples_processed)):
                try:
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                    img = img.astype(np.uint8)
                    if img.shape[0] > 0 and img.shape[1] > 0:
                        cv2.imwrite(os.path.join(attention_dir, f'sample_{samples_processed + 1}_original.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        for idx, attn_map in enumerate(attention_maps):
                            if i < attn_map.shape[0]:
                                attn = attn_map[i, 0].astype(np.float32)
                                if attn.shape[0] > 2 and attn.shape[1] > 2 and not np.any(np.isnan(attn)) and not np.any(np.isinf(attn)):
                                    attn = cv2.resize(attn, (299, 299), interpolation=cv2.INTER_CUBIC)
                                    if attn.max() - attn.min() > 1e-8:
                                        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8) * 255
                                        attn = attn.astype(np.uint8)
                                        heatmap = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
                                        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0.0)
                                        cv2.imwrite(os.path.join(attention_dir, f'sample_{samples_processed + 1}_attn_{idx + 1}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                                    else:
                                        print(f"Skipping normalization for attention map {idx + 1}: constant values")
                                else:
                                    print(f"Skipping attention map {idx + 1} for sample {samples_processed + 1}: invalid shape or values")
                    samples_processed += 1
                except Exception as e:
                    print(f"Error processing attention map for sample {samples_processed + 1}: {e}")
            attention_maps.clear()
    for hook in hooks:
        hook.remove()

def train_advanced_xception():
    train_dataset = DeepfakeDataset(f"{DATA_PATH}/train.csv", transform=train_transform)
    val_dataset = DeepfakeDataset(f"{DATA_PATH}/val.csv", transform=val_test_transform)
    test_dataset = DeepfakeDataset(f"{DATA_PATH}/test.csv", transform=val_test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    model = improved_xception(num_classes=1, pretrained='imagenet').to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {params:.2f}M")
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    criterion = FocalLoss(alpha=torch.tensor([ALPHA, 1 - ALPHA]), gamma=GAMMA)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_auc': [], 'val_auc': []}
    best_val_loss = float('inf')

    for name, param in model.named_parameters():
        param.requires_grad = 'last_linear' in name or 'attention' in name.lower()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
        print(f"    Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        generate_attention_maps(model, val_loader, DEVICE, OUTPUT_PATH, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, f"{MODEL_PATH}/best_model.pth")

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
        print(f"    Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"    Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        generate_attention_maps(model, val_loader, DEVICE, OUTPUT_PATH, epoch + PRETRAIN_EPOCHS)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, f"{MODEL_PATH}/best_model.pth")

    torch.save({'model_state_dict': model.state_dict()}, f"{MODEL_PATH}/final_model.pth")
    checkpoint = torch.load(f"{MODEL_PATH}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, DEVICE)
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)
            with autocast(enabled=DEVICE.type == 'cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_pred_prob.extend(probs)
    y_true, y_pred_prob = np.array(y_true), np.array(y_pred_prob)
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
    misclassified = analyze_misclassifications(model, test_loader, DEVICE, OUTPUT_PATH, "Advanced Xception")
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    print(f"  Inference: {inference_time_ms:.2f} ms/sample ({fps:.1f} FPS)")
    print(f"  Misclassified: {len(misclassified)} samples")
    save_plots(history, test_metrics, y_true, y_pred_prob, OUTPUT_PATH, "Advanced Xception")
    results = {
        'model_name': "Advanced Xception",
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
    with open(f"{OUTPUT_PATH}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    with open(f"{OUTPUT_PATH}/results.txt", 'w') as f:
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
    print("Advanced Xception Training")
    print("==========================\n")
    results = train_advanced_xception()
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{results['model_name']:<20} Acc: {results['accuracy']:.4f} F1: {results['f1']:.4f} Params: {results['parameters_millions']:.1f}M FPS: {results['fps']:.1f}")
    print("=" * 50)

if __name__ == "__main__":
    main()