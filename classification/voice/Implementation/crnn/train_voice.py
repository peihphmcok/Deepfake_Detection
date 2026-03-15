import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from PIL import Image
from torch.amp import GradScaler, autocast

from classification.voice.Implementation.crnn.crnn_model import ImprovedCRNN
from classification.voice.Implementation.transform import train_transforms, test_transforms
from classification.voice.Implementation.utils import EarlyStopping, calculate_eer

import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets, num_classes=2).float()
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        loss = -focal_weight * log_probs * targets_onehot
        return loss.mean()

# Project root (4 levels up from crnn/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data_preprocessing", "asvsproof_data")
OUTPUT_BASE_DIR = os.path.join(_PROJECT_ROOT, "classification", "voice", "output")
MODEL_DIR = os.path.join(_PROJECT_ROOT, "classification", "voice", "models")

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8

EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.0005
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

class VoiceTrainer:
    def __init__(self):
        self.model_name = "crnn_2"
        self.output_dir = ensure_dir(os.path.join(OUTPUT_BASE_DIR, self.model_name))
        self.model_dir = ensure_dir(MODEL_DIR)
        self._load_data()
        self.model = ImprovedCRNN(num_classes=len(self.train_dataset.classes), dropout_rate=DROPOUT_RATE)
        self.model.to(DEVICE)
        if not hasattr(self.model, 'forward'):
            raise AttributeError("Model is missing the 'forward' method")
        self._setup_training()
        stats = get_model_stats(self.model)
        print(f"\n{self.model_name.upper()} Model (Mel spectrogram input, 299x299):")
        print(f"  Total parameters: {stats['total_params']:,}")
        print(f"  Trainable parameters: {stats['trainable_params']:,}")
        print(f"  Size: {stats['size_mb']:.2f} MB")
        self.best_model_path = os.path.join(self.model_dir, f"{self.model_name}_best_model.pth")

    def _load_data(self):
        """Load Mel spectrogram data (299x299) and remap labels to ensure 0=real, 1=fake"""
        self.train_dataset = datasets.ImageFolder(
            os.path.join(DATA_DIR, "train"),
            transform=train_transforms
        )
        self.val_dataset = datasets.ImageFolder(
            os.path.join(DATA_DIR, "validation"),
            transform=test_transforms
        )
        self.test_dataset = datasets.ImageFolder(
            os.path.join(DATA_DIR, "test"),
            transform=test_transforms
        )
        self.class_to_idx = {'real': 0, 'fake': 1}
        original_class_to_idx = self.train_dataset.class_to_idx
        self.label_map = {original_class_to_idx['fake']: 1, original_class_to_idx['real']: 0}

        self.train_dataset.samples = [(path, self.label_map[label]) for path, label in self.train_dataset.samples]
        self.val_dataset.samples = [(path, self.label_map[label]) for path, label in self.val_dataset.samples]
        self.test_dataset.samples = [(path, self.label_map[label]) for path, label in self.test_dataset.samples]

        self.train_dataset.classes = ['real', 'fake']
        self.val_dataset.classes = ['real', 'fake']
        self.test_dataset.classes = ['real', 'fake']

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        print(f"Number of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        print(f"Number of test samples: {len(self.test_dataset)}")
        print(f"Classes: {self.train_dataset.classes}")

    def _setup_training(self):
        """Set up training components"""
        class_counts = np.bincount([y for _, y in self.train_dataset.samples])
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        self.class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {self.class_weights}")
        self.criterion = FocalLoss(alpha=0.8, gamma=2)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
        )
        self.scaler = GradScaler('cuda')
        self.early_stopping = EarlyStopping(EARLY_STOPPING_PATIENCE, MIN_DELTA)

    def _train_epoch(self):
        """Train one epoch with Mixup"""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        alpha = 0.2
        for images, labels in tqdm(self.train_loader, desc=f"Epoch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images = images.mean(dim=1, keepdim=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outputs = self.model(images)
                if np.random.rand() < 0.5:
                    lam = np.random.beta(alpha, alpha)
                    batch_size = images.size(0)
                    idx = torch.randperm(batch_size).to(DEVICE)
                    mixed_images = lam * images + (1 - lam) * images[idx]
                    outputs_mixed = self.model(mixed_images)
                    loss = lam * self.criterion(outputs_mixed, labels) + (1 - lam) * self.criterion(outputs_mixed, labels[idx])
                else:
                    loss = self.criterion(outputs, labels)
                preds = torch.softmax(outputs, dim=-1)[:, 1].cpu().detach().numpy()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        return total_loss, all_preds, all_labels

    def train(self):
        """Train the model"""
        print(f"\nStarting training for {self.model_name}...")
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        train_aucs, val_aucs = [], []
        best_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            train_loss, train_preds, train_labels = self._train_epoch()
            train_loss /= len(self.train_loader)
            train_pred_labels = [1 if p > 0.45 else 0 for p in train_preds]
            train_acc = accuracy_score(train_labels, train_pred_labels)
            train_f1 = f1_score(train_labels, train_pred_labels, zero_division=0)
            train_auc = roc_auc_score(train_labels, train_preds)
            val_loss, val_preds, val_labels = self._validate()
            val_loss /= len(self.val_loader)
            val_pred_labels = [1 if p > 0.45 else 0 for p in val_preds]
            val_acc = accuracy_score(val_labels, val_pred_labels)
            val_f1 = f1_score(val_labels, val_pred_labels, zero_division=0)
            val_auc = roc_auc_score(val_labels, val_preds)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            print(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, Train AUC={train_auc:.4f}")
            print(
                f"           Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved at {self.best_model_path}")

            self.scheduler.step(val_loss)
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        final_model_path = os.path.join(self.model_dir, f"{self.model_name}_final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

        self.model.load_state_dict(torch.load(self.best_model_path))
        print(f"Loaded best model from {self.best_model_path} for evaluation")
        return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs

    def _validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images = images.mean(dim=1, keepdim=True)
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        return total_loss, all_preds, all_labels

    def evaluate(self):
        """Evaluate on the test set"""
        print(f"Evaluating {self.model_name} on the test set...")
        self.model.eval()
        test_preds, test_labels, test_paths = [], [], []
        misclassified_samples = []
        test_dataset_paths = [sample[0] for sample in self.test_dataset.samples]
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                images = images.mean(dim=1, keepdim=True)
                with autocast('cuda'):
                    outputs = self.model(images)
                batch_preds = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
                batch_labels = labels.cpu().numpy()
                batch_indices = range(batch_idx * BATCH_SIZE, min((batch_idx + 1) * BATCH_SIZE, len(self.test_dataset)))
                for pred, true_label, path_idx in zip(batch_preds, batch_labels, batch_indices):
                    test_preds.append(pred)
                    test_labels.append(true_label)
                    test_paths.append(test_dataset_paths[path_idx])
                    pred_label = 1 if pred > 0.45 else 0
                    if pred_label != true_label:
                        misclassified_samples.append({
                            'path': test_dataset_paths[path_idx],
                            'true_label': self.test_dataset.classes[true_label],
                            'pred_label': self.test_dataset.classes[pred_label],
                            'confidence': pred if pred_label == 1 else 1 - pred
                        })
        inference_time = time.time() - start_time
        inference_per_sample = inference_time / len(self.test_dataset)
        test_pred_labels = [1 if p > 0.45 else 0 for p in test_preds]
        test_acc = accuracy_score(test_labels, test_pred_labels)
        test_precision = precision_score(test_labels, test_pred_labels, zero_division=0)
        test_recall = recall_score(test_labels, test_pred_labels, zero_division=0)
        test_f1 = f1_score(test_labels, test_pred_labels, zero_division=0)
        test_auc = roc_auc_score(test_labels, test_preds)
        fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
        precision, recall, _ = precision_recall_curve(test_labels, test_preds)
        ap = np.trapz(recall[::-1], precision[::-1])
        eer, _ = calculate_eer(fpr, tpr, thresholds)
        cm = confusion_matrix(test_labels, test_pred_labels)
        print("Confusion Matrix:")
        print(cm)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test EER: {eer:.4f}")
        print(f"Number of Misclassified Samples: {len(misclassified_samples)}")
        print(f"Inference Time per Sample: {inference_per_sample * 1000:.2f}ms")
        return test_acc, test_precision, test_recall, test_f1, test_auc, eer, cm, misclassified_samples, fpr, tpr, precision, recall, ap, test_paths, inference_per_sample

    def save_results(self, metrics, cm, misclassified_samples, fpr, tpr, precision, recall, ap, train_losses,
                     val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs, inference_per_sample):
        """Save evaluation results and plots"""
        test_acc, test_precision, test_recall, test_f1, test_auc, eer = metrics
        results_file = os.path.join(self.output_dir, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n")
            f.write(f"Test F1-Score: {test_f1:.4f}\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            f.write(f"Test EER: {eer:.4f}\n")
            f.write(f"Number of Misclassified Samples: {len(misclassified_samples)}\n")
            f.write(f"Inference Time per Sample: {inference_per_sample * 1000:.2f}ms\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
        misclassified_file = os.path.join(self.output_dir, "misclassified_samples.csv")
        with open(misclassified_file, "w") as f:
            f.write("Path,True Label,Predicted Label,Confidence\n")
            for sample in misclassified_samples:
                f.write(f"{sample['path']},{sample['true_label']},{sample['pred_label']},{sample['confidence']:.4f}\n")
        num_samples_to_plot = min(5, len(misclassified_samples))
        if num_samples_to_plot > 0:
            plt.figure(figsize=(15, 3 * num_samples_to_plot))
            for i, sample in enumerate(misclassified_samples[:num_samples_to_plot]):
                img = Image.open(sample['path'])
                plt.subplot(num_samples_to_plot, 1, i + 1)
                plt.imshow(img)
                plt.title(f"Path: {os.path.basename(sample['path'])}\n"
                          f"True: {sample['true_label']}, Pred: {sample['pred_label']}, "
                          f"Confidence: {sample['confidence']:.4f}")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "misclassified_samples.png"))
            plt.close()
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(25, 5))
        plt.subplot(1, 5, 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 5, 2)
        plt.plot(epochs, train_accs, label="Train Acc")
        plt.plot(epochs, val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 5, 3)
        plt.plot(epochs, train_f1s, label="Train F1")
        plt.plot(epochs, val_f1s, label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 5, 4)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {test_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.scatter(eer, 1 - eer, color='red', label=f"EER = {eer:.4f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 5, 5)
        plt.plot(recall, precision, label=f"PR (AP = {ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_plots.png"))
        plt.close()
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()
        print(f"Results saved for {self.model_name}!")

def get_model_stats(model):
    """Calculate model statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': model_size_mb
    }

def ensure_dir(path):
    """Create directory if it does not exist"""
    os.makedirs(path, exist_ok=True)
    return path

def train_voice():
    """Train the ImprovedCRNN model"""
    trainer = VoiceTrainer()
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs = trainer.train()
    test_acc, test_precision, test_recall, test_f1, test_auc, eer, cm, misclassified_samples, fpr, tpr, precision, recall, ap, test_paths, inference_per_sample = trainer.evaluate()
    trainer.save_results(
        (test_acc, test_precision, test_recall, test_f1, test_auc, eer),
        cm, misclassified_samples, fpr, tpr, precision, recall, ap, train_losses, val_losses, train_accs, val_accs,
        train_f1s, val_f1s, train_aucs, val_aucs, inference_per_sample
    )
    return {
        'model_name': trainer.model_name,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_eer': eer,
        'misclassified_samples': len(misclassified_samples),
        'inference_per_sample_ms': inference_per_sample * 1000
    }

if __name__ == "__main__":
    train_voice()