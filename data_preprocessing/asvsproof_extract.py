import os
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA = os.path.join(PROJECT_ROOT, "data", "ASVspoof2017_V2")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "asvsproof_data")
TRAIN_WAV_DIR = os.path.join(_DATA, "ASVspoof2017_V2_train")
DEV_WAV_DIR = os.path.join(_DATA, "ASVspoof2017_V2_dev")
TRAIN_PROTOCOL = os.path.join(_DATA, "protocol_V2", "ASVspoof2017_V2_train.txt")
DEV_PROTOCOL = os.path.join(_DATA, "protocol_V2", "ASVspoof2017_V2_dev.txt")

# Tạo thư mục
os.makedirs(DATA_DIR, exist_ok=True)
for split in ['train', 'validation', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(DATA_DIR, split, label), exist_ok=True)

# Hàm chuyển .wav thành Mel spectrogram
def wav_to_melspectrogram(wav_path, img_path):
    y, sr = librosa.load(wav_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(2.99, 2.99))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

# Đọc protocol train
train_files = []
train_labels = []
with open(TRAIN_PROTOCOL, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            file_name, label_str = parts[0], parts[1]
            label_num = 0 if label_str == 'genuine' else 1  # genuine=0, spoof=1
            train_files.append(file_name)
            train_labels.append(label_num)

# Chia train thành 80% train, 20% validation
train_files_train, train_files_val, train_labels_train, train_labels_val = train_test_split(
    train_files, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# Xử lý tập train
for file_name, label_num in tqdm(zip(train_files_train, train_labels_train), desc='Xử lý train'):
    wav_path = os.path.join(TRAIN_WAV_DIR, file_name)
    save_dir = os.path.join(DATA_DIR, 'train', 'real' if label_num == 0 else 'fake')
    img_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.png")
    wav_to_melspectrogram(wav_path, img_path)

# Xử lý validation
for file_name, label_num in tqdm(zip(train_files_val, train_labels_val), desc='Xử lý validation'):
    wav_path = os.path.join(TRAIN_WAV_DIR, file_name)
    save_dir = os.path.join(DATA_DIR, 'validation', 'real' if label_num == 0 else 'fake')
    img_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.png")
    wav_to_melspectrogram(wav_path, img_path)

# Đọc protocol dev (test)
test_files = []
test_labels = []
with open(DEV_PROTOCOL, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            file_name, label_str = parts[0], parts[1]
            label_num = 0 if label_str == 'genuine' else 1
            test_files.append(file_name)
            test_labels.append(label_num)

# Xử lý test
for file_name, label_num in tqdm(zip(test_files, test_labels), desc='Xử lý test'):
    wav_path = os.path.join(DEV_WAV_DIR, file_name)
    save_dir = os.path.join(DATA_DIR, 'test', 'real' if label_num == 0 else 'fake')
    img_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.png")
    wav_to_melspectrogram(wav_path, img_path)