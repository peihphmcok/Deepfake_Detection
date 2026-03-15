# 5

import os
import csv
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio", "mel_to_video_mapping_clean.csv")
BASE_OUT = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audios_videos")
AUDIO_OUT = os.path.join(BASE_OUT, "audio")
VIDEO_OUT = os.path.join(BASE_OUT, "video")

for label in ["fake", "real"]:
    os.makedirs(os.path.join(AUDIO_OUT, label), exist_ok=True)
    os.makedirs(os.path.join(VIDEO_OUT, label), exist_ok=True)

# Gom mẫu
fake_samples = []
real_samples = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        mel = row["mel_spectrogram_path"]
        vid = row["video_path"]

        if not (mel and vid and os.path.exists(mel) and os.path.exists(vid)):
            continue

        if "fakevideo" in vid.lower():
            fake_samples.append((mel, vid))
        elif "realvideo" in vid.lower():
            real_samples.append((mel, vid))

print(f"Tổng fake samples: {len(fake_samples)}")
print(f"Tổng real samples: {len(real_samples)}")

# Copy function
def copy_group(pairs, label, start_idx=0):
    for idx, (mel_path, vid_path) in enumerate(pairs):
        base = f"sample_{start_idx + idx:04d}"
        dst_mel = os.path.join(AUDIO_OUT, label, base + ".png")
        dst_vid = os.path.join(VIDEO_OUT, label, base + ".mp4")
        shutil.copy(mel_path, dst_mel)
        shutil.copy(vid_path, dst_vid)

# Copy real
copy_group(real_samples, "real", start_idx=0)

# Copy fake (sau real để không trùng tên)
copy_group(fake_samples, "fake", start_idx=len(real_samples))

print(f"Đã copy toàn bộ {len(real_samples)} real và {len(fake_samples)} fake mẫu vào: {BASE_OUT}")
