# 7

import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audios_videos")
FRAME_DIR = os.path.join(ROOT_DIR, "video_frame_faces")
OUTPUT_CSV = os.path.join(ROOT_DIR, "labels.csv")

rows = []

for label in ["real", "fake"]:
    label_dir = os.path.join(FRAME_DIR, label)
    for sample_id in os.listdir(label_dir):
        sample_path = os.path.join(label_dir, sample_id)
        if os.path.isdir(sample_path):
            for fname in os.listdir(sample_path):
                if fname.endswith(".jpg"):
                    # Use absolute path
                    frame_abs_path = os.path.join(FRAME_DIR, label, sample_id, fname).replace("\\", "/")
                    rows.append({"image_path": frame_abs_path, "label": label})

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda x: x["image_path"]))

print(f"Đã tạo {OUTPUT_CSV} với {len(rows)} frames.")