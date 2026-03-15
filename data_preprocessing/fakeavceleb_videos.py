# 4

import os
import csv
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio", "mel_to_video_mapping_clean.csv")
VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "FakeAVCeleb_v1.2")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

copied_count = 0
video_set = set()  # để tránh copy trùng

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_path = row["video_path"]
        if video_path and video_path != "NOT_FOUND" and os.path.exists(video_path):
            if video_path not in video_set:
                rel_path = os.path.relpath(video_path, VIDEO_ROOT)
                dst_path = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                try:
                    shutil.copy(video_path, dst_path)
                    copied_count += 1
                    video_set.add(video_path)
                except Exception as e:
                    print(f"Lỗi khi copy: {video_path} → {dst_path} | {e}")

print(f"Đã copy {copied_count} video sang {OUTPUT_DIR}")
