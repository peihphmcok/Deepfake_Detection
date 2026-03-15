# 3

import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEL_ROOT = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio")
VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "FakeAVCeleb_v1.2")

# Chuyển đổi tên dataset từ "_" sang "-"
NAME_MAP = {
    "FakeVideo_FakeAudio": "FakeVideo-FakeAudio",
    "FakeVideo_RealAudio": "FakeVideo-RealAudio",
    "RealVideo_FakeAudio": "RealVideo-FakeAudio",
    "RealVideo_RealAudio": "RealVideo-RealAudio"
}

mel_to_video = []

for root, _, files in os.walk(MEL_ROOT):
    for file in files:
        if file.endswith(".png"):
            mel_path = os.path.join(root, file)

            # Đường dẫn tương đối từ MEL_ROOT
            rel_path = os.path.relpath(mel_path, MEL_ROOT)
            parts = rel_path.split(os.sep)

            # Thay phần tên dataset (index 0)
            parts[0] = NAME_MAP.get(parts[0], parts[0])

            # Gộp lại thành đường dẫn tương đối video
            video_rel_path = os.path.join(*parts)
            video_rel_path = os.path.splitext(video_rel_path)[0] + ".mp4"

            # Ghép với VIDEO_ROOT
            video_path = os.path.join(VIDEO_ROOT, video_rel_path)

            mel_to_video.append((mel_path, video_path if os.path.exists(video_path) else "NOT_FOUND"))

# Ghi ra file
output_csv = os.path.join(MEL_ROOT, "mel_to_video_mapping_clean.csv")
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["mel_spectrogram_path", "video_path"])
    writer.writerows(mel_to_video)

print(f"Đã ghi kết quả mapping vào: {output_csv}")
