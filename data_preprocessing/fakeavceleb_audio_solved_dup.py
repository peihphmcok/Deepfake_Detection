# 2

import os
import shutil
import hashlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio_dup")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio")
TEMP_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    all_files = []
    labels = []
    file_hashes = []

    # Step 1: Quét toàn bộ file
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, DATA_DIR).lower()

                # Nhận nhãn từ đường dẫn chứa từ khóa "fake" hoặc "real"
                if "fake" in rel_path:
                    label = 1
                elif "real" in rel_path:
                    label = 0
                else:
                    print(f"Không xác định nhãn cho file: {file_path}")
                    continue

                all_files.append(file_path)
                labels.append(label)
                file_hashes.append(compute_md5(file_path))

    if not all_files:
        print("Không tìm thấy file nào trong thư mục fakeavceleb_audio_dup!")
        return

    print(f"Tổng số file: {len(all_files)}")

    # Step 2: Loại bỏ trùng lặp dựa trên MD5
    seen_hashes = set()
    unique_files = []
    unique_labels = []
    for file_path, label, hash_val in zip(all_files, labels, file_hashes):
        if hash_val not in seen_hashes:
            unique_files.append(file_path)
            unique_labels.append(label)
            seen_hashes.add(hash_val)

    print(f"Số file duy nhất sau khi loại bỏ trùng lặp: {len(unique_files)}")

    # Step 3: Copy giữ nguyên đường dẫn, chỉ thay gốc từ DATA_DIR → OUTPUT_DIR
    copied_count = 0
    for file_path in tqdm(unique_files, desc="Copying files"):
        rel_path = os.path.relpath(file_path, DATA_DIR)
        dst_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            shutil.copy(file_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Lỗi khi copy {file_path} → {dst_path}: {e}")

    print(f"Đã copy {copied_count} file duy nhất vào {OUTPUT_DIR}")

    # Step 4: Xoá thư mục tạm nếu có
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()