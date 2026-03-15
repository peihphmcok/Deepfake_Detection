# 6

import os
import cv2
import logging
import multiprocessing
from mtcnn import MTCNN
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FAKEAVCELEB = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audios_videos")
INPUT_VIDEO_DIR = os.path.join(_FAKEAVCELEB, "video")
OUTPUT_FRAME_DIR = os.path.join(_FAKEAVCELEB, "video_frame_faces")
LOG_FILE = "process_error_log.txt"
TARGET_SIZE = (299, 299)
FRAMES_PER_VIDEO = 2
NUM_PROCESSES = max(cpu_count() - 2, 1)  # Giữ lại 2 core cho hệ thống, còn lại chạy full

# Thiết lập logging để ghi lại video lỗi
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR,
                    format='%(asctime)s - %(message)s')


def mute_tensorflow():
    """Tắt log rác của Tensorflow/MTCNN"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')


def extract_faces_from_video(args):
    video_path, label = args

    # Init MTCNN trong process con để tránh lỗi pickle
    # Chú ý: Việc init MTCNN tốn thời gian, nhưng bắt buộc với multiprocessing spawn/forkserver
    try:
        mute_tensorflow()
        detector = MTCNN(min_face_size=40)  # min_face_size giúp chạy nhanh hơn chút
    except Exception as e:
        return f"Error Init MTCNN: {e}"

    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_subdir = os.path.join(OUTPUT_FRAME_DIR, label, basename)

    # 1. KIỂM TRA ĐÃ LÀM XONG CHƯA
    if os.path.exists(output_subdir):
        valid_images = [f for f in os.listdir(output_subdir) if f.endswith(('.jpg', '.png'))]
        if len(valid_images) >= FRAMES_PER_VIDEO:
            return None  # Trả về None để không spam log console

    try:
        os.makedirs(output_subdir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 or not cap.isOpened():
            cap.release()
            logging.error(f"Corrupt video: {video_path}")
            return f"Error: {basename} (corrupt/empty)"

        # Logic nhảy frame
        step = max(total_frames // FRAMES_PER_VIDEO, 1)
        count = 0
        frame_idx = 0

        while count < FRAMES_PER_VIDEO:
            success, frame = cap.read()
            if not success:
                break

                # Chỉ xử lý frame ở các bước nhảy (step) hoặc frame đầu tiên
            if frame_idx % step == 0 or (count == 0 and frame_idx == 0):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb)

                if faces:
                    # Lấy khuôn mặt to nhất (thường là chính chủ)
                    faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
                    x, y, w, h = faces[0]['box']
                    x, y = max(x, 0), max(y, 0)

                    # Mở rộng vùng mặt 30%
                    new_w, new_h = int(w * 1.3), int(h * 1.3)
                    cx, cy = x + w // 2, y + h // 2
                    new_x = max(cx - new_w // 2, 0)
                    new_y = max(cy - new_h // 2, 0)
                    new_x_end = min(new_x + new_w, frame.shape[1])
                    new_y_end = min(new_y + new_h, frame.shape[0])

                    face_crop = frame[new_y:new_y_end, new_x:new_x_end]

                    if face_crop.size > 0:
                        face_crop = cv2.resize(face_crop, TARGET_SIZE)
                        out_path = os.path.join(output_subdir, f"frame_{count}.jpg")
                        cv2.imwrite(out_path, face_crop)
                        count += 1

            frame_idx += 1

            # Safety break: nếu duyệt quá nhiều frame mà không tìm thấy mặt
            if frame_idx > total_frames:
                break

        cap.release()

        if count < FRAMES_PER_VIDEO:
            logging.error(f"Not enough faces: {basename} (found {count}/{FRAMES_PER_VIDEO})")
            return f"Warning: {basename} (only found {count} faces)"

        return f"Processed: {basename}"

    except Exception as e:
        logging.error(f"Exception processing {basename}: {str(e)}")
        return f"Exception: {basename}"


def get_video_tasks():
    tasks = []
    print("Đang quét thư mục video...")
    for label in ["real", "fake"]:
        label_dir = os.path.join(INPUT_VIDEO_DIR, label)
        if not os.path.exists(label_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục {label_dir}")
            continue

        for file in os.listdir(label_dir):
            if file.endswith((".mp4", ".avi", ".mov")):
                full_path = os.path.join(label_dir, file)
                tasks.append((full_path, label))
    return tasks


def verify_results(total_inputs):
    """Hàm kiểm tra lại số lượng output"""
    print("\n--- KẾT QUẢ KIỂM TRA ---")
    processed_count = 0
    missing_count = 0

    for label in ["real", "fake"]:
        out_label_dir = os.path.join(OUTPUT_FRAME_DIR, label)
        if not os.path.exists(out_label_dir): continue

        folders = os.listdir(out_label_dir)
        for folder in folders:
            folder_path = os.path.join(out_label_dir, folder)
            if len(os.listdir(folder_path)) >= FRAMES_PER_VIDEO:
                processed_count += 1

    print(f"Tổng đầu vào: {total_inputs}")
    print(f"Tổng đã xử lý thành công (đủ {FRAMES_PER_VIDEO} frames): {processed_count}")
    print(f"Tỷ lệ hoàn thành: {(processed_count / total_inputs) * 100:.2f}%")

    if processed_count < total_inputs:
        print(f"⚠️ Có {total_inputs - processed_count} video bị lỗi hoặc không tìm thấy mặt.")
        print(f"Vui lòng xem file log: {LOG_FILE}")
    else:
        print("✅ Đã xử lý đầy đủ tất cả video!")


def main():
    # 1. Tắt log TF ở main process
    mute_tensorflow()

    # 2. Lấy danh sách task
    tasks = get_video_tasks()
    total_tasks = len(tasks)
    print(f"Tổng số video tìm thấy: {total_tasks}")
    print(f"Sử dụng {NUM_PROCESSES} CPU cores để xử lý.")

    if total_tasks == 0:
        return

    # 3. Chạy Multiprocessing
    # Chunksize: Giao 10 video cho mỗi process một lúc để đỡ overhead giao tiếp
    with Pool(processes=NUM_PROCESSES) as pool:
        for result in tqdm(pool.imap_unordered(extract_faces_from_video, tasks, chunksize=5), total=total_tasks):
            if result and "Error" in result:
                # In ra lỗi ngay lập tức để user biết (nhưng không dừng chương trình)
                tqdm.write(result)

    print("Hoàn tất trích xuất.")

    # 4. Kiểm tra lại số lượng
    verify_results(total_tasks)


if __name__ == "__main__":
    # Fix cho Windows Multiprocessing
    multiprocessing.freeze_support()
    main()