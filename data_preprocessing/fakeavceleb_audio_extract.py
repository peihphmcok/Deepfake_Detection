# 1

import os
import shutil
import moviepy.editor as mp
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.join(PROJECT_ROOT, "data", "FakeAVCeleb_v1.2")
output_dir = os.path.join(PROJECT_ROOT, "data_preprocessing", "fakeavceleb_audio_dup")
os.makedirs(output_dir, exist_ok=True)

def process_video(args):
    video_path, output_path = args
    try:
        # Nếu file đã tồn tại thì bỏ qua
        if os.path.exists(output_path):
            return f"Skipped (exists): {output_path}"

        # Extract audio from video
        video = mp.VideoFileClip(video_path)
        audio_path = f"temp_audio_{os.getpid()}.wav"
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Save spectrogram as image
        plt.figure(figsize=(2.99, 2.99))
        plt.axis('off')
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Resize spectrogram to 299x299
        img = cv2.imread(output_path)
        img_resized = cv2.resize(img, (299, 299))
        cv2.imwrite(output_path, img_resized)

        # Clean up
        os.remove(audio_path)
        return f"Processed: {output_path}"

    except Exception as e:
        return f"Error processing {video_path}: {e}"

def main():
    video_tasks = []
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        for race in os.listdir(category_path):
            race_path = os.path.join(category_path, race)
            for gender in os.listdir(race_path):
                gender_path = os.path.join(race_path, gender)
                for id_folder in os.listdir(gender_path):
                    id_path = os.path.join(gender_path, id_folder)
                    for file in os.listdir(id_path):
                        if file.endswith(".mp4"):
                            video_path = os.path.join(id_path, file)
                            output_subdir = os.path.join(output_dir, category.replace("-", "_"), race, gender, id_folder)
                            os.makedirs(output_subdir, exist_ok=True)
                            output_path = os.path.join(output_subdir, f"{file.replace('.mp4', '.png')}")
                            # ✅ Bỏ qua nếu đã có file
                            if not os.path.exists(output_path):
                                video_tasks.append((video_path, output_path))

    print(f"Found {len(video_tasks)} videos to process.")

    # Process videos in parallel
    if video_tasks:
        with Pool(processes=6) as pool:
            for result in tqdm(pool.imap_unordered(process_video, video_tasks), total=len(video_tasks)):
                print(result)
    else:
        print("All videos already processed!")

if __name__ == "__main__":
    main()
