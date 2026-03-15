import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "ff_labels")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.csv")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "val.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

def split_data():
    """Split data by video source, ensuring all source folders are represented."""
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")

    df = pd.read_csv(LABELS_PATH)

    # Group ff_frames by video source
    video_groups = df.groupby('video_source')
    videos = [[name, group['label'].iloc[0], group['source_folder'].iloc[0]] for name, group in video_groups]

    # Create stratification key combining label and source folder
    video_names, labels, source_folders = zip(*videos)
    stratify_key = [f"{label}_{folder}" for label, folder in zip(labels, source_folders)]

    # Split videos
    train_videos, temp_videos = train_test_split(
        video_names, test_size=0.3, stratify=stratify_key, random_state=42
    )
    val_videos, test_videos = train_test_split(
        temp_videos, test_size=0.5, stratify=[stratify_key[video_names.index(v)] for v in temp_videos], random_state=42
    )

    # Assign ff_frames to splits
    train_df = df[df['video_source'].isin(train_videos)]
    val_df = df[df['video_source'].isin(val_videos)]
    test_df = df[df['video_source'].isin(test_videos)]

    # Save splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print("Data split completed!")

if __name__ == "__main__":
    split_data()