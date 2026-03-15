import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing", "ff_frames")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "ff_labels", "labels.csv")

def create_labels():
    """Create labels CSV from frame directories."""
    data = []

    for root, _, files in os.walk(OUTPUT_DIR):
        if not files:
            continue

        folder_name = os.path.basename(root)
        label = "real" if folder_name.startswith("real_") else "fake" if folder_name.startswith("fake_") else None
        if not label:
            continue

        source_folder = folder_name.split('_')[1]  # Extract source folder (e.g., youtube, Deepfakes)
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                data.append([img_path, label, root, source_folder])

    df = pd.DataFrame(data, columns=["image_path", "label", "video_source", "source_folder"])
    df.to_csv(LABELS_PATH, index=False)
    print("Labels CSV created!")

if __name__ == "__main__":
    create_labels()