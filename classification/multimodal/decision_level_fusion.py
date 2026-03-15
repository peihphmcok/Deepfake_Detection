import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os

def extract_sample_id(image_path):
    filename = image_path.split('/')[-1]
    return filename.split('.')[0]

def load_and_combine_predictions(face_file, voice_file, face_weight=0.5, voice_weight=0.5):
    try:
        face_df = pd.read_csv(face_file)
        voice_df = pd.read_csv(voice_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")

    if face_df.empty or voice_df.empty:
        raise ValueError("One or both CSV files are empty or invalid")

    voice_df['sample_id'] = voice_df['image_path'].apply(extract_sample_id)

    merged_df = pd.merge(
        face_df[['sample_id', 'label', 'face_prob_avg']],
        voice_df[['sample_id', 'true_label', 'probability_fake']],
        left_on=['sample_id', 'label'],
        right_on=['sample_id', 'true_label'],
        how='inner'
    )

    if merged_df.empty:
        raise ValueError("No matching data between the two CSV files")

    merged_df['final_prob'] = (face_weight * merged_df['face_prob_avg'] +
                               voice_weight * merged_df['probability_fake'])

    merged_df['predicted_label'] = merged_df['final_prob'].apply(lambda x: 'fake' if x > 0.5 else 'real')
    merged_df['true_label_binary'] = merged_df['true_label'].map({'fake': 1, 'real': 0})
    merged_df = merged_df.rename(columns={'face_prob_avg': 'face_prob', 'probability_fake': 'audio_prob'})

    return merged_df

def evaluate_predictions(df):
    y_true = df['true_label']
    y_pred = df['predicted_label']
    y_true_binary = df['true_label_binary']
    y_score = df['final_prob']

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='fake')
    auc = roc_auc_score(y_true_binary, y_score)
    cm = confusion_matrix(y_true, y_pred, labels=['real', 'fake'])
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_score)

    return acc, f1, auc, cm, fpr, tpr, precision, recall

def find_special_cases(df, threshold=0.5):
    face_real_voice_fake_final_real = df[
        (df['face_prob'] <= threshold) &
        (df['audio_prob'] > threshold) &
        (df['final_prob'] <= threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_real_voice_fake_final_fake = df[
        (df['face_prob'] <= threshold) &
        (df['audio_prob'] > threshold) &
        (df['final_prob'] > threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_fake_voice_real_final_real = df[
        (df['face_prob'] > threshold) &
        (df['audio_prob'] <= threshold) &
        (df['final_prob'] <= threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_fake_voice_real_final_fake = df[
        (df['face_prob'] > threshold) &
        (df['audio_prob'] <= threshold) &
        (df['final_prob'] > threshold)
    ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    return (face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
            face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

# Hàm lưu kết quả
def save_results(df, acc, f1, auc, cm, fpr, tpr, precision, recall, output_file, special_cases_file,
                 face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
                 face_fake_voice_real_final_real, face_fake_voice_real_final_fake):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Lưu kết quả chính với cột đã đổi tên
    result_df = df[['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob', 'predicted_label']]
    result_df.to_csv(output_file, index=False)

    special_cases = pd.concat([
        face_real_voice_fake_final_real.assign(case='Face Real Voice Fake Final Real'),
        face_real_voice_fake_final_fake.assign(case='Face Real Voice Fake Final Fake'),
        face_fake_voice_real_final_real.assign(case='Face Fake Voice Real Final Real'),
        face_fake_voice_real_final_fake.assign(case='Face Fake Voice Real Final Fake')
    ], ignore_index=True)
    special_cases.to_csv(special_cases_file, index=False)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'roc_curve.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'pr_curve.png'))
    plt.close()

    print(f"Results saved to {output_file}")
    print(f"Special cases saved to {special_cases_file}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

def main():
    face_file = "C:/Personal/df/classification/multimodal/output/output_face/aggregated_face_probs.csv"
    voice_file = "C:/Personal/df/classification/multimodal/output/output_audio_crnn2/all_predictions.csv"
    output_file = "C:/Personal/df/classification/multimodal/output/output_multimodal/final_predictions_with_metrics.csv"
    special_cases_file = "C:/Personal/df/classification/multimodal/output/output_multimodal/special_cases.csv"

    merged_df = load_and_combine_predictions(face_file, voice_file, face_weight=0.5, voice_weight=0.5)
    acc, f1, auc, cm, fpr, tpr, precision, recall = evaluate_predictions(merged_df)
    (face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
     face_fake_voice_real_final_real, face_fake_voice_real_final_fake) = find_special_cases(merged_df)
    save_results(merged_df, acc, f1, auc, cm, fpr, tpr, precision, recall, output_file, special_cases_file,
                 face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
                 face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nFace Real Voice Fake Final Real cases:")
    print(face_real_voice_fake_final_real)
    print("\nFace Real Voice Fake Final Fake cases:")
    print(face_real_voice_fake_final_fake)
    print("\nFace Fake Voice Real Final Real cases:")
    print(face_fake_voice_real_final_real)
    print("\nFace Fake Voice Real Final Fake cases:")
    print(face_fake_voice_real_final_fake)

    return (merged_df, acc, f1, auc, cm,
            face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
            face_fake_voice_real_final_real, face_fake_voice_real_final_fake)

if __name__ == "__main__":
    (result_df, acc, f1, auc, cm,
     face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
     face_fake_voice_real_final_real, face_fake_voice_real_final_fake) = main()