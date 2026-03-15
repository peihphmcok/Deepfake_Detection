import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os


# --- CÁC HÀM HỖ TRỢ ---

def extract_sample_id(image_path):
    # Chuẩn hóa: thay thế tất cả dấu gạch ngược (\) thành gạch xuôi (/)
    clean_path = image_path.replace('\\', '/')
    # Lấy phần tử cuối cùng sau khi tách
    filename = clean_path.split('/')[-1]
    # Bỏ phần mở rộng .png/.jpg
    return filename.split('.')[0]


def load_merged_data(face_file, voice_file):
    """
    Hàm này chỉ thực hiện đọc và ghép dữ liệu (Merge), chưa tính toán final_prob.
    Giúp tối ưu hiệu năng khi chạy vòng lặp tìm weight.
    """
    try:
        face_df = pd.read_csv(face_file)
        voice_df = pd.read_csv(voice_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Không tìm thấy file: {e}")

    if face_df.empty or voice_df.empty:
        raise ValueError("Một trong hai file CSV rỗng hoặc không hợp lệ")

    voice_df['sample_id'] = voice_df['image_path'].apply(extract_sample_id)

    merged_df = pd.merge(
        face_df[['sample_id', 'label', 'face_prob_avg']],
        voice_df[['sample_id', 'true_label', 'probability_fake']],
        left_on=['sample_id', 'label'],
        right_on=['sample_id', 'true_label'],
        how='inner'
    )

    if merged_df.empty:
        raise ValueError("Không có dữ liệu khớp giữa hai file CSV")

    # Đổi tên cột cho dễ xử lý
    merged_df = merged_df.rename(columns={'face_prob_avg': 'face_prob', 'probability_fake': 'audio_prob'})

    # Chuyển true_label thành nhị phân
    merged_df['true_label_binary'] = merged_df['true_label'].map({'fake': 1, 'real': 0})

    return merged_df


def calculate_metrics_at_weight(df, face_weight):
    """Tính toán các chỉ số tại một mức weight cụ thể"""
    voice_weight = 1.0 - face_weight

    # Tính xác suất tổng hợp
    final_probs = (face_weight * df['face_prob'] + voice_weight * df['audio_prob'])
    predicted_labels = final_probs.apply(lambda x: 'fake' if x > 0.5 else 'real')

    # Lấy nhãn thực tế
    y_true = df['true_label']
    y_true_binary = df['true_label_binary']

    # Tính các chỉ số
    acc = accuracy_score(y_true, predicted_labels)
    f1 = f1_score(y_true, predicted_labels, pos_label='fake')
    auc = roc_auc_score(y_true_binary, final_probs)

    return acc, f1, auc


def analyze_weight_impact(merged_df, output_dir):
    """Chạy vòng lặp weight và vẽ biểu đồ"""
    weights = np.arange(0.0, 1.01, 0.01)  # Chạy từ 0% đến 100% bước nhảy 1%
    results = []

    print("Đang chạy mô phỏng thay đổi trọng số...")

    best_auc = 0
    best_weight = 0.5

    for w in weights:
        acc, f1, auc = calculate_metrics_at_weight(merged_df, w)
        results.append({
            'face_weight': w,
            'voice_weight': 1.0 - w,
            'Accuracy': acc,
            'F1-Score': f1,
            'AUC': auc
        })

        # Tìm weight tối ưu theo AUC (hoặc có thể đổi thành F1/Acc tùy mục đích)
        if auc > best_auc:
            best_auc = auc
            best_weight = w

    results_df = pd.DataFrame(results)

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['face_weight'], results_df['Accuracy'], label='Accuracy', linestyle='-', linewidth=2)
    plt.plot(results_df['face_weight'], results_df['F1-Score'], label='F1 Score', linestyle='--', linewidth=2)
    plt.plot(results_df['face_weight'], results_df['AUC'], label='AUC', linestyle='-.', linewidth=2)

    # Đánh dấu điểm tối ưu
    plt.axvline(x=best_weight, color='red', linestyle=':', alpha=0.5, label=f'Best Weight (AUC): {best_weight:.2f}')

    plt.title('Impact of Face Weight on Model Performance')
    plt.xlabel('Face Weight (Voice Weight = 1 - Face Weight)')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'weight_analysis_chart.png')
    plt.savefig(plot_path)
    plt.close()

    # Lưu bảng dữ liệu phân tích weight
    results_df.to_csv(os.path.join(output_dir, 'weight_analysis_data.csv'), index=False)

    print(f"Biểu đồ phân tích weight đã lưu tại: {plot_path}")
    print(f"Trọng số tối ưu tìm được (theo AUC): Face={best_weight:.2f}, Voice={1 - best_weight:.2f}")

    return best_weight


# --- CÁC HÀM CŨ (GIỮ LẠI ĐỂ XUẤT KẾT QUẢ CUỐI CÙNG) ---

def apply_final_prediction(df, face_weight):
    """Áp dụng weight cuối cùng để ra dataframe kết quả"""
    voice_weight = 1.0 - face_weight
    df['final_prob'] = (face_weight * df['face_prob'] + voice_weight * df['audio_prob'])
    df['predicted_label'] = df['final_prob'].apply(lambda x: 'fake' if x > 0.5 else 'real')
    return df


def find_special_cases(df, threshold=0.5):
    # (Giữ nguyên logic cũ của bạn)
    face_real_voice_fake_final_real = df[
        (df['face_prob'] <= threshold) & (df['audio_prob'] > threshold) & (df['final_prob'] <= threshold)
        ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_real_voice_fake_final_fake = df[
        (df['face_prob'] <= threshold) & (df['audio_prob'] > threshold) & (df['final_prob'] > threshold)
        ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_fake_voice_real_final_real = df[
        (df['face_prob'] > threshold) & (df['audio_prob'] <= threshold) & (df['final_prob'] <= threshold)
        ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    face_fake_voice_real_final_fake = df[
        (df['face_prob'] > threshold) & (df['audio_prob'] <= threshold) & (df['final_prob'] > threshold)
        ][['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob']]

    return (face_real_voice_fake_final_real, face_real_voice_fake_final_fake,
            face_fake_voice_real_final_real, face_fake_voice_real_final_fake)


def save_final_outputs(df, output_file, special_cases_file):
    # Tính lại metrics lần cuối cho weight tốt nhất
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

    # Lưu file CSV chính
    df[['sample_id', 'true_label', 'face_prob', 'audio_prob', 'final_prob', 'predicted_label']].to_csv(output_file,
                                                                                                       index=False)

    # Tìm và lưu special cases
    (c1, c2, c3, c4) = find_special_cases(df)
    special_cases = pd.concat([
        c1.assign(case='Face Real Voice Fake Final Real'),
        c2.assign(case='Face Real Voice Fake Final Fake'),
        c3.assign(case='Face Fake Voice Real Final Real'),
        c4.assign(case='Face Fake Voice Real Final Fake')
    ], ignore_index=True)
    special_cases.to_csv(special_cases_file, index=False)

    # Lưu Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.title(f'Confusion Matrix (Best Weight Acc: {acc:.4f})')
    plt.savefig(os.path.join(os.path.dirname(output_file), 'best_confusion_matrix.png'))
    plt.close()

    print(f"Đã lưu kết quả tối ưu vào {output_file}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


# --- MAIN ---

def main():
    # ĐƯỜNG DẪN FILE (Bạn hãy thay đổi nếu cần)
    face_file = "C:/Personal/df/classification/multimodal/output/output_face/aggregated_face_probs.csv"
    voice_file = "C:/Personal/df/classification/multimodal/output/output_audio_crnn2/all_predictions.csv"
    output_dir = "C:/Personal/df/classification/multimodal/output/output_multimodal_2/"

    output_file = os.path.join(output_dir, "final_predictions_optimized.csv")
    special_cases_file = os.path.join(output_dir, "special_cases_optimized.csv")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load và Merge dữ liệu (Chưa tính toán)
    print("Đang tải dữ liệu...")
    merged_df = load_merged_data(face_file, voice_file)

    # 2. Phân tích Weight và Vẽ biểu đồ
    best_weight = analyze_weight_impact(merged_df, output_dir)

    # 3. Áp dụng Weight tốt nhất để ra kết quả cuối cùng
    print(f"\nÁp dụng trọng số tối ưu (Face: {best_weight:.2f}) để xuất báo cáo cuối cùng...")
    final_df = apply_final_prediction(merged_df, best_weight)

    # 4. Lưu các file kết quả chi tiết
    save_final_outputs(final_df, output_file, special_cases_file)


if __name__ == "__main__":
    main()