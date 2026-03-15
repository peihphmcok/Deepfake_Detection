import pandas as pd


# Đọc file CSV
def load_and_aggregate_face_probs(csv_file):
    # Đọc dữ liệu
    df = pd.read_csv(csv_file)

    # Kiểm tra dữ liệu rỗng hoặc không hợp lệ
    if df.empty:
        raise ValueError("File CSV rỗng hoặc không hợp lệ")

    # Nhóm theo sample_id và tính trung bình face_prob
    aggregated = df.groupby(['sample_id', 'label'])['face_prob'].mean().reset_index()

    # Đổi tên cột face_prob thành face_prob_avg để rõ ràng
    aggregated.rename(columns={'face_prob': 'face_prob_avg'}, inplace=True)

    return aggregated


# Lưu kết quả vào file CSV
def save_aggregated_probs(aggregated_df, output_file):
    aggregated_df.to_csv(output_file, index=False)
    print(f"Kết quả đã được lưu vào {output_file}")


# Hàm chính
def main():
    input_file = "C:/Personal/df/classification/multimodal/output/output_face/face_frame_predictions.csv"
    output_file = "C:/Personal/df/classification/multimodal/output/output_face/aggregated_face_probs.csv"

    # Tính xác suất tổng
    aggregated_df = load_and_aggregate_face_probs(input_file)

    # Lưu kết quả
    save_aggregated_probs(aggregated_df, output_file)

    return aggregated_df


if __name__ == "__main__":
    result = main()
    print(result.head())