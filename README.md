# Visual Question Answering (VQA) with PyTorch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

Dự án này triển khai một hệ thống **Visual Question Answering (VQA)**, cho phép máy tính trả lời các câu hỏi ngôn ngữ tự nhiên dựa trên nội dung hình ảnh. Dự án so sánh hiệu quả của việc trích xuất đặc trưng ảnh từ các mạng CNN khác nhau (ResNet, MobileNet, EfficientNet) kết hợp với LSTM và cơ chế Attention.

## 🚀 Tính năng nổi bật

* **Đa dạng Backbone:** Hỗ trợ MobileNetV2, ResNet50, EfficientNetB3 và Custom CNN.
* **Xử lý ngôn ngữ:** Sử dụng LSTM để mã hóa câu hỏi và Word Embedding.
* **Cơ chế Attention:** Giúp mô hình tập trung vào vùng ảnh quan trọng liên quan đến câu hỏi.
* **Pipeline hoàn chỉnh:** Từ tiền xử lý dữ liệu, huấn luyện, đánh giá đến dự đoán thực tế.

## 📂 Cấu trúc dữ liệu

Dữ liệu đầu vào bao gồm ảnh và các file JSON định nghĩa câu hỏi và câu trả lời:
* `questions.json`: Chứa `question_id`, `image_id` và nội dung câu hỏi.
* `annotations.json`: Chứa câu trả lời chuẩn (ground truth).
* `images/`: Thư mục chứa ảnh gốc.

## 🏗️ Kiến trúc mô hình

Hệ thống hoạt động theo cơ chế **Joint Embedding**:
1.  **Image Encoder:** CNN trích xuất vector đặc trưng từ ảnh.
2.  **Question Encoder:** LSTM trích xuất vector đặc trưng từ câu hỏi.
3.  **Fusion:** Kết hợp hai vector này (Element-wise multiplication).
4.  **Classifier:** Phân lớp câu trả lời qua các lớp Fully Connected.



## 📊 Kết quả Thực nghiệm

Chúng tôi đã tiến hành huấn luyện và đánh giá trên tập Test. Dưới đây là kết quả chi tiết:

| Mô hình | Backbone | Attention | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **MobileNetV2** | MobileNetV2 | ✅ | **[Điền số]%** | **[Điền số]** |
| **ResNet50** | ResNet50 | ✅ | [Điền số]% | [Điền số] |
| **EfficientNetB3** | EfficientNetB3 | ✅ | [Điền số]% | [Điền số] |
| **Custom CNN** | 4-Block CNN | ✅ | [Điền số]% | [Điền số] |
| **Custom CNN** | 4-Block CNN | ❌ | [Điền số]% | [Điền số] |

> **Nhận xét:** Các mô hình sử dụng Pretrained weights (như ResNet, EfficientNet) thường hội tụ nhanh hơn và đạt độ chính xác cao hơn so với mạng tự xây dựng.

## 🛠️ Cài đặt & Sử dụng

1.  **Cài đặt thư viện:**
    ```bash
    pip install torch torchvision matplotlib pillow scikit-learn numpy
    ```

2.  **Chuẩn bị dữ liệu:**
    Chạy các bước tiền xử lý trong notebook để tạo file `train.json` và `test.json`.

3.  **Huấn luyện:**
    ```python
    python train.py # hoặc chạy cell training trong notebook
    ```

4.  **Dự đoán (Demo):**
    Sử dụng hàm `implement` để kiểm tra trên ảnh bất kỳ:
    ```python
    implement(model, "path/to/image.png", vocab_path, ans_path, transform, device)
    ```

## 📝 Tác giả
* **Thực hiện bởi:** [Tên của bạn] - Sinh viên Khoa học Máy tính
* **Môn học:** Deep Learning / Computer Vision
