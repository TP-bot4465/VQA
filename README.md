# Visual Question Answering (VQA) System

##  Giới thiệu (Overview)
Dự án này xây dựng một hệ thống **Visual Question Answering (VQA)** hoàn chỉnh, có khả năng tự động trả lời các câu hỏi ngôn ngữ tự nhiên dựa trên nội dung hình ảnh đầu vào. Hệ thống kết hợp kỹ thuật Học sâu đa phương thức (Multimodal Deep Learning), tận dụng sức mạnh của **Convolutional Neural Networks (CNN)** cho xử lý ảnh và **Long Short-Term Memory (LSTM)** cho xử lý ngôn ngữ, được tăng cường bởi cơ chế **Attention**.

##  Tính năng nổi bật

  * **Xử lý đa phương thức (Multimodal):** Kết hợp luồng xử lý thị giác và ngôn ngữ song song.
  * **Backbone đa dạng:** Hỗ trợ nhiều kiến trúc CNN để trích xuất đặc trưng ảnh:
      * Pre-trained models: **MobileNetV2**, **ResNet50**, **EfficientNet-B3**.
      * Custom model: **CNN\_Self\_Build** (được thiết kế và huấn luyện từ đầu).
  * **Xử lý ngôn ngữ:** Sử dụng mạng LSTM để mã hóa câu hỏi, giúp nắm bắt sự phụ thuộc tuần tự của từ ngữ.
  * **Cơ chế Attention:** Áp dụng lên đầu ra của LSTM để mô hình tập trung vào các từ khóa quan trọng trong câu hỏi, tạo ra vector ngữ cảnh (Context Vector) chính xác hơn.
  * **Fusion Mechanism:** Sử dụng phép nhân Hadamard (Element-wise multiplication) để kết hợp đặc trưng ảnh và đặc trưng câu hỏi.

##  Kiến trúc Hệ thống

Mô hình VQA được chia thành 3 module chính:

1.  **Image Encoder (Bộ mã hóa hình ảnh):**

      * Đầu vào: Ảnh RGB đã được resize về kích thước `224x224`.
      * Mô hình: Sử dụng các CNN Backbone (đã loại bỏ lớp Classifier cuối cùng).
      * Đầu ra: Vector đặc trưng được đưa qua lớp Fully Connected (FC) để đưa về kích thước `feature_size` (ví dụ: 512) và chuẩn hóa L2.

2.  **Question Encoder (Bộ mã hóa câu hỏi):**

      * Đầu vào: Câu hỏi dạng text được tokenize và chuyển thành vector chỉ số (indices).
      * Embedding: Chuyển đổi từ vựng sang vector dày đặc (dense vector).
      * Mô hình: LSTM nhiều lớp.
      * **Attention:** Tính toán trọng số attention cho các hidden states của LSTM để tạo ra vector đặc trưng câu hỏi cuối cùng.

3.  **Classifier (Bộ phân loại):**

      * Kết hợp hai vector đặc trưng từ Image Encoder và Question Encoder.
      * Đi qua các lớp FC kết hợp Dropout và hàm kích hoạt Tanh.
      * Lớp cuối cùng trả về xác suất cho các câu trả lời trong tập từ điển (Answer Vocabulary).

##  Cấu trúc Dữ liệu (Dataset Structure)

Dữ liệu huấn luyện và kiểm thử được lưu trữ dưới dạng JSON kết hợp với thư mục chứa ảnh. Dưới đây là định dạng chi tiết của các file:

### 1\. File Câu hỏi (`questions.json`)

Chứa danh sách các câu hỏi, mỗi câu hỏi liên kết với một hình ảnh thông qua `image_id`.

```json
{
    "questions": [
        {
            "question_id": 1,
            "image_id": 1,
            "question": "Địa điểm trong hình là gì?"
        },
        {
            "question_id": 2,
            "image_id": 1,
            "question": "Ảnh này có người không?"
        }
    ]
}
```

  * `question_id`: Định danh duy nhất cho câu hỏi.
  * `image_id`: ID của hình ảnh tương ứng (Hệ thống sẽ tìm file ảnh có tên dạng `id_{image_id}.png`).
  * `question`: Nội dung câu hỏi dạng văn bản.

### 2\. File Nhãn/Câu trả lời (`annotations.json`)

Chứa câu trả lời chuẩn (ground truth) cho quá trình huấn luyện.

```json
{
    "annotations": [
        {
            "question_id": 1,
            "image_id": 1,
            "answers": [
                {
                    "answer": "Bảo tàng Chứng tích Chiến tranh",
                    "answer_confidence": "yes"
                }
            ],
            "multiple_choice_answer": "Bảo tàng Chứng tích Chiến tranh",
            "answer_type": "other"
        }
    ]
}
```

  * `question_id`: Khớp với ID trong file câu hỏi.
  * `answers`: Danh sách các câu trả lời (có thể có nhiều câu trả lời từ các annotator khác nhau, hệ thống hiện tại lấy câu trả lời đầu tiên làm nhãn chính).
  * `multiple_choice_answer`: Câu trả lời chính xác nhất.

### 3\. Từ điển (Vocabularies)

Hệ thống sẽ tự động sinh ra 2 file text trong quá trình tiền xử lý:

  * `question_vocabs.txt`: Danh sách toàn bộ từ vựng xuất hiện trong tập câu hỏi.
  * `answer_vocabs.txt`: Danh sách các nhãn (classes) đầu ra mà mô hình có thể dự đoán.

##  Cài đặt và Yêu cầu

Dự án được phát triển bằng Python và PyTorch.

**Thư viện yêu cầu:**

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

## Hướng dẫn sử dụng

### 1\. Tiền xử lý dữ liệu (Preprocessing)

Trước khi train, dữ liệu ảnh cần được resize và text cần được tạo từ điển:

```python
# Resize ảnh
resize_images(input_dir="/path/to/raw/images", output_dir="/path/to/resized")

# Tạo vocab và file JSON huấn luyện
process_data(question_file, annotation_file, image_dir, output_path='/working/train.json')
```

### 2\. Huấn luyện (Training)

Cấu hình các tham số và chạy hàm `train`:

```python
# Ví dụ sử dụng Custom CNN
cnn_self_build = CNN_Self_Build(feature_dim=2048)
model = VQAModel(cnn_self_build, ..., with_att=True)

train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)
```

### 3\. Kiểm thử (Inference)

Sử dụng hàm `implement` để dự đoán trên một ảnh bất kỳ:

```python
image_path = "path/to/image.jpg"
implement(model, image_path, ques_vocab_path, ans_vocab_path, transform, device)
```

##  Kết quả

Hệ thống đã được thử nghiệm với các backbone khác nhau. Việc tích hợp Attention và sử dụng các mô hình backbone hiện đại giúp cải thiện đáng kể độ chính xác so với các phương pháp cơ bản.
