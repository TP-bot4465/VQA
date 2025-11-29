# Visual Question Answering (VQA) with PyTorch

 
**TravelLens VQA** là một hệ thống trí tuệ nhân tạo đa phương thức (Multi-modal AI), có khả năng hiểu và trả lời các câu hỏi ngôn ngữ tự nhiên dựa trên nội dung của hình ảnh đầu vào. Dự án tập trung vào việc so sánh hiệu năng giữa các kiến trúc CNN Backbone khác nhau và đánh giá tác động của cơ chế Attention.

-----
##  Giới thiệu

Bài toán **Visual Question Answering (VQA)** yêu cầu máy tính phải kết hợp kiến thức từ hai lĩnh vực:

1.  **Thị giác máy tính (Computer Vision):** Để hiểu nội dung bức ảnh ("Trong ảnh có gì?", "Màu sắc ra sao?").
2.  **Xử lý ngôn ngữ tự nhiên (NLP):** Để hiểu ý định của câu hỏi.

Hệ thống này nhận đầu vào là một bức ảnh và một câu hỏi, sau đó đưa ra câu trả lời chính xác nhất từ bộ từ điển đã học.

-----
##  Tính năng nổi bật

  * **Đa dạng Backbone CNN:** Hỗ trợ nhiều mô hình trích xuất đặc trưng ảnh mạnh mẽ:
      * **Pretrained Models:** MobileNetV2, ResNet50, EfficientNetB3.
      * **Custom Model:** Mạng CNN tự xây dựng (4 blocks) để so sánh hiệu năng.
  * **Xử lý ngôn ngữ:** Sử dụng mạng **LSTM (Long Short-Term Memory)** kết hợp với Word Embedding để mã hóa câu hỏi.
  * **Cơ chế Attention:** Tích hợp cơ chế Attention giúp mô hình tập trung vào các vùng ảnh quan trọng liên quan đến từ khóa trong câu hỏi.
  * **End-to-End Pipeline:** Bao gồm quy trình từ tiền xử lý dữ liệu, huấn luyện, đánh giá đến dự đoán thực tế.

-----

##  Kiến trúc Mô hình

1.  **Image Encoder:**
      * Ảnh đầu vào được đưa qua mạng CNN (ví dụ: ResNet50) để trích xuất Feature Map.
      * Đầu ra là một vector đặc trưng (Feature Vector).
2.  **Question Encoder:**
      * Câu hỏi được mã hóa thành các chỉ số (tokens) và đưa qua lớp Embedding.
      * Mạng LSTM xử lý chuỗi và trả về vector ngữ cảnh.
3.  **Fusion (Kết hợp):**
      * Hai vector đặc trưng (Ảnh & Câu hỏi) được kết hợp thông qua phép nhân từng phần tử (**Element-wise Multiplication**).
4.  **Classifier:**
      * Đưa qua các lớp Fully Connected (FC), Dropout và hàm kích hoạt Tanh.
      * Lớp cuối cùng sử dụng Softmax để dự đoán xác suất của câu trả lời.

-----

## Kết quả Thực nghiệm

Chúng tôi đã tiến hành thử nghiệm trên tập dữ liệu kiểm thử (Test Set) với các cấu hình mô hình khác nhau. Dưới đây là bảng tổng hợp kết quả:

| Training Type       | Attention          | Model           | Accuracy | F1_Score |
|---------------------|--------------------|-----------------|----------|----------|
| Pretrained Model    | With Attention     | MobileNetV2     | 82.46    | 0.8074   |
| Pretrained Model    | With Attention     | ResNet50        | 82.21    | 0.8084   |
| Pretrained Model    | With Attention     | EfficientNet-B3 | 81.02    | 0.7972   |
| Pretrained Model    | Without Attention  | MobileNetV2     | 82.54    | 0.8111   |
| Pretrained Model    | Without Attention  | ResNet50        | 81.58    | 0.8008   |
| Pretrained Model    | Without Attention  | EfficientNet-B3 | 81.08    | 0.8000   |
| Build From Scratch  | With Attention     | -               | 79.58    | 0.7815   |
| Build From Scratch  | Without Attention  | -               | 77.17    | 0.7604   |

-----

##  Cấu trúc Dữ liệu (JSON Format)

Hệ thống yêu cầu dữ liệu đầu vào tuân thủ cấu trúc JSON sau:

### 1\. File Câu hỏi (`questions.json`)

Mỗi câu hỏi được liên kết với một hình ảnh thông qua `image_id`.

```json
{
    "questions": [
        {
            "question_id": 1,
            "image_id": 101,
            "question": "Địa điểm trong hình là gì?"
        },
        {
            "question_id": 2,
            "image_id": 101,
            "question": "Ảnh này có người không?"
        }
    ]
}
```

### 2\. File Nhãn (`annotations.json`)

Chứa câu trả lời chuẩn (Ground Truth) dùng để huấn luyện.

```json
{
    "annotations": [
        {
            "question_id": 1,
            "image_id": 101,
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

### 3\. Từ điển (Vocabularies)

Được sinh ra tự động trong quá trình tiền xử lý:

  * `question_vocabs.txt`: Danh sách toàn bộ từ vựng trong tập câu hỏi.
  * `answer_vocabs.txt`: Danh sách các nhãn (classes) đầu ra (One-hot encoding targets).

-----

##  Cài đặt & Sử dụng

### 1\. Yêu cầu hệ thống

  * Python 3.8+
  * PyTorch, Torchvision
  * Numpy, Matplotlib, Pillow, Scikit-learn

Cài đặt các thư viện cần thiết:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

### 2\. Tiền xử lý dữ liệu

Chạy phần Preprocessing trong notebook để resize ảnh và tạo từ điển:

```python
# Resize ảnh về 224x224
resize_images(input_dir, output_dir)
# Tạo file JSON và Vocab
process_data(...)
```

### 3\. Huấn luyện (Training)

Lựa chọn mô hình và bắt đầu huấn luyện:

```python
# Ví dụ huấn luyện với ResNet50
model = VQAModel(resnet, ..., with_att=True)
train(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=50)
```

### 4\. Kiểm thử (Inference)

Dự đoán câu trả lời cho một ảnh bất kỳ:

```python
image_path = "path/to/image.png"
implement(model, image_path, ques_vocab, ans_vocab, transform, device)
```

-----

