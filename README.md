

<h1 align="center">
    Hand Gesture Detector
</h1>

<p align="center">
    <i>Nền tảng hỗ trợ giao tiếp thông qua ký hiệu tay</i>
</p>

---

## Tổng quan sản phẩm
Người dùng có thể tự thu thập dữ liệu và huấn luyện mô hình
để nhận diện các ký hiệu riêng tùy theo nhu cầu sử dụng.

Trích xuất camera trên thiết bị, nhận diện các cử chỉ tay để dự đoán câu lệnh tương ứng. Phần mềm sử dụng một mô hình học sâu (do người phát triển sản phẩm huấn luyện) để nhận diện các ký hiệu tay.

Hiện tại, sản phẩm đã có 2 chế độ là nhận diện ký hiệu đơn lẻ và nhận diện chuỗi ký hiệu (ghép các từ thành câu).

Output khi nhận diện ký hiệu bao gồm chữ hiển thị trên màn hình giao diện phần mềm và âm thanh phát ra loa.

## Ứng dụng
Sản phẩm hướng đến các tình huống sử dụng cụ thể, quy mô nhỏ như trường học, bệnh viện, các điểm dịch vụ công, ...

Mô hình deep learning có thể được huấn luyện riêng cho từng đối tượng và môi trương sử dụng.

---

## Chi tiết sản phẩm

Có 4 module chính:
- Thu thập dữ liệu của các ký hiệu muốn dùng để nhận diện.
- Huấn luyện mô hình để nhận diện các ký hiệu.
- Nhận diện các ký hiệu riêng lẻ.
- Nhận diện các chuỗi ký hiệu, ghép các từ thành câu.

### Thu thập dữ liệu
- Người dùng đặt tên cho file ghi bộ nhãn.
- Nhập các câu muốn nhận diện
- Sau khi gán nhãn cho các ký hiệu xong, cửa sổ camera sẽ hiện ra. Người dùng nhấn start sau đó thực hiện ký hiệu trước camera để thu thập dữ liệu huấn luyện.

### Huấn luyện mô hình
- Người dùng đặt tên model.
- Chọn bộ nhãn đã ghi để huấn luyện mô hình.
- Quá trình huấn luyện mô hình sẽ tự động chạy, sau khi kết thúc sẽ hiện biểu đồ đánh giá quá trình huấn luyện và confusion matrix.

### Nhận diện ký hiệu
- Cửa sổ camera sẽ hiện ra, người dùng thực hiện cử chỉ trước camera và model sẽ dự đoán câu/từ.
- Nhấn nút read aloud để bật/tắt chế độ đọc thành tiếng.
- Nhấn nut stop để tạm dừng chương trình, đổi model và bộ nhãn.

---

## Nguyên lý hoạt động
### Xử lý dữ liệu
- Dùng MediaPipe để trích xuất tọa độ của các landmark.
- Lấy các tọa độ đã thu được cùng trừ cho một điểm gốc *(tương ứng với vị trí của cổ tay đối với hand landmarks và vị trí mũi đối với pose landmarks)*.
- Chia các giá trị vừa tính được cho giá trị tuyệt đối lớn nhất.
- Mỗi khung hình đều sẽ bao gồm các dữ liệu trên, ghép các dữ liệu đã xử lý của các khung hình theo thứ tự thời gian để cho ra chuỗi dữ liệu.

### Huấn luyện mô hình
**Kiến trúc mô hình:**
- **BiLSTM**: Mô hình sẽ học chuỗi dữ liệu từ dataset theo cả chiều thuận *(quá khứ -> tương lai)* và chiều nghịch *(tương lai -> quá khứ)*.
- **LSTM**: Layer lstm dùng để học các thông tin đặc trưng cuối cùng trước khi qua các fully-connected layer.
- **BatchNormalization**: Ổn định các output của layer trước để tăng hiệu quả huấn luyện.
- **Dropout**: Bỏ đi ngẫu nhiên (30%) các output của layer trước để tránh model học thuộc dư liệu (overfit).
- **Dense** *(fully-connected layer)*: Phân loại các nhãn (các câu) thành dạng xác xuất để dự đoán câu.

---

## Cấu trúc dự án
```text
d:/Project/
│
├── data/
│   ├── processed/
│   │   ├── Xin chào/
│   │   │   ├── 0.npy
│   │   │   ├── 1.npy
│   │   │   ├── ...
│   │   │   └── 19.npy
│   │   └── ...
│   │
│   └── Training_plot/
│       ├── plot.png
│       └── confusion_matrix.png
│
├── voices/
│   ├── voices en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
│
├── labels/
│   ├── labels.json
│   └── ...
│
├── models/
│   ├── model.keras
│   └── ...
│
├── resources/
│   ├── icon.ico
│   ├── Page1.PNG
│   └── ...
│
├── resources.qrc
├── resources_rc.py
│
├── _configurations.py
├── _detector.py
├── _landmark_module.py
├── _tts_module.py
│
├── a1_data_collect_module.py    <- Thu thập dữ liệu
├── a2_model_training_module.py  <- Huấn luyện mô hình
├── b1_module1_single_sign.py    <- Nhận diện ký hiệu lẻ
├── a2_module2_multi_sign.py     <- Nhận diện chuỗi ký hiệu
│
├── A_main.py
│
└── requirements.txt
```

---

## Yêu cầu hệ thống & cài đặt
- `python >= 3.9`

- `requirements.txt`
    ```text
    mediapipe==0.10.21
    opencv-python==4.8.1.78
    tensorflow==2.16.1
    scikit-learn==1.7.2
    numpy==1.26.4
    piper-tts
    pyside6
    ```

- Cài đặt: `pip install -r requirements.txt`

- Tải model và giọng đọc: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium

- Lưu **en_US-lessac-medium.onnx** và **en_US-lessac-medium.onnx.json** trong thư mục voices.

---

## Mã nguồn

### `_detector.py`
Module nhận diện ký hiệu từ model và nhãn đã được chọn.

### `_landmark_module.py`
Module xử lý dữ liệu từ tọa độ của các điểm lanmarks và vẽ khung xương.

### `_tts_module.py`
Module đọc văn bản thành tiếng.

### Lệnh chạy chương trình
```bash
python A_main.py
```

### Biên dịch mã nguồn sang file thực thi
#### Cài đặt:
```
pip install pyinstaller
```

#### Lệnh biên dịch:
```bash
pyinstaller A_main.py --name HandGestureDetector --windowed --onedir --collect-all mediapipe --collect-all tensorflow --collect-all piper --add-data "voices;voices"
```