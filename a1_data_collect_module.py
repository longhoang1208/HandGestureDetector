# ==================================================
# DATA COLLECT MODULE
# ==================================================
# File name   : a1_data_collect_module.py
# Description : module thu thập dữ liệu
#               huấn luyện mô hình.
# 
# --------------------------------------------------
# CẤU TRÚC TỔ CHỨC DỮ LIỆU
# --------------------------------------------------
# data/
# │
# ├── processed/        <- dữ liệu để huấn luyện
# │   ├── Xin chào/     <- tên nhãn
# │   │   ├── 0.npy     <- video mẫu
# │   │   ├── 1.npy
# │   │   └── ...
# │   └── ...
# │
# └── training_plot/    <- biểu đồ
#     ├── training_plot.png     <- accuracy, loss
#     └── confusion_matrix.png  <- ma trận nhầm lẫn
# --------------------------------------------------


from _detector import DetectorConfigurations
from _landmark import extract_landmarks
from _landmark import draw_landmarks
from _landmark import lm_shape
from _config   import config

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer
from PySide6.QtGui  import QPixmap
from PySide6.QtGui  import QImage
from PySide6.QtGui  import QFont

from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QLineEdit
from PySide6.QtWidgets import QSpinBox
from PySide6.QtWidgets import QStackedWidget

CFG    = config()
Dt_CFG = DetectorConfigurations()


class Interface(QWidget):
    def __init__(self):
        super().__init__()

        self.mainLayout    = QHBoxLayout(self)
        self.camera_layout = QVBoxLayout()
        self.right_layout  = QVBoxLayout()

        # Setup pages
        self.stack = QStackedWidget()
        self.mainLayout.addWidget(self.stack)

        # ---------------------------------------
        # TRANG 1 - TẠO FILE LƯU BỘ NHÃN
        # ---------------------------------------
        # Khởi tạo trang
        self.page_create_file = QWidget()
        self.page_create_file_layout = QVBoxLayout(self.page_create_file)

        # Đặt tên cho file bộ nhãn
        self.label_name_input = QLineEdit()
        self.label_name_input.setFixedSize(
            CFG.inputSize[0],
            CFG.inputSize[1]
        )
        self.label_name_input.setPlaceholderText("Enter label file name")

        # Lưu file bộ nhãn lần đầu tiên tạo
        self.save_label_file_btn1 = QPushButton("Save file")
        self.save_label_file_btn1.setFixedWidth(CFG.button_width)

        # Sắp xếp bố cục trang
        self.page_create_file_layout.addStretch()
        self.page_create_file_layout.addWidget(self.label_name_input)
        self.page_create_file_layout.addWidget(self.save_label_file_btn1)
        self.page_create_file_layout.addStretch()


        # ---------------------------------------
        # TRANG 2 - GHI NHÃN
        # ---------------------------------------
        # Khởi tạo trang
        self.page_labeling = QWidget()
        self.page_labeling_layout = QVBoxLayout(self.page_labeling)

        # Nhập nhãn
        self.label_input = QLineEdit()
        self.label_input.setFixedSize(
            CFG.inputSize[0],
            CFG.inputSize[1]
        )
        self.label_input.setPlaceholderText("Enter your label")

        # Xác nhận ghi nhãn
        self.confirm_lb_btn = QPushButton("Confirm")
        self.confirm_lb_btn.setFixedWidth(CFG.button_width)

        # Lưu file bộ nhãn sau khi ghi nhãn xong
        self.save_label_file_btn2 = QPushButton("Save file")
        self.save_label_file_btn2.setFixedWidth(CFG.button_width)

        # Sắp xếp bố cục trang
        self.page_labeling_layout.addStretch()
        self.page_labeling_layout.addWidget(self.label_input)
        self.page_labeling_layout.addWidget(self.confirm_lb_btn)
        self.page_labeling_layout.addWidget(self.save_label_file_btn2)
        self.page_labeling_layout.addStretch()


        # ---------------------------------------
        # TRANG 3 - CHỌN SỐ LƯỢNG MẪU
        # ---------------------------------------
        # Khởi tạo trang
        self.page_num_sample = QWidget()
        self.page_num_sample_layout = QVBoxLayout(self.page_num_sample)

        # Select number of samples
        self.num_sample_select = QSpinBox()
        self.num_sample_select.setMinimum(CFG.minNumSample)
        self.num_sample_select.setMaximum(CFG.maxNumSample)
        self.num_sample_select.setValue(CFG.minNumSample)

        self.num_sample_select.setFixedSize(100, 50)

        # Xác nhận số lượng mẫu
        self.num_samp_confirm_btn = QPushButton("Confirm")
        self.num_samp_confirm_btn.setFixedWidth(CFG.button_width)

        # Sắp xếp bố cục trang
        self.page_num_sample_layout.addStretch()
        self.page_num_sample_layout.addWidget(self.num_sample_select)
        self.page_num_sample_layout.addWidget(self.num_samp_confirm_btn)
        self.page_num_sample_layout.addStretch()


        # ---------------------------------------
        # TRANG 4 - GHI HÌNH
        # ---------------------------------------
        # Khởi tạo trang
        self.page_collect = QWidget()
        self.page_collect_layout = QHBoxLayout(self.page_collect)

        # Khởi tạo khung hình camera
        self.cameraLabel = QLabel()
        self.cameraLabel.setMinimumSize(
            CFG.cameraFrameSize[0],
            CFG.cameraFrameSize[1]
        )

        # Start collect button
        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(CFG.button_width)

        # Return button
        self.redo_btn = QPushButton("Again")
        self.redo_btn.setFixedWidth(CFG.button_width)

        # Start timer - turn camera ON
        self.on_btn = QPushButton("ON")
        self.on_btn.setFixedWidth(CFG.button_width)

        # Stop timer - turn camera OFF
        self.off_btn = QPushButton("OFF")
        self.off_btn.setFixedWidth(CFG.button_width)

        self.frame_count_bar = QProgressBar()
        self.frame_count_bar.setStyleSheet(CFG.bar_style)
        self.frame_count_bar.setFixedWidth(CFG.barMinWidth)
        
        self.seq_count_bar = QProgressBar()
        self.seq_count_bar.setStyleSheet(CFG.bar_style)
        self.seq_count_bar.setFixedWidth(CFG.barMinWidth)

        # Sắp xếp bố cục trang
        self.page_collect_layout.addLayout(self.camera_layout)
        self.page_collect_layout.addSpacing(10)
        self.page_collect_layout.addLayout(self.right_layout)

        self.camera_layout.addWidget(self.cameraLabel)

        # Set font chữ
        font = QFont("Arial", 16)
        font.setBold(True)

        self.current_label = QLabel("Current label:")
        self.current_label.setFont(font)

        # Sắp xếp các thành phần của SIDEBAR bên phải
        self.right_layout.addWidget(self.current_label)

        self.right_layout.addWidget(QLabel("frame count"))
        self.right_layout.addWidget(self.frame_count_bar)

        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(QLabel("Progress"))
        self.right_layout.addWidget(self.seq_count_bar)

        self.right_layout.addSpacing(15)

        self.right_layout.addStretch()

        # Sắp xếp các nút bấm
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.on_btn)
        button_layout.addWidget(self.off_btn)
        button_layout.addWidget(self.redo_btn)

        self.right_layout.addLayout(button_layout)

        # --------------------------------------------------
        # SẮP XẾP THỨ TỰ CÁC TRANG
        # --------------------------------------------------
        self.stack.addWidget(self.page_create_file)
        self.stack.addWidget(self.page_labeling)
        self.stack.addWidget(self.page_num_sample)
        self.stack.addWidget(self.page_collect)


# ----------------------------------------------------------
# DATA COLLECTOR
# ----------------------------------------------------------
# 
# Description:
#   - Tạo file
#   - Ghi nhãn
#   - Lưu file
#   - Thu dữ liệu
# ----------------------------------------------------------
class CollectModule(Interface):
    def __init__(self):
        super().__init__()

        # Landmark list
        self.lm_list = []

        # Timestep - model.input_shape[1]
        self.timestep = 30

        # Init data file index
        self.data_file_idx = 0

        # Init class index
        self.class_idx = 0

        self.labels = {}
        self.label_idx = 0
        self.start_collecting_data = False

        self.save_label_file_btn1.clicked.connect(self.save_label_file1)
        self.save_label_file_btn2.clicked.connect(self.save_label_file2)

        self.confirm_lb_btn.clicked.connect(self.confirm_label)
        self.num_samp_confirm_btn.clicked.connect(self.confirm_num_sample)

        self.start_btn.clicked.connect(self.enable_collect_data)
        self.redo_btn.clicked.connect(self.reset)

        # QTimer - Cập nhật frame mỗi 30ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_camera)

        self.on_btn.clicked.connect(self.start_timer)

        self.off_btn.clicked.connect(self.timer.stop)
        self.off_btn.clicked.connect(self.stop_camera)

        # Số frame đã xử lý
        self.frame_count = 0
        
    # Ghi nhãn vào bộ nhãn
    def confirm_label(self):
        label = self.label_input.text().strip()
        self.labels[str(self.label_idx)] = label
        self.label_idx += 1

        self.label_input.setText("")

    # Tạo file để ghi bộ nhãn.
    def save_label_file1(self):
        self.label_file_name = self.label_name_input.text().strip()
        if not self.label_file_name:
            pass

        self.label_dir = Path(CFG.labels_dir) / f"{self.label_file_name}.json"
        Path(self.label_dir).parent.mkdir(exist_ok=True)

        self.labels = {}
        
        with open(self.label_dir, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.stack.setCurrentWidget(self.page_labeling)


    # Lưu bộ nhãn và0 file ghi bộ nhãn.
    def save_label_file2(self):
        with open(self.label_dir, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.stack.setCurrentWidget(self.page_num_sample)

    # xác nhận số lượng mẫu
    def confirm_num_sample(self):
        # Set number of samples
        self.num_sample = self.num_sample_select.value()

        # Enable collect data
        self.stack.setCurrentWidget(self.page_collect)
        self.camera_init()
        self.mp_init()

    # khởi tạo đối tượng đọc camera
    def camera_init(self, frame_size: tuple = CFG.default_frame_size):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        if not self.cap.isOpened():
            print("Can't open camera")

        self.timer.start(30)

    """
    Vòng lặp camera. Tạo đường dẫn lưu
    dữ liệu. thu thập dữ và xử lý dữ liệu
    từ camera. Lưu file dữ liệu đã xử lý.
    """
    def run_camera(self):
        data_save_dir = Path(CFG.data_dir)
        data_save_dir.mkdir(parents=True, exist_ok=True)

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.hand_res = self.hands.process(rgb)
        self.pose_res = self.pose.process(rgb)

        draw_landmarks(frame, self.hand_res, self.pose_res)

        total = sum(self.count_seq(data_save_dir, label)
                    for label in self.labels.values())

        self.class_idx = self.get_current_class_idx(data_save_dir)

        if self.class_idx < len(self.labels):
            curr_class = self.labels[str(self.class_idx)]
            self.current_label.setText("Current label: " + curr_class)
        
            if total < self.num_sample * len(self.labels):
                self.collect_data(data_save_dir, curr_class)
        else:
            self.start_btn.setDisabled(True)

        # Cập nhật quá trình thu thập dữ liệu
        self.draw_progress_bar(total)

        # Cập nhật frame mới
        self.update_frame(frame)

    def enable_collect_data(self):
        self.start_collecting_data = not self.start_collecting_data

    """
    Xử lý dữ liệu, lưu dữ liệu, cập nhật tiến độ.
    """
    def collect_data(self, data_save_dir, curr_class):
        if self.start_collecting_data:
            self.data_file_idx = self.count_seq(data_save_dir, curr_class)
            if self.data_file_idx < self.num_sample:
                # Extract features
                if self.hand_res and self.hand_res.multi_hand_landmarks:
                    lm = extract_landmarks(self.hand_res, self.pose_res)
                else:
                    lm = [0] * lm_shape
                    
                self.current_label.setText(
                    "Current label: " + curr_class
                )

                # Append data to landmark list
                if len(self.lm_list) < self.timestep:
                    self.lm_list.append(lm)
                    self.frame_count += 1
                    self.draw_frame_count_bar()
                    self.start_btn.setDisabled(True)
                # Save data file
                else:
                    data_save_path = Path(data_save_dir) / curr_class / f"{self.data_file_idx}.npy"
                    Path(data_save_path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(
                        data_save_path,
                        np.array(self.lm_list, dtype=np.float32)
                    )

                    self.lm_list.clear()
                    self.start_collecting_data = False
                    self.start_btn.setEnabled(True)

                    self.data_file_idx = self.count_seq(data_save_dir, curr_class)
            else:
                self.data_file_idx = 0

    """
    Hiển thị tiến độ thu thập dữ liệu
    bằng dạng progress bar.
    """
    # Phần trăm khung hình đã xử lý
    def draw_frame_count_bar(self):
        # Cập nhật giá trị trên frame_count_bar
        self.frame_count_bar.setValue(
            int(100*self.frame_count/self.timestep)
        )

        # Reset frame_count về 0 sau khi đủ video
        if self.frame_count >= self.timestep:
            self.frame_count = 0

    # Phần trăm số dữ liệu đã thu
    def draw_progress_bar(self, total):
        # Đếm số chuỗi đã thu
        self.seq_count_bar.setValue(
            int(100*total/(self.num_sample*len(self.labels)))
        )

    # Đếm số lượng mẫu trong một thư mục
    def count_seq(self, data_save_dir, curr_class):
        label_path = Path(data_save_dir) / curr_class

        if not label_path.exists():
            return 0

        num_seq = len(os.listdir(label_path))
        return num_seq

    def get_current_class_idx(self, data_save_dir):
        for idx in range(len(self.labels)):
            label_name = self.labels[str(idx)]
            if self.count_seq(
                data_save_dir, label_name
            ) < self.num_sample:
                return idx
        return len(self.labels)

    # Khởi tạo các đối tượng của mediapipe
    def mp_init(self):
        self.mp_hand = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.hands = self.mp_hand.Hands(
            max_num_hands = 2,
            min_detection_confidence = 0.7,
            min_tracking_confidence  = 0.5
        )

        self.pose = self.mp_pose.Pose(
            min_detection_confidence = 0.7,
            min_tracking_confidence  = 0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    # Cập nhật khung hình
    def update_frame(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frameRGB.shape
        image = QImage(
            frameRGB.data,
            w, h, w * 3,
            QImage.Format.Format_RGB888
        )

        image = image.scaled(
            self.cameraLabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.cameraLabel.setPixmap(QPixmap.fromImage(image))

    # Hiện màn hình đen khi tạm dừng chương trình
    def stop_camera(self):
        self.timer.stop()

        h = self.cameraLabel.height()
        w = self.cameraLabel.width()

        black = np.zeros((h, w, 3), dtype=np.uint8)
        self.update_frame(black)

    # Bật QTimer
    def start_timer(self):
        # Tự động gọi hàm run_camera mỗi 30ms
        self.timer.start(30)

    # Reset
    def reset(self):
        self.timer.stop()
        self.cap.release()

        self.frame_count = 0
        self.frame_count_bar.setValue(0)

        self.label_name_input.clear()
        self.label_input.clear()

        self.lm_list.clear()

        self.start_btn.setEnabled(True)
        self.label_dir = None

        self.class_idx = 0
        self.data_file_idx = 0

        self.stack.setCurrentWidget(self.page_create_file)


# def main():
#     from PySide6.QtWidgets import QApplication
#     app = QApplication()

#     window = CollectModule()
#     window.show()

#     app.exec()

#     if hasattr(window, "cap"):
#         window.cap.release()

# if __name__=="__main__":
#     main()