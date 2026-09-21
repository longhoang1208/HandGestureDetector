# ==================================================
# DATA COLLECT MODULE
# ==================================================
# File name   : a1_data_collect_module.py
# Description : Training dataset collecting module.
# 
# --------------------------------------------------
# DATA SAVING STRUCTURE
# --------------------------------------------------
# data/
# │
# ├── processed/        <- Collected dataset
# │   ├── Xin chào/     <- Label name
# │   │   ├── 0.npy     <- processed data
# │   │   ├── 1.npy
# │   │   └── ...
# │   └── ...
# │
# └── training_plot/    <- biểu đồ
#     ├── training_plot.png     <- accuracy, loss
#     └── confusion_matrix.png  <- Confusion matrix
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
        # PAGE 1 - CREATE LABEL FILE
        # ---------------------------------------
        # Create page
        self.page_create_file = QWidget()
        self.page_create_file_layout = QVBoxLayout(self.page_create_file)

        # Name label file
        self.label_name_input = QLineEdit()
        self.label_name_input.setFixedSize(
            CFG.inputSize[0],
            CFG.inputSize[1]
        )
        self.label_name_input.setPlaceholderText("Enter label file name")

        # Create file button
        self.create_label_file_btn = QPushButton("Save file")
        self.create_label_file_btn.setFixedWidth(CFG.button_width)

        # Arrange elements
        self.page_create_file_layout.addStretch()
        self.page_create_file_layout.addWidget(self.label_name_input)
        self.page_create_file_layout.addWidget(self.create_label_file_btn)
        self.page_create_file_layout.addStretch()


        # ---------------------------------------
        # PAGE 2 - LABELING
        # ---------------------------------------
        # Create page
        self.page_labeling = QWidget()
        self.page_labeling_layout = QVBoxLayout(self.page_labeling)

        # Label input
        self.label_input = QLineEdit()
        self.label_input.setFixedSize(
            CFG.inputSize[0],
            CFG.inputSize[1]
        )
        self.label_input.setPlaceholderText("Enter your label")

        # Confirm label button
        self.confirm_lb_btn = QPushButton("Confirm")
        self.confirm_lb_btn.setFixedWidth(CFG.button_width)

        # Save label file after labeling button
        self.save_label_file_btn = QPushButton("Save file")
        self.save_label_file_btn.setFixedWidth(CFG.button_width)

        # Arrange elements
        self.page_labeling_layout.addStretch()
        self.page_labeling_layout.addWidget(self.label_input)
        self.page_labeling_layout.addWidget(self.confirm_lb_btn)
        self.page_labeling_layout.addWidget(self.save_label_file_btn)
        self.page_labeling_layout.addStretch()


        # ---------------------------------------
        # PAGE 3 - SELECT NUMBER OF SAMPLES
        # ---------------------------------------
        # Create page
        self.page_num_sample = QWidget()
        self.page_num_sample_layout = QVBoxLayout(self.page_num_sample)

        # Select number of samples
        self.num_sample_select = QSpinBox()
        self.num_sample_select.setMinimum(CFG.minNumSample)
        self.num_sample_select.setMaximum(CFG.maxNumSample)
        self.num_sample_select.setValue(CFG.minNumSample)

        self.num_sample_select.setFixedSize(100, 50)

        # Confirm numer of samples
        self.num_samp_confirm_btn = QPushButton("Confirm")
        self.num_samp_confirm_btn.setFixedWidth(CFG.button_width)

        # Arrange elements
        self.page_num_sample_layout.addStretch()
        self.page_num_sample_layout.addWidget(self.num_sample_select)
        self.page_num_sample_layout.addWidget(self.num_samp_confirm_btn)
        self.page_num_sample_layout.addStretch()


        # ---------------------------------------
        # PAGE 4 - RECORD
        # ---------------------------------------
        # Create page
        self.page_collect = QWidget()
        self.page_collect_layout = QHBoxLayout(self.page_collect)

        # Create camera frame
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

        # Arrange elements
        self.page_collect_layout.addLayout(self.camera_layout)
        self.page_collect_layout.addSpacing(10)
        self.page_collect_layout.addLayout(self.right_layout)

        self.camera_layout.addWidget(self.cameraLabel)

        # Set font
        font = QFont("Arial", 16)
        font.setBold(True)

        self.current_label = QLabel("Current label:")
        self.current_label.setFont(font)

        # Arrange SIDEBAR elements
        self.right_layout.addWidget(self.current_label)

        self.right_layout.addWidget(QLabel("frame count"))
        self.right_layout.addWidget(self.frame_count_bar)

        self.right_layout.addSpacing(15)

        self.right_layout.addWidget(QLabel("Progress"))
        self.right_layout.addWidget(self.seq_count_bar)

        self.right_layout.addSpacing(15)

        self.right_layout.addStretch()

        # Arrange buttons
        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.on_btn)
        button_layout.addWidget(self.off_btn)
        button_layout.addWidget(self.redo_btn)

        self.right_layout.addLayout(button_layout)

        # --------------------------------------------------
        # ARRANGE PAGES ORDER
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
#   - Create file
#   - Labeling
#   - Save file
#   - Save processed data
# ----------------------------------------------------------
class CollectModule(Interface):
    def __init__(self, user_name):
        super().__init__()

        self.user_name = user_name
        self.root_dir = Path("Users") / self.user_name

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

        self.create_label_file_btn.clicked.connect(self.create_label_file)
        self.save_label_file_btn.clicked.connect(self.save_label_file)

        self.confirm_lb_btn.clicked.connect(self.confirm_label)
        self.num_samp_confirm_btn.clicked.connect(self.confirm_num_sample)

        self.start_btn.clicked.connect(self.enable_collect_data)
        self.redo_btn.clicked.connect(self.reset)

        # QTimer - Update frame every 30ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_camera)

        self.on_btn.clicked.connect(self.start_timer)

        self.off_btn.clicked.connect(self.timer.stop)
        self.off_btn.clicked.connect(self.stop_camera)

        # Number of processed frames
        self.frame_count = 0
        
    # Save labels
    def confirm_label(self):
        label = self.label_input.text().strip()
        self.labels[str(self.label_idx)] = label
        self.label_idx += 1

        self.label_input.setText("")

    # Create file
    def create_label_file(self):
        self.label_file_name = self.label_name_input.text().strip()
        if not self.label_file_name:
            pass

        self.label_dir = Path(self.root_dir) / CFG.labels_dir / f"{self.label_file_name}.json"
        Path(self.label_dir).parent.mkdir(exist_ok=True)
        
        with open(self.label_dir, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.stack.setCurrentWidget(self.page_labeling)


    # Save label file after labeling
    def save_label_file(self):
        with open(self.label_dir, "w") as f:
            json.dump(self.labels, f, indent=2)

        self.stack.setCurrentWidget(self.page_num_sample)

    # Confirm number of samples
    def confirm_num_sample(self):
        # Set number of samples
        self.num_sample = self.num_sample_select.value()

        # Enable collect data
        self.stack.setCurrentWidget(self.page_collect)
        self.camera_init()
        self.mp_init()

    # Initialize camera frame reader object
    def camera_init(self, frame_size: tuple = CFG.default_frame_size):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        if not self.cap.isOpened():
            print("Can't open camera")

        self.timer.start(30)

    # Camera loop
    def run_camera(self):
        data_save_dir = Path(self.root_dir) / CFG.data_dir
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

        # Update data collect process
        self.draw_progress_bar(total)

        # Update new frame
        self.update_frame(frame)

    def enable_collect_data(self):
        self.start_collecting_data = not self.start_collecting_data

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

    # Percentage of processed frames
    def draw_frame_count_bar(self):
        # Update new value on frame_count_bar
        self.frame_count_bar.setValue(
            int(100*self.frame_count/self.timestep)
        )

        # Reset frame_count to 0 after finishing a video
        if self.frame_count >= self.timestep:
            self.frame_count = 0

    # Percentage of collected files
    def draw_progress_bar(self, total):
        # Count totle processed files (sequences)
        self.seq_count_bar.setValue(
            int(100*total/(self.num_sample*len(self.labels)))
        )

    # Count number of files (sequences) in a folder
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

    # Initialze mediapipe objects
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


    def stop_camera(self):
        self.timer.stop()

        h = self.cameraLabel.height()
        w = self.cameraLabel.width()

        black = np.zeros((h, w, 3), dtype=np.uint8)
        self.update_frame(black)


    def start_timer(self):
        # Automatically call run_camera every 30ms
        self.timer.start(30)


    def reset(self):
        self.timer.stop()
        self.cap.release()

        self.labels = {}

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