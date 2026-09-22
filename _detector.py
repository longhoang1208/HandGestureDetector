"""
==================================================
DETECTOR MODULE
==================================================
File name   : _detector.py
Description : Camera data processing module for
              hand gesture recognition using a
              BiLSTM model. It also initializes
              a base shared interface for two
              future gesture recognition modules:
              single_sign and multi_sign.
--------------------------------------------------
"""


import cv2
import mediapipe as mp
import numpy as np
import time
import os
import keras
import gc
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer
from PySide6.QtCore import QFileSystemWatcher
from PySide6.QtGui  import QImage
from PySide6.QtGui  import QPixmap
from PySide6.QtGui  import QFont

from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QStackedWidget

from _landmark   import extract_landmarks
from _config     import config
from _tts_module import delete_speaker

# -----------------------------------
# CONFIGURATIONS
# -----------------------------------
# 
# max_num_hand             : 2 hands
# min_detection_confidence : 70%
# min_tracking_confidence  : 50%
# label_duration           : 0.1s
# -----------------------------------

class DetectorConfigurations:
    def __init__(self):
        self.max_num_hand = 2
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence  = 0.5

        self.label_duration = 0.1

Dt_CFG = DetectorConfigurations()
CFG = config()


# --------------------------------------------
# DETECTOR
# --------------------------------------------
# 
# - Extract landmarks.
# - Normalize data.
# - Detect gestures.
# --------------------------------------------

class Detector:
    def __init__(self, model, labels):
        self._mp_init()
        self._main_init(model, labels)
    
    def _main_init(self, model, labels):
        self.model       = model
        self.confidence  = 0
        self.probs       = None
        self.sequence    = []

        self.labels      = labels

        self.last_label  = None
        self.label       = ""
        self.final_label = ""

        self.speak       = ""
        self.last_spoke  = ""

        self.stable_time = 0

    def _camera_init(self, frame_size: tuple):
        """
        frame_size có dạng (frame_width, frame_height)
        """
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        if not self.cap.isOpened():
            print("Can't open camera")

    def _mp_init(self):
        """
        Initialize hands and pose
        objects from mediapipe solutions
        """
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=Dt_CFG.max_num_hand,
            min_detection_confidence=Dt_CFG.min_detection_confidence,
            min_tracking_confidence=Dt_CFG.min_tracking_confidence
        )
        self.hand_results = None

        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            min_detection_confidence=Dt_CFG.min_detection_confidence,
            min_tracking_confidence=Dt_CFG.min_tracking_confidence
        )
        self.pose_results = None


    def detect(self, sequence):
        sequence = np.array(sequence)
        sequence = np.expand_dims(sequence, axis=0)

        results    = self.model.predict(sequence, verbose=0)[0]  # lấy [0]
        self.probs = results                                     # lưu toàn bộ

        self.confidence = np.max(results)
        pred = np.argmax(results)

        if self.confidence >= 0.7:
            self.label = self.labels[str(pred)]
        else:
            self.label = ""
            self.speak = ""
            self.last_spoke = ""
        
        if self.label != self.last_label:
            self.stable_time = time.time() + Dt_CFG.label_duration
            self.last_label  = self.label
            self.final_label = ""
            self.sequence.clear()
        
        if time.time() >= self.stable_time:
            self.final_label = self.label
            self.speak = self.final_label


    def detection(self, timestep: int, stride: int) -> str:
        if self.hand_results.multi_hand_landmarks:
            lm = extract_landmarks(self.hand_results, self.pose_results)
        else:
            lm = [0]*(63*2+33*3)
        
        self.sequence.append(lm)
        if len(self.sequence) == timestep:
            self.detect(self.sequence)
            del self.sequence[:stride]

        if not self.hand_results.multi_hand_landmarks:
            self.final_label = ""
            self.speak = self.last_spoke = ""

        self.final_label = self.final_label.replace("_", " ")

        return self.final_label
    

    def reset(self):
        self.sequence.clear()
        self.probs = None
        self.label = ""
        self.last_label = ""
        self.final_label = ""

        self.model = None
        self.labels = None

        self.hands.close()
        self.hands = None
        self.hand_results = None

        self.pose.close()
        self.pose = None
        self.pose_results = None


# --------------------------------------------
# GENERAL UI SETUP
# --------------------------------------------
class Interface(QWidget):
    def __init__(self):
        super().__init__()

        self.cameraLabel = QLabel()
        self.cameraLabel.setMinimumSize(
            CFG.cameraFrameSize[0],
            CFG.cameraFrameSize[1]
        )
        self.cameraLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cameraLabel.setStyleSheet("background:black;")

        font = QFont("Arial", 16)
        font.setBold(True)

        self.lbPredict = QLabel("Prediction: ")
        self.lbPredict.setFont(font)

        self.lbConfidence = QLabel("Confidence: ")
        self.lbConfidence.setFont(font)
        
        self.FPS = QLabel("FPS: ")
        self.FPS.setFont(font)

        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(CFG.button_width)

        self.stop_btn  = QPushButton("Stop")
        self.stop_btn.setFixedWidth(CFG.button_width)
        self.stop_btn.setDisabled(True)

        self.aud_btn = QPushButton("Read aloud")
        self.aud_btn.setFixedWidth(CFG.button_width)
        self.aud_btn.setCheckable(True)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.aud_btn)

        bar1 = QProgressBar()
        bar1.setStyleSheet(CFG.bar_style)
        bar1.setFixedWidth(CFG.barMinWidth)
        bar1.setValue(0)
        
        bar2 = QProgressBar()
        bar2.setStyleSheet(CFG.bar_style)
        bar2.setFixedWidth(CFG.barMinWidth)
        bar2.setValue(0)
        
        bar3 = QProgressBar()
        bar3.setStyleSheet(CFG.bar_style)
        bar3.setFixedWidth(CFG.barMinWidth)
        bar3.setValue(0)

        prob_lbl1 = QLabel()
        prob_lbl2 = QLabel()
        prob_lbl3 = QLabel()

        self.prob_elements = {
            "labels": [prob_lbl1, prob_lbl2, prob_lbl3],
            "bars": [bar1, bar2, bar3]
        }

        self.rightLayout = QVBoxLayout()
        self.rightLayout.addWidget(self.lbPredict)
        self.rightLayout.addWidget(self.lbConfidence)
        self.rightLayout.addWidget(self.FPS)

        self.rightLayout.addSpacing(20)

        self.rightLayout.addWidget(prob_lbl1)
        self.rightLayout.addWidget(bar1)

        self.rightLayout.addSpacing(20)

        self.rightLayout.addWidget(prob_lbl2)
        self.rightLayout.addWidget(bar2)

        self.rightLayout.addSpacing(20)

        self.rightLayout.addWidget(prob_lbl3)
        self.rightLayout.addWidget(bar3)

        self.rightLayout.addStretch()

        self.rightLayout.addLayout(button_layout)

        self.stack = QStackedWidget()

        self.mainLayout = QHBoxLayout(self)
        self.mainLayout.addWidget(self.stack)
        self.mainLayout.addSpacing(10)
        self.mainLayout.addLayout(self.rightLayout)

        self.camera_page = QWidget()
        self.camera_page_layout = QVBoxLayout(self.camera_page)

        self.selection_page = QWidget()
        self.selection_page_layout = QVBoxLayout(self.selection_page)


class ModuleSetUp(Interface):
    def __init__(self, user_name):
        super().__init__()

        self.user_name = user_name
        self.models_dir = Path(CFG.users_dir) / self.user_name / CFG.models_dir
        self.labels_dir = Path(CFG.users_dir) / self.user_name / CFG.labels_dir

        self.start_btn.clicked.connect(self.detector_init)
        self.stop_btn.clicked.connect(self.stop_camera)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)

        self.frame_size = (1280, 720)

        self.labels_file = CFG.labels
        self.model_name  = CFG.model_name

        self.prev_time = time.perf_counter()

        self.selection_page_setup()

        self.stack.addWidget(self.selection_page)
        self.stack.addWidget(self.camera_page)

    # Để subclass override
    def detector_init(self): ...
    def read_frame(self): ...

    def selection_page_setup(self):
        self.selection_page_layout.addStretch()

        self.model_drop_list = QComboBox()
        self.model_drop_list.setFixedWidth(200)
        self.model_drop_list.addItems(os.listdir(self.models_dir))
        self.model_drop_list.setCurrentText(CFG.model_name)
        self.selection_page_layout.addWidget(self.model_drop_list)

        self.labels_drop_list = QComboBox()
        self.labels_drop_list.setFixedWidth(200)
        self.labels_drop_list.addItems(os.listdir(self.labels_dir))
        self.labels_drop_list.setCurrentText(CFG.labels)
        self.selection_page_layout.addWidget(self.labels_drop_list)

        self.refresh_m_l()

        self.watcher = QFileSystemWatcher()
        self.watcher.addPaths([str(self.labels_dir), str(self.models_dir)])
        self.watcher.directoryChanged.connect(self.refresh_m_l)
        self.selection_page_layout.addStretch()


    def refresh_m_l(self, changed_path=None):
        self.model_drop_list.blockSignals(True)
        self.model_drop_list.clear()
        self.model_drop_list.addItems(
            os.listdir(self.models_dir)
        )
        if CFG.model_name in os.listdir(self.models_dir):
            self.model_drop_list.setCurrentText(CFG.model_name)

        self.model_drop_list.blockSignals(False)

        self.labels_drop_list.blockSignals(True)
        self.labels_drop_list.clear()
        self.labels_drop_list.addItems(
            os.listdir(self.labels_dir)
        )
        if CFG.labels in os.listdir(self.labels_dir):
            self.labels_drop_list.setCurrentText(CFG.labels)

        self.labels_drop_list.blockSignals(False)


    def prob_bar(self, top_idx: int = 3):
        if self.detector.probs is None:
            return
        
        probs = self.detector.probs
        top_indices = np.argsort(probs)[-top_idx:][::-1]

        for i, idx in enumerate(top_indices):
            prob = probs[idx]

            self.prob_elements["labels"][i].setText(self.labels[str(idx)])
            self.prob_elements["bars"][i].setValue(int(prob*100))


    def update_frame(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = QImage(
            frameRGB.data,
            frameRGB.shape[1],
            frameRGB.shape[0],
            frameRGB.shape[1] * 3,
            QImage.Format.Format_RGB888
        )

        image = image.scaled(
            self.cameraLabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.prob_bar()

        self.cameraLabel.setPixmap(QPixmap.fromImage(image))

    def stop_camera(self):
        self.stack.setCurrentWidget(self.selection_page)
        self.reset()

    def reset(self):
        if self.timer.isActive():
            self.timer.stop()

        self.cap.release()

        if hasattr(self, "collector"):
            self.collector = None
            
        delete_speaker(self.speaker_thread)
        self.speaker_thread = None
        
        self.detector.reset()
        self.detector = None

        self.model = None
        self.labels = None

        keras.backend.clear_session()
        gc.collect()

        # --------------------------------------------
        # RESET UI
        # --------------------------------------------
        self.start_btn.setEnabled(True)
        self.stop_btn.setDisabled(True)

        self.lbPredict.setText("Prediction: ")
        self.lbConfidence.setText("Confidence: ")
        self.FPS.setText("FPS: ")

        for label in self.prob_elements["labels"]:
            label.setText("")

        for bar in self.prob_elements["bars"]:
            bar.setValue(0)

