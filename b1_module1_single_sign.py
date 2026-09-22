# ==================================================
# SINGLE SIGN MODULE
# ==================================================
# File name   : b1_module1_single_sign.py
# Description : Module nhận diện từng cử chỉ
#               riêng lẻ.
# --------------------------------------------------


from _tts_module import speaker_init
from _detector import ModuleSetUp
from _detector import Detector
from _landmark import draw_landmarks
from _config  import config

import cv2
import os
import os
import json
import tensorflow as tf

from PySide6.QtWidgets import QLabel

import time

CFG = config()


class Module1(ModuleSetUp):
    def __init__(self):
        super().__init__()

        self.camera_page_layout.addWidget(self.cameraLabel)
        self.selection_page_layout.addWidget(QLabel("Single Sign"))

    def detector_init(self):
        self.model_name  = self.model_drop_list.currentText()
        self.labels_file = self.labels_drop_list.currentText()
        
        labels_path = os.path.join(
            CFG.labels_dir,
            self.labels_file
        )

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        
        self.model = tf.keras.models.load_model(
            os.path.join(CFG.models_dir, self.model_name),
            compile=False)
        
        self.timestep = self.model.input_shape[1]
        self.detector = Detector(self.model, self.labels)
        self.detector._camera_init(self.frame_size)
        self.cap = self.detector.cap

        self.stack.setCurrentWidget(self.camera_page)

        self.speaker_thread = speaker_init(self.detector, self.aud_btn)
        self.speaker_thread.start()

        self.timer.start(30)
        self.start_btn.setDisabled(True)
        self.stop_btn.setEnabled(True)

    def prob_bar(self, top_idx = 3):
        return super().prob_bar(top_idx)

    def update_frame(self, frame):
        return super().update_frame(frame)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't read frame from camera...")
            return

        frame    = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.detector.hand_results = self.detector.hands.process(frameRGB)
        self.detector.pose_results = self.detector.pose.process(frameRGB)

        text = self.detector.detection(
            timestep=self.timestep,
            stride=CFG.stride
        )

        self.lbPredict.setText(f"Prediction: {text}")
        self.lbConfidence.setText(f"Confidence: {self.detector.confidence:.2f}")

        draw_landmarks(
            frame,
            hand_results=self.detector.hand_results,
            pose_results=self.detector.pose_results
        )

        # FPS
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        self.FPS.setText(f"FPS: {fps:.1f}")
        
        self.update_frame(frame)


# def main():
#     app = QApplication()
    
#     window = Module1()
#     window.show()

#     app.exec()

#     if hasattr(window, "speaker_thread"):
#         delete_speaker(window.speaker_thread)

# if __name__=="__main__":
#     main()
