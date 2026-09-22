"""
==================================================
SINGLE SIGN MODULE
==================================================
File name   : b2_module2_multi_signs.py
Description : Module nhận diện chuỗi cử chỉ.
--------------------------------------------------
"""


from _tts_module import speaker_init
from _detector import ModuleSetUp
from _detector import Detector
from _landmark import draw_landmarks
from _config  import config

import cv2
import time
from pathlib import Path
import json
import tensorflow as tf

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QHBoxLayout

CFG = config()


class SequenceModule:
    def __init__(self, detector):
        self.detector = detector
        self.gloss_list = []
        self.gloss = ""

        self.last_spoke = ""
        self.speak = ""
        self.confirmed = False

    def make_gloss_seq(self):
        merged = []
        spell_word = ""

        for token in self.gloss_list:
            # If token is a SINGLE LETTER
            if token.startswith("'") and token.endswith("'"):
                spell_word += token.replace("'", "")

            else:
                # Push in if spelling word
                if spell_word:
                    merged.append(spell_word)
                    spell_word = ""

                merged.append(token)

        # Append LAST spelled word
        if spell_word:
            merged.append(spell_word)

        self.gloss = " ".join(merged)

        return self.gloss
    

    def get_gloss(self, text):
        text = str(text).strip()

        if text == "":
            return

        if not self.gloss_list or text != self.gloss_list[-1]:
            self.gloss_list.append(text)

    def collect(self, frame, timestep, qlabel: QLabel):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.detector.hand_results = self.detector.hands.process(frameRGB)
        self.detector.pose_results = self.detector.pose.process(frameRGB)
        text = self.detector.detection(timestep, CFG.stride)

        draw_landmarks(frame,
                       self.detector.hand_results,
                       self.detector.pose_results
                    )

        self.get_gloss(text)
        self.gloss = self.make_gloss_seq()
        self.speak = self.make_gloss_seq()

        qlabel.setText(self.gloss)


class Module2(ModuleSetUp):
    def __init__(self, user_name):
        super().__init__(user_name)

        _text = QLabel("Text: ")
        _text.setFixedHeight(30)

        font = QFont("Arial", 16)
        font.setBold(True)

        self.text_bar = QLabel()
        self.text_bar.setFont(font)
        self.text_bar.setFixedHeight(15)

        btn_layout = QHBoxLayout()

        confirm_btn = QPushButton("Confirm")
        confirm_btn.setFixedWidth(CFG.button_width)
        confirm_btn.clicked.connect(self.confirm)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(CFG.button_width)
        clear_btn.clicked.connect(self.clear)

        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(confirm_btn)
        btn_layout.addStretch()

        self.camera_page_layout.addWidget(_text)
        self.camera_page_layout.addWidget(self.text_bar)
        self.camera_page_layout.addLayout(btn_layout)
        self.camera_page_layout.addWidget(self.cameraLabel)
        self.selection_page_layout.addWidget(QLabel("Multi Signs"))

    def clear(self):
        self.collector.gloss = ""
        self.collector.gloss_list.clear()
        self.collector.speak = self.collector.last_spoke = ""

    def confirm(self):
        self.collector.confirmed = True

    def detector_init(self):
        self.model_name  = self.model_drop_list.currentText()
        self.labels_file = self.labels_drop_list.currentText()
        
        labels_path = Path(self.labels_dir) / self.labels_file

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        
        self.model = tf.keras.models.load_model(
            Path(self.models_dir) / self.model_name,
            compile=False)
        
        self.timestep = self.model.input_shape[1]
        self.detector = Detector(self.model, self.labels)
        self.detector._camera_init(self.frame_size)
        self.cap = self.detector.cap

        self.collector = SequenceModule(detector=self.detector)

        self.stack.setCurrentWidget(self.camera_page)

        self.speaker_thread = speaker_init(self.collector, self.aud_btn)
        self.speaker_thread.start()

        self.timer.start(30)
        self.start_btn.setDisabled(True)
        self.stop_btn.setEnabled(True)

    def update_frame(self, frame):
        return super().update_frame(frame)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Can't read frame from camera")
            return
        frame = cv2.flip(frame, 1)
        
        self.collector.collect(frame, self.timestep, self.text_bar)

        self.lbPredict.setText(f"Prediction: {self.detector.final_label}")
        self.lbConfidence.setText(f"Confidence: {self.detector.confidence:.2f}")

        # FPS
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        self.FPS.setText(f"FPS: {fps:.1f}")

        self.update_frame(frame)


"""
-----------------------------------------------
Module testing function
-----------------------------------------------

def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication()
    
    window = Module2()
    window.show()

    app.exec()

if __name__=="__main__":
    main()
"""