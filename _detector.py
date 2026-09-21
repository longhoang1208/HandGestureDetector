

"""
══════════════════════════════════════════════════════
HAND SIGN DETECT MODULE
══════════════════════════════════════════════════════

Module nhận diện cử chỉ tay sử dụng
mô hình BiLSTM.

- Load mô hình và nhãn.

- Trích xuất đặc trưng từ camera
  và chuẩn hóa dữ liệu bằng
  module _landmark_module.py.

- Nhận diện ký hiệu bằng mô hình
  BiLSTM.

- Vẽ các điểm landmark và khung xương.

- Vẽ thanh xác suất của 5 nhãn có
  xác suất cao nhất.
"""


import cv2
import mediapipe as mp
from _landmark_module import extract_landmarks
import numpy as np
import time
from _configurations import color
from _draw_ui_module import ui_cfg


# ════════════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════════════

class config:
    def __init__(self):
        self.max_num_hand = 2
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence  = 0.5

        self.label_duration = 0.1

cfg = config()
ui_CFG = ui_cfg()
COL = color()

# ════════════════════════════════════════════════════════════════════════════════════
# LOGIC XỬ LÝ CHÍNH
# - trích xuất đặc trưng
# - chuẩn hóa dữ
# - ghép chuỗi & nhận diện ký hiệu
# ════════════════════════════════════════════════════════════════════════════════════

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
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

        if not self.cap.isOpened():
            print("Can't open camera")

    def _mp_init(self):
        """
        Khởi tạo đối tượng hands và pose
        từ mediapipe solutions
        """
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=cfg.max_num_hand,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.hand_results = None

        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )
        self.pose_results = None

    def detect(self, sequence):
        """
        Dự đoán xác suất dựa trên chuỗi dữ liệu,
        xác nhận kết quả nếu xác suất lớn hơn 70%.

        Sau khi có kết quả dự đoán, đợi 0.1 giây.
        Nếu trong lúc đợi mà nhãn không đổi mới hiển thị,
        nếu không thì reset thời gian đợi và xóa chuỗi dữ liệu.
        """
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
            self.stable_time = time.time() + cfg.label_duration
            self.last_label  = self.label
            self.final_label = ""
            self.sequence.clear()
        
        if time.time() >= self.stable_time:
            self.final_label = self.label
            self.speak = self.final_label

    def detection(self, timestep: int, stride: int) -> str:
        """
        Trích xuất các landmarks và chuẩn hóa,
        nếu không thấy tay thì mặc định dữ liệu
        trong frame đó là 0.

        Ghép dữ liệu các frame thành chuỗi dữ liệu,
        gọi hàm dự đoán nhãn. Thay ký tự "_" thành
        khoảng trắng.
        """
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
        """
        Hủy toàn bộ dữ liệu và kết quả dự đoán
        khi dừng chương trình để đổi mô hình.

        - Xóa chuỗi dữ liệu
        - Xóa xác suất dự đoán
        - Xóa các nhãn đã dự đoán
        """
        self.sequence.clear()
        self.probs = None
        self.label = ""
        self.last_label = ""
        self.final_label = ""


def draw_landmarks(frame, hand_results, pose_results):
    """
    Vẽ khung xương:
    - Điểm khớp (landmarks)
    - Đường nối (HAND_CONNECTIONS, POSE_CONNECTIONS)
    """
    mp_hands   = mp.solutions.hands
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    if hand_results and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=COL.CYAN,
                    thickness=-1,
                    circle_radius=8
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 220, 100),
                    thickness=5
                ),
            )

    if pose_results and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=COL.CYAN,
                thickness=-1,
                circle_radius=8
            ),
            mp_drawing.DrawingSpec(
                color=(0, 220, 100),
                thickness=5
            ),
        )
