

"""
══════════════════════════════════════════════════════
LANDMARKS EXTRACTION & NORMALIZATION MODULE
══════════════════════════════════════════════════════

Trích xuất đặc trưng:
    - 21 landmarks tay: 21 x 2(2 tay) x 3(3 trục) = 126
    - 33 landmarks cơ thể: 33 x 3 = 99
    -> Tổng 225

Chuẩn hóa:
    - Xác định điểm gốc tọa độ
    - Trừ các landmarks cho gốc tọa độ
    - Chia các vector cho giá trị tuyệt đối lớn nhất
"""

import numpy as np

def normalize_landmarks(points, origin_idx):
    """
    Trừ các landmark cho điểm gốc và
    chia cho giá trị tuyệt đối lớn nhất.
    """
    pts = np.array(points, dtype=np.float32)
    pts -= pts[origin_idx].copy()

    max_dist = np.max(np.abs(pts))
    if max_dist > 0:
        pts /= max_dist
    
    return pts.flatten().tolist()


def extract_landmarks(hand_results, pose_results):
    """
    Tạo các list để nhận tọa độ các landmarks từ
    2 tay và của pose landmark. Mặc định các phần
    tử là 0 nếu không thấy tay.

    Trích xuất các đặc trưng và gán cho list
    tương ứng sau đó chuẩn hóa.

    Ghép dữ liệu từ 2 tay và cơ thể thành 1.
    """
    hand_landmarks = hand_results.multi_hand_landmarks
    handedness     = hand_results.multi_handedness
    pose_landmarks = pose_results.pose_landmarks

    left_hand  = [0]*3*21
    right_hand = [0]*3*21
    body = [0]*3*33

    if pose_results and pose_landmarks:
        pose_lm = []

        for lm in pose_landmarks.landmark:
            lm = np.array([lm.x, lm.y, lm.z])
            pose_lm.append(lm)
            
        body = normalize_landmarks(pose_lm, origin_idx=0)


    if hand_results and hand_landmarks:
        for hand_idx in range(len(hand_landmarks)):
            landmarks = hand_landmarks[hand_idx]
            l_r = handedness[hand_idx].classification[0].label

            hand_lm = []

            for lm in landmarks.landmark:
                lm = np.array([lm.x, lm.y, lm.z])
                hand_lm.append(lm)

            hand_flat = normalize_landmarks(hand_lm, origin_idx=0)

            if l_r == "Left":
                left_hand = hand_flat
            else:
                right_hand = hand_flat

    return left_hand + right_hand + body