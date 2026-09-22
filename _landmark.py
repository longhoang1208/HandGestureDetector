"""
==================================================
LANDMARK PROCESSOR
==================================================
File name   : _landmark_module.py
Description : Extracts the coordinates of key
              landmarks and normalizes them
              to the range [-1, 1].
--------------------------------------------------
"""


import numpy as np
import mediapipe as mp


"""
--------------------------
DATA SHAPES
--------------------------
"""

# MediaPipe Hand Landmarks
single_hand_shape = 3 * 21

# MediaPipe Pose Landmarks
pose_shape = 3 * 33

# Frame features shape
lm_shape = 2 * single_hand_shape + pose_shape


def normalize_landmarks(points, origin_idx):
    pts = np.array(points, dtype=np.float32)
    pts -= pts[origin_idx].copy()

    max_dist = np.max(np.abs(pts))
    if max_dist > 0:
        pts /= max_dist
    
    return pts.flatten().tolist()


def extract_landmarks(hand_results, pose_results):
    hand_landmarks = hand_results.multi_hand_landmarks
    handedness     = hand_results.multi_handedness
    pose_landmarks = pose_results.pose_landmarks

    left_hand  = [0] * single_hand_shape
    right_hand = [0] * single_hand_shape
    body = [0] * pose_shape

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


def draw_landmarks(frame, hand_results, pose_results):
    """
    Draw landmarks:
    - Keypoints (landmarks)
    - Connect lines (HAND_CONNECTIONS, POSE_CONNECTIONS)
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
                    color=(20, 171, 226),
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
                color=(20, 171, 226),
                thickness=-1,
                circle_radius=8
            ),
            mp_drawing.DrawingSpec(
                color=(0, 220, 100),
                thickness=5
            ),
        )