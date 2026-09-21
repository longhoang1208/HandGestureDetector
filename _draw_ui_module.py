

"""
══════════════════════════════════════════════════════
USER INTERFACE MODULE
══════════════════════════════════════════════════════

- HEAD BAR:
    Ghi tiêu đề và hiện trạng thái có tay
    trong khung hình hay không.

- FOOTER:
    Hướng dẫn thao tác: thoát chương trình,
    thay đổi mô hình, bật/tắt tính năng đọc
    thành tiếng,...

    Hiển thị tên mô hình đang được sử dụng.

- CLEAR TERMINAL:
    Hàm clear_terminal() để các module khác
    gọi khi cần xóa terminal trước khi hiện 
    thông tin mới. Sử dụng subprocess và
    platform.
"""


import cv2
import platform
import subprocess
from _configurations import color, ui_cfg
from _write_text_vi import draw_text
import numpy as np


COL = color()
UI_CFG = ui_cfg()


def draw_ui(frame, model_name, hand_results, guides: str):
    h, w = frame.shape[:2]
    def_w, def_h = UI_CFG.default_frame_size

    # scale
    sx = w/def_w
    sy = h/def_h

    scale = min(sx, sy)
    thickness = max(1, int(2*scale))

    # Head Bar
    cv2.rectangle(
        frame,
        (0, 0),
        (w, int(50 * sy)),
        COL.NAVY,
        -1
    )
    cv2.line(
        frame,
        (0, int(50 * sy)),
        (w, int(50 * sy)),
        COL.AMBER,
        thickness=thickness
    )
    cv2.putText(
        frame,
        "HAND SIGN DETECTOR",
        (int(150 * sx), int(40 * sy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 * scale,
        COL.CYAN,
        thickness
    )

    # Kiểm tra có tay trong khung hình không
    if hand_results and hand_results.multi_hand_landmarks:
        is_hand = True
    else:
        is_hand = False

    cv2.putText(
        frame,
        f'Hand in frame: {"YES" if is_hand else "NO"}',
        (w - int(400 * sx), int(40 * sy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 * scale,
        COL.GREEN if is_hand else COL.RED,
        thickness=thickness
    )


    # Footer Background
    cv2.rectangle(
        frame,
        (0, h - int(80 * sy)),
        (w, h),
        (20, 20, 20),
        -1
    )
    cv2.line(
        frame,
        (0, h - int(80 * sy)),
        (w, h - int(80 * sy)),
        COL.CYAN,
        thickness=thickness
    )
    
    # Guides
    cv2.putText(
        frame,
        guides,
        (int(30 * sx), h - int(50 * sy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * scale,
        color=(100, 100, 100),
        thickness=thickness
    )

    # Tên model đang sử dụng
    if model_name:
        cv2.putText(
            frame,
            f"Model: {model_name}",
            (int(30 * sx), h - int(10 * sy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            color=(150, 150, 150),
            thickness=thickness
        )


# Khung hiển thị output
def write_on_frame(frame, text):
    h, w = frame.shape[:2]
    def_w, def_h = UI_CFG.default_frame_size

    # scale
    sx = w/def_w
    sy = h/def_h

    scale = min(sx, sy)

    # Background
    cv2.rectangle(
        frame,
        (int(UI_CFG.Text_bg_pos_1[0]*sx), int(UI_CFG.Text_bg_pos_1[1]*sy)),
        (int(UI_CFG.Text_bg_pos_2[0]*sx), int(UI_CFG.Text_bg_pos_2[1]*sy)),
        (20, 20, 20), -1
    )
    cv2.rectangle(
        frame,
        (int(UI_CFG.Text_bg_pos_1[0]*sx), int(UI_CFG.Text_bg_pos_1[1]*sy)),
        (int(UI_CFG.Text_bg_pos_2[0]*sx), int(UI_CFG.Text_bg_pos_2[1]*sy)),
        COL.AMBER, 2
    )

    # Text
    cv2.putText(
        frame, "Text: ", (int(50 * sx), int(180 * sy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * scale,
        color=(90, 90, 90),
        thickness=int(2*scale),
    )

    text_pos = (int(UI_CFG.Text_pos[0]*sx), int(UI_CFG.Text_pos[1]*sy))

    draw_text(
        frame,
        text,
        text_pos,
        40 * scale,
        COL.WHITE
    )


# Vẽ thanh xác suất dự đoán
def draw_multi_bars(frame, detector, labels, top_k=5):
    h, w = frame.shape[:2]
    def_w, def_h = UI_CFG.default_frame_size

    # scale
    sx = w/def_w
    sy = h/def_h

    scale = min(sx, sy)

    if detector.probs is None:
        return

    probs = detector.probs

    # lấy top K label cao nhất
    top_indices = np.argsort(probs)[-top_k:][::-1]

    start_x = 50
    gap = 30
    font_size = 0.5 * scale

    for i, idx in enumerate(top_indices):
        prob  = probs[idx]
        label = labels[str(idx)]

        # Đổi màu
        if prob > 0.7:
            color = COL.GREEN

        elif prob > 0.3:
            color = COL.AMBER
        else:
            color = COL.RED

        # Đổi kích thước, vị trí bar
        if i == 0:
            start_y    = 300
            bar_width  = 250
            bar_height = 25
        else:
            start_y    = 330
            bar_width  = 200
            bar_height = 15

        y = start_y + i * gap

        # background
        cv2.rectangle(frame,
                    (int(start_x * sx), int(y*sy)),
                    (int((start_x + bar_width)*sx), int((y + bar_height)*sy)),
                    (50, 50, 50), -1)

        # fill
        cv2.rectangle(frame,
                    (int(start_x*sx), int(y*sy)),
                    (int((start_x + int(prob * bar_width))*sx), int((y + bar_height)*sy)),
                    color, -1)

        # border
        cv2.rectangle(frame,
                    (int(start_x*sx), int(y*sy)),
                    (int((start_x + bar_width)*sx), int((y + bar_height)*sy)),
                    COL.WHITE, 1)

        # Ghi nhãn
        draw_text(frame, label,
                  (int(start_x * sx), int((y - 13) * sy)),
                  size=int(15*scale),
                  col_bgr=COL.WHITE
                )

        # Xác suất
        cv2.putText(frame, f"{prob*100:.1f}%",
                    (int((start_x + bar_width + 10)*sx), int((y + 15)*sy)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    COL.WHITE, 1)


def clear_terminal():
    if platform.system() == "Windows":
        subprocess.run("cls", shell=True)
    else:  # Linux và macOS
        subprocess.run("clear", shell=True)