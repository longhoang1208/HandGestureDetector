

"""
══════════════════════════════════════════════════════
VIETNAMESE TEXT WRITTING MODULE
══════════════════════════════════════════════════════

Module viết chữ tiếng Việt có dấu
sử dụng pillow.

Cài đặt thư viện:
    pip install pillow

1 - Chọn font chữ từ thiết bị hoặc
    lấy font mặc định.

2 - Cache fonts

3 - Vẽ chữ lên ROI.

4 - Thay vùng cần hiện chữ trên
    frame bằng ROI.
"""


import cv2
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np


def load_font(size: int = 24):
    """
    Tìm font chữ có sẵn trong máy,
    load font mặc định nếu không tìm thấy.
    """
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
        
    print("Using default font")
    return ImageFont.load_default()


# Cache font chữ để chỉ cần load font 1 lần
_font_cache_ = {}
def get_font(size: int = 24):
    if size not in _font_cache_:
        _font_cache_[size] = load_font(size)
    return _font_cache_[size]


# Vẽ chữ lên ROI và hiển thị lên frame
def draw_text(
    frame: np.ndarray, text: str, pos: tuple,
    size: int = 24, col_bgr: tuple = (255, 255, 255),
    bg_bgr: tuple = None
):

    col_rgb = (col_bgr[2], col_bgr[1], col_bgr[0])
    font = get_font(size)

    # Get image size
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)

    # bbox = (x1, y1, x2, y2)
    # tw = x2 - x1
    # th = y2 - y1
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x, y = pos

    if bg_bgr is not None:
        bg_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        pad = 4
        cv2.rectangle(
            frame,
            (x - pad, y - pad),
            (x + tw + pad +10, y + th + pad +10),
            bg_rgb,
            -1
        )
    
    margin = 4
    rx1 = max(0, x - margin)
    ry1 = max(0, y - margin)
    rx2 = min(frame.shape[1], x + tw + margin + 10)
    ry2 = min(frame.shape[0], y + th + margin + 10)

    if rx2 <= rx1 or ry2 <= ry1:
        return
    
    roi = frame[ry1:ry2, rx1:rx2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    draw    = ImageDraw.Draw(pil_img)

    # Vẽ text (vị trí tương đối trong ROI)
    draw.text((x - rx1, y - ry1), text, font=font, fill=col_rgb)

    # Convert lại BGR
    roi_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[ry1:ry2, rx1:rx2] = roi_bgr