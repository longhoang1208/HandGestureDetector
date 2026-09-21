

"""
══════════════════════════════════════════════════════
TEXT TO SPEECH MODULE
══════════════════════════════════════════════════════

Load:
    voice/
    ├── en_US-lessac-medium.onnx
    └── en_US-lessac-medium.onnx.json

Luồng đọc văn bản thành tiếng, chạy độc lập với
luồng chính nhận diện ký hiệu.

Luồng được khởi tạo khi người dùng bật tính năng
đọc thành tiếng, tự động gọi hàm hủy luồng hoạt
động khi người dùng tắt tính năng này.
"""


from piper.voice import PiperVoice
import sounddevice as sd
import threading
import numpy as np
import time
import cv2
from _configurations import ui_cfg


GREEN = (76, 153, 0)
RED = (50, 50, 220)
UI_CFG = ui_cfg()


# Khóa ngắt luồng khi muốn ngưng đọc thành tiếng
stop_thread = threading.Event()

# Tải model giọng đọc tiếng Anh
voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")
def speak(text):
    """
    Chuyển văn bản thành vector các byte
    dữ liệu và tách thành phoneme.

    Phát âm thanh bằng sounddevice.
    """
    audio = bytearray()

    for chunk in voice.synthesize(text):
        audio.extend(chunk.audio_int16_bytes)

    pcm = np.frombuffer(
        audio,
        dtype=np.int16
    )

    sd.play(
        pcm,
        samplerate=voice.config.sample_rate
    )
    sd.wait()


def get_constraints(detector):
    """
    Xét điều kiện, kiểm tra có yêu cầu
    ngưỡng thời gian (set_time) không.
    """
    const1 = detector.speak != detector.last_spoke
    const2 = detector.speak.strip()
    if hasattr(detector, "set_time"):
        return const1 and const2 and time.time() >= detector.set_time
    return const1 and const2


# Luồng đọc văn bản thành tiếng
def call_speak(detector: object):
    """
    Khi thread vẫn đang tồn tại (stop_thread
    chưa được set), vòng lặp sẽ xét điều kiện
    từ hàm get constraint và gọi hàm speak để
    chuyển văn bản thành âm thanh, sau đó gán
    biến last_spoke bằng từ vừa mới đọc để
    tránh lặp lại một từ nhiều lần.
    """
    while not stop_thread.is_set():
        constraints = get_constraints(detector)
        if constraints:
            try:
                speak(detector.speak)
                detector.last_spoke = detector.speak

            except Exception as e:
                print(f"Speaker error: {e}")

        time.sleep(0.05)


def speaker_init(detector: object) -> threading.Thread:
    """
    Khởi tạo luồng đọc thành tiếng
    """
    speaker_thread = None
    try:
        speaker_thread = threading.Thread(
            target=call_speak,
            args=(detector,),
            daemon=True
        )

    except Exception as e:
        print(f"Error: {e}")
        print("Can't play audio")
        pass
    return speaker_thread


def delete_speaker(speaker_thread: threading.Thread):
    """
    Hủy luồng khi người dùng tắt
    tính năng đọc thành tiếng.
    """
    stop_thread.set()
    speaker_thread.join()


# Kiểm tra có đang sử dụng tính năng đọc thành tiếng không
def audio_state(frame, use_audio):
    h, w = frame.shape[:2]
    def_w, def_h = UI_CFG.default_frame_size

    # scale
    sx = w/def_w
    sy = h/def_h

    scale = min(sx, sy)

    if use_audio:
        speaker_stat = "Using Audio"
        col = GREEN
    else:
        speaker_stat = "Not Using Audio"
        col = RED
    
    cv2.putText(
        frame,
        speaker_stat,
        (w-int(300*sx), h-int(20*sy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8*scale, col, max(1, int(2*scale))
    )
