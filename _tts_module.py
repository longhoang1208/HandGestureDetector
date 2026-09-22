"""
==================================================
TEXT TO SPEECH PROCESSOR
==================================================
File name   : _tts_module.py
Description : Initializes the text-to-speech
              process so it runs in parallel
              with the main application flow.
--------------------------------------------------
"""


from piper.voice import PiperVoice
import sounddevice as sd
import threading
import numpy as np
import time
import sys
import os


# Khóa dừng luồng
stop_thread = threading.Event()

# load model giọng đọc tiếng Anh
"""
Nguyên lý hoạt động:
    Khi compile bằng lệnh
    `pyinstaller main.py --onedir --add-data "voices;voices"`,
    khi chạy phần mềm thì file main.exe sẽ giải nén
    các thư mục được thêm vào từ `--add-data` vào một
    thư mục các file tạm.
    
    Hàm resource_path là để tìm đường dẫn tới model voice
    trong thư mục tạm đó, thay vì dùng đường dẫn tương đối
    cần phải copy trực tiếp thư mục voices vào cùng với
    main.exe.
"""
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        # Tìm trong thư mục giải nén của pyinstaller
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path("voices/en_US-lessac-medium.onnx")
voice = PiperVoice.load(model_path)

# ------------------------------------------
# Funciton name : speak
# Description   : Đọc văn bản thành tiếng.
# ------------------------------------------
def speak(text):
    """
    Chia nhỏ các âm tiết trong văn bản,
    sau đó chyển thành các bytes. Cuối cùng
    phát ra loa bằng sounddevice.
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
        samplerate=voice.config.sample_rate,
        blocksize=2048
    )
    sd.wait()


# -----------------------------------------------
# Function name : get_constraints
# Description   : Xét điều kiện để gọi hàm speak.
# -----------------------------------------------
def get_constraints(detector):
    """
    Xét điều kiện, kiểm tra có yêu cầu
    ngưỡng thời gian (set_time) không:
    - Trong single_sign_module, detector
      không yêu cầu ngưỡng thời gian chờ.
    - Trong multi_sign_module, detector
      (trong module này là Collector)
      yêu cầu thời gian chờ để người dùng
      ghép từ, sau khi hết thời gian chờ
      mới đọc cả câu.
    """
    const1 = detector.speak != detector.last_spoke
    const2 = detector.speak.strip()
    if hasattr(detector, "confirmed"):
        return const1 and const2 and detector.confirmed
    return const1 and const2


# -----------------------------------------------------
# Funciton name : call_speak
# Description   : Vòng lặp gọi hàm speak.
# -----------------------------------------------------
def call_speak(detector: object, aud_btn):
    """
    Luồng chạy liên tục, bất cứ khi nào
    thỏa mãn các điều kiện ràng buộc và
    chế độ đọc thành tiếng được bật lên
    (aud_btn.isChecked) thì phát âm thanh.
    """
    while not stop_thread.is_set():
        try:
            constraints = get_constraints(detector)

            if aud_btn.isChecked() and constraints:
                speak(detector.speak)
                detector.last_spoke = detector.speak
                if hasattr(detector, "confirmed"):
                    detector.confirmed  = False
                    detector.last_spoke = ""

        except RuntimeError:
            break

        except Exception as e:
            print(f"Speaker error: {e}")

        time.sleep(0.05)


# ---------------------------------------------------------------
# Function name : speaker_init
# Description   : Khởi tạo luồng đọc văn bản thành tiếng.
# ---------------------------------------------------------------
def speaker_init(detector: object, aud_btn) -> threading.Thread:
    speaker_thread = None
    stop_thread.clear()
    try:
        speaker_thread = threading.Thread(
            target=call_speak,
            args=(detector, aud_btn),
            daemon=True
        )

    except Exception as e:
        print(f"Error: {e}")
        print("Can't play audio")
        return
    return speaker_thread


# -------------------------------------------------------
# Function name : delete_speaker
# Description   : Dừng luồng đọc văn bản
#                 và xóa khỏi bộ nhớ đệm.
# -------------------------------------------------------
def delete_speaker(speaker_thread: threading.Thread):
    stop_thread.set()
    speaker_thread.join()
