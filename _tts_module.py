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


# Thread lock
stop_thread = threading.Event()

# Load English voice model
"""
How it works:
    When building the app with:
    `pyinstaller main.py --onedir --add-data "voices;voices"`,
    the generated `main.exe` will extract the added folders
    from `--add-data` into a temporary directory.

    The `resource_path` function is used to locate the voice model
    inside that temporary folder instead of relying on a relative path.
    In other words, the `voices` folder must be copied alongside
    the `main.exe` file.
"""

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        # Look in the PyInstaller extraction directory
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

model_path = resource_path("voices/en_US-lessac-medium.onnx")
voice = PiperVoice.load(model_path)

"""
------------------------------------------
Function name : speak
Description   : Reads text aloud
------------------------------------------
"""
def speak(text):
    """
    Split the text into smaller chunks,
    then convert them into bytes. Finally,
    play the audio through the speaker using sounddevice.
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


"""
-----------------------------------------------
Function name : get_constraints
Description   : Checks the conditions before
                calling speak.
-----------------------------------------------
"""
def get_constraints(detector):
    """
    Check whether the required conditions are met:
    - In single_sign_module, the detector does not require a waiting time threshold.
    - In multi_sign_module, the detector (in this module, the Collector)
      requires a waiting period so the user can combine words.
      After the timeout, the whole sentence is read aloud.
    """
    const1 = detector.speak != detector.last_spoke
    const2 = detector.speak.strip()
    if hasattr(detector, "confirmed"):
        return const1 and const2 and detector.confirmed
    return const1 and const2


"""
-----------------------------------------------------
Function name : call_speak
Description   : Loop that calls the speak function.
-----------------------------------------------------
"""
def call_speak(detector: object, aud_btn):
    """
    This runs continuously. Whenever the conditions are met
    and the text-to-speech mode is enabled
    (`aud_btn.isChecked()`), audio is played.
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


"""
---------------------------------------------------------------
Function name : speaker_init
Description   : Initializes the text-to-speech thread.
---------------------------------------------------------------
"""
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


"""
-------------------------------------------------------
Function name : delete_speaker
Description   : Stops the text-to-speech thread
                and clears it from memory.
-------------------------------------------------------
"""
def delete_speaker(speaker_thread: threading.Thread):
    stop_thread.set()
    speaker_thread.join()
