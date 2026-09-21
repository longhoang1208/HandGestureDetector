

import cv2
import tensorflow as tf
from _draw_ui_module import draw_ui, clear_terminal, write_on_frame, draw_multi_bars
import json
import os
import argparse
from _tts_module import speaker_init, delete_speaker, audio_state, stop_thread
from _detector import Detector, draw_landmarks
from _select_model_n_labels import list_models, select_model, list_labels, select_labels
from _configurations import config, color, ANSI_code


CFG  = config()
COL  = color()
ANSI = ANSI_code()


title = rf"""{ANSI.CYAN}
███████╗██╗███╗   ██╗ ██████╗ ██╗     ███████╗
██╔════╝██║████╗  ██║██╔════╝ ██║     ██╔════╝
███████╗██║██╔██╗ ██║██║  ███╗██║     █████╗
╚════██║██║██║╚██╗██║██║   ██║██║     ██╔══╝
███████║██║██║ ╚████║ ██████╔╝███████║███████╗
╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝

███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║██╔════╝ ████╗  ██║
███████╗██║██║  ███╗██╔██╗ ██║
╚════██║██║██║   ██║██║╚██╗██║
███████║██║ ██████╔╝██║ ╚████║
╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
{ANSI.RESET}
"""


# ════════════════════════════════════════════════════════════════════════════════════
# ARGUEMENTS
# - Tên model
# - Định dạng model
# ════════════════════════════════════════════════════════════════════════════════════

# Lựa chọn model và định dạng của model
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=CFG.model_name,
        choices=[model_name for model_name in os.listdir(CFG.models_dir)]
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=CFG.labels,
        choices=[labels_file for labels_file in os.listdir(CFG.labels_dir)]
    )

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════
def main():
    frame_size = (800, 450)

    args = parse_args()
    labels_file = args.labels
    labels = os.path.join(
        CFG.labels_dir,
        labels_file
    )

    with open(labels, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    model_list = os.listdir(CFG.models_dir)
    labels_list = os.listdir(CFG.labels_dir)

    model_name = args.model
    model = tf.keras.models.load_model(
        os.path.join(CFG.models_dir, model_name),
        compile=False)
    
    timestep = model.input_shape[1]     # (None, timestep, n_features)
    detector = Detector(model, labels)

    detector._camera_init(frame_size)
    cap = detector.cap
    
    is_detecting = True

    speaker_thread = None
    use_audio = False

    clear_terminal()

    print(title)

    print(f"\n{ANSI.GREEN}✔  Load{ANSI.RESET}: {ANSI.YELLOW}{model_name}{ANSI.RESET}")
    print(f"{ANSI.GREEN}✔  Load{ANSI.RESET}: {ANSI.YELLOW}{labels_file}{ANSI.RESET}\n")

    while True:
        if is_detecting:
            ret, frame = cap.read()
            if not ret:
                print("Can't read frame from camera...")
                break

            frame    = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detector.hand_results = detector.hands.process(frameRGB)
            detector.pose_results = detector.pose.process(frameRGB)

            text = detector.detection(
                timestep=timestep,
                stride=CFG.stride
            )

            draw_landmarks(
                frame,
                hand_results=detector.hand_results,
                pose_results=detector.pose_results
            )
            write_on_frame(frame, text)
            draw_multi_bars(frame, detector, detector.labels)

            draw_ui(
                frame,
                model_name,
                detector.hand_results,
                'Press SPACE to start/stop detecting    |    Press "A" to use audio    |    Press ESC to quit'
            )
            audio_state(frame, use_audio)
            
            cv2.imshow("HandSignDetector", frame)
        
        else:
            if cap.isOpened():
                cap.release()
                cv2.destroyAllWindows()

            detector.reset()

            if speaker_thread:
                delete_speaker(speaker_thread)
                speaker_thread = None
                use_audio = False

            clear_terminal()
            print(title)

            # Change model and labels
            list_models(model_list)
            model_name = select_model(model_list)

            if model_name:
                list_labels(labels_list)
                labels_file = select_labels(labels_list)

                if labels_file:
                    model = tf.keras.models.load_model(
                        os.path.join(CFG.models_dir, model_name),
                        compile=False)
                    
                    labels = os.path.join(
                        CFG.labels_dir,
                        labels_file
                    )

                    with open(labels, "r", encoding="utf-8") as f:
                        labels = json.load(f)

                    timestep = model.input_shape[1]
                    detector = Detector(model, labels)

                    detector._camera_init(frame_size)
                    cap = detector.cap

                    is_detecting = True

                    clear_terminal()
                    print(title)

                    list_models(model_list)
                    print(f"{ANSI.GREEN}✔  Load{ANSI.RESET}: {ANSI.YELLOW}{model_name}{ANSI.RESET}\n")

                    list_labels(labels_list)
                    print(f"{ANSI.GREEN}✔  Load{ANSI.RESET}: {ANSI.YELLOW}{labels_file}{ANSI.RESET}\n")


        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            is_detecting = not is_detecting

        if key == ord("a"):
            use_audio = not use_audio

        if use_audio and speaker_thread is None:
            stop_thread.clear()

            try:
                speaker_thread = speaker_init(detector)
                speaker_thread.start()

            except Exception as e:
                print(f"Error: {e}")
        
        elif not use_audio and not stop_thread.is_set():
            if speaker_thread:
                delete_speaker(speaker_thread)
                speaker_thread = None

        if key == 27:
            break

    cv2.destroyAllWindows()
    print("\n👋 Quit")

if __name__ == "__main__":
    main()