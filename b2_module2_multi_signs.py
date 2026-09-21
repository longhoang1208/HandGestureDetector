

import cv2
from _detector import Detector, draw_landmarks
import tensorflow as tf
import json
import time
from _tts_module import speaker_init, delete_speaker, audio_state, stop_thread
from _draw_ui_module import draw_ui, clear_terminal, write_on_frame, draw_multi_bars
import os
from _write_text_vi import draw_text
from _select_model_n_labels import list_models, select_model, list_labels, select_labels
import argparse
from _configurations import config, color, ANSI_code, ui_cfg


CFG  = config()
COL  = color()
ANSI = ANSI_code()
UI_CFG = ui_cfg()


title = rf"""{ANSI.CYAN}
███╗    ███╗██╗   ██╗██╗  ████████╗██╗
████╗  ████║██║   ██║██║  ╚══██╔══╝██║
██╔██╗██╔██║██║   ██║██║     ██║   ██║
██║╚███╔╝██║██║   ██║██║     ██║   ██║
██║ ╚══╝ ██║╚██████╔╝███████║██║   ██║
╚═╝      ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═╝

███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║██╔════╝ ████╗  ██║
███████╗██║██║  ███╗██╔██╗ ██║
╚════██║██║██║   ██║██║╚██╗██║
███████║██║ ██████╔╝██║ ╚████║
╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
{ANSI.RESET}
"""


class SequenceModule:
    def __init__(self, dur, detector):
        self.detector = detector
        self.gloss_list = []
        self.gloss = ""
        self.init_len = 0
        self.set_time = 0
        self.dur = dur

        self.last_spoke = ""
        self.speak = ""
    

    def make_gloss_seq(self):
        merged = []
        spell_word = ""

        for token in self.gloss_list:
            # nếu là 1 chữ cái
            if token.startswith("'") and token.endswith("'"):
                spell_word += token.replace("'", "")

            else:
                # nếu đang spelling thì push vào
                if spell_word:
                    merged.append(spell_word)
                    spell_word = ""

                merged.append(token)

        # append cuối
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

    def collect(self, frame, timestep):
        h, w = frame.shape[:2]
        def_w, def_h = UI_CFG.default_frame_size

        # scale
        sx = w/def_w
        sy = h/def_h

        scale = min(sx, sy)

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1],int(120*sy)),
            (25, 25, 25), -1
        )

        self.detector.hand_results = self.detector.hands.process(frameRGB)
        self.detector.pose_results = self.detector.pose.process(frameRGB)
        text = self.detector.detection(timestep, CFG.stride)

        draw_landmarks(frame,
                       self.detector.hand_results,
                       self.detector.pose_results
                    )
        write_on_frame(frame, text)
        draw_multi_bars(frame,
                        self.detector,
                        self.detector.labels
                    )

        self.get_gloss(text)
        self.gloss = self.make_gloss_seq()
        self.speak = self.make_gloss_seq()

        if self.init_len != len(self.gloss):
            self.init_len = len(self.gloss)
            self.set_time = time.time() + self.dur
        else:
            if time.time() <= self.set_time:
                draw_text(
                    frame, self.gloss,
                    (int(30*sx), int(60*sy)), 45 * scale,
                    COL.WHITE
                )

            elif time.time() > self.set_time + 5.0:
                self.gloss = ""
                self.gloss_list.clear()
                self.speak = ""
                self.last_spoke = ""


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
        compile=False
    )
    
    detector = Detector(model, labels)
    detector._camera_init(frame_size)
    cap = detector.cap

    timestep = model.input_shape[1]
    collector = SequenceModule(dur=8.0, detector=detector)

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
                print("Can't read frame from camera")
                break
            frame = cv2.flip(frame, 1)
            collector.collect(frame, timestep)

            draw_ui(
                frame,
                model_name,
                collector.detector.hand_results,
                'Press "A" to use audio    |    Press ESC to quit',
            )
            audio_state(frame, use_audio)
            
            cv2.imshow("HandSignDetector", frame)

        else:
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

                    collector = SequenceModule(dur=8.0, detector=detector)

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

        if key == 27:
            break
        
        if key == ord("a"):
            use_audio = not use_audio

        if use_audio and speaker_thread is None:
            stop_thread.clear()

            try:
                speaker_thread = speaker_init(collector)
                speaker_thread.start()

            except Exception as e:
                print(f"Error: {e}")
        
        elif not use_audio and not stop_thread.is_set():
            if speaker_thread:
                delete_speaker(speaker_thread)
                speaker_thread = None
    
    detector.cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()