

"""
══════════════════════════════════════════════════════
DATA COLLECT MODULE
══════════════════════════════════════════════════════

Module thu thập dữ liệu để huấn luyện mô hình.

1 - Tạo thư mục để lưu dữ liệu.

2 - Tạo khung input để người dùng
    nhập các nhãn muốn sử dụng.

3 - Lưu các nhãn theo thứ tự vào một
    file json trong thư mục labels.

4 - Mở cửa sổ camera, người dùng nhấn
    phím cách để thực hiện ký hiệu
    trước camera để bắt đầu thu thập 
    dữ liệu.

5 - In bảng tổng kết quá trình thu
    data: các nhãn và số chuỗi đã
    lưu.

Thư mục data gồm 2 thư mục con:
    processed: dữ liệu đã xử lý và có
               thể đưa vào huấn luyện.

    training plot: lưu biểu đồ đánh
                   giá quá trình huấn
                   luyện và confusion
                   matrix.
"""


import cv2
from _detector import Detector, draw_landmarks
import numpy as np
from pathlib import Path
from _draw_ui_module import draw_ui, clear_terminal
from _landmark_module import extract_landmarks
from _write_text_vi import draw_text
import json
import time
from _configurations import color


data_dir = "data"
default_num_samples = 20

COL = color()


def get_labels(labels_dir):
    labels = {}
    is_labeling = True
    index = 0

    while is_labeling:
        label = str(input(f'Enter label for index [{index}] (enter "f" to finish labeling): '))
        labels[index] = label
        
        if label.lower() == "f":
            labels.pop(index)

            if len(labels) < 2:
                print("There must be at least 2 labels")
                break

            with open(labels_dir, "w") as f:
                json.dump(labels, f, indent=2)

            return labels
        index += 1


def count_down(frame, start_time, duration):
    remaining = int(duration + 1 - (time.time() - start_time))

    if remaining > 0:
        # nền mờ
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (frame.shape[1]//2 - 80, frame.shape[0]//2 - 80),
            (frame.shape[1]//2 + 80, frame.shape[0]//2 + 80),
            (30, 30, 30), -1
        )
        
        frame[:] = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # số đếm
        cv2.putText(
            frame,
            str(remaining),
            (frame.shape[1]//2 - 28, frame.shape[0]//2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 3,
            COL.AMBER, 5
        )

        cv2.putText(
            frame,
            str(remaining),
            (frame.shape[1]//2 - 28, frame.shape[0]//2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 3,
            COL.AMBER, 5
        )
        return False  # chưa xong
    
    else:
        return True   # đã xong


class CollectModule(Detector):
    def __init__(self, timestep, num_sample, labels_dir):
        self._mp_init()
        self.timestep = timestep

        self.sequence = []
        self.labels = {}

        self.num_sample = num_sample
        self.labels_dir = labels_dir

        self.is_recording = False
        self.start_time = None

    def count_seq(self, label):
        label_path = Path(data_dir) / "processed" / label

        if not label_path.exists():
            return 0

        num_seq = len(list(label_path.glob("*.npy")))
        return num_seq


    def make_sequence(self, hand_results, pose_results):
        if hand_results.multi_hand_landmarks:
            lm = extract_landmarks(hand_results, pose_results)
        else:
            lm = [0]*(63*2+33*3)
        
        if len(self.sequence) < self.timestep:
            self.sequence.append(lm)
    

    def collect(self, hand_results, pose_resulte, current_index):
        self.make_sequence(hand_results, pose_resulte)
        
        save_dir = Path(data_dir) / "processed" / self.labels[current_index]
        save_dir.mkdir(parents=True, exist_ok=True)

        seq_id = self.count_seq(self.labels[current_index])
        save_path = save_dir / f"{seq_id}.npy"
        
        if seq_id == self.num_sample:
            self.is_recording = False

        if len(self.sequence) == self.timestep:
            np.save(
                save_path,
                np.array(self.sequence, dtype=np.float32)
            )
            self.is_recording = False
            self.sequence.clear()


    def draw_progress_bar(self, frame, curr_idx):
        # MAIN BAR
        main_bar_start_x = 30
        main_bar_start_y = 300
        bar_w = 250
        bar_h = 20

        cv2.putText(
            frame,
            "Overall process",
            (main_bar_start_x, main_bar_start_y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL.WHITE, 1
        )

        # BLANK BAR
        cv2.rectangle(
            frame,
            (main_bar_start_x, main_bar_start_y),
            (
                main_bar_start_x + bar_w,
                main_bar_start_y + bar_h
            ),
            COL.BLACK, -1
        )

        cv2.rectangle(
            frame,
            (main_bar_start_x, main_bar_start_y),
            (
                main_bar_start_x + bar_w,
                main_bar_start_y + bar_h
            ),
            COL.WHITE, 2
        )

        # FILL
        total = sum(self.count_seq(label)
                    for label in self.labels.values())
        fill_size = int(bar_w * total/(self.num_sample * len(self.labels)))

        cv2.rectangle(
            frame,
            (main_bar_start_x, main_bar_start_y),
            (
                main_bar_start_x + fill_size,
                main_bar_start_y + bar_h
            ),
            COL.WHITE, -1
        )

        # CURRENT PROCESS BAR
        curr_start_x = main_bar_start_x
        curr_start_y = main_bar_start_y+bar_h+30

        cv2.putText(
            frame,
            "Current process",
            (curr_start_x, curr_start_y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL.WHITE, 1
        )

        # BLANK BAR
        cv2.rectangle(
            frame,
            (curr_start_x, curr_start_y),
            (
                curr_start_x + bar_w,
                curr_start_y + bar_h
            ),
            COL.BLACK, -1
        )

        fill_size = int(bar_w * self.count_seq(self.labels[curr_idx])/self.num_sample)
        cv2.rectangle(
            frame,
            (curr_start_x, curr_start_y),
            (
                curr_start_x+fill_size,
                curr_start_y + bar_h
            ),
            COL.WHITE, -1
        )

        cv2.rectangle(
            frame,
            (curr_start_x, curr_start_y),
            (
                curr_start_x + bar_w,
                curr_start_y + bar_h
            ),
            COL.WHITE, 2
        )


    def frame_count_bar(self, frame, frame_count):
        # FRAME COUNT
        fc_start_x = 30
        fc_start_y = 220
        fc_w = 200
        fc_h = 10

        cv2.rectangle(
            frame,
            (fc_start_x, fc_start_y), (fc_start_x+fc_w, fc_start_y+fc_h),
            COL.NAVY, -1
        )

        if frame_count <= self.timestep:
            cv2.rectangle(
                frame,
                (fc_start_x, fc_start_y),
                (fc_start_x+int(fc_w*frame_count/self.timestep), fc_start_y+fc_h),
                COL.CYAN, -1
            )

        cv2.rectangle(
            frame,
            (fc_start_x, fc_start_y), (fc_start_x+fc_w, fc_start_y+fc_h),
            COL.AMBER, 1
        )


    def run(self):
        self._camera_init()

        clear_terminal()
        print("═"*50)
        print(" DATA COLLECT MODULE")
        print(" Press SPACE to start collecting")
        print("═"*50)

        self.labels = get_labels(self.labels_dir)
        if not self.labels:
            return

        Path(data_dir).mkdir(exist_ok=True)
        
        label_idx = 0
        frame_count = 0

        for label_idx in self.labels.keys():
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Can't read frame from camera")
                    break
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_res = self.hands.process(framergb)
                pose_res = self.pose.process(framergb)

                # Process box
                cv2.rectangle(
                    frame,
                    (0, 0), (320, 400),
                    COL.GRAY, -1
                )

                cv2.rectangle(
                    frame,
                    (0, 0), (320, 400),
                    COL.AMBER, 2
                )

                draw_ui(
                    frame,
                    hand_results=hand_res,
                    model_name=None,
                    guides='Press SPACE to start collecting    |    Press ESC to quit'
                )
                self.draw_progress_bar(frame, label_idx)
                draw_text(
                    frame,
                    f"LABEL: {self.labels[label_idx]}",
                    (25,75),
                    35, COL.WHITE,
                    bg_bgr=(10, 30, 60)
                )

                draw_landmarks(frame, hand_res, pose_res)
                current_n_seq = self.count_seq(self.labels[label_idx])

                if self.is_recording:
                    started = count_down(frame, self.start_time, 3)
                    if started:
                        current_n_seq += 1
                        cv2.putText(
                            frame,
                            "Recording",
                            (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, COL.RED, 2
                        )

                        self.collect(hand_res, pose_res, label_idx)
                        frame_count += 1
                        
                        self.frame_count_bar(frame, frame_count)

                    if (not self.is_recording
                        and self.count_seq(self.labels[label_idx])==self.num_sample):
                        break                        

                else:
                    frame_count = 0

                cv2.putText(
                    frame,
                    f'{current_n_seq}/{self.num_sample}',
                    (30, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                )
                cv2.imshow("DataCollectModule", frame)

                key = cv2.waitKey(1) & 0xFF
                if (key == 27
                    or self.count_seq(self.labels[label_idx])>=self.num_sample):
                    break

                if key == ord(" ") and not self.is_recording:
                    self.is_recording = True
                    self.start_time = time.time()

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    clear_terminal()

    labels_name = None
    while not labels_name or not labels_name.strip():
        clear_terminal()
        labels_name = str(input("Enter labels file name: "))

    labels_dir = f"labels/{labels_name}.json"
    Path(labels_dir).parent.mkdir(parents=True, exist_ok=True)

    n_samples = default_num_samples
    try:
        n_samples = int(input(f"Enter number of samples for each label (default num-samples: {default_num_samples}): "))
        if n_samples < 20:
            print("⚠️ There must be at least 20 samples for each label!")
            print(f"Using default number of samples: {default_num_samples}")
            time.sleep(1.0)
    except ValueError:
        print("⚠️ You must enter an interger!")
        print(f"Using default number of samples: {default_num_samples}")

    if n_samples >= 20: num_sample = n_samples

    collect_module = CollectModule(
        timestep=30,
        num_sample=num_sample,
        labels_dir=labels_dir
    )

    collect_module.run()
    if not collect_module.labels:
        return

    print("\n" + "═"*50)
    print(" 📊 DATA COLLECT RESULTS")
    print("═"*50)
    print(f'{"LABELS":<15}{"SAMPLES":<8}\n')

    for label in collect_module.labels.values():
        num_collected_samples = f"{collect_module.count_seq(label)}/{num_sample}"
        print(f"{label:<15}{num_collected_samples:<8}")

if __name__=="__main__":
    main()