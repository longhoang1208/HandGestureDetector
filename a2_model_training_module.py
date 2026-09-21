

"""
══════════════════════════════════════════════════════
MODEL TRAINING MODULE
══════════════════════════════════════════════════════

Module huấn luyện mô hình BiLSTM, sử dụng thư viện
keras từ tensorflow, scikit-learn và dùng matplotlib
để vẽ biểu đồ đánh giá quá trình huấn luyện và
confusion matrix.

Kiến trúc mô hình:
    BiLSTM (128 units): Huấn luyện mô hình bằng chuỗi
                        dữ liệu theo cả 2 chiều: từ
                        quá khứ đến tương lai và ngược
                        lại.

    LSTM (64 units):    Mô hình học các thông tin đăc
                        trưng cuối cùng trước khi qua
                        các lớp Dense.

    Dense:              Chuyển các nhãn thành xác các
                        xác suất.

EarlyStop: Dừng quá trình huấn luyện khi có dấu hiệu
           overfit và khôi phục các trọng số ở lần
           hiệu quả nhất.

ReduceLearningRate: Giảm tốc độ huấn luyện khi sai số
                    không còn quá lớn, bắt đầu tinh chỉnh.

Mô hình sau khi huấn luyện được lưu trong thư mục models
với tên mặc định là model.keras.
"""


import numpy as np
from keras.layers import LSTM, Input, Dense, Dropout, BatchNormalization, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import json
import time
import argparse
import tensorflow as tf
from _draw_ui_module import clear_terminal
from _select_model_n_labels import list_labels, select_labels


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

class config:
    def __init__(self):
        self.data_dir   = "data/processed"
        self.labels_dir = "labels"

        self.models_dir = "models"
        self.model_name = "model"

        self.training_plot_dir = "data/training_plot"

        self.LSTM_layer_1_units = 128
        self.LSTM_layer_2_units = 64

        self.epochs    = 1500
        self.test_size = 0.25
        self.patience  = 20

CFG = config()


# ══════════════════════════════════════════════════════════════════
# TRAIN MODULE
# ══════════════════════════════════════════════════════════════════

class BiLSTM:
    def __init__(self, epochs, patience, model_name, model_type, labels_path):
        self.epochs     = epochs
        self.patience   = patience
        self.model_name = f"{model_name}.{model_type}"

        self.labels_path = labels_path

    # LOAD DATAS
    def load_data(self, samples):
        x = []
        y = []

        for path, label in samples:
            data = np.load(path)

            x.append(data)
            y.append(label)
        return np.array(x, dtype=np.float32), np.array(y)
    

    def get_samples(self):
        samples = []

        with open(self.labels_path, "r") as f:
            label_map = json.load(f)

        # Lặp qua tất cả file và thêm vào samples list
        for idx, label in label_map.items():
            folder = os.path.join(CFG.data_dir, label)

            for file in os.listdir(folder):
                if file.endswith(".npy"):
                    samples.append((os.path.join(folder, file), int(idx)))

        # Tách train data và test data theo file
        train_samples, test_samples = train_test_split(
            samples,
            test_size=CFG.test_size,
            stratify=[s[1] for s in samples],
            shuffle=True,
            random_state=42
        )

        x_train, y_train = self.load_data(train_samples)
        x_test, y_test   = self.load_data(test_samples)

        print(f"✔ Train samples: {len(train_samples)} samples")
        print(f"✔ Test samples: {len(test_samples)} samples")

        print(f"Train shape: {x_train.shape}")
        print(f"Test shape: {x_test.shape}\n")

        return x_train, y_train, x_test, y_test, label_map


    # TRAIN
    def train_model(self):
        x_train, y_train, x_test, y_test, label_map = self.get_samples()
        (timestep, n_features) = (x_train.shape[1], x_train.shape[2])

        # Sắp xếp layers
        model = Sequential()
        model.add(Input(shape=(timestep, n_features)))
        model.add(Bidirectional(
            LSTM(
                CFG.LSTM_layer_1_units,
                return_sequences=True
                )
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(LSTM(CFG.LSTM_layer_2_units))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))

        model.add(Dense(len(label_map), activation="softmax"))

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            metrics = ["accuracy"],
            loss = "sparse_categorical_crossentropy"
            )

        # Tự động dừng khi có dấu hiệu overfit
        early_stop = EarlyStopping(
            monitor="val_loss", 
            patience=self.patience,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        # Điều chỉnh Learning Rate (Tốc độ học của model)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )

        # Train model
        train_history = model.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr]
        )

        history = {
            "train_loss": train_history.history["loss"],
            "val_loss":   train_history.history["val_loss"],
            "train_acc":  train_history.history["accuracy"],
            "val_acc":    train_history.history["val_accuracy"],
        }
        model.save(os.path.join(CFG.models_dir, self.model_name))

        # Test model để vẽ confusion matrix
        y_pred_probs = model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        self.evaluation(
            model,
            history,
            x_test,
            y_test,
        )

        self.plot_history(history)
        self.plot_confusion_matrix(y_test, y_pred, label_map)


    # PLOT
    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid()

        # Accuracy
        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid()

        t = time.localtime()
        save_time = f"{t.tm_year}{t.tm_mon}{t.tm_mday}_{t.tm_hour}_{t.tm_min}"
        save_path = f"{CFG.training_plot_dir}/train_plot_{save_time}"

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✔ Training plot saved: {save_path}")
        plt.show()
    
    # CONFUSION MATRIX
    def plot_confusion_matrix(self, y_true, y_pred, label_map):
        cm = confusion_matrix(y_true, y_pred, normalize='true')

        class_names = [
            label_map[str(i)] for i in range(len(label_map))
        ]

        # Vẽ confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        display.plot(
            cmap='Blues',
            ax=ax,
            xticks_rotation=45
        )

        plt.title('Confusion Matrix')

        t = time.localtime()
        save_time = f"{t.tm_year}{t.tm_mon}{t.tm_mday}_{t.tm_hour}_{t.tm_min}"
        save_path = f"{CFG.training_plot_dir}/confusion_matrix_{save_time}.png"

        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )
        print(f"✔ Confusion matrix saved: {save_path}")
        plt.show()
    

    # EVALUATION
    def evaluation(self, model, history, x_test, y_test):
        # Evalutate
        test_loss, test_acc = model.evaluate(
            x_test,
            y_test,
            verbose=0
        )

        # Prediction
        y_pred = np.argmax(
            model.predict(x_test, verbose=0),
            axis=1
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        # Print result
        print("\n" + "═"*50)
        print(" TRAINING RESULT")
        print("═"*50)

        print(f'{"Epoch":<25}:{len(history["train_loss"]):>4.4f}')
        print(f'{"Best Validation Accuracy":<25}:{max(history["val_acc"]):>4.4f}')
        print(f'{"Best Validation Loss":<25}:{max(history["val_loss"]):>4.4f}')

        print("-" * 55)

        print(f'{"Test Accuracy":<25}:{test_acc:>4.4f}')
        print(f'{"Test Loss":<25}:{test_loss:>4.4f}')
        print(f'{"Precision":<25}:{precision:>4.4f}')
        print(f'{"Recall":<25}:{recall:>4.4f}')
        print(f'{"F1-score":<25}:{f1:>4.4f}')

        print("═" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════
# ARGUEMENTS
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patience",
        type=int,
        default=CFG.patience
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    patience = args.patience

    os.makedirs(CFG.models_dir, exist_ok=True)
    os.makedirs(CFG.training_plot_dir, exist_ok=True)

    labels_list = os.listdir(CFG.labels_dir)

    model_name = None
    while not model_name or not model_name.strip():
        clear_terminal()
        model_name = str(input("Name your model: "))

    try:
        epochs = int(input(f"Enter epochs number (default: {CFG.epochs} epochs): "))
        if not epochs:
            print("⚠️ Invalid value!")
            print(f"Using default epochs number: {CFG.epochs}")
            time.sleep(1.0)
    except ValueError:
        print("⚠️ You must enter an interger!")
        print(f"Using default epochs number: {CFG.epochs}")

    list_labels(labels_list)
    labels_file = select_labels(labels_list)

    if labels_file:
        labels_path = os.path.join(CFG.labels_dir, labels_file)

        clear_terminal()
        print("═"*50)
        print(" MODEL TRAINING MODULE")
        print("═"*50 + "\n")

        # CHECK DEVICE
        device = tf.config.list_physical_devices('GPU')
        print(f"GPU available: {device}" if device else "GPU not found. Using CPU")

        TRAIN_MODULE = BiLSTM(
            epochs=epochs,
            patience=patience,
            model_name=model_name,
            model_type="keras",
            labels_path=labels_path
        )

        print("✔ Datas loaded")
        print("READY TO TRAIN!\n")

        print("═"*50)
        print(" TRAINING")
        print("═"*50 + "\n")

        TRAIN_MODULE.train_model()

if __name__ == "__main__":
    main()