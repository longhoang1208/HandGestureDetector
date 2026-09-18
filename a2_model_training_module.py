# ==================================================
# MODEL TRAINING MODULE
# ==================================================
# File name   : a2_model_training_module.py
# Description : Huấn luyện mô hình BiLSTM dựa trên
#               tập dữ liệu đã được thu thập.
# --------------------------------------------------


import numpy as np

from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Bidirectional

from keras.models       import Sequential
from keras.callbacks    import EarlyStopping
from keras.callbacks    import Callback
from keras.optimizers   import Adam
from keras.callbacks    import ReduceLROnPlateau
from keras.regularizers import l2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from pathlib import Path

import os
import json
import time
from _config import config as cf

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6.QtCore import Qt

from PySide6.QtGui import QPixmap

from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QProgressBar
from PySide6.QtWidgets import QComboBox
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QTableWidget
from PySide6.QtWidgets import QTableWidgetItem
from PySide6.QtWidgets import QLineEdit


# ---------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------
# Description : Khởi tạo các đường dẫn và các tham số
#               trong kiến trúc mô hình và quá trình
#               huấn luyện.
# ---------------------------------------------------------

class config(cf):
    def __init__(self):
        super().__init__()

        self.training_plot_dir = "data/training_plot"

        self.LSTM_layer_1_units = 128
        self.LSTM_layer_2_units = 64

        self.epochs    = 150
        self.test_size = 0.25
        self.patience  = 20

CFG = config()


def load_data(samples):
    x = []
    y = []

    for path, label in samples:
        data = np.load(path)

        x.append(data)
        y.append(label)
    return np.array(x, dtype=np.float32), np.array(y)

def get_samples(labels_path):
    samples = []

    with open(labels_path, "r") as f:
        label_map = json.load(f)

    # Lặp qua tất cả file và thêm vào samples list
    for idx, label in label_map.items():
        folder = os.path.join(CFG.data_dir, label)

        for file in os.listdir(folder):
            if file.endswith(".npy"):
                samples.append((os.path.join(folder, file), int(idx)))
    return samples, label_map

def split_train_test(samples, label_map):
    train_samples, test_samples = train_test_split(
        samples,
        test_size=CFG.test_size,
        stratify=[s[1] for s in samples],
        shuffle=True,
        random_state=42
    )

    x_train, y_train = load_data(train_samples)
    x_test, y_test   = load_data(test_samples)

    return x_train, y_train, x_test, y_test, label_map


def train_model(labels_path, model_name, epochs, patience):
    x_train, y_train, x_test, y_test, label_map = get_samples(labels_path)
    (timestep, n_features) = (x_train.shape[1], x_train.shape[2])

def build_model(timestep, n_features, label_map):
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
    return model


def plot_history(history, train_plot: QLabel, plots_dir: Path):
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
    save_path = f"{plots_dir}/train_plot_{save_time}.png"

    # Tạo thư mục lưu biểu đồ
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    pixmap = QPixmap(save_path)
    train_plot.setPixmap(
        pixmap.scaled(
            train_plot.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    )

# CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, label_map, cfs_matrix: QLabel, plots_dir: Path):
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
    save_path = f"{plots_dir}/confusion_matrix_{save_time}.png"

    # Tạo thư mục lưu biểu đồ
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches='tight',
        dpi=300
    )
    plt.close()
    pixmap = QPixmap(save_path)
    cfs_matrix.setPixmap(
        pixmap.scaled(
            cfs_matrix.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    )


# EVALUATION
def evaluation(model, history, x_test, y_test):
    # Evalutate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred    = np.argmax(model.predict(x_test, verbose=0), axis=1)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return {
        "Epoch": len(history["train_loss"]),
        "Best Validation Accuracy": max(history["val_acc"]),
        "Best Validation Loss"    : min(history["val_loss"]),

        "Test Accuracy": test_acc,
        "Test Loss"    : test_loss,
        "Precision"    : precision,
        "Recall"       : recall,
        "F1-score"     : f1
    }


"""
Tạo luồn traing model trong QThread để
không đóng băng UI.
"""
class TrainingWorker(QThread):
    # current epoch, total epoch, logs
    epoch_progress = Signal(int, int, dict)

    # history, eval_metrics, y_test, y_pred, label_map
    training_done  = Signal(dict, dict, np.ndarray, np.ndarray, dict)

    # Thông báo lỗi
    training_error = Signal(str)

    def __init__(self, labels_path, model_name, epochs, patience, models_dir: Path):
        super().__init__()
        self.labels_path = labels_path
        self.model_name = model_name
        self.epochs = epochs
        self.patience = patience

        # Tạo thư mục lưu model
        self.models_dir = models_dir
        Path(self.models_dir).parent.mkdir(exist_ok=True)

    # Hàm run() mặc định được QThread gọi trong vòng lặp riêng.
    # Hàm này liên tục phát đi thông tin của quá trình huấn luyện.
    def run(self):
        try:
            history, eval_metrics, y_test, y_pred, label_map = self.train()
            self.training_done.emit(history, eval_metrics, y_test, y_pred, label_map)
        except Exception as e:
            self.training_error.emit(str(e))

    # Huấn luyện và lưu mô hình
    def train(self):
        samples, label_map = get_samples(self.labels_path)
        x_train, y_train, x_test, y_test, label_map = split_train_test(samples, label_map)

        (timestep, n_features) = (x_train.shape[1], x_train.shape[2])
        # x_train.shape = (None, timestep, n_features)

        model = build_model(timestep, n_features, label_map)

        early_stop = EarlyStopping(
            monitor="val_loss", 
            patience=self.patience,
            min_delta=0.001,
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-5,
        )

        worker = self
        class QtBridgeCallback(Callback):
            # Callback của Keras tự động gọi hàm on_epoch_end()
            # Override on_epoch_end của callback
            def on_epoch_end(self, epoch, logs=None):
                worker.epoch_progress.emit(epoch+1, worker.epochs, logs or {})

        train_history = model.fit(
            x_train, y_train,
            epochs          = self.epochs,
            batch_size      = 32,
            validation_data = (x_test, y_test),
            callbacks       = [
                early_stop,
                reduce_lr,
                QtBridgeCallback()
            ]
        )

        history = {
            "train_loss": train_history.history["loss"],
            "val_loss"  : train_history.history["val_loss"],
            "train_acc" : train_history.history["accuracy"],
            "val_acc"   : train_history.history["val_accuracy"],
        }


        # Lưu model
        model.save(self.models_dir)

        y_pred_probs = model.predict(x_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        eval_metrics = evaluation(model, history, x_test, y_test)

        return history, eval_metrics, y_test, y_pred, label_map


class TrainingModule(QWidget):
    def __init__(self, user_name):
        super().__init__()

        self.user_name = user_name
        self.labels_dir = Path("Users") / self.user_name / CFG.labels_dir
        self.models_dir = Path("Users") / self.user_name / CFG.models_dir
        self.plots_dir  = Path("Users") / self.user_name / CFG.training_plot_dir

        self.mainLayout = QHBoxLayout(self)
        self.stack = QStackedWidget()
        self.mainLayout.addWidget(self.stack)

        # ---------------------------------------
        # TRANG 1 - CHỌN BỘ NHÃN & ĐẶT TÊN MODEL
        # ---------------------------------------
        self.MnL_page = QWidget()
        self.MnL_page_layout = QVBoxLayout(self.MnL_page)

        # Droplist chọn bộ nhãn
        self.labels_drop_list = QComboBox()
        self.labels_drop_list.setFixedWidth(200)
        self.labels_drop_list.addItems(os.listdir(self.labels_dir))

        # Ô nhập liệu nhập tên model
        self.model_name_input = QLineEdit()
        self.model_name_input.setFixedWidth(200)
        self.model_name_input.setPlaceholderText("Enter your model name")

        # Nút tải lại để cập nhật các file bộ nhãn mới
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(90)
        self.refresh_btn.clicked.connect(self.refresh_label_list)

        # Nút xác nhận bộ nhãn và tên model
        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setFixedWidth(90)
        self.confirm_btn.clicked.connect(self.confirm_MnL)

        # Sắp xếp bố cục trang
        self.MnL_page_layout.addStretch()
        self.MnL_page_layout.addWidget(QLabel("Select labels"))
        self.MnL_page_layout.addWidget(self.labels_drop_list)
        self.MnL_page_layout.addWidget(QLabel("Name your model"))
        self.MnL_page_layout.addWidget(self.model_name_input)
        self.MnL_page_layout.addWidget(self.refresh_btn)
        self.MnL_page_layout.addWidget(self.confirm_btn)
        self.MnL_page_layout.addStretch()

        # ---------------------------------------
        # TRANG 2 - BẮT ĐẦU HUẤN LUYỆN
        # ---------------------------------------
        # Khởi tạo trang
        self.training_page = QWidget()
        self.training_page_layout = QVBoxLayout(self.training_page)

        self.status_label = QLabel("Ready!")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(CFG.bar_style)

        # Nút bắt đầu huấn luyện
        self.start_training_btn = QPushButton("Start training")
        self.start_training_btn.setFixedWidth(CFG.button_width)
        self.start_training_btn.clicked.connect(self.start_training)

        # Nút quay lại trang trước
        self.go_back_btn = QPushButton("Back")
        self.go_back_btn.setFixedWidth(CFG.button_width)
        self.go_back_btn.clicked.connect(
            lambda: (
                self.stack.setCurrentWidget(
                    self.MnL_page
                )
            )
        )

        # Sắp xếp bố cục trang
        self.training_page_layout.addStretch()
        self.training_page_layout.addWidget(self.status_label)
        self.training_page_layout.addWidget(self.progress_bar)
        self.training_page_layout.addWidget(self.start_training_btn)
        self.training_page_layout.addWidget(self.go_back_btn)
        self.training_page_layout.addStretch()

        # ---------------------------------------
        # TRANG 3 - KẾT QUẢ HUẤN LUYỆN
        # ---------------------------------------
        self.result_page = QWidget()
        self.result_page_layout = QVBoxLayout(self.result_page)

        # Bảng thông sô kết quả huấn luyện
        self.metrics_table = QTableWidget()
        self.metrics_table.setFixedWidth(500)  # chiều rộng bảng

        self.plot_layout = QHBoxLayout()

        # Ô hiển thị training plot
        self.train_plot = QLabel()
        self.train_plot.setFixedSize(700, 350)

        # Ô hiển thị confusion matrix
        self.cfs_matrix = QLabel()
        self.cfs_matrix.setFixedSize(400, 350)

        # Sắp xếp các biểu đò
        self.plot_layout.addWidget(self.train_plot)
        self.plot_layout.addWidget(self.cfs_matrix)

        # Nút quay về trang đầu, huấn luyện lại
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setFixedWidth(90)
        self.redo_btn.clicked.connect(self.redo)

        # Sắp xếp bố cục trang
        self.result_page_layout.addStretch()
        self.result_page_layout.addLayout(self.plot_layout)
        self.result_page_layout.addWidget(self.metrics_table)
        self.result_page_layout.addWidget(self.redo_btn)
        self.result_page_layout.addStretch()

        # ---------------------------------------
        # SẮP XẾP THỨ TỰ CÁC TRANG
        # ---------------------------------------
        self.stack.addWidget(self.MnL_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.result_page)

        self.selected_labels_file = None

    # Tải lại để cập nhật các file bộ nhãn mới
    def refresh_label_list(self):
        self.labels_drop_list.clear()
        self.labels_drop_list.addItems(os.listdir(self.labels_dir))

    # Xác nhận bộ nhãn và tên model
    def confirm_MnL(self):
        if not self.model_name_input.text():
            self.model_name_input.setText(
                "Please enter a model name"
            )
            return
        
        self.selected_labels_file = self.labels_drop_list.currentText()
        self.model_name = self.model_name_input.text()

        self.stack.setCurrentWidget(self.training_page)

    # Bắt đầu huấn luyện model
    def start_training(self):
        if not self.selected_labels_file:
            self.status_label.setText("⚠️ Chưa chọn bộ nhãn!")
            return

        labels_path = os.path.join(
            self.labels_dir,
            self.selected_labels_file
        )

        self.start_training_btn.setEnabled(False)
        self.worker = TrainingWorker(
            labels_path,
            self.model_name,
            CFG.epochs,
            CFG.patience,
            self.models_dir
        )

        self.worker.epoch_progress.connect(self.on_epoch_progress)
        self.worker.training_done.connect(self.on_training_done)
        self.worker.training_error.connect(self.on_training_error)
        self.worker.start()

    def on_epoch_progress(self, epoch, total_epochs, logs):
        self.progress_bar.setMaximum(total_epochs)
        self.progress_bar.setValue(epoch)
        self.status_label.setText(
            f"Epoch {epoch}/{total_epochs}\n"
            f"loss: {logs.get('loss'):.4f}\n"
            f"val_loss: {logs.get('val_loss'):.4f}\n"
            f"acc: {logs.get('accuracy'):.4f}\n"
            f"val_acc: {logs.get('val_accuracy'):.4f}"
        )

    def on_training_done(self, history, eval_metrics, y_test, y_pred, label_map):
        self.stack.setCurrentWidget(self.result_page)

        self.start_training_btn.setEnabled(True)

        self.metrics_table.setRowCount(len(eval_metrics))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        for row, (k, v) in enumerate(eval_metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{v:.4f}" if isinstance(v, float) else str(v)))

        plot_history(history, self.train_plot, self.plots_dir)
        plot_confusion_matrix(y_test, y_pred, label_map, self.cfs_matrix, self.plots_dir)

    def on_training_error(self, message):
        self.start_training_btn.setEnabled(True)
        self.status_label.setText(f"Lỗi: {message}")

    def redo(self):
        self.stack.setCurrentWidget(self.MnL_page)


# def main():
#     from PySide6.QtWidgets import QApplication

#     device = tf.config.list_physical_devices('GPU')
#     print(f"GPU available: {device}" if device else "GPU not found. Using CPU")

#     app = QApplication()
#     window = TrainingModule()
#     window.show()
#     app.exec()
    
# if __name__ == "__main__":
#     main()