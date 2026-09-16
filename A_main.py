# =========================================================
# HAND GESTURE DETECTOR
# =========================================================
# File name   : A_main.py
# Description : Module điều khiển chính
#               của phần mềm.
# 
# ---------------------------------------------------------
# CẤU TRÚC DỰ ÁN
# ---------------------------------------------------------
# Project/
# │
# ├── data/
# │   │
# │   ├── processed/        <- dữ liệu để huấn luyện
# │   │   ├── Xin chào/     <- tên nhãn
# │   │   │   ├── 0.npy     <- video mẫu
# │   │   │   ├── 1.npy
# │   │   │   └── ...
# │   │   └── ...
# │   │
# │   └── training_plot/    <- biểu đồ
# │       ├── training_plot.png     <- accuracy, loss
# │       └── confusion_matrix.png  <- ma trận nhầm lẫn
# │
# ├── labels/
# │   │
# │   ├── asl_labels.json
# │   └── ...
# │
# ├── models/
# │   │
# │   ├── asl_model.keras
# │   └── ...
# │
# ├── voices/
# │   │
# │   ├── en_US-lessac-medium.onnx
# │   └── en_US-lessac-medium.onnx.json
# │
# ├── _configurations.py
# ├── _detector.py
# ├── _landmarks_module.py
# ├── _tts_module.py
# │
# ├── A_main.py
# │
# ├── a1_data_collect_module.py
# ├── a2_model_training_module.py
# ├── b1_module1_single_sign.py
# └── b2_module2_multi_signs.py
# ---------------------------------------------------------


from a1_data_collect_module   import CollectModule
from a2_model_training_module import TrainingModule
from b1_module1_single_sign   import Module1
from b2_module2_multi_signs   import Module2

import resources_rc
import sys
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QStackedWidget
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QComboBox

from PySide6.QtGui  import QIcon
from PySide6.QtGui  import QPixmap
from PySide6.QtGui  import QFont

from PySide6.QtCore import Qt


# -------------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------------
USERS_DIR = "Users"
button_width = 150


# -------------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------------
# 
# Giao diện trang giới thiệu, hướng dẫn người dùng sử dụng
# các tính năng của phần mềm.
# -------------------------------------------------------------
class HomaPage(QWidget):
    def __init__(self):
        super().__init__()
        home_page_layout = QVBoxLayout(self)

        # WIDGET STACK
        stack = QStackedWidget()

        # CREATE PAGES
        num_page = 13
        pages = {}
        for i in range(num_page):
            label = QLabel()
            label.setPixmap(
                QPixmap(f":/resources/Slide{i+1}.PNG")
            )
            label.setScaledContents(True)
            pages[i] = QWidget()
            layout = QVBoxLayout(pages[i])
            layout.addWidget(label)

            stack.addWidget(pages[i])

        # NEXT PAGE BUTTON
        next_btn = QPushButton("Next page")
        next_btn.setFixedWidth(150)
        next_btn.clicked.connect(
            lambda: (
                stack.setCurrentIndex(
                    stack.currentIndex() + 1
                    if stack.currentIndex() < num_page - 1
                    else 0
                )
            )
        )

        # PREVIOUS PAGE BUTTON
        prev_btn = QPushButton("Previous page")
        prev_btn.setFixedWidth(150)
        prev_btn.clicked.connect(
            lambda: (
                stack.setCurrentIndex(
                    stack.currentIndex() - 1
                    if stack.currentIndex() > 0
                    else num_page - 1
                )
            )
        )
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(prev_btn)
        button_layout.addWidget(next_btn)
        button_layout.addStretch()

        home_page_layout.addWidget(stack)
        home_page_layout.addLayout(button_layout)


# -------------------------------------------------------------
# SELECT USER PAGE
# -------------------------------------------------------------
class SelectUserPage(QWidget):
    def __init__(self):
        super().__init__()

        self.main_layout = QVBoxLayout(self)

        self.user_list = QComboBox()
        self.user_list.setFont(QFont('Arial', 18))
        self.user_list.setFixedWidth(300)
        self.user_list.setEditable(True)

        # Hiển thị danh sách 
        os.makedirs(USERS_DIR, exist_ok=True)
        self.user_list.addItems(os.listdir(USERS_DIR))

        # Select user button
        self.select_btn = QPushButton("Select user name")
        self.select_btn.setFixedWidth(button_width)

        # Arrange layout
        self.main_layout.addStretch()
        self.main_layout.addWidget(self.user_list)
        self.main_layout.addWidget(self.select_btn)
        self.main_layout.addStretch()

        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


# -------------------------------------------------------------
# MAIN WINDOW
# -------------------------------------------------------------
# 
# Giao diện chính điều khiển phần mềm, bao gồm tất cả các trang
# -------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Gesture Detector") # Title
        self.app_layout = QVBoxLayout(self)          # Layout
        self.stack = QStackedWidget()                # Widgets Stack

        self.user_name = ""

        self.select_user_page = SelectUserPage()
        self.select_user_page.select_btn.clicked.connect(self.select_user)

        self.stack.addWidget(self.select_user_page)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit_user)

        self.app_layout.addWidget(self.stack)
        self.app_layout.addWidget(self.quit_button)

    # Select user
    def select_user(self):
        if (
            not self.select_user_page.user_list.currentText() or
            self.select_user_page.user_list.currentText() not in os.listdir(USERS_DIR)
        ):
            self.select_user_page.user_list.setCurrentText("Invalid user name")
        else:
            self.user_name = self.select_user_page.user_list.currentText()

        # Debug
        print("user name: ", self.user_name if self.user_name else "Invalid")

    # Quit user
    def quit_user(self):
        self.user_name = ""
        self.stack.setCurrentWidget(self.select_user_page)

        # Debug
        print("user name: ", self.user_name if self.user_name else "Invalid")


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.setWindowIcon(QIcon(":/resources/icon.ico"))
    window.resize(1200, 700)
    window.show()

    app.exec()

if __name__=="__main__":
    main()