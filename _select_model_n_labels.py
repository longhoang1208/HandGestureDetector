

"""
══════════════════════════════════════════════════════
MODEL & LABELS SELECTION MODULE
══════════════════════════════════════════════════════

Tạo bảng và cho phép người dùng nhập
chỉ mục của model và bộ nhãn muốn sử dụng.

Hiện thông báo lỗi nếu giá trị nhập vào
không phải là số nguyên sau đó thoát hàm.
"""


import time
import os


# ══════════════════════════════════════════════════════
# ANSI COLORS
# ══════════════════════════════════════════════════════

RESET  = "\033[0m"
GRAY   = "\033[90m"


def list_models(model_list):
    print(f"{GRAY}Available models:{RESET}")

    print("┌" + "─" * 30 + "┐")
    for idx, model in enumerate(model_list):
        print(f'│ [{idx:<2}] │ {os.path.splitext(model)[0]:>20}  │')
    print("└" + "─" * 30 + "┘")


def list_labels(labels_list):
    print(f"{GRAY}Available labels:{RESET}")
    
    print("┌" + "─" * 30 + "┐")
    for idx, labels in enumerate(labels_list):
        print(f'│ [{idx:<2}] │ {os.path.splitext(labels)[0]:>20}  │')
    print("└" + "─" * 30 + "┘")

def select_model(model_list):
    try:
        select_m_idx = int(input("Select model's index: "))
    except ValueError:
        print("You must enter an interger")
        time.sleep(1.0)
        return

    if select_m_idx < len(model_list) and model_list[select_m_idx]:
        model_name = model_list[select_m_idx]
        return model_name
    else:
        print("invalid index")
        time.sleep(1.0)
        return



def select_labels(labels_list):
    try:
        select_l_idx = int(input("Select labels' index: "))
    except ValueError:
        print("You must enter an interger")
        time.sleep(1.0)
        return

    if select_l_idx < len(labels_list) and labels_list[select_l_idx]:
        labels_file = labels_list[select_l_idx]
        return labels_file
    else:
        print("invalid index")
        time.sleep(1.0)
        return

