

"""
══════════════════════════════════════════════════════
MODULE CHÍNH CỦA PHẦN MỀM
══════════════════════════════════════════════════════

import các module:
    - a1_data_collect_module
    - a2_model_training_module
    - b1_module1_single_sign
    - b2_module2_multi_signs

clear terminal để hiện giao diện CLI.
Tạo ô input để user chọn module muốn sử dụng.

Cách chạy:
    python main.py
    Chọn module, nhấn 'q' để thoát chương trình
    a1 -> Thu thập dữ liệu huấn luyện
    a2 -> Huấn luyện mô hình 
    b1 -> Nhận diện ký hiệu đơn lẻ: 1 ký hiệu = 1 từ/câu
    b2 -> Chuỗi ký hiệu: nhiều từ ghép thành câu
"""


import a1_data_collect_module as collect_module
import a2_model_training_module as train_module
import b1_module1_single_sign as module_1
import b2_module2_multi_signs as module_2
import subprocess
import platform

# ══════════════════════════════════════════════════════
# ANSI COLORS
# ══════════════════════════════════════════════════════
RESET  = "\033[0m"
BOLD   = "\033[1m"

WHITE  = "\033[97m"
GRAY   = "\033[90m"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"

# ══════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════

title = rf"""{CYAN}
██╗  ██╗ █████╗ ███╗   ██╗██████╗   ███████╗██╗ ██████╗ ███╗   ██╗
██║  ██║██╔══██╗████╗  ██║██╔══██╗  ██╔════╝██║██╔════╝ ████╗  ██║
███████║███████║██╔██╗ ██║██║  ██║  ███████╗██║██║  ███╗██╔██╗ ██║
██╔══██║██╔══██║██║╚██╗██║██║  ██║  ╚════██║██║██║   ██║██║╚██╗██║
██║  ██║██║  ██║██║ ╚████║██████╔╝  ███████║██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝   ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝

██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗
██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
{RESET}
"""

def clear_terminal():
    if platform.system() == "Windows":
        subprocess.run("cls", shell=True)
    else:  # Linux và macOS
        subprocess.run("clear", shell=True)


def main():
    while True:
        clear_terminal()

        print(title)

        print(f"{GRAY}{'─'*18} Real-time Hand Sign Recognition {'─'*18}{RESET}\n")
        print(f"{GRAY}┌{'─'*67}┐{RESET}")
        print(f"{GRAY}│{RESET} {CYAN}Controls:{RESET}{' '*57}{GRAY}│{RESET}")
        
        controls = [
            ("a1", "Collect data for model training"),
            ("a2", "Train model"),
            ("b1", "Detect single signs"),
            ("b2", "Detect multi signs"),
            ("space", "Start/stop detection"),
            ("q", "Quit"),
        ]

        for keys, text in controls:
            key_text = f"[{keys}]"
            key = f"[{YELLOW}{keys}{WHITE}]"

            line = f"{GRAY}│{RESET} {key} {' '*(12 - len(key_text))} {text} {GRAY}{' '*(51-len(text)) + '│'}{RESET}"
            print(line)

        print(f"{GRAY}└{'─'*67}┘{RESET}")

        print()
        print(f"{GRAY}{'─'*90}{RESET}")
        print()

        # USER'S SELECTION
        print(f"{PURPLE}SELECT > {RESET}", end="")
        choose = str(input())
        choose = choose.lower().strip()

        if choose == "q":
            break
        elif choose == "a1":
            collect_module.main()
        elif choose == "a2":
            train_module.main()
        elif choose == "b1":
            module_1.main()
        elif choose == "b2":
            module_2.main()
        else:
            print("Invalid choice...")

if __name__=="__main__":
    main()