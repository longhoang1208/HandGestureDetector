

"""
Dưới đây là các tham số chung cơ bản của các module,
trong đó quy định model và bộ nhãn mặc định, đường
dẫn lưu các model và các bộ nhãn, bước (stride) của
sliding window, bảng màu BGR và bảng màu ANSI.
"""


class config:
    def __init__(self):
        self.labels_dir = 'labels'
        self.labels     = 'asl_labels.json'

        self.models_dir = 'models'
        self.model_name = 'asl_model.keras'

        self.stride = 5


class ui_cfg:
    def __init__(self):
        self.default_frame_size = (1280, 720)

        self.Text_pos      = ( 50, 200)
        self.Text_bg_pos_1 = ( 30, 140)
        self.Text_bg_pos_2 = (400, 480)


class color:
    def __init__(self):
        self.BLACK = (  0,   0,   0)
        self.GRAY  = ( 30,  30,  30)
        self.GREEN = ( 76, 153,   0)
        self.CYAN  = (255, 200,   0)
        self.WHITE = (255, 255, 255)
        self.RED   = ( 50,  50, 220)
        self.AMBER = (  0, 180, 255)
        self.NAVY  = ( 60,  30,  10)


class ANSI_code:
    def __init__(self):
        self.RESET  = "\033[0m"
        self.BOLD   = "\033[1m"

        self.WHITE  = "\033[97m"
        self.GRAY   = "\033[90m"

        self.CYAN   = "\033[96m"
        self.GREEN  = "\033[92m"
        self.YELLOW = "\033[93m"
        self.PURPLE = "\033[95m"
