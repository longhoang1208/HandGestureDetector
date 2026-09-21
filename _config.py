# ==================================================
# CONFIGURATIONS
# ==================================================
# File name   : _configurations.py
# Description : khởi tạo các biến chung giữa các
#               module, bao gồm các đường dẫn và
#               các cấu hình giao diện.
# --------------------------------------------------


class config:
    def __init__(self):
        self.users_dir  = "Users"
        
        self.labels_dir = "labels"
        self.labels     = "asl_labels.json"

        self.models_dir = "models"
        self.model_name = "asl_model.keras"

        self.data_dir   = "data/processed"

        self.stride = 5

        self.cameraFrameSize    = (900, 506)
        self.default_frame_size = (1280, 720)

        self.inputSize    = (400, 30)
        self.button_width = 150

        self.barMinWidth = self.default_frame_size[0]//5
        self.bar_style = """
            QProgressBar {
                border: 1px solid #555;
                border-radius: 6px;
                text-align: center;
                background-color: #222;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #30C77C;
                border-radius: 6px;
            }
        """

        self.minNumSample = 10
        self.maxNumSample = 50


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
