# ==================================================
# CONFIGURATIONS
# ==================================================
# File name   : _configurations.py
# Description : Initialize the common variables
#               among modules.
# --------------------------------------------------


class config:
    def __init__(self):
        self.users_dir  = "Users"
        self.default_user_dir = "default_user"
        
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

