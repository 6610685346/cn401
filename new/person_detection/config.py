"""Settings. Edit these, then run:  python connect.py"""

from pathlib import Path


# Folders relative to the folder holding connect.py

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"


# --------------------------------------------------------------- video source
# IP Webcam: keep the /video suffix

SOURCE = "rtsp://admin:123456@192.168.1.21:554/media/video1"


MAX_READ_FAILURES = 5
RECONNECT_DELAY_SEC = 1.0

# ------------------------------------------------------------------ display

ROTATE = 0
DISPLAY_WIDTH = 1020
DISPLAY_HEIGHT = 500
FRAME_SKIP = 1
WINDOW_NAME = "Person Detection"
PRINT_MOUSE_POSITION = True


# ----------------------------------------------------------------- models
# downloaded automatically if missing

MODEL_PATH = str(MODELS_DIR / "yolo11n-pose.pt")
CONFIDENCE = 0.4

ENABLE_FACE_DETECTION = True
FACE_MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
FACE_CONFIDENCE = 0.6


# VLM-PAR clothing recognition in models/vlm_par.pt

ENABLE_VLM_PAR = True
VLM_PAR_MODEL_PATH = str(MODELS_DIR / "vlm_par.pt")
VLM_PAR_MIN_FRAMES = 3  # wait for a stable track before running the large model


# Vlm_par color

VLM_PAR_COLOR_LABELS = (
    "black", "blue", "brown", "gray", "green", "orange", "pink",
    "purple", "red", "white", "yellow",
)


# ENABLE_DIRECTION (direction.py).

ENABLE_DIRECTION = False

# Face recognition to match the name using sface
# To load model run
# curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

ENABLE_FACE_RECOGNITION = True
FACE_RECOGNITION_MODEL_PATH = str(
    MODELS_DIR / "face_recognition_sface_2021dec.onnx")
FACE_MATCH_THRESHOLD = 0.363
PEOPLE_DB_PATH = OUTPUT_DIR / "people.json"
WARDROBE_LOG_PATH = OUTPUT_DIR / "wardrobe.json"


# ----------------------------------------------------------------- outputs

SAVE_CSV = True
SAVE_CROPS = False
CSV_PATH = None
CROPS_DIR = OUTPUT_DIR / "cropped_persons"
MIN_FRAMES_BEFORE_LOG = 5
