"""Settings. Edit these, then run:  python connect.py"""
import os

from urllib.parse import quote
from pathlib import Path


# Folders relative to the folder holding connect.py

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"


# --------------------------------------------------------------- video source

# Credentials come from environment variables, so no password is in git:
#   Windows cmd:  set CAM_PASSWORD=yourpassword      bash:  export CAM_PASSWORD='yourpassword'
# connect.py runs every camera in CAMERAS (one window each). The desktop app
# does not use this list: cameras are added on its "กล้อง IP" page.

CAMERA_USER = os.environ.get("CAM_USER", "admin")
CAMERA_PASSWORD = quote(os.environ.get("CAM_PASSWORD", ""), safe="")  # '#' -> %23


def _rtsp(ip: str, path: str = "stream1") -> str:
    return f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{ip}:554/{path}"


CAMERAS = {
    "cam1": _rtsp("192.168.137.5X"),
    "cam2": _rtsp("192.168.137.5X"),   
}

SOURCE = CAMERAS["cam1"]


MAX_READ_FAILURES = 5
RECONNECT_DELAY_SEC = 1.0

# ------------------------------------------------------------- performance
# Work-savers around the models (PERFORMANCE.md). All can be changed on the
# "ตั้งค่าระบบ" page except YOLO_BACKEND.
# "auto" = PyTorch on an NVIDIA GPU, OpenVINO on CPU when installed (pip install openvino);
# "pytorch" / "openvino" to force one.
YOLO_BACKEND = "auto"
TRACKER_STATIC = True  # cameras are fixed: no camera-motion compensation in BoT-SORT
TRACK_BUFFER = 60  # frames a hidden person keeps their ID (default 30)
MOTION_GATE = True  # skip YOLO while the picture is empty and nothing moves
IDLE_CHECK_SEC = 2.0  # ...but still look every this many seconds
SMART_FACES = True  # find faces only around people still without a name
FACE_EVERY = 2  # ...on every N-th processed frame
FACE_VOTES = 3  # faces averaged before deciding a name (1 = first face decides)
FACE_MIN_SIZE = 24  # px; smaller faces are not used for recognition
LOG_COOLDOWN_SEC = 30  # same named person, same camera: one sighting row per this many seconds
FACE_FULL_RES = True  # find/embed faces in the camera's full-size picture (far faces get more pixels)
TRACKER_REID = False  # BoT-SORT also compares appearance (ReID) to keep IDs through occlusions
REID_MODEL = "auto"  # "auto" = YOLO's own features (cheap) | "yolo26n-reid.onnx" / "yolo26s-reid.onnx"

# ------------------------------------------------------------------ display

ROTATE = 0

# Every camera frame is scaled so its LONG side is this many pixels before
# detection, whatever the camera's own resolution (480p ... 4K, portrait too).
# Raise it (e.g. 1280) to catch smaller/farther faces; lower it for speed.
PROCESS_LONG_SIDE = 960

# Size of the OpenCV preview window only (the frame is shrunk to fit).
DISPLAY_WIDTH = 1020
DISPLAY_HEIGHT = 500
FRAME_SKIP = 1
WINDOW_NAME = "Person Detection"
PRINT_MOUSE_POSITION = False


# ----------------------------------------------------------------- models
# downloaded automatically if missing

MODEL_PATH = str(MODELS_DIR / "yolo11n-pose.pt")
# Which person model runs (choices in vision/detector.py PERSON_MODELS,
# changeable on the "ตั้งค่าระบบ" page); downloaded to models/<name>.pt.
PERSON_MODEL = "yolo11n-pose"
CONFIDENCE = 0.4
# YOLO input size: smaller = faster, bigger = finds smaller / farther people.
# 640 is YOLO's default; 480 was used for two cameras on one laptop GPU.
YOLO_IMGSZ = 640
YOLO_HALF = True  # fp16 on an NVIDIA GPU (ignored on CPU)

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


# ENABLE_DIRECTION: count people walking in / out (counting/line.py).
# connect.py uses DOOR_LINES[camera name] if given (copy it from the app's
# "เส้นประตู" page), otherwise the old fixed zones of direction.py.
#   DOOR_LINES = {"cam1": {"a": [0.30, 0.70], "b": [0.75, 0.62], "inside": "left"}}

ENABLE_DIRECTION = False
DOOR_LINES = {}

# Face recognition to match the name using sface
# To load model run
# curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

ENABLE_FACE_RECOGNITION = True
FACE_RECOGNITION_MODEL_PATH = str(
    MODELS_DIR / "face_recognition_sface_2021dec.onnx")
FACE_MATCH_THRESHOLD = 0.363  # for SFace
# Face recognition model (vision/face_models.py): "sface" | "arcface_mbf" | "arcface_r50".
# ArcFace models are downloaded to models/insightface/ the first time.
FACE_RECOGNIZER = "sface"
# ArcFace vectors are spread differently from SFace's: matches score lower.
# 0.35 is a starting point; tune it on the "ทดสอบด้วยวิดีโอ" page.
FACE_MATCH_THRESHOLD_ARCFACE = 0.35
PEOPLE_DB_PATH = OUTPUT_DIR / "people.json"
# Enrollment photos, one folder per person (capture_faces.py and the web page)
PHOTOS_DIR = MODELS_DIR / "photos"
WARDROBE_LOG_PATH = OUTPUT_DIR / "wardrobe.json"


# ----------------------------------------------------------------- outputs

SAVE_CSV = True
SAVE_CROPS = True  # one small photo per history row (shown on the history page)
CSV_PATH = None
CROPS_DIR = OUTPUT_DIR / "cropped_persons"
MIN_FRAMES_BEFORE_LOG = 5

# ---------------------------------------------------------------- database
# PostgreSQL (see DATABASE.md). Empty = no database: people.json + CSV only.
# Set it before starting the program, e.g. (Windows cmd):
#   set DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/person_detection
# then create the tables once:  python register.py init
# With a database, faces live in persons/face_embeddings and every logged
# person is a row in detection_logs. SAVE_CSV still writes the CSV as a backup.
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_TIMEZONE = "Asia/Bangkok"
# direction.py reports "Up" / "Down"; which one means walking IN depends on
# where the camera hangs. Swap these if the counts come out reversed.
DIRECTION_NAMES = {"Up": "in", "Down": "out"}

# ------------------------------------------------- settings saved from the web
# The web portal's "ตั้งค่าระบบ" page saves changed values to settings.json next
# to connect.py. They override the values above for the web AND connect.py.
# Delete the file (or press "ค่าเริ่มต้น" on that page) to go back to this file.
SETTINGS_PATH = PROJECT_DIR / "settings.json"
FILE_DEFAULTS = {}  # name -> value written above, for settings that were overridden
# Allowed values of the text settings
CHOICES = {
    "FACE_RECOGNIZER": ("sface", "arcface_mbf", "arcface_r50"),
    "PERSON_MODEL": ("yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo26n-pose", "yolo26s-pose"),
    "REID_MODEL": ("auto", "yolo26n-reid.onnx", "yolo26s-reid.onnx"),
}


def apply_saved_settings(path=None) -> dict:
    """Apply settings.json on top of this module; returns what was applied."""
    import json

    path = SETTINGS_PATH if path is None else path
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    module = globals()
    applied = {}
    for name, value in saved.items() if isinstance(saved, dict) else ():
        if not name.isupper() or name not in module:
            continue  # other parts' keys (e.g. "hub.MATCH_RADIUS_M") or unknown
        current = module[name]
        same_type = type(value) is type(current) or (isinstance(current, float) and type(value) is int)
        if not same_type:
            continue  # never let a bad file turn a number into text
        if name in CHOICES and value not in CHOICES[name]:
            continue  # e.g. a model name that does not exist
        FILE_DEFAULTS.setdefault(name, current)
        module[name] = float(value) if isinstance(current, float) else value
        applied[name] = module[name]
    return applied


apply_saved_settings()
