"""Settings. Edit these, then run:  python connect.py"""

from pathlib import Path


# Folders relative to the folder holding connect.py

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"


# --------------------------------------------------------------- video source
# IP Webcam: keep the /video suffix

SOURCE = "rtsp://admin:%23Oung2548@192.168.1.13:554/media/video2"


MAX_READ_FAILURES = 5
RECONNECT_DELAY_SEC = 1.0

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
# Enrollment photos, one folder per person (capture_faces.py and the web page)
PHOTOS_DIR = MODELS_DIR / "photos"
WARDROBE_LOG_PATH = OUTPUT_DIR / "wardrobe.json"


# ----------------------------------------------------------------- outputs

SAVE_CSV = True
SAVE_CROPS = False
CSV_PATH = None
CROPS_DIR = OUTPUT_DIR / "cropped_persons"
MIN_FRAMES_BEFORE_LOG = 5

# ------------------------------------------------- settings saved from the web
# The web portal's "ตั้งค่าระบบ" page saves changed values to settings.json next
# to connect.py. They override the values above for the web AND connect.py.
# Delete the file (or press "ค่าเริ่มต้น" on that page) to go back to this file.
SETTINGS_PATH = PROJECT_DIR / "settings.json"
FILE_DEFAULTS = {}  # name -> value written above, for settings that were overridden


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
        FILE_DEFAULTS.setdefault(name, current)
        module[name] = float(value) if isinstance(current, float) else value
        applied[name] = module[name]
    return applied


apply_saved_settings()
