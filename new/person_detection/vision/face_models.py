"""Face-recognition models the user can pick on the "ตั้งค่าระบบ" page.

Every model turns an aligned face into a vector; two faces of the same person
give vectors pointing the same way (cosine similarity close to 1).

    sface        OpenCV SFace, 128 numbers, ~5 MB. The original model.
    arcface_mbf  ArcFace MobileFaceNet (InsightFace "buffalo_s", w600k_mbf),
                 512 numbers, 13 MB. More accurate than SFace and still fast.
    arcface_r50  ArcFace ResNet-50 (InsightFace "buffalo_l", w600k_r50),
                 512 numbers, 174 MB. The most accurate, ~10x slower on CPU.

The ArcFace models are trained on WebFace600K and run with OpenCV's DNN module
(no extra Python package). They are downloaded from the InsightFace GitHub
release the first time they are chosen. InsightFace's pretrained models are for
non-commercial research use only - fine for this university project.

Vectors from different models cannot be compared, so every model has its own
face gallery (people.json for SFace, people_<model>.json for the others, or
the "model" column in the database). enrollment.GalleryBuilder fills a new
model's gallery from the enrollment photos in models/photos/<name>/.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..core.geometry import Frame
from .face import FaceRecognizer

INSIGHTFACE_RELEASE = "https://github.com/deepinsight/insightface/releases/download/v0.7/"


@dataclass(frozen=True)
class FaceModel:
    key: str
    label: str
    dim: int
    file: str  # file name in models/ (SFace) or models/insightface/
    url: str = ""  # zip that holds the file
    member: str = ""  # name inside the zip
    sha256: str = ""
    download_mb: int = 0
    threshold_setting: str = "FACE_MATCH_THRESHOLD"

    @property
    def path(self) -> Path:
        if self.key == "sface":
            return Path(config.FACE_RECOGNITION_MODEL_PATH)
        return Path(config.MODELS_DIR) / "insightface" / self.file

    @property
    def threshold(self) -> float:
        return float(getattr(config, self.threshold_setting))


FACE_MODELS: dict[str, FaceModel] = {
    "sface": FaceModel("sface", "SFace (OpenCV) · 128 มิติ · เร็ว · ค่าเดิม", 128,
                       "face_recognition_sface_2021dec.onnx"),
    "arcface_mbf": FaceModel(
        "arcface_mbf", "ArcFace MobileFaceNet (buffalo_s) · 512 มิติ · แม่นขึ้น เร็วพอ ๆ กัน", 512,
        "w600k_mbf.onnx", INSIGHTFACE_RELEASE + "buffalo_s.zip", "w600k_mbf.onnx",
        "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f", 122,
        "FACE_MATCH_THRESHOLD_ARCFACE"),
    "arcface_r50": FaceModel(
        "arcface_r50", "ArcFace ResNet-50 (buffalo_l) · 512 มิติ · แม่นที่สุด ช้า (ควรมี GPU)", 512,
        "w600k_r50.onnx", INSIGHTFACE_RELEASE + "buffalo_l.zip", "w600k_r50.onnx",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", 276,
        "FACE_MATCH_THRESHOLD_ARCFACE"),
}
CHOICES = [(m.key, m.label) for m in FACE_MODELS.values()]


def get(key: str | None = None) -> FaceModel:
    key = key or config.FACE_RECOGNIZER
    if key not in FACE_MODELS:
        raise ValueError(f"unknown face model {key!r} (choose {', '.join(FACE_MODELS)})")
    return FACE_MODELS[key]


def threshold_for(key: str | None = None) -> float:
    return get(key).threshold


# ------------------------------------------------------------------ download
_download_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(key: str | None = None, progress=None) -> Path:
    """Path of the model file, downloading it first when it is missing.

    progress(done_bytes, total_bytes) is called while downloading.
    """
    model = get(key)
    path = model.path
    if path.exists():
        return path
    if not model.url:
        raise RuntimeError(f"Face recognition model not found: {path}")
    with _download_lock:
        if path.exists():  # another thread finished it meanwhile
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model.key} face model (~{model.download_mb} MB) from {model.url}")
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            archive = Path(tmp) / "model.zip"
            with urllib.request.urlopen(model.url, timeout=60) as response, open(archive, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            with zipfile.ZipFile(archive) as zf:
                member = next(n for n in zf.namelist() if Path(n).name == model.member)
                extracted = Path(tmp) / model.member
                with zf.open(member) as src, open(extracted, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if model.sha256 and _sha256(extracted) != model.sha256:
                raise RuntimeError(f"{model.member}: checksum does not match, download again")
            extracted.replace(path)
    return path


def is_downloaded(key: str) -> bool:
    return get(key).path.exists()


# ------------------------------------------------------------------ ArcFace
# Where the 5 landmarks go in a 112x112 ArcFace input (InsightFace's template).
# YuNet gives them in the same order: right eye, left eye (as seen in the
# picture: the left one first), nose tip, right and left mouth corner.
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_arcface(frame: Frame, raw_face_row: np.ndarray, size: int = 112) -> np.ndarray:
    """112x112 face, rotated and scaled so the eyes / nose / mouth sit where ArcFace expects."""
    landmarks = np.asarray(raw_face_row[4:14], dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:  # degenerate landmarks: fall back to the box
        x, y, w, h = (float(v) for v in raw_face_row[:4])
        src = np.array([[x, y], [x + w, y], [x, y + h]], np.float32)
        matrix = cv2.getAffineTransform(src, np.array([[0, 0], [size, 0], [0, size]], np.float32))
    return cv2.warpAffine(frame, matrix, (size, size), borderValue=0.0)


class ArcFaceRecognizer(FaceRecognizer):
    """Same interface as FaceRecognizer (SFace), with an InsightFace ArcFace model."""

    def __init__(self, model_path: str | Path, match_threshold: float = 0.35, key: str = "arcface"):
        if not Path(model_path).exists():
            raise RuntimeError(f"Face recognition model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.match_threshold = match_threshold
        self.model_key = key
        self._lock = threading.Lock()  # one cv2.dnn net is not safe to share between threads

    def embed(self, frame: Frame, raw_face_row: np.ndarray) -> np.ndarray:
        face = align_arcface(frame, raw_face_row)
        # BGR -> RGB, (x - 127.5) / 127.5, NCHW
        blob = cv2.dnn.blobFromImage(face, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        with self._lock:
            self.net.setInput(blob)
            vector = self.net.forward().flatten().astype(np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        a = np.asarray(embedding_a, np.float32).ravel()
        b = np.asarray(embedding_b, np.float32).ravel()
        if a.shape != b.shape:
            return -1.0  # vector from another model: never a match
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def create_recognizer(key: str | None = None, threshold: float | None = None) -> FaceRecognizer:
    """The recognizer for `key` (default: config.FACE_RECOGNIZER), downloading it if needed."""
    model = get(key)
    path = ensure_model(model.key)
    threshold = model.threshold if threshold is None else threshold
    if model.key == "sface":
        recognizer = FaceRecognizer(str(path), threshold)
    else:
        recognizer = ArcFaceRecognizer(path, threshold, model.key)
    recognizer.model_key = model.key
    return recognizer
