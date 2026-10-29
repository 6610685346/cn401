"""YOLO person detection with tracking."""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from ..core.geometry import Box, Frame

PERSON_CLASS_ID = 0


# COCO keypoint indices used by the pose model
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
KEYPOINT_MIN_CONF = 0.5


@dataclass(frozen=True)
class Detection:
    """One tracked person in a single frame."""

    box: Box
    track_id: int  # -1 until the tracker assigns one
    confidence: float
    # Where the person touches the floor, in frame pixels: between the ankles
    # when the pose model sees them, else the bottom-centre of the box.
    foot: tuple[float, float] | None = None

    @property
    def foot_point(self) -> tuple[float, float]:
        if self.foot is not None:
            return self.foot
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, float(y2))

    @property
    def is_tracked(self) -> bool:
        return self.track_id >= 0


# Person models that can be chosen on the "ตั้งค่าระบบ" page (config.PERSON_MODEL).
# Ultralytics downloads the .pt file into models/ the first time it is used.
# All are pose models: the ankles give the foot point used by the door line.
PERSON_MODELS = {
    "yolo11n-pose": "YOLO11n-pose · 2.9M พารามิเตอร์ · เร็วที่สุด (ค่าเดิม)",
    "yolo11s-pose": "YOLO11s-pose · 9.9M · แม่นขึ้น ช้ากว่า ~2 เท่า",
    "yolo11m-pose": "YOLO11m-pose · 20.9M · ควรมี GPU",
    "yolo26n-pose": "YOLO26n-pose · รุ่นใหม่ (2026) เร็วบน CPU ไม่ต้องใช้ NMS",
    "yolo26s-pose": "YOLO26s-pose · รุ่นใหม่ แม่นขึ้น",
}


def person_model_path(key: str | None = None) -> str:
    """models/<key>.pt for config.PERSON_MODEL (or MODEL_PATH when it names another file)."""
    from pathlib import Path

    from .. import config

    key = key or config.PERSON_MODEL
    default = Path(config.MODEL_PATH)
    if key == default.stem:
        return str(default)
    if key not in PERSON_MODELS:
        raise ValueError(f"unknown person model {key!r} (choose {', '.join(PERSON_MODELS)})")
    return str(Path(config.MODELS_DIR) / f"{key}.pt")


class PersonDetector:
    """Runs YOLO tracking on a frame and returns the people it found.

    Uses the NVIDIA GPU whenever PyTorch can see one, in fp16 (half precision)
    there, which is what keeps several cameras smooth at once.
    """

    def __init__(self, model_path: str, conf: float, imgsz: int = 640,
                 device: str | None = None, half: bool = True, backend: str = "pytorch",
                 tracker: str | None = None):
        from ..perf.backend import resolve_model

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.model_name = str(model_path)
        # "auto": PyTorch on an NVIDIA GPU, OpenVINO on CPU when installed (perf/backend.py)
        path, self.backend = resolve_model(model_path, backend, "cpu" if self.device == "cpu" else "cuda", imgsz)
        try:
            self.model = YOLO(path, task="pose") if self.backend == "openvino" else YOLO(path)
        except Exception as exc:
            raise RuntimeError(f"Error loading YOLO model '{path}'") from exc
        self.conf = conf
        # BoT-SORT settings file (None = Ultralytics default), or a function
        # backend -> file: ReID "auto" reuses PyTorch YOLO's own features, which
        # an OpenVINO model doesn't expose, so that one gets a ReID model instead.
        if callable(tracker):
            tracker = tracker(self.backend)
        self.tracker = tracker
        self.reid = bool(tracker and "_reid-" in str(tracker))
        self.half = half and self.device.startswith("cuda") and self.backend == "pytorch"  # fp16 only on GPU
        print(f"YOLO device: {self.device} [{self.backend}]" + (" (fp16)" if self.half else "") + f", imgsz={self.imgsz}")
        self._warm_up()

    def _options(self) -> dict:
        options = {"imgsz": self.imgsz, "verbose": False}
        if self.backend == "pytorch":
            options.update(device=self.device, quantize=16 if self.half else None)
        return options

    def _warm_up(self) -> None:
        """The first call pays for CUDA start-up; do it now, not on the first camera frame."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, **self._options())

    def detect(self, frame: Frame) -> list[Detection]:
        extra = {"tracker": self.tracker} if self.tracker else {}
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            **extra,
            **self._options(),
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [-1] * len(coordinates)
        )
        feet = self._feet(results[0], len(coordinates))
        # The sole is a little below the ankle joint: nudge down, not past the box.
        feet = [None if f is None else (f[0], min(float(b[3]), f[1] + 0.03 * (b[3] - b[1])))
                for f, b in zip(feet, coordinates)]
        return [
            Detection(tuple(box), track_id, confidence, foot)
            for box, track_id, confidence, foot in zip(coordinates, track_ids, confidences, feet)
        ]

    @staticmethod
    def _feet(result, count: int) -> list:
        """Ankle midpoint per person (pose models only), else None."""
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or len(keypoints.data) != count:
            return [None] * count
        feet = []
        for person in keypoints.data.cpu().numpy():  # (17, 3): x, y, confidence
            if person.shape[0] <= RIGHT_ANKLE or person.shape[1] < 3:
                feet.append(None)
                continue
            ankles = [person[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if person[i][2] >= KEYPOINT_MIN_CONF]
            if ankles:
                feet.append((float(sum(a[0] for a in ankles) / len(ankles)),
                             float(max(a[1] for a in ankles))))
            else:
                feet.append(None)
        return feet
