"""YOLO person detection with tracking."""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from ..core.geometry import Box, Frame

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """One tracked person in a single frame."""

    box: Box
    track_id: int  # -1 until the tracker assigns one
    confidence: float

    @property
    def is_tracked(self) -> bool:
        return self.track_id >= 0


class PersonDetector:
    """Runs YOLO tracking on a frame and returns the people it found."""

    def __init__(self, model_path: str, conf: float,
                 imgsz: int = 480, device: str | None = None,
                 half: bool = True):
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Error loading YOLO model '{model_path}'") from exc
        self.conf = conf
        self.imgsz = imgsz
        # Use the GPU whenever PyTorch can see one.
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.startswith("cuda")  # fp16 only on GPU
        print(f"YOLO device: {self.device}"
              + (" (fp16)" if self.half else "")
              + f", imgsz={self.imgsz}")
        self._warm_up()

    def _warm_up(self) -> None:
        """First call pays CUDA init; do it now, not on the first camera frame."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=self.device,
                           quantize=16 if self.half else None, verbose=False)

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            imgsz=self.imgsz,
            device=self.device,
            quantize=16 if self.half else None,
            verbose=False,
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
        return [
            Detection(tuple(box), track_id, confidence)
            for box, track_id, confidence in zip(coordinates, track_ids, confidences)
        ]