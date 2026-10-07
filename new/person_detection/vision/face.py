"""Face detection and recognition with OpenCV's YuNet + SFace models."""

from pathlib import Path

import cv2
import numpy as np

from ..core.geometry import Box, Frame


class FaceDetector:
    """Finds faces in a frame."""

    def __init__(self, model_path: str, score: float):
        if not Path(model_path).exists():
            raise RuntimeError(f"Face model not found: {model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=score
        )

    def detect(self, frame: Frame) -> list[Box]:
        """Face boxes only (kept for anything that just wants boxes)."""
        return [box for box, _ in self.detect_raw(frame)]

    def detect_raw(self, frame: Frame) -> list[tuple[Box, np.ndarray]]:
        """(box, raw_yunet_row) pairs. The raw row is what FaceRecognizer
        needs to align the face before embedding it."""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        out = []
        for f in faces:
            box = (int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3]))
            out.append((box, f))
        return out


class FaceRecognizer:
    """Turns an aligned face into an embedding and compares it to references."""

    def __init__(self, model_path: str, match_threshold: float = 0.363):
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Face recognition model not found: {model_path}")
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        self.match_threshold = match_threshold

    def embed(self, frame: Frame, raw_face_row: np.ndarray) -> np.ndarray:
        """Align and embed a face found by FaceDetector.detect_raw."""
        aligned = self.recognizer.alignCrop(frame, raw_face_row)
        return self.recognizer.feature(aligned).flatten()

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return float(
            self.recognizer.match(
                embedding_a.reshape(1, -1),
                embedding_b.reshape(1, -1),
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

    def best_match(
        self, embedding: np.ndarray, gallery: dict[str, np.ndarray]
    ) -> tuple[str | None, float]:
        """Closest gallery entry above the threshold, or (None, best_score)."""
        best_name, best_score = None, -1.0
        for name, ref in gallery.items():
            score = self.similarity(embedding, ref)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.match_threshold:
            return None, best_score
        return best_name, best_score
