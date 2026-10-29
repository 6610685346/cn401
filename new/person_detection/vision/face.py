"""Face detection and recognition with OpenCV's YuNet + SFace models."""

from pathlib import Path

import cv2
import numpy as np

from ..core.geometry import Box, Frame


class FaceHit(tuple):
    """(box, raw_yunet_row) like detect_raw gives, plus the same face in the
    camera's full-resolution picture when that was available (FACE_FULL_RES).

    Box and row are in the processed (scaled-down) frame, so drawing and
    matching faces to people work as before; the recognizer embeds `source` /
    `source_row`, which has many more pixels on a small, far-away face.
    """

    def __new__(cls, box: Box, row: np.ndarray, source: Frame | None = None,
                source_row: np.ndarray | None = None):
        hit = super().__new__(cls, (box, row))
        hit.source = source
        hit.source_row = source_row
        return hit

    @property
    def source_width(self) -> float:
        """Face width in pixels of the picture the recognizer will see."""
        row = self.source_row if self.source_row is not None else self[1]
        return float(row[2])


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


    def detect_in_regions(self, frame: Frame, regions: list[Box],
                          full_frame_above: float = 0.5, source: Frame | None = None,
                          scale: float = 1.0, max_side: int = 480) -> list[FaceHit]:
        """Like detect_raw, but only looks inside `regions` (heads of people
        whose name is still unknown). Much cheaper than the whole frame when
        one or two people are in view; falls back to the whole frame when the
        regions cover most of it anyway.

        source / scale: the camera's full-resolution picture (already rotated)
        and how many times bigger it is than `frame`. The regions are then cut
        from `source` (shrunk to at most `max_side` px for YuNet), so a face
        that is 30 px wide in the processed frame is found and embedded from
        the 60-120 px it really has on a 1080p-4K camera.
        """
        hires = source is not None and scale > 1.05
        height, width = frame.shape[:2]
        clipped = []
        for x1, y1, x2, y2 in regions:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 - x1 >= 24 and y2 - y1 >= 24:
                clipped.append((x1, y1, x2, y2))
        if not clipped:
            return []
        if sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in clipped) > full_frame_above * width * height:
            return self.detect_full(frame, source, scale)
        found: list[FaceHit] = []
        for x1, y1, x2, y2 in clipped:
            if hires:
                hits = self._detect_hires(source, scale, (x1, y1, x2, y2), max_side)
            else:
                hits = []
                for _, row in self.detect_raw(frame[y1:y2, x1:x2]):
                    row = row.copy()
                    row[_X] += x1
                    row[_Y] += y1
                    hits.append(FaceHit(_box(row), row))
            for hit in hits:
                box = hit[0]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if any(abs(cx - (b[0] + b[2]) / 2) < (b[2] - b[0]) / 2 and abs(cy - (b[1] + b[3]) / 2) < (b[3] - b[1]) / 2
                       for b, _ in found):
                    continue  # same face seen through two overlapping regions
                found.append(hit)
        return found

    def detect_full(self, frame: Frame, source: Frame | None = None, scale: float = 1.0) -> list[FaceHit]:
        """Whole processed frame; with `source`, each face also points at the
        full-resolution picture (landmarks scaled up) for the recognizer."""
        hits = []
        for box, row in self.detect_raw(frame):
            if source is not None and scale > 1.05:
                hits.append(FaceHit(box, row, source, _scaled(row, scale)))
            else:
                hits.append(FaceHit(box, row))
        return hits

    def _detect_hires(self, source: Frame, scale: float, region: Box, max_side: int) -> list[FaceHit]:
        sh, sw = source.shape[:2]
        x1, y1, x2, y2 = region
        sx1, sy1 = max(0, int(x1 * scale)), max(0, int(y1 * scale))
        sx2, sy2 = min(sw, int(x2 * scale)), min(sh, int(y2 * scale))
        crop = source[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return []
        shrink = min(1.0, max_side / max(crop.shape[:2]))
        if shrink < 1.0:
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * shrink)), max(1, round(crop.shape[0] * shrink))),
                              interpolation=cv2.INTER_AREA)
        hits = []
        for _, row in self.detect_raw(crop):
            source_row = _scaled(row, 1.0 / shrink)
            source_row[_X] += sx1
            source_row[_Y] += sy1
            small = _scaled(source_row, 1.0 / scale)
            hits.append(FaceHit(_box(small), small, source, source_row))
        return hits


# YuNet row: x, y, w, h, then 5 landmarks (x, y), then score
_X = [0, 4, 6, 8, 10, 12]
_Y = [1, 5, 7, 9, 11, 13]
_XY = [0, 1, 2, 3, *range(4, 14)]


def _scaled(row: np.ndarray, factor: float) -> np.ndarray:
    out = np.array(row, dtype=np.float32, copy=True)
    out[_XY] *= factor
    return out


def _box(row: np.ndarray) -> Box:
    return (int(row[0]), int(row[1]), int(row[0] + row[2]), int(row[1] + row[3]))


def head_region(box: Box, pad: float = 0.12) -> Box:
    """Upper half of a person box (where the face is), a little wider."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - w * pad, y1 - h * 0.05, x2 + w * pad, y1 + h * 0.5)


class FaceRecognizer:
    """Turns an aligned face into an embedding and compares it to references."""

    def __init__(self, model_path: str, match_threshold: float = 0.363):
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Face recognition model not found: {model_path}")
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        self.match_threshold = match_threshold
        self.model_key = "sface"  # vision/face_models.py

    def embed(self, frame: Frame, raw_face_row: np.ndarray) -> np.ndarray:
        """Align and embed a face found by FaceDetector.detect_raw."""
        aligned = self.recognizer.alignCrop(frame, raw_face_row)
        return self.recognizer.feature(aligned).flatten()

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        if np.size(embedding_a) != np.size(embedding_b):
            return -1.0  # vector from another model (e.g. ArcFace): never a match
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
