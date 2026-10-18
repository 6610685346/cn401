"""Shared types and helpers for boxes and frames.

Resolution independence
-----------------------
Cameras come in every size (640x480 webcams, 1080p IP cameras, 4K, portrait
phones). Every frame is first scaled so its LONG side equals PROCESS_LONG_SIDE
(config.py), keeping the aspect ratio. After that step all pixel thresholds
mean the same thing for every camera, and anything that has to survive a
change of camera (zones, results sent to the web page) is stored as fractions
of the frame (0..1) instead of pixels.
"""

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame pixels
NormBox = tuple[float, float, float, float]  # (x1, y1, x2, y2) as 0..1 fractions
Frame = np.ndarray  # a BGR image

# Clockwise rotations accepted by the ROTATE setting.
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def fit_frame(frame: Frame, max_width: int, max_height: int) -> Frame:
    """Scale a frame to fit the given box, keeping its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(frame, (round(width * scale), round(height * scale)))


def scale_to_long_side(frame: Frame, long_side: int) -> Frame:
    """Scale a frame so its longer side is `long_side`, keeping aspect ratio."""
    height, width = frame.shape[:2]
    scale = long_side / max(height, width)
    if abs(scale - 1.0) < 1e-6:
        return frame
    # INTER_AREA is the right filter when shrinking, LINEAR when enlarging.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=interpolation)


def clip_box(frame: Frame, box: Box) -> Box:
    """Clamp a box to the frame bounds."""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def crop_box(frame: Frame, box: Box) -> Frame:
    """Cut a box out of a frame, clamped to the frame bounds."""
    x1, y1, x2, y2 = clip_box(frame, box)
    return frame[y1:y2, x1:x2]


def crop_band(frame: Frame, box: Box, band: tuple[float, float]) -> Frame:
    """Cut a horizontal slice out of a box, given as fractions of its height."""
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    top = y1 + int(box_height * band[0])
    bottom = y1 + int(box_height * band[1])
    return crop_box(frame, (x1, top, x2, bottom))


def short_side(frame: Frame) -> int:
    return min(frame.shape[:2])


def normalize_box(box: Box, width: int, height: int) -> NormBox:
    """Pixel box -> 0..1 fractions of the frame, rounded for compact JSON."""
    x1, y1, x2, y2 = box
    return (
        round(x1 / width, 4), round(y1 / height, 4),
        round(x2 / width, 4), round(y2 / height, 4),
    )


def to_pixels(points, width: int, height: int) -> np.ndarray:
    """0..1 (x, y) points -> int32 pixel points for this frame size."""
    pts = np.asarray(points, dtype=np.float32) * np.array([width, height], np.float32)
    return np.round(pts).astype(np.int32)


class FramePreprocessor:
    """Straightens a raw frame and scales it to the standard processing size.

    Any input resolution or aspect ratio comes out with its long side equal to
    `long_side`, so detection thresholds behave the same for every camera.
    """

    def __init__(self, rotate: int = 0, long_side: int = 960):
        if rotate not in (0, *ROTATIONS):
            raise ValueError(f"rotate must be 0, 90, 180 or 270, got {rotate}")
        if long_side < 160:
            raise ValueError(f"long_side too small: {long_side}")
        self.rotation = ROTATIONS.get(rotate)  # None when rotate == 0
        self.long_side = long_side

    def __call__(self, frame: Frame) -> Frame:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected a BGR colour frame")
        if self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)
        return scale_to_long_side(frame, self.long_side)
