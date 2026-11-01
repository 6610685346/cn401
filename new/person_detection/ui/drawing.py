"""Everything painted onto the frame."""

import functools
import glob
import os

import cv2
import numpy as np

from ..vision.colors import Appearance
from ..vision.detector import Detection
from ..core.geometry import Box, Frame
from ..vision.vlm_par import PedestrianAttributes

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOR = (255, 0, 255)
LABEL_BG = (255, 0, 255)
COUNTER_BG = (0, 128, 0)
FACE_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


# OpenCV's own font has only ASCII: a Thai name would show as "?????".
# Labels with other characters are drawn with Pillow and a font that has Thai.
UNICODE_FONTS = [
    r"C:\Windows\Fonts\LeelawUI.ttf", r"C:\Windows\Fonts\leelawad.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf", "/usr/share/fonts/opentype/tlwg/Loma.otf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", "/System/Library/Fonts/Thonburi.ttc",
]


@functools.lru_cache(maxsize=8)
def _unicode_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    candidates = UNICODE_FONTS + sorted(glob.glob("/usr/share/fonts/**/*[Tt]hai*.[ot]tf", recursive=True))
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def _draw_unicode_label(frame: Frame, text: str, x: int, y: int, scale: float, bg_color) -> bool:
    font = _unicode_font(max(10, round(scale * 32)))
    if font is None:
        return False
    from PIL import Image, ImageDraw
    left, top, right, bottom = font.getbbox(text)
    text_w, text_h = right - left, bottom - top
    height, width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y - text_h - 10)
    x2, y2 = min(width, x + text_w + 8), min(height, max(y, y1 + 1))
    if x2 <= x1 or y2 <= y1:
        return True
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    roi = frame[y1:y2, x1:x2]
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((4 - left, 5 - top), text, font=font, fill=TEXT_COLOR[::-1])
    frame[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return True


def draw_label(
    frame: Frame,
    text: str,
    position: tuple[int, int],
    scale: float = 0.6,
    thickness: int = 1,
    bg_color: tuple[int, int, int] = LABEL_BG,
) -> None:
    """Draw text on a filled rectangle, anchored at its bottom-left corner."""
    x, y = position
    if not text.isascii() and _draw_unicode_label(frame, text, x, y, scale, bg_color):
        return
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    top_left = (x, y - text_h - baseline - 4)
    bottom_right = (x + text_w + 8, y)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, cv2.FILLED)
    text_origin = (x + 4, y - baseline - 2)
    cv2.putText(
        frame, text, text_origin, FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA
    )


class Overlay:
    """Draws people, faces and the head count."""

    @staticmethod
    def person_label(detection: Detection, appearance: Appearance | None, name: str | None = None,
                     global_id: int | None = None) -> str:
        label = name if name else (
            f"ID: {detection.track_id}" if detection.is_tracked else "person")
        if global_id is not None:  # same G-number on every camera
            label = f"G{global_id} {label}"
        detail = str(
            appearance) if appearance else f"{detection.confidence:.2f}"

        return f"{label} {detail}"

    def draw_person(self, frame: Frame, detection: Detection,
                    appearance: Appearance | None = None,
                    attributes: PedestrianAttributes | None = None,
                    name: str | None = None,
                    global_id: int | None = None) -> None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        draw_label(frame, self.person_label(
            detection, appearance, name, global_id), (x1, max(y1 - 4, 20)))
        if attributes:
            draw_label(frame, str(attributes), (x1, min(y2, frame.shape[0] - 4)),
                       scale=0.48, bg_color=(90, 70, 20))

    def draw_face(self, frame: Frame, box: Box) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), FACE_COLOR, 2)

    def draw_count(self, frame: Frame, count: int) -> None:
        draw_label(frame, f"Person: {count}", (10, 30),
                   scale=0.7, thickness=2, bg_color=COUNTER_BG)
