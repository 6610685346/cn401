"""Everything painted onto the frame."""

import cv2

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
    def person_label(detection: Detection, appearance: Appearance | None, name: str | None = None) -> str:

        label = name if name else (
            f"ID: {detection.track_id}" if detection.is_tracked else "person")
        detail = str(
            appearance) if appearance else f"{detection.confidence:.2f}"

        return f"{label} {detail}"

    def draw_person(self, frame: Frame, detection: Detection,
                    appearance: Appearance | None = None,
                    attributes: PedestrianAttributes | None = None,
                    name: str | None = None) -> None:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        draw_label(frame, self.person_label(
            detection, appearance, name), (x1, max(y1 - 4, 20)))
        if attributes:
            draw_label(frame, str(attributes), (x1, min(y2, frame.shape[0] - 4)),
                       scale=0.48, bg_color=(90, 70, 20))

    def draw_face(self, frame: Frame, box: Box) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), FACE_COLOR, 2)

    def draw_count(self, frame: Frame, count: int) -> None:
        draw_label(frame, f"Person: {count}", (10, 30),
                   scale=0.7, thickness=2, bg_color=COUNTER_BG)
