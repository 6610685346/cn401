"""The OpenCV preview window."""

import cv2

from ..core.geometry import Frame, fit_frame


class DisplayWindow:
    """Shows frames and reports when the user presses the quit key.

    Frames are shrunk to fit `max_size` for display only; detection always runs
    on the full processing frame.
    """

    def __init__(self, name: str, print_mouse: bool = False, quit_key: str = "q",
                 max_size: tuple[int, int] | None = None):
        self.name = name
        self.quit_key = ord(quit_key)
        self.max_size = max_size
        self._shown_size = (1, 1)
        cv2.namedWindow(name)
        if print_mouse:
            cv2.setMouseCallback(name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        """Print cursor position as 0..1 fractions, used to lay out zones."""
        if event == cv2.EVENT_MOUSEMOVE:
            width, height = self._shown_size
            print(f"Mouse Position: ({x}, {y}) px  ->  ({x / width:.4f}, {y / height:.4f})")

    def show(self, frame: Frame) -> bool:
        """Show a frame; return False once the quit key is pressed."""
        if self.max_size:
            height, width = frame.shape[:2]
            if width > self.max_size[0] or height > self.max_size[1]:
                frame = fit_frame(frame, *self.max_size)
        self._shown_size = (frame.shape[1], frame.shape[0])
        cv2.imshow(self.name, frame)
        return cv2.waitKey(1) & 0xFF != self.quit_key

    def close(self) -> None:
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
