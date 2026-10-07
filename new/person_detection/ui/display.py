"""The OpenCV preview window."""

import cv2

from ..core.geometry import Frame


class DisplayWindow:
    """Shows frames and reports when the user presses the quit key."""

    def __init__(self, name: str, print_mouse: bool = False, quit_key: str = "q"):
        self.name = name
        self.quit_key = ord(quit_key)
        cv2.namedWindow(name)
        if print_mouse:
            cv2.setMouseCallback(name, self._on_mouse)

    @staticmethod
    def _on_mouse(event, x, y, flags, param):
        """Print cursor coordinates, used to lay out the ROI polygons."""
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def show(self, frame: Frame) -> bool:
        """Show a frame; return False once the quit key is pressed."""
        cv2.imshow(self.name, frame)
        return cv2.waitKey(1) & 0xFF != self.quit_key

    def close(self) -> None:
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
