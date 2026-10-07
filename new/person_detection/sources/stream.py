"""Video input that reconnects when a network stream drops."""

import time
from collections.abc import Iterator

import cv2

from ..core.geometry import Frame


class VideoStream:
    """Yields frames from a URL, webcam index, or file."""

    def __init__(self, source, max_failures: int = 5, reconnect_delay: float = 1.0):
        self.source = source
        self.max_failures = max_failures
        self.reconnect_delay = reconnect_delay
        self.cap = self._open()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        # keep network streams near real time
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source}\n"
                "  - Make sure the phone and this computer are on the same Wi-Fi\n"
                "  - Open the URL in a browser first to confirm it streams\n"
                "  - IP Webcam URLs end with /video, DroidCam uses /mjpegfeed"
            )
        return cap

    def frames(self) -> Iterator[Frame]:
        """Yield frames, reconnecting on failure until the stream gives up."""
        failures = 0
        while True:
            ok, frame = self.cap.read()
            if ok:
                failures = 0
                yield frame
                continue

            failures += 1
            print(f"Frame read failed ({failures}/{self.max_failures})")
            if failures >= self.max_failures:
                print("Stream ended or failed.")
                return
            try:
                self._reconnect()
            except RuntimeError as exc:
                print(exc)
                return

    def _reconnect(self) -> None:
        self.cap.release()
        time.sleep(self.reconnect_delay)
        self.cap = self._open()

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.release()
