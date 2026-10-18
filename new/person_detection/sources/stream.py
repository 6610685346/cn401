"""Video input: files, webcams, and network cameras (RTSP / HTTP / MJPEG)."""

import threading
import time
from collections.abc import Iterator

import cv2

from ..core.geometry import Frame


def parse_source(source):
    """Turn "0" / "1" into a webcam index; leave URLs and paths alone."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


def is_live(source) -> bool:
    """True for cameras and network streams, False for video files."""
    source = parse_source(source)
    if isinstance(source, int):
        return True
    return str(source).lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://"))


class VideoStream:
    """Yields frames from a URL, webcam index, or file, reconnecting on drops."""

    def __init__(self, source, max_failures: int = 5, reconnect_delay: float = 1.0):
        self.source = parse_source(source)
        self.max_failures = max_failures
        self.reconnect_delay = reconnect_delay
        self.cap = self._open()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        # keep network streams near real time (not every backend honours this)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source}\n"
                "  - Make sure the camera and this computer are on the same network\n"
                "  - Open the URL in VLC or a browser first to confirm it streams\n"
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


class LatestFrameStream:
    """Wraps a live VideoStream and always hands out the NEWEST frame.

    A background thread keeps reading the camera, so when processing is slower
    than the camera, old frames are dropped instead of piling up in a buffer.
    This is what stops the picture from lagging further and further behind.
    Do not use it for video files (it would skip frames).
    """

    def __init__(self, source, max_failures: int = 5, reconnect_delay: float = 1.0):
        self._stream = VideoStream(source, max_failures, reconnect_delay)
        self.source = self._stream.source
        self._frame: Frame | None = None
        self._frame_id = 0
        self._ended = False
        self._stop = threading.Event()
        self._new_frame = threading.Condition()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        try:
            for frame in self._stream.frames():
                if self._stop.is_set():
                    break
                with self._new_frame:
                    self._frame = frame
                    self._frame_id += 1
                    self._new_frame.notify_all()
        finally:
            with self._new_frame:
                self._ended = True
                self._new_frame.notify_all()

    def frames(self, timeout: float = 10.0) -> Iterator[Frame]:
        """Yield the newest frame each time one arrives; stop when the camera does."""
        last_id = 0
        while not self._stop.is_set():
            with self._new_frame:
                got = self._new_frame.wait_for(
                    lambda: self._frame_id != last_id or self._ended or self._stop.is_set(),
                    timeout=timeout,
                )
                if not got:
                    raise TimeoutError(f"No frame from {self.source} for {timeout:.0f}s")
                if self._frame_id == last_id:  # ended or stopped, nothing new
                    return
                last_id, frame = self._frame_id, self._frame
            yield frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self._stream.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.release()


def open_stream(source, max_failures: int = 5, reconnect_delay: float = 1.0):
    """LatestFrameStream for live cameras, plain VideoStream for files."""
    if is_live(source):
        return LatestFrameStream(source, max_failures, reconnect_delay)
    return VideoStream(source, max_failures, reconnect_delay)
