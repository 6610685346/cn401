"""Video input that reconnects when a network stream drops."""

import os
import threading
import time
from collections.abc import Iterator

import cv2

from ..core.geometry import Frame


DEFAULT_RTSP_TRANSPORT = "tcp"


def configure_ffmpeg(rtsp_transport: str = DEFAULT_RTSP_TRANSPORT) -> None:
    # Must be set before the first VideoCapture opens.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        f"rtsp_transport;{rtsp_transport}|fflags;nobuffer|flags;low_delay",
    )


def _is_network_source(source) -> bool:
    return isinstance(source, str) and "://" in source


class VideoStream:
    """Yields frames from a URL, webcam index, or file."""

    def __init__(self, source, max_failures: int = 5, reconnect_delay: float = 1.0,
                 low_latency: bool | None = None,
                 rtsp_transport: str = DEFAULT_RTSP_TRANSPORT):
        self.source = source
        self.max_failures = max_failures
        self.reconnect_delay = reconnect_delay
        # None -> automatic: live network streams drop stale frames,
        # files and webcams are read normally (a file must not skip frames).
        self.low_latency = (
            _is_network_source(source) if low_latency is None else low_latency
        )
        if _is_network_source(source):
            configure_ffmpeg(rtsp_transport)

        self.cap = self._open()

        self._lock = threading.Condition()
        self._latest: Frame | None = None
        self._latest_id = 0
        self._finished = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.low_latency:
            self._thread = threading.Thread(
                target=self._reader_loop, name="video-reader", daemon=True
            )
            self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source}\n"
                "  - Make sure the camera and this computer are on the same network\n"
                "  - Open the URL in VLC first to confirm it streams\n"
                "  - IP Webcam URLs end with /video, DroidCam uses /mjpegfeed"
            )
        return cap

    # ------------------------------------------------------- threaded reader

    def _reader_loop(self) -> None:
        """Grab frames as fast as the camera sends them; keep only the newest."""
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                failures = 0
                with self._lock:
                    self._latest = frame
                    self._latest_id += 1
                    self._lock.notify_all()
                continue

            failures += 1
            print(f"Frame read failed ({failures}/{self.max_failures})")
            if failures >= self.max_failures:
                print("Stream ended or failed.")
                break
            try:
                self._reconnect()
            except RuntimeError as exc:
                print(exc)
                break

        with self._lock:
            self._finished = True
            self._lock.notify_all()

    def _frames_latest(self) -> Iterator[Frame]:
        last_id = 0
        while True:
            with self._lock:
                while self._latest_id == last_id and not self._finished:
                    self._lock.wait(timeout=0.5)
                if self._latest_id == last_id:  # finished, nothing new
                    return
                frame, last_id = self._latest, self._latest_id
            yield frame

    # --------------------------------------------------------- plain reader

    def _frames_plain(self) -> Iterator[Frame]:
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

    def frames(self) -> Iterator[Frame]:
        """Yield frames, reconnecting on failure until the stream gives up."""
        return self._frames_latest() if self.low_latency else self._frames_plain()

    def _reconnect(self) -> None:
        self.cap.release()
        time.sleep(self.reconnect_delay)
        self.cap = self._open()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Blocked inside cap.read(); releasing from here could crash.
                # The thread is a daemon and dies with the process.
                return
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.release()
