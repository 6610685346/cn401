"""Command-line app: every camera in config.CAMERAS -> pipeline -> one window each.

    python connect.py           (camera password: set CAM_PASSWORD=... first)

Each camera runs in its own thread with its own processor (YOLO's tracker
remembers the previous frames of *its* camera); the big shared pieces (VLM-PAR,
face gallery, wardrobe) are loaded once by PipelineFactory. All cameras write
to output/person_log_<date>.csv, told apart by the "camera" column.

The frame-by-frame work lives in pipeline.py so the desktop app reuses it.
"""

from __future__ import annotations

import threading
import time

import cv2

from . import config
from .pipeline import CameraOptions, FrameProcessor, PipelineFactory
from .sources.stream import LatestFrameStream, open_stream
from .ui.display import DisplayWindow

RETRY_SEC = 5  # wait before reconnecting a camera that dropped


class CameraWorker(threading.Thread):
    """Reads one camera, runs its processor, and keeps only the newest result."""

    def __init__(self, name: str, source, processor: FrameProcessor, quit_event: threading.Event):
        super().__init__(name=f"camera-{name}", daemon=True)
        self.camera_name = name
        self.source = source
        self.processor = processor
        self.quit_event = quit_event
        self._lock = threading.Lock()
        self._result = None
        self._result_id = 0

    def latest(self, last_id: int):
        """(frame, id) if there is something newer than last_id, else (None, last_id)."""
        with self._lock:
            if self._result_id == last_id:
                return None, last_id
            return self._result, self._result_id

    def run(self) -> None:
        while not self.quit_event.is_set():
            try:
                with open_stream(self.source, config.MAX_READ_FAILURES,
                                 config.RECONNECT_DELAY_SEC) as stream:
                    print(f"[{self.camera_name}] reading stream")
                    frames = (stream.frames(timeout=15) if isinstance(stream, LatestFrameStream)
                              else stream.frames())
                    for count, frame in enumerate(frames, start=1):
                        if self.quit_event.is_set():
                            return
                        if count % max(1, config.FRAME_SKIP):
                            continue
                        annotated = self.processor(frame)
                        with self._lock:
                            self._result = annotated
                            self._result_id += 1
                print(f"[{self.camera_name}] stream ended")
            except Exception as exc:  # one camera failing must not stop the others
                print(f"[{self.camera_name}] {str(exc).splitlines()[0] if str(exc) else exc!r}")
            if self.quit_event.wait(RETRY_SEC):
                return
            print(f"[{self.camera_name}] reconnecting ...")


class MultiCameraApp:
    """Shows one window per camera; 'q' in a window or 'q' + Enter in the terminal quits."""

    def __init__(self, workers: list[CameraWorker], quit_event: threading.Event):
        self.workers = workers
        self._quit = quit_event

    def _watch_terminal_quit(self) -> None:
        while not self._quit.is_set():
            try:
                if input().strip().lower() == "q":
                    self._quit.set()
                    break
            except (EOFError, OSError):
                break

    def run(self) -> None:
        threading.Thread(target=self._watch_terminal_quit, daemon=True).start()
        # OpenCV windows must be made and refreshed by the main thread.
        windows = {
            w.camera_name: DisplayWindow(
                f"{config.WINDOW_NAME} - {w.camera_name}", config.PRINT_MOUSE_POSITION,
                max_size=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            for w in self.workers
        }
        last_ids = {w.camera_name: 0 for w in self.workers}
        for w in self.workers:
            w.start()
        print("Press 'q' in a window or type 'q' + Enter here to quit")
        try:
            while not self._quit.is_set():
                shown = False
                for w in self.workers:
                    frame, last_ids[w.camera_name] = w.latest(last_ids[w.camera_name])
                    if frame is None:
                        continue
                    shown = True
                    if not windows[w.camera_name].show(frame):
                        self._quit.set()
                        break
                if not shown and cv2.waitKey(10) & 0xFF == ord("q"):
                    self._quit.set()  # no new frames yet: keep the windows responsive
        finally:
            self._quit.set()
            print("Exiting...")
            for w in self.workers:
                w.join(timeout=3)
            for window in windows.values():
                window.close()


# ------------------------------------------------------------------ builders
def cameras() -> dict:
    """name -> source from config.CAMERAS (falls back to config.SOURCE)."""
    found = dict(getattr(config, "CAMERAS", None) or {})
    return found or {"cam1": config.SOURCE}


def build_processor(factory: PipelineFactory, camera: str) -> FrameProcessor:
    """One processor per camera, each with its own YOLO tracker and face detector."""
    return factory.create(CameraOptions(camera=camera, door_line=config.DOOR_LINES.get(camera)))


def main() -> None:
    factory = PipelineFactory()
    print(f"Models: person={config.PERSON_MODEL}, face={config.FACE_RECOGNIZER}"
          + (f", ReID={config.REID_MODEL}" if config.TRACKER_REID else ""))
    if config.ENABLE_FACE_RECOGNITION:
        if config.FACE_RECOGNIZER != "sface":  # e.g. ArcFace: vectors made from the enrollment photos
            from .enrollment import prepare_gallery
            prepare_gallery(factory)
        names = factory.people_db().names()
        print(f"Loaded {len(names)} enrolled people: {names}")
    if config.SAVE_CSV:
        print(f"Logging people to: {config.CSV_PATH or config.OUTPUT_DIR / 'person_log_<date>.csv'}")
    quit_event = threading.Event()
    workers = []
    for name, source in cameras().items():
        print(f"Loading models for {name} ...")
        workers.append(CameraWorker(name, source, build_processor(factory, name), quit_event))
    started = time.monotonic()
    MultiCameraApp(workers, quit_event).run()
    print(f"Ran for {time.monotonic() - started:.0f}s")
