"""Background work for the portal: IP-camera workers and browser-camera sessions.

Everything here lives inside the single web-server process:

* WorkerManager  - one thread per running IP camera. It keeps reading the
                   camera, runs the detection pipeline, keeps the newest
                   annotated JPEG for the live page, and writes CSV rows.
                   Runs whether or not anybody has the page open.
* SessionManager - one pipeline per open "camera on my computer" page. The
                   browser sends frames; we return people as 0..1 coordinates.

Both share one PipelineFactory, so the big VLM-PAR model is loaded only once.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from django.conf import settings as django_settings

from person_detection import config
from person_detection.multicam.floor import FloorCalibration
from person_detection.multicam.hub import IdentityHub
from person_detection.multicam.link import CameraLink
from person_detection.pipeline import CameraOptions, PipelineFactory
from person_detection.sources.stream import LatestFrameStream, open_stream

log = logging.getLogger("cameras")

JPEG_QUALITY = 80
STREAM_TIMEOUT_SEC = 15  # no frame for this long -> reconnect
MAX_BACKOFF_SEC = 30

factory = PipelineFactory()
# Joins people across the IP cameras (same spot on the floor = same person).
hub = IdentityHub()
SNAPSHOT_EVERY_SEC = 2.0  # clean picture kept for the floor-calibration page
BROWSER_MAX_FPS = 10  # "camera on my computer" page: frames sent per second (ตั้งค่าระบบ)

# The LAN history page served next to the desktop app (filled in by
# desktop.py, shown on the app's dashboard). managed=False: not run by the app.
web_info = {"managed": False, "enabled": False, "urls": [], "error": ""}


def mask_passwords(text: str) -> str:
    """rtsp://user:secret@host -> rtsp://user:****@host (for messages shown on pages)."""
    return re.sub(r"(://[^:/@\s]+:)[^@\s]*@", r"\1****@", text)


def device_info() -> dict:
    """Which processor the models run on, for the dashboard."""
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": True, "name": torch.cuda.get_device_name(0)}
        return {"gpu": False, "name": "CPU (ไม่พบ CUDA)"}
    except Exception as exc:  # torch missing or broken driver
        return {"gpu": False, "name": f"CPU ({exc.__class__.__name__})"}


# --------------------------------------------------------------- IP cameras
@dataclass(frozen=True)
class CameraSpec:
    """What a worker needs to know about a camera (a copy, not the DB row)."""

    id: int
    name: str
    source: str
    options: CameraOptions
    calibration: FloorCalibration | None = None

    @classmethod
    def from_model(cls, camera) -> "CameraSpec":
        return cls(
            id=camera.id,
            name=camera.name,
            source=camera.source.strip(),
            options=CameraOptions(
                rotate=camera.rotate,
                long_side=camera.long_side,
                enable_face_detection=True,
                enable_face_recognition=camera.enable_face_recognition,
                enable_vlm_par=camera.enable_vlm_par,
                enable_direction=camera.enable_direction,
                csv_tag=camera.csv_tag,
                save_csv=config.SAVE_CSV,
                save_crops=config.SAVE_CROPS,
            ),
            calibration=FloorCalibration.from_dict(camera.floor_calibration),
        )


class CameraWorker:
    """Runs one camera in a background thread until stop() is called."""

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.status = "starting"
        self.message = ""
        self.fps = 0.0
        self.source_size: tuple[int, int] | None = None
        self.process_size: tuple[int, int] | None = None
        self.people_now = 0
        self.started_at = time.time()
        self.recent = deque(maxlen=30)  # newest first: people written to CSV
        self.jpeg: bytes | None = None
        self.jpeg_id = 0
        self.snapshot_jpeg: bytes | None = None  # un-annotated, for calibration
        self._snapshot_at = 0.0
        self.link = CameraLink(hub, spec.name, spec.calibration)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{spec.name}", daemon=True)
        self._thread.start()

    # ---- control --------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5) -> None:
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    # ---- the loop ---------------------------------------------------------
    def _run(self) -> None:
        processor = None
        backoff = 1
        while not self._stop.is_set():
            try:
                if processor is None:
                    self._set("loading", "กำลังโหลดโมเดล")
                    processor = factory.create(self.spec.options, on_record=self.recent.appendleft,
                                               identity=self.link)
                self._set("connecting", "กำลังเชื่อมต่อกล้อง")
                with open_stream(self.spec.source, max_failures=3, reconnect_delay=1) as stream:
                    self._set("running", "")
                    backoff = 1
                    self._pump(stream, processor)
                if self._stop.is_set():
                    break
                raise ConnectionError("กล้องหยุดส่งภาพ")
            except Exception as exc:  # keep the thread alive whatever happens
                if self._stop.is_set():
                    break
                first_line = mask_passwords((str(exc).strip().splitlines() or [exc.__class__.__name__])[0])
                state = "error" if processor is None else "reconnecting"
                self._set(state, f"{first_line} — ลองใหม่ใน {backoff} วินาที")
                log.warning("camera %s: %s", self.spec.name, first_line)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SEC)
        self.link.close()  # this camera's track ids are gone; tell the hub
        self._set("stopped", "")

    def _pump(self, stream, processor) -> None:
        last = time.perf_counter()
        if isinstance(stream, LatestFrameStream):  # live camera: newest frame only
            frames = stream.frames(timeout=STREAM_TIMEOUT_SEC)
        else:  # a video file (handy for testing): every frame, then loop
            frames = stream.frames()
        received = 0
        for frame in frames:
            if self._stop.is_set():
                return
            received += 1
            if received % max(1, config.FRAME_SKIP):  # frame skip (ตั้งค่าระบบ), read live
                continue
            if time.monotonic() - self._snapshot_at > SNAPSHOT_EVERY_SEC:
                clean = processor.preprocess(frame)  # same size/rotation as detection
                ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    self.snapshot_jpeg = buf.tobytes()
                    self._snapshot_at = time.monotonic()
            result = processor.process(frame)
            ok, buf = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.jpeg = buf.tobytes()
                self.jpeg_id += 1
            self.source_size = (result.source_width, result.source_height)
            self.process_size = (result.width, result.height)
            self.people_now = len(result.people)
            now = time.perf_counter()
            instant = 1.0 / max(now - last, 1e-3)
            self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
            last = now

    def _set(self, status: str, message: str) -> None:
        self.status, self.message = status, message

    def snapshot(self) -> dict:
        return {
            "id": self.spec.id,
            "name": self.spec.name,
            "status": self.status,
            "message": self.message,
            "fps": round(self.fps, 1),
            "source_size": self.source_size,
            "process_size": self.process_size,
            "people_now": self.people_now,
            "recent": list(self.recent)[:10],
        }


class WorkerManager:
    def __init__(self):
        self._workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()

    def start(self, camera) -> None:
        """Start (or restart with new settings) the worker for a camera."""
        spec = CameraSpec.from_model(camera)
        with self._lock:
            old = self._workers.pop(spec.id, None)
        if old:
            # Let the old thread finish first: on exit it clears its tracks from
            # the hub, which must not hit the new worker's tracks.
            old.stop()
            old.join(5)
        with self._lock:
            self._workers[spec.id] = CameraWorker(spec)
        log.info("camera %s started", spec.name)

    def stop(self, camera_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            log.info("camera %s stopped", worker.spec.name)

    def get(self, camera_id: int) -> CameraWorker | None:
        return self._workers.get(camera_id)

    def set_calibration(self, camera_id: int, calibration: FloorCalibration | None) -> None:
        """Apply a new floor calibration to a running camera, no restart needed."""
        worker = self._workers.get(camera_id)
        if worker:
            worker.link.calibration = calibration

    def snapshot(self, camera_id: int) -> dict | None:
        worker = self.get(camera_id)
        return worker.snapshot() if worker else None

    def running_ids(self) -> list[int]:
        with self._lock:
            return list(self._workers)

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(3)


workers = WorkerManager()


# ---------------------------------------------------------- browser cameras
class BrowserSession:
    def __init__(self, key, options: CameraOptions):
        self.key = key
        self.options = options
        self.processor = factory.create(options)
        self.frame_shape = None
        self.lock = threading.Lock()  # one frame at a time per session
        self.last_used = time.monotonic()

    def process(self, frame):
        with self.lock:
            if self.frame_shape is not None and frame.shape != self.frame_shape:
                # Picture size changed (user rotated it or switched camera):
                # the tracker compares consecutive frames, so start it fresh.
                self.processor = factory.create(self.options)
            self.frame_shape = frame.shape
            return self.processor.process(frame, annotate=False)


class SessionLimitError(RuntimeError):
    pass


class SessionManager:
    """Pipelines for 'camera on my computer' pages, closed when idle."""

    IDLE_SEC = 60

    def __init__(self, max_sessions: int = 6):
        self.max_sessions = max_sessions
        self._sessions: dict[tuple, BrowserSession] = {}
        self._lock = threading.Lock()

    def process(self, user, session_id: str, jpeg: bytes, options: CameraOptions) -> dict:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("ไม่ใช่ไฟล์ภาพ JPEG/PNG")
        session = self._session(user, session_id, options)
        started = time.perf_counter()
        result = session.process(frame)
        session.last_used = time.monotonic()
        data = result.to_dict()
        data["ms"] = round((time.perf_counter() - started) * 1000)
        return data

    def _session(self, user, session_id: str, options: CameraOptions) -> BrowserSession:
        key = (user.pk, session_id)
        with self._lock:
            self._close_idle_locked()
            session = self._sessions.get(key)
            if session and session.options == options:
                return session
            if session is None and len(self._sessions) >= self.max_sessions:
                raise SessionLimitError(
                    f"มีคนใช้กล้องจากเบราว์เซอร์ครบ {self.max_sessions} หน้าต่างแล้ว ลองใหม่ภายหลัง")
        # Model loading takes a moment; do it outside the lock.
        session = BrowserSession(key, options)
        with self._lock:
            self._sessions[key] = session
        log.info("browser session opened for %s (%d open)", user, len(self._sessions))
        return session

    def clear(self) -> None:
        """Close every session (settings changed; pages reopen one on the next frame)."""
        with self._lock:
            self._sessions.clear()

    def close(self, user, session_id: str) -> None:
        with self._lock:
            self._sessions.pop((user.pk, session_id), None)

    def _close_idle_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, s in self._sessions.items() if now - s.last_used > self.IDLE_SEC]:
            del self._sessions[key]

    def count(self) -> int:
        with self._lock:
            self._close_idle_locked()
            return len(self._sessions)


sessions = SessionManager(django_settings.PORTAL_MAX_BROWSER_SESSIONS)

# ------------------------------------------------------------- enrollment
_enroller = None
_enroller_lock = threading.Lock()


def enroller():
    """The face enroller, sharing the face gallery used by the running cameras."""
    global _enroller
    from person_detection import config
    from person_detection.enrollment import FaceEnroller
    from person_detection.vision.face import FaceDetector, FaceRecognizer

    with _enroller_lock:
        if _enroller is None:
            _enroller = FaceEnroller(
                FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE),
                FaceRecognizer(config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD),
                factory.people_db(),
                config.PHOTOS_DIR,
            )
        return _enroller


def reset_enroller() -> None:
    """Rebuild the enroller next time (face settings changed)."""
    global _enroller
    with _enroller_lock:
        _enroller = None


def reset_shared_state() -> None:
    """Forget cached gallery/enroller (tests point config at a temp folder)."""
    global _enroller
    with _enroller_lock:
        _enroller = None
    factory._people_db = None
    factory._wardrobe = None
    global hub
    hub = IdentityHub()


# ------------------------------------------------------------- start-up
_started = False
_start_lock = threading.Lock()


def start_background() -> None:
    """Start every camera marked 'running'. Safe to call more than once."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True

    def _boot():
        from . import system_settings
        from .models import Camera
        system_settings.load_and_apply()  # settings.json from the "ตั้งค่าระบบ" page
        try:
            for camera in Camera.objects.filter(should_run=True):
                workers.start(camera)
        except Exception as exc:  # e.g. database not migrated yet
            log.error("could not auto-start cameras: %s", exc)

    threading.Thread(target=_boot, name="portal-boot", daemon=True).start()
