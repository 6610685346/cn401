"""Run the real pipeline on a video file and collect what it would have logged.

The same FrameProcessor the cameras use (YOLO + tracker, YuNet, SFace,
VLM-PAR, direction zones) runs on every frame, but instead of writing the CSV
/ database each logged person becomes an *event* handed to `on_event`, with:

* the video time and how long after the person first appeared it was logged;
* the best face match for ANY threshold (candidate name + cosine score), so
  FAR / FRR can be recomputed later for other thresholds;
* a photo of the person, and up to 5 photos taken when a face was visible
  (to register the person from the video).

With `realtime=True` a slow computer drops frames exactly like a live camera
does (it always takes the newest frame when it is ready for the next one), so
the results show what the real system would have done on this hardware.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import config
from ..core.geometry import crop_box
from ..storage.recorder import FaceIdentifier, PersonRecorder, TrackGate, _center_inside

JPEG_QUALITY = 85


def jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes | None:
    if image is None or not image.size:
        return None
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def padded_crop(frame: np.ndarray, box, pad: float = 0.08) -> np.ndarray:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return crop_box(frame, (int(x1 - dx), int(y1 - dy), int(x2 + dx), int(y2 + dy)))


def _normalized(box, frame) -> list[float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    return [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4)]


class EvalFaceIdentifier(FaceIdentifier):
    """The real FaceIdentifier (it keeps the best candidate below the threshold too)."""


class EvalRecorder(PersonRecorder):
    """PersonRecorder that hands events to a callback instead of saving them."""

    MAX_FACE_CROPS = 5
    FACE_CROP_EVERY = 6  # processed frames between two enrollment photos of one track

    def __init__(self, face_identifier: EvalFaceIdentifier | None, on_event: Callable[[dict], None],
                 on_face_crops: Callable[[int, list[bytes]], None] | None = None,
                 min_frames: int | None = None):
        super().__init__(None, gate=TrackGate(min_frames or config.MIN_FRAMES_BEFORE_LOG),
                         face_identifier=face_identifier, camera="eval",
                         log_cooldown_sec=config.LOG_COOLDOWN_SEC,
                         clock=lambda: self.video_time)  # cooldown in video time
        self.on_event = on_event
        self.on_face_crops = on_face_crops
        self.video_time = 0.0
        self.frame_index = 0
        self.first_seen: dict[int, float] = {}
        self.face_crops: dict[int, list[bytes]] = {}
        self._last_crop: dict[int, int] = {}
        self._logged: set[int] = set()

    def observe(self, frame, detection, faces_raw=None):
        if detection.is_tracked:
            track_id = detection.track_id
            self.first_seen.setdefault(track_id, self.video_time)
            crops = self.face_crops.setdefault(track_id, [])
            if (faces_raw and len(crops) < self.MAX_FACE_CROPS
                    and self.frame_index - self._last_crop.get(track_id, -10**9) >= self.FACE_CROP_EVERY
                    and any(_center_inside(box, detection.box) for box, _ in faces_raw)):
                data = jpeg(padded_crop(frame, detection.box), 92)
                if data:
                    crops.append(data)
                    self._last_crop[track_id] = self.frame_index
        return super().observe(frame, detection, faces_raw)

    def _save(self, frame, detection, appearance, direction=None):
        track_id = detection.track_id
        name, global_id, _, _ = self._identity(detection)
        candidate, score = (None, None)
        face_seen = False
        if isinstance(self.face_identifier, EvalFaceIdentifier) and track_id in self.face_identifier.candidates:
            candidate, score = self.face_identifier.candidates[track_id]
            face_seen = True
        if direction is None:
            self._logged.add(track_id)
        self.on_event({
            "kind": direction or "sighting",
            "track_id": track_id,
            "time_s": round(self.video_time, 3),
            "latency_s": round(self.video_time - self.first_seen.get(track_id, self.video_time), 3),
            "name": name or "unknown",
            "face_seen": face_seen,
            "candidate": candidate,
            "score": None if score is None else round(score, 4),
            "shirt_color": appearance.shirt_color if appearance else "",
            "pants_color": appearance.pants_color if appearance else "",
            "confidence": round(float(detection.confidence), 3),
            "snapshot": jpeg(padded_crop(frame, detection.box)),
            "box": _normalized(detection.box, frame),
        })
        return name, global_id, time.localtime()

    def flush_face_crops(self, track_id: int) -> None:
        crops = self.face_crops.get(track_id)
        if crops and track_id in self._logged and self.on_face_crops:
            self.on_face_crops(track_id, list(crops))

    def forget(self, track_id: int) -> None:
        self.flush_face_crops(track_id)
        super().forget(track_id)
        self.first_seen.pop(track_id, None)
        self.face_crops.pop(track_id, None)
        self._last_crop.pop(track_id, None)

    def finish(self) -> None:
        for track_id in list(self.face_crops):
            self.flush_face_crops(track_id)
        self.face_crops.clear()


@dataclass
class EvalOptions:
    rotate: int = 0
    long_side: int = field(default_factory=lambda: config.PROCESS_LONG_SIDE)
    realtime: bool = True  # drop frames like a live camera when processing is slower
    enable_direction: bool = False
    enable_vlm_par: bool = False
    door_line: dict | None = None  # a camera's door line, to count in/out the same way
    frame_skip: int = field(default_factory=lambda: config.FRAME_SKIP)
    preview_every_sec: float = 0.5


@dataclass
class EvalStats:
    video_fps: float = 0.0
    total_frames: int = 0  # from the file header (may be 0 / approximate)
    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0  # skipped because the computer was still busy (realtime)
    frames_skipped: int = 0  # skipped by FRAME_SKIP
    duration_s: float = 0.0  # video length
    processing_s: float = 0.0  # time spent in the pipeline
    width: int = 0
    height: int = 0
    threshold: float = 0.0
    person_model: str = ""
    face_model: str = ""
    backend: str = ""
    reid: bool = False
    full_res: bool = False

    @property
    def ms_per_frame(self) -> float:
        return 1000 * self.processing_s / self.frames_processed if self.frames_processed else 0.0

    @property
    def processing_fps(self) -> float:
        """Frames the pipeline can do per second on this computer."""
        return self.frames_processed / self.processing_s if self.processing_s else 0.0

    @property
    def effective_fps(self) -> float:
        """Frames per second of video that were actually looked at."""
        return self.frames_processed / self.duration_s if self.duration_s else 0.0

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data.update(ms_per_frame=round(self.ms_per_frame, 1), processing_fps=round(self.processing_fps, 2),
                    effective_fps=round(self.effective_fps, 2))
        return data


class VideoEvaluator:
    """Runs one video through a fresh FrameProcessor."""

    def __init__(self, factory, options: EvalOptions | None = None):
        self.factory = factory
        self.options = options or EvalOptions()

    def build_processor(self, on_event, on_face_crops):
        from ..pipeline import CameraOptions
        from ..vision import face_models

        o = self.options
        processor = self.factory.create(CameraOptions(
            rotate=o.rotate, long_side=o.long_side, enable_face_detection=True,
            enable_face_recognition=True, enable_vlm_par=o.enable_vlm_par,
            enable_direction=o.enable_direction, save_csv=False, save_crops=False, camera="eval",
            door_line=o.door_line))
        recognizer = face_models.create_recognizer()  # the face model chosen in settings
        identifier = EvalFaceIdentifier(recognizer, self.factory.people_db(recognizer.model_key),
                                        votes=config.FACE_VOTES,
                                        min_face_px=config.FACE_MIN_SIZE)
        recorder = EvalRecorder(identifier, on_event, on_face_crops)
        processor.recorder = recorder
        self.models = {  # stored with the results, to compare models on the same video
            "person_model": Path(processor.detector.model_name).stem,
            "backend": processor.detector.backend,
            "face_model": recognizer.model_key,
            "reid": processor.detector.reid,
            "full_res": bool(processor.face_full_res),
        }
        if processor.scheduler is not None:
            processor.scheduler.clock = lambda: recorder.video_time  # idle checks in video time
        return processor, recognizer.match_threshold

    def run(self, path: str, on_event: Callable[[dict], None],
            on_face_crops: Callable[[int, list[bytes]], None] | None = None,
            on_progress: Callable[[EvalStats], None] | None = None,
            on_preview: Callable[[bytes], None] | None = None,
            stop: threading.Event | None = None) -> EvalStats:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("เปิดไฟล์วิดีโอไม่ได้ (ลองแปลงเป็น MP4 H.264)")
        stats = EvalStats()
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            stats.video_fps = fps if fps and 1 <= fps <= 240 else 25.0
            stats.total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            processor, stats.threshold = self.build_processor(on_event, on_face_crops)
            for key, value in self.models.items():
                setattr(stats, key, value)
            self.processor = processor  # benchmark.py reads processor.perf()
            recorder: EvalRecorder = processor.recorder
            skip = max(1, int(self.options.frame_skip or 1))
            busy_until = 0.0  # video time when the (simulated) computer is free again
            last_preview = 0.0
            index = -1
            while True:
                if stop is not None and stop.is_set():
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                stats.frames_read += 1
                t = index / stats.video_fps
                stats.duration_s = (index + 1) / stats.video_fps
                if (index + 1) % skip:
                    stats.frames_skipped += 1
                    continue
                # A live camera hands over its newest frame when the computer is
                # free; frames that arrive while it is busy are never seen.
                if self.options.realtime and (index + 1) / stats.video_fps <= busy_until:
                    stats.frames_dropped += 1
                    continue
                recorder.video_time = t
                recorder.frame_index = index
                started = time.perf_counter()
                result = processor.process(frame, annotate=on_preview is not None)
                spent = time.perf_counter() - started
                stats.processing_s += spent
                stats.frames_processed += 1
                busy_until = max(busy_until, t) + spent
                stats.height, stats.width = result.source_height, result.source_width
                now = time.monotonic()
                if on_preview and now - last_preview >= self.options.preview_every_sec:
                    data = jpeg(result.frame, 75)
                    if data:
                        on_preview(data)
                    last_preview = now
                if on_progress and stats.frames_processed % 5 == 0:
                    on_progress(stats)
            recorder.finish()
        finally:
            cap.release()
        if on_progress:
            on_progress(stats)
        return stats
