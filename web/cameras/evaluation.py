"""Video tests ("ทดสอบด้วยวิดีโอ"): queue uploaded videos and run them one at a time.

Each video is played through the real pipeline (person_detection.evaluation),
and every row the system would have logged is stored as an EvalEvent with
a photo, so it can be checked on the review page and turned into metrics.
Nothing is written to the real history (CSV / database) or the wardrobe.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from pathlib import Path

from django.db import close_old_connections
from django.utils import timezone

from person_detection import config
from person_detection.evaluation import metrics
from person_detection.evaluation.runner import EvalOptions, VideoEvaluator

from . import services

log = logging.getLogger(__name__)

VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_VIDEO_BYTES = 4 * 1024 ** 3  # 4 GB


def eval_dir() -> Path:
    return Path(config.OUTPUT_DIR) / "eval"


def video_dir(video_id: int) -> Path:
    return eval_dir() / str(int(video_id))


class EvalJobs:
    """One background thread that runs queued videos in order."""

    PROGRESS_SAVE_SEC = 1.0

    def __init__(self):
        self._queue: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel: dict[int, threading.Event] = {}
        self._preview: dict[int, bytes] = {}
        self.current: int | None = None

    # ---- control ------------------------------------------------------------
    def submit(self, video_id: int) -> None:
        with self._lock:
            self._cancel[video_id] = threading.Event()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="video-tests", daemon=True)
                self._thread.start()
        self._queue.put(video_id)

    def cancel(self, video_id: int) -> None:
        event = self._cancel.get(video_id)
        if event:
            event.set()

    def preview(self, video_id: int) -> bytes | None:
        return self._preview.get(video_id)

    def wait_idle(self, timeout: float = 600) -> bool:
        """For tests: wait until every queued video is finished."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.1)
        return False

    def resume_after_restart(self) -> None:
        """Videos left queued/running when the app closed are run again."""
        from .models import EvalVideo
        for video in EvalVideo.objects.filter(status__in=["queued", "running"]).order_by("created_at"):
            video.events.all().delete()
            video.status, video.progress, video.message = "queued", 0, "รอคิว (เริ่มใหม่หลังเปิดแอป)"
            video.save(update_fields=["status", "progress", "message"])
            self.submit(video.id)

    # ---- worker ---------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            video_id = self._queue.get()
            try:
                self._run(video_id)
            except Exception:  # never let one bad video stop the queue
                log.exception("video test %s failed", video_id)
            finally:
                self._preview.pop(video_id, None)
                self._cancel.pop(video_id, None)
                self.current = None
                close_old_connections()
                self._queue.task_done()

    def _run(self, video_id: int) -> None:
        from .models import EvalEvent, EvalVideo
        video = EvalVideo.objects.filter(id=video_id).first()
        if video is None or video.status not in ("queued",):
            return
        stop = self._cancel.get(video_id) or threading.Event()
        if stop.is_set():
            return
        self.current = video_id
        folder = video_dir(video.id)
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("ev_*.jpg"):
            old.unlink(missing_ok=True)
        for old in folder.glob("face_*.jpg"):
            old.unlink(missing_ok=True)
        # Run again (e.g. with another model): keep a summary of the last run and
        # the ground truth typed in, to put back on the matching new rows.
        old_truth = [{"kind": e.kind, "time_s": e.time_s, "box": e.box, "gt_status": e.gt_status,
                      "gt_identity": e.gt_identity, "gt_direction": e.gt_direction}
                     for e in video.events.all() if e.gt_status or e.gt_identity or e.gt_direction]
        if video.stats and video.events.exists():
            video.runs = (list(video.runs or []) + [run_summary(video)])[-MAX_RUNS:]
        video.events.all().delete()
        video.status, video.progress, video.message, video.stats = "running", 0, "กำลังโหลดโมเดล", {}
        video.save(update_fields=["status", "progress", "message", "stats", "runs"])

        track_event: dict[int, int] = {}  # track id -> its sighting EvalEvent id
        counter = {"n": 0}

        def on_event(e: dict) -> None:
            counter["n"] += 1
            snapshot = ""
            if e.get("snapshot"):
                path = folder / f"ev_{counter['n']:05d}.jpg"
                path.write_bytes(e["snapshot"])
                snapshot = str(path)
            row = EvalEvent.objects.create(
                video=video, kind=e["kind"], track_id=e["track_id"], time_s=e["time_s"],
                latency_s=e.get("latency_s"), name=(e.get("name") or "unknown")[:100],
                face_seen=bool(e.get("face_seen")), candidate=(e.get("candidate") or "")[:100],
                score=e.get("score"), shirt_color=e.get("shirt_color") or "", pants_color=e.get("pants_color") or "",
                confidence=e.get("confidence") or 0, snapshot=snapshot, box=e.get("box"))
            if e["kind"] == "sighting":
                track_event[e["track_id"]] = row.id

        def on_face_crops(track_id: int, crops: list[bytes]) -> None:
            event_id = track_event.get(track_id)
            if not event_id:
                return
            paths = []
            for i, data in enumerate(crops, start=1):
                path = folder / f"face_{event_id}_{i}.jpg"
                path.write_bytes(data)
                paths.append(str(path))
            EvalEvent.objects.filter(id=event_id).update(face_photos=paths)

        last_save = {"t": 0.0}

        def on_progress(stats) -> None:
            now = time.monotonic()
            if now - last_save["t"] < self.PROGRESS_SAVE_SEC:
                return
            last_save["t"] = now
            total = stats.total_frames or 0
            progress = min(0.99, stats.frames_read / total) if total else 0
            EvalVideo.objects.filter(id=video.id).update(
                progress=progress, stats=stats.to_dict(),
                message=f"เฟรม {stats.frames_read}/{total or '?'} · {stats.processing_fps:.1f} fps")

        def on_preview(data: bytes) -> None:
            self._preview[video.id] = data

        opts = video.options or {}
        options = EvalOptions(
            rotate=int(opts.get("rotate", 0)), long_side=int(opts.get("long_side", config.PROCESS_LONG_SIDE)),
            realtime=bool(opts.get("realtime", True)), enable_direction=bool(opts.get("direction", False)),
            enable_vlm_par=bool(opts.get("vlm", False)), door_line=opts.get("door_line"),
            frame_skip=config.FRAME_SKIP)
        try:
            stats = VideoEvaluator(services.factory, options).run(
                video.file, on_event, on_face_crops, on_progress, on_preview, stop)
        except Exception as exc:
            first = (str(exc).strip().splitlines() or [exc.__class__.__name__])[0]
            EvalVideo.objects.filter(id=video.id).update(status="error", message=first[:300],
                                                         finished_at=timezone.now())
            return
        cancelled = stop.is_set()
        message = "ยกเลิกแล้ว (ผลถึงจุดที่หยุด)" if cancelled else ""
        if old_truth:
            new_events = list(EvalEvent.objects.filter(video_id=video.id))
            carried = carry_ground_truth(old_truth, new_events)
            message = (message + " · " if message else "") + (
                f"ใช้ผลตรวจเดิมกับ {carried}/{len(new_events)} แถวที่ตรงกับรอบก่อน (คนเดิม เวลาใกล้กัน) "
                "ตรวจแถวที่เหลือ และจำนวนคนที่ระบบพลาด อีกครั้ง")
        EvalVideo.objects.filter(id=video.id).update(
            status="cancelled" if cancelled else "done", progress=1, stats=stats.to_dict(),
            message=message[:300], finished_at=timezone.now())


jobs = EvalJobs()


def delete_video(video) -> None:
    jobs.cancel(video.id)
    shutil.rmtree(video_dir(video.id), ignore_errors=True)
    video.delete()


# ------------------------------------------------------- comparing models
MAX_RUNS = 10
MATCH_SECONDS = 1.0  # a row of the new run is the same as an old one within this time...
MATCH_IOU = 0.3  # ...and with boxes overlapping at least this much


def model_threshold(stats: dict | None) -> float:
    """Face threshold of the face model a run used (scores of different models differ)."""
    from person_detection.vision import face_models
    try:
        return face_models.threshold_for((stats or {}).get("face_model") or "sface")
    except ValueError:
        return float(config.FACE_MATCH_THRESHOLD)


def model_label(stats: dict | None) -> str:
    stats = stats or {}
    person = stats.get("person_model") or "yolo11n-pose"
    face = {"sface": "SFace", "arcface_mbf": "ArcFace-MBF", "arcface_r50": "ArcFace-R50"}.get(
        stats.get("face_model") or "sface", stats.get("face_model"))
    extra = [x for x, on in (("ReID", stats.get("reid")), ("full-res", stats.get("full_res"))) if on]
    return " + ".join([person, face, *extra])


def run_summary(video, threshold: float | None = None) -> dict:
    stats = video.stats or {}
    threshold = model_threshold(stats) if threshold is None else threshold
    m = video_metrics(video, threshold)
    return {
        "label": model_label(stats), "finished_at": video.finished_at.isoformat() if video.finished_at else "",
        "threshold": threshold, "ms_per_frame": stats.get("ms_per_frame"), "processing_fps": stats.get("processing_fps"),
        "events": m["events"], "reviewed": m["reviewed"],
        "precision": m["detection"]["precision"], "recall": m["detection"]["recall"], "f1": m["detection"]["f1"],
        "far": m["face"]["far"], "frr": m["face"]["frr"], "accuracy": m["face"]["accuracy"],
        "coverage": m["face"]["coverage"], "latency_s": m["latency_s"],
    }


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def carry_ground_truth(old: list[dict], new_events) -> int:
    """Copy the ground truth of the last run onto the rows of the new run that
    are the same person at the same moment (same kind of row, within
    MATCH_SECONDS, boxes overlapping). Returns how many rows got it."""
    unused = list(old)
    carried = 0
    for event in sorted(new_events, key=lambda e: e.time_s):
        best, best_key = None, None
        for row in unused:
            if row["kind"] != event.kind or abs(row["time_s"] - event.time_s) > MATCH_SECONDS:
                continue
            overlap = _iou(row["box"], event.box) if row.get("box") and event.box else None
            if overlap is not None and overlap < MATCH_IOU:
                continue
            key = (abs(row["time_s"] - event.time_s), -(overlap or 0))
            if best_key is None or key < best_key:
                best, best_key = row, key
        if best is None:
            continue
        unused.remove(best)
        event.gt_status, event.gt_identity, event.gt_direction = best["gt_status"], best["gt_identity"], best["gt_direction"]
        event.save(update_fields=["gt_status", "gt_identity", "gt_direction"])
        carried += 1
    return carried


# ---------------------------------------------------------------- metrics
def video_metrics(video, threshold: float) -> dict:
    return metrics.compute([([e.as_metric() for e in video.events.all()], video.missed)], threshold)


def group_metrics(videos, threshold: float) -> dict:
    return metrics.compute([([e.as_metric() for e in v.events.all()], v.missed) for v in videos], threshold)


def threshold_sweep(videos, face_model: str = "sface") -> dict:
    events = [e.as_metric() for v in videos for e in v.events.all()]
    if face_model == "sface":
        return metrics.sweep(events)
    # ArcFace scores sit lower than SFace's: look from 0.20 up
    return metrics.sweep(events, [round(0.20 + 0.02 * i, 2) for i in range(16)])
