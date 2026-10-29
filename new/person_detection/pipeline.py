"""One frame in, people out. Shared by the command-line app and the web portal.

    raw frame (any resolution) -> FramePreprocessor (rotate + scale to the
    standard size) -> YOLO tracking -> faces / names -> colours -> VLM-PAR
    -> FrameResult (annotated picture + people as 0..1 coordinates)

PipelineFactory owns the heavy, shareable pieces (the VLM-PAR model, the face
gallery, the wardrobe file) and hands out one FrameProcessor per camera. Each
camera needs its own processor because YOLO's tracker remembers the previous
frames of *that* camera.
"""

from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from . import config
from .core.geometry import Frame, FramePreprocessor, crop_box, normalize_box, short_side
from .counting.direction import DirectionTracker
from .counting.line import DoorLine, LineCrossCounter
from .perf.backend import tracker_config
from .perf.motion import DetectionScheduler, MotionGate
from .storage.people_db import PeopleDB, people_db_path
from .storage.recorder import (CropSaver, CsvLog, DailyCsvLog, FaceIdentifier,
                               PersonRecorder, TrackGate)
from .storage.wardrobe import WardrobeLog
from .ui.drawing import Overlay
from .vision import face_models
from .vision.detector import PersonDetector, person_model_path
from .vision.face import FaceDetector, head_region
from .vision.vlm_par import VLMParReader

# A track not seen for this many processed frames is forgotten, so memory stays
# flat when a camera runs for days (track IDs only ever go up).
STALE_TRACK_FRAMES = 150


@dataclass
class PersonResult:
    """One person in one frame, ready to be sent to a web page as JSON."""

    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 as 0..1 fractions
    confidence: float
    label: str  # same text the CLI draws above the box
    name: str | None = None
    clothing: str | None = None  # "shirt/pants" from the colour reader
    attributes: str | None = None  # VLM-PAR summary
    foot: tuple[float, float] | None = None  # where the feet are, 0..1 of the frame
    # filled in when several cameras are joined (multicam/):
    global_id: int | None = None  # same number on every camera that sees this person
    floor: tuple[float, float] | None = None  # metres on the floor plan
    cameras: list[str] | None = None  # cameras that see this person right now
    name_from_other_camera: bool = False  # the face was recognised by another camera

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameResult:
    frame: Frame  # processed (rotated + scaled) frame, annotated if asked
    width: int  # size of `frame`
    height: int
    source_width: int  # size of the frame as it came from the camera
    source_height: int
    people: list[PersonResult] = field(default_factory=list)
    faces: list[tuple[float, float, float, float]] = field(default_factory=list)
    directions: list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "count": len(self.people),
            "people": [p.to_dict() for p in self.people],
            "faces": self.faces,
            "directions": self.directions,
        }


class FrameProcessor:
    """Turns one raw frame into a FrameResult; holds no camera or window I/O."""

    def __init__(
        self,
        preprocess: FramePreprocessor,
        detector: PersonDetector,
        face_detector: FaceDetector | None = None,
        recorder: PersonRecorder | None = None,
        direction_tracker: DirectionTracker | None = None,
        overlay: Overlay | None = None,
        vlm_par: VLMParReader | None = None,
        vlm_min_frames: int = config.VLM_PAR_MIN_FRAMES,
        identity=None,
        vlm_pool: ThreadPoolExecutor | None = None,
        scheduler: DetectionScheduler | None = None,
        smart_faces: bool = False,
        face_every: int = 1,
        face_full_res: bool = False,
    ):
        self.preprocess = preprocess
        # perf/motion.py: skip YOLO while the picture is empty and still (None = never skip)
        self.scheduler = scheduler
        # Look for faces only around people whose name is still unknown, every
        # `face_every` frames, instead of the whole picture on every frame.
        self.smart_faces = smart_faces
        self.face_every = max(1, int(face_every))
        # Find and embed faces in the camera's full-resolution picture instead
        # of the scaled-down one (more pixels on far-away faces).
        self.face_full_res = face_full_res
        self._source = None  # (full-size upright frame, scale) of the current frame
        self._had_people = True
        self.stats = {"frames": 0, "detect_runs": 0, "idle_skips": 0, "face_runs": 0, "face_regions": 0}
        self.detector = detector
        self.face_detector = face_detector
        self.recorder = recorder
        self.direction_tracker = direction_tracker
        self.overlay = overlay or Overlay()
        self.vlm_par = vlm_par
        self.vlm_min_frames = vlm_min_frames
        self.vlm_attributes = {}  # track_id -> attributes, inferred once per track
        self.vlm_frames_seen = {}
        # VLM-PAR is slow, so it runs in a background thread and its answer is
        # picked up on a later frame: the video does not stall while it thinks.
        # None = run it inline (blocking), e.g. in tests.
        self.vlm_pool = vlm_pool
        self._vlm_pending: dict[int, Future] = {}
        # Optional callable(FrameResult) run before drawing; multicam's CameraLink
        # uses it to add global ids and names recognised by other cameras.
        self.identity = identity
        self._frame_index = 0
        self._last_seen: dict[int, int] = {}  # track_id -> frame index

    # Kept so old code that called processor(frame) still gets a picture back.
    def __call__(self, raw: Frame) -> Frame:
        return self.process(raw).frame

    def process(self, raw: Frame, annotate: bool = True) -> FrameResult:
        source_height, source_width = raw.shape[:2]
        upright = self.preprocess.upright(raw)
        frame = self.preprocess.scale(upright)
        height, width = frame.shape[:2]
        scale = max(upright.shape[:2]) / max(height, width)
        self._source = (upright, scale) if self.face_full_res and scale > 1.05 else (None, 1.0)
        self._frame_index += 1
        self.stats["frames"] += 1

        if self.scheduler is not None and not self.scheduler.should_detect(frame, self._had_people):
            self.stats["idle_skips"] += 1  # empty, still picture: nothing to detect
            result = FrameResult(frame, width, height, source_width, source_height)
            if annotate:
                if self.direction_tracker:
                    self.direction_tracker.draw_zones(frame)
                self.overlay.draw_count(frame, 0)
            self._forget_stale_tracks()
            return result

        detections = self.detector.detect(frame)
        self.stats["detect_runs"] += 1
        self._had_people = bool(detections)
        faces_raw = self._faces(frame, detections)
        faces = [box for box, _ in faces_raw]

        # Sample colours before drawing, which would paint over other people.
        appearances = [
            self.recorder.observe(frame, detection, faces_raw) if self.recorder else None
            for detection in detections
        ]

        result = FrameResult(frame, width, height, source_width, source_height)
        found = []  # (detection, appearance, attributes, person)
        for detection, appearance in zip(detections, appearances):
            if detection.is_tracked:
                self._last_seen[detection.track_id] = self._frame_index
            attributes = self._vlm_attributes(frame, detection)
            name = self.recorder.name_for(detection.track_id) if self.recorder else None
            foot_x, foot_y = detection.foot_point
            person = PersonResult(
                track_id=detection.track_id,
                box=normalize_box(detection.box, width, height),
                confidence=round(detection.confidence, 2),
                label="",
                name=name,
                clothing=str(appearance) if appearance else None,
                attributes=str(attributes) if attributes else None,
                foot=(round(foot_x / width, 4), round(foot_y / height, 4)),
            )
            result.people.append(person)
            found.append((detection, appearance, attributes, person))

        if self.identity:  # other cameras may know this person's name
            self.identity(result)

        for detection, appearance, attributes, person in found:
            person.label = Overlay.person_label(detection, appearance, person.name, person.global_id)
            if annotate:
                self.overlay.draw_person(frame, detection, appearance, attributes, person.name,
                                         person.global_id)
            if self.direction_tracker and detection.is_tracked:
                direction = self.direction_tracker.update(detection, width, height)
                if direction:
                    result.directions.append((detection.track_id, direction))
                    print(f"Track ID {detection.track_id} -> {direction}")
                    if self.recorder:
                        self.recorder.record_direction(
                            frame, detection, config.DIRECTION_NAMES.get(direction, direction.lower()))

        result.faces = [normalize_box(box, width, height) for box in faces]
        if annotate:
            if self.direction_tracker:
                self.direction_tracker.draw_zones(frame)
            for face in faces:
                self.overlay.draw_face(frame, face)
            self.overlay.draw_count(frame, len(detections))

        self._forget_stale_tracks()
        return result

    def _faces(self, frame: Frame, detections) -> list:
        if self.face_detector is None:
            return []
        source, scale = self._source or (None, 1.0)
        if not self.smart_faces:
            self.stats["face_runs"] += 1
            return self.face_detector.detect_full(frame, source, scale)
        need = [d for d in detections
                if d.is_tracked and self.recorder is not None and self.recorder.wants_face(d.track_id)]
        if not need or self._frame_index % self.face_every:
            return []
        self.stats["face_runs"] += 1
        self.stats["face_regions"] += len(need)
        return self.face_detector.detect_in_regions(frame, [head_region(d.box) for d in need],
                                                    source=source, scale=scale)

    def perf(self) -> dict:
        """Work counters for the live page and benchmark.py."""
        data = dict(self.stats)
        identifier = getattr(self.recorder, "face_identifier", None)
        data["face_embeddings"] = getattr(identifier, "embeddings_made", 0)
        data["rows_skipped"] = getattr(self.recorder, "suppressed", 0)
        data["backend"] = getattr(self.detector, "backend", "pytorch")
        data["person_model"] = Path(getattr(self.detector, "model_name", "") or "").stem
        data["face_model"] = getattr(getattr(identifier, "recognizer", None), "model_key", None)
        data["reid"] = bool(getattr(self.detector, "reid", False))
        data["full_res"] = bool(self._source and self._source[0] is not None)
        frames = data["frames"] or 1
        data["detect_share"] = round(data["detect_runs"] / frames, 3)
        data["face_share"] = round(data["face_runs"] / frames, 3)
        return data

    def _vlm_attributes(self, frame: Frame, detection):
        if not (self.vlm_par and detection.is_tracked):
            return None
        track_id = detection.track_id
        attributes = self.vlm_attributes.get(track_id)
        if attributes is None:
            attributes = self._collect_vlm(track_id)
        if attributes is None and track_id not in self._vlm_pending:
            seen = self.vlm_frames_seen.get(track_id, 0) + 1
            self.vlm_frames_seen[track_id] = seen
            if seen >= self.vlm_min_frames:
                if self.vlm_pool is None:
                    attributes = self.vlm_par.read(frame, detection.box)
                else:
                    self._submit_vlm(frame, detection)
        if attributes is not None:
            self.vlm_attributes[track_id] = attributes
        return attributes

    def _submit_vlm(self, frame: Frame, detection) -> None:
        x1, y1, x2, y2 = detection.box
        # Size check against the whole frame (the reader would only see the crop).
        if y2 - y1 < self.vlm_par.MIN_BOX_HEIGHT_FRAC * short_side(frame):
            return
        # Copy the crop now: the overlay paints over `frame` before the worker runs.
        crop = crop_box(frame, detection.box).copy()
        if crop.size == 0:
            return
        height, width = crop.shape[:2]
        self._vlm_pending[detection.track_id] = self.vlm_pool.submit(
            self.vlm_par.read, crop, (0, 0, width, height))

    def _collect_vlm(self, track_id: int):
        """A finished background read for this track, if there is one."""
        future = self._vlm_pending.get(track_id)
        if future is None or not future.done():
            return None
        del self._vlm_pending[track_id]
        try:
            return future.result()
        except Exception as exc:  # keep the video going if the model fails
            print(f"VLM-PAR failed for track {track_id}: {exc}")
            return None

    def _forget_stale_tracks(self) -> None:
        if self._frame_index % 30:
            return  # a cheap sweep every 30 frames is plenty
        cutoff = self._frame_index - STALE_TRACK_FRAMES
        for track_id in [t for t, seen in self._last_seen.items() if seen < cutoff]:
            del self._last_seen[track_id]
            self.vlm_attributes.pop(track_id, None)
            self.vlm_frames_seen.pop(track_id, None)
            self._vlm_pending.pop(track_id, None)
            if self.recorder:
                self.recorder.forget(track_id)
            if self.direction_tracker:
                self.direction_tracker.forget(track_id)


@dataclass
class CameraOptions:
    """Per-camera choices. Defaults come from config.py."""

    # Read from config when the options are made (not when this file is
    # imported), so settings changed later still apply.
    rotate: int = field(default_factory=lambda: config.ROTATE)
    long_side: int = field(default_factory=lambda: config.PROCESS_LONG_SIDE)
    enable_face_detection: bool = field(default_factory=lambda: config.ENABLE_FACE_DETECTION)
    enable_face_recognition: bool = field(default_factory=lambda: config.ENABLE_FACE_RECOGNITION)
    enable_vlm_par: bool = field(default_factory=lambda: config.ENABLE_VLM_PAR)
    enable_direction: bool = field(default_factory=lambda: config.ENABLE_DIRECTION)
    save_csv: bool = field(default_factory=lambda: config.SAVE_CSV)
    save_crops: bool = field(default_factory=lambda: config.SAVE_CROPS)
    csv_tag: str | None = None  # None -> output/person_log_<date>.csv
    camera: str = ""  # camera name for the CSV "camera" column and crop file names
    # Door line drawn on the "เส้นประตู" page: {"a": [x, y], "b": [x, y], "inside": "left"|"right"}
    # (0..1 of the processed frame). None = the old fixed zones of direction.py.
    door_line: dict | None = None


class PipelineFactory:
    """Loads shared models once and builds a FrameProcessor per camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._vlm_par: VLMParReader | None = None
        self._vlm_pool: ThreadPoolExecutor | None = None
        self._people_dbs: dict[str, PeopleDB] = {}  # face model -> its gallery
        self._wardrobe: WardrobeLog | None = None
        self._database = None  # storage.database.Database, when DATABASE_URL is set
        self._log_writer = None

    # ---- shared, loaded on first use -------------------------------------
    def vlm_par(self) -> VLMParReader:
        with self._lock:
            if self._vlm_par is None:
                self._vlm_par = VLMParReader(config.VLM_PAR_MODEL_PATH,
                                             config.VLM_PAR_COLOR_LABELS)
            return self._vlm_par

    def vlm_pool(self) -> ThreadPoolExecutor:
        """One background thread for VLM-PAR, shared by all cameras (one model, one GPU)."""
        with self._lock:
            if self._vlm_pool is None:
                self._vlm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-par")
            return self._vlm_pool

    def database(self):
        """The PostgreSQL connection, or None when DATABASE_URL is empty."""
        if not config.DATABASE_URL:
            return None
        with self._lock:
            if self._database is None:
                from .storage.database import Database  # sqlalchemy only needed with a DB
                self._database = Database(config.DATABASE_URL, config.DB_TIMEZONE)
            return self._database

    def log_writer(self):
        """One background writer for detection_logs shared by every camera."""
        database = self.database()
        if database is None:
            return None
        with self._lock:
            if self._log_writer is None:
                from .storage.database import DetectionLogWriter
                self._log_writer = DetectionLogWriter(database)
            return self._log_writer

    def people_db(self, model: str | None = None) -> PeopleDB:
        """The face gallery of a face model (default: the one chosen in settings).

        Each model has its own vectors: people.json / people_<model>.json, or
        the face_embeddings.model column in the database.
        """
        model = model or config.FACE_RECOGNIZER
        database = self.database()
        with self._lock:
            if model not in self._people_dbs:
                if database is not None:
                    from .storage.database import SqlPeopleDB
                    self._people_dbs[model] = SqlPeopleDB(database, model)
                else:
                    self._people_dbs[model] = PeopleDB(people_db_path(model))
            return self._people_dbs[model]

    @property
    def _people_db(self):  # old name, kept for code that resets it
        return self._people_dbs.get(config.FACE_RECOGNIZER)

    @_people_db.setter
    def _people_db(self, value):
        if value is None:
            self._people_dbs.clear()
        else:
            self._people_dbs[config.FACE_RECOGNIZER] = value

    def wardrobe(self) -> WardrobeLog:
        with self._lock:
            if self._wardrobe is None:
                self._wardrobe = WardrobeLog(config.WARDROBE_LOG_PATH)
            return self._wardrobe

    # ---- per camera --------------------------------------------------------
    def create(self, options: CameraOptions | None = None,
               on_record: Callable[[dict], None] | None = None,
               identity=None) -> FrameProcessor:
        """identity: optional multicam CameraLink joining this camera to others."""
        options = options or CameraOptions()
        face_detector = None
        if options.enable_face_detection or options.enable_face_recognition:
            face_detector = FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)

        tracker = self._tracker_for  # chosen once the detector knows its backend
        scheduler = None
        if config.MOTION_GATE:
            scheduler = DetectionScheduler(MotionGate(), idle_check_sec=config.IDLE_CHECK_SEC)
        return FrameProcessor(
            preprocess=FramePreprocessor(options.rotate, options.long_side),
            detector=PersonDetector(person_model_path(), config.CONFIDENCE,
                                    imgsz=config.YOLO_IMGSZ, half=config.YOLO_HALF,
                                    backend=config.YOLO_BACKEND, tracker=tracker),
            scheduler=scheduler,
            smart_faces=config.SMART_FACES,
            face_every=config.FACE_EVERY,
            face_full_res=config.FACE_FULL_RES,
            face_detector=face_detector,
            recorder=self._recorder(options, on_record, identity),
            direction_tracker=self._direction_tracker(options),
            vlm_par=self.vlm_par() if options.enable_vlm_par else None,
            vlm_min_frames=config.VLM_PAR_MIN_FRAMES,  # read now: may be changed from the web
            identity=identity,
            vlm_pool=self.vlm_pool() if options.enable_vlm_par else None,
        )

    @staticmethod
    def _tracker_for(backend: str) -> str:
        """BoT-SORT settings (perf/backend.py) for this detector backend."""
        reid_model = config.REID_MODEL
        if reid_model == "auto" and backend != "pytorch":
            reid_model = "yolo26n-reid.onnx"  # OpenVINO YOLO has no features to reuse
        if reid_model != "auto":
            reid_model = str(Path(config.MODELS_DIR) / reid_model)  # downloaded there by Ultralytics
        return tracker_config(Path(config.OUTPUT_DIR) / "cache", config.TRACK_BUFFER, config.TRACKER_STATIC,
                              reid=config.TRACKER_REID, reid_model=reid_model)

    @staticmethod
    def _direction_tracker(options: CameraOptions):
        if not options.enable_direction:
            return None
        line = DoorLine.from_dict(options.door_line)
        return LineCrossCounter(line) if line else DirectionTracker()

    def _recorder(self, options: CameraOptions, on_record, identity=None) -> PersonRecorder | None:
        db_log = self.log_writer()
        if not options.save_csv and db_log is None:
            return None
        if not options.save_csv:
            csv_log = None
        elif config.CSV_PATH and options.csv_tag is None:
            csv_log = CsvLog(config.CSV_PATH)
        else:
            csv_log = DailyCsvLog(config.OUTPUT_DIR, options.csv_tag)
        face_identifier = None
        if options.enable_face_recognition:
            recognizer = face_models.create_recognizer()  # SFace or ArcFace (settings)
            face_identifier = FaceIdentifier(recognizer, self.people_db(recognizer.model_key),
                                             votes=config.FACE_VOTES,
                                             min_face_px=config.FACE_MIN_SIZE)
        return PersonRecorder(
            csv_log,
            CropSaver(config.CROPS_DIR) if options.save_crops else None,
            TrackGate(config.MIN_FRAMES_BEFORE_LOG),
            face_identifier=face_identifier,
            wardrobe=self.wardrobe() if options.enable_face_recognition else None,
            on_record=on_record,
            identity_lookup=identity.lookup if identity else None,
            camera=options.camera or (options.csv_tag or ""),
            db_log=db_log,
            person_ids=self.people_db().person_id if db_log else None,
            log_cooldown_sec=config.LOG_COOLDOWN_SEC,
        )
