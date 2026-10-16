"""Wire the pieces together: stream -> preprocess -> detect -> record -> draw -> show."""

import time
import threading
import cv2
import time

from . import config

from .vision.detector import PersonDetector
from .vision.face import FaceDetector, FaceRecognizer
from .vision.vlm_par import VLMParReader

from .counting.direction import DirectionTracker

from .ui.display import DisplayWindow
from .ui.drawing import Overlay

from .storage.people_db import PeopleDB
from .storage.wardrobe import WardrobeLog
from .storage.recorder import FaceIdentifier
from .storage.recorder import CropSaver, CsvLog, PersonRecorder, TrackGate

from concurrent.futures import Future, ThreadPoolExecutor

from .core.geometry import Frame, FramePreprocessor, crop_box

from .sources.stream import VideoStream


class FrameProcessor:
    """Turns one raw frame into an annotated frame; holds no I/O of its own."""

    def __init__(
        self,
        preprocess: FramePreprocessor,
        detector: PersonDetector,
        face_detector: FaceDetector | None = None,
        recorder: PersonRecorder | None = None,
        direction_tracker: DirectionTracker | None = None,
        overlay: Overlay | None = None,
        vlm_par: VLMParReader | None = None,
        vlm_pool: ThreadPoolExecutor | None = None,
    ):
        self.preprocess = preprocess
        self.detector = detector
        self.face_detector = face_detector
        self.recorder = recorder
        self.direction_tracker = direction_tracker
        self.overlay = overlay or Overlay()
        self.vlm_par = vlm_par
        self.vlm_attributes = {}  # track_id -> attributes, inferred once per track
        self.vlm_frames_seen = {}
        self._vlm_pool = vlm_pool or (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-par")
            if vlm_par else None)
        self._vlm_pending: dict[int, Future] = {}

    def _poll_vlm(self, track_id: int):
        """Collect a finished background read for this track, if any."""
        future = self._vlm_pending.get(track_id)
        if future is None or not future.done():
            return None
        del self._vlm_pending[track_id]
        try:
            return future.result()
        except Exception as exc:  # keep the video alive if the model fails
            print(f"VLM-PAR failed for track {track_id}: {exc}")
            return None

    def _submit_vlm(self, frame: Frame, detection) -> None:

        x1, y1, x2, y2 = detection.box
        if y2 - y1 < self.vlm_par.MIN_BOX_HEIGHT_PX:
            return
        
        track_id = detection.track_id
        if track_id in self._vlm_pending:
            return
        # Copy the crop now: the overlay paints over `frame` before the
        # worker gets to it. The worker reads the whole crop.
        crop = crop_box(frame, detection.box).copy()
        if crop.size == 0:
            return
        height, width = crop.shape[:2]
        self._vlm_pending[track_id] = self._vlm_pool.submit(
            self.vlm_par.read, crop, (0, 0, width, height))

    def __call__(self, raw: Frame) -> Frame:
        frame = self.preprocess(raw)
        detections = self.detector.detect(frame)
        faces_raw = self.face_detector.detect_raw(
            frame) if self.face_detector else []
        faces = [box for box, _ in faces_raw]

        # Sample colours before drawing, which would paint over other people.
        appearances = [
            self.recorder.observe(
                frame, detection, faces_raw) if self.recorder else None
            for detection in detections
        ]

        for detection, appearance in zip(detections, appearances):
            attributes = None
            if self.vlm_par and detection.is_tracked:
                track_id = detection.track_id
                attributes = self.vlm_attributes.get(track_id)
                if attributes is None:
                    attributes = self._poll_vlm(track_id)
                    if attributes is not None:
                        self.vlm_attributes[track_id] = attributes
                if attributes is None:
                    seen = self.vlm_frames_seen.get(track_id, 0) + 1
                    self.vlm_frames_seen[track_id] = seen
                    if seen >= config.VLM_PAR_MIN_FRAMES:
                        self._submit_vlm(frame, detection)


            name = self.recorder.name_for(
                detection.track_id) if self.recorder else None
            self.overlay.draw_person(
                frame, detection, appearance, attributes, name)
            if self.direction_tracker:
                direction = self.direction_tracker.update(detection)
                if direction:
                    print(f"Track ID {detection.track_id} -> {direction}")

        if self.direction_tracker:
            self.direction_tracker.draw_zones(frame)

        for face in faces:
            self.overlay.draw_face(frame, face)

        self.overlay.draw_count(frame, len(detections))
        return frame


class PersonDetectionApp:
    """Reads frames from a stream, processes them, and shows the result."""

    def __init__(self, source, processor: FrameProcessor, frame_skip: int = 1):
        self.source = source
        self.processor = processor
        self.frame_skip = max(1, frame_skip)
        self._quit = threading.Event()

    def _watch_terminal_quit(self) -> None:
        """Lets 'q' + Enter in the terminal also stop the app."""
        while not self._quit.is_set():
            try:
                if input().strip().lower() == "q":
                    self._quit.set()
                    break
            except (EOFError, OSError):
                break

    def run(self) -> None:
        threading.Thread(target=self._watch_terminal_quit, daemon=True).start()
        with VideoStream(
            self.source, config.MAX_READ_FAILURES, config.RECONNECT_DELAY_SEC
        ) as stream, DisplayWindow(
            config.WINDOW_NAME, config.PRINT_MOUSE_POSITION
        ) as window:
            print(
                f"Reading stream from: {self.source}  (press 'q' in the window or type 'q' + Enter here to quit)")
            for frame_count, frame in enumerate(stream.frames(), start=1):
                if self._quit.is_set():
                    print("Exiting...")
                    break
                if frame_count % self.frame_skip != 0:
                    continue
                if not window.show(self.processor(frame)):
                    print("Exiting...")
                    break



# ------------------------------------------------------------------ new class for muti threading multiple camera


class CameraWorker(threading.Thread):
    """Reads one camera, runs its own processor, and keeps only the newest result."""

    def __init__(self, name: str, source, processor: FrameProcessor,
                 frame_skip: int, quit_event: threading.Event):
        super().__init__(name=f"camera-{name}", daemon=True)
        self.camera_name = name
        self.source = source
        self.processor = processor
        self.frame_skip = max(1, frame_skip)
        self.quit_event = quit_event
        self.finished = threading.Event()
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
        try:
            with VideoStream(
                self.source, config.MAX_READ_FAILURES, config.RECONNECT_DELAY_SEC
            ) as stream:
                print(f"[{self.camera_name}] reading stream")
                for count, frame in enumerate(stream.frames(), start=1):
                    if self.quit_event.is_set():
                        break
                    if count % self.frame_skip != 0:
                        continue
                    annotated = self.processor(frame)
                    with self._lock:
                        self._result = annotated
                        self._result_id += 1
        except Exception as exc:  # one camera failing must not kill the other
            print(f"[{self.camera_name}] stopped: {exc}")
        finally:
            self.finished.set()


class MultiCameraApp:
    """Shows one window per camera; 'q' in a window or in the terminal quits."""

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
        windows = {
            w.camera_name: DisplayWindow(
                f"{config.WINDOW_NAME} - {w.camera_name}", False)
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
                    name = w.camera_name
                    frame, last_ids[name] = w.latest(last_ids[name])
                    if frame is None:
                        continue
                    shown = True
                    if not windows[name].show(frame):
                        self._quit.set()
                        break
                if not shown:
                    # no new frames yet: keep the windows responsive
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        self._quit.set()
                if all(w.finished.is_set() for w in self.workers):
                    print("All cameras stopped.")
                    break
        finally:
            self._quit.set()
            for w in self.workers:
                w.join(timeout=3)
            for window in windows.values():
                window.close()



# ------------------------------------------------------------------ builders

class SharedResources:
    """Loaded once and shared by every camera."""

    def __init__(self):
        self.vlm_par = (
            VLMParReader(config.VLM_PAR_MODEL_PATH,
                         config.VLM_PAR_COLOR_LABELS)
            if config.ENABLE_VLM_PAR else None
        )
        self.vlm_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-par")
            if self.vlm_par else None
        )

        self.people_db = None
        self.wardrobe = None
        if config.ENABLE_FACE_RECOGNITION:
            self.people_db = PeopleDB(config.PEOPLE_DB_PATH)
            self.wardrobe = WardrobeLog(config.WARDROBE_LOG_PATH)
            print(f"Loaded {len(self.people_db.names())} enrolled people: "
                  f"{self.people_db.names()}")

        self.csv_log = None
        if config.SAVE_CSV:
            csv_path = config.CSV_PATH or (
                config.OUTPUT_DIR / f"person_log_{time.strftime('%Y-%m-%d')}.csv"
            )
            print(f"Logging people to: {csv_path}")
            self.csv_log = CsvLog(csv_path)


def build_recorder(shared: SharedResources, camera: str) -> PersonRecorder | None:
    """One recorder per camera: track IDs are only unique within a camera."""
    if shared.csv_log is None:
        return None
    face_identifier = None
    if config.ENABLE_FACE_RECOGNITION:
        recognizer = FaceRecognizer(
            config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD)
        face_identifier = FaceIdentifier(recognizer, shared.people_db)
    return PersonRecorder(
        shared.csv_log,
        CropSaver(config.CROPS_DIR) if config.SAVE_CROPS else None,
        TrackGate(config.MIN_FRAMES_BEFORE_LOG),
        face_identifier=face_identifier,
        wardrobe=shared.wardrobe,
        camera=camera,
    )


def build_processor(shared: SharedResources, camera: str) -> FrameProcessor:
    """One processor per camera, each with its own YOLO tracker and face detector."""
    return FrameProcessor(
        preprocess=FramePreprocessor(
            config.ROTATE, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
        ),
        detector=PersonDetector(config.MODEL_PATH, config.CONFIDENCE),
        face_detector=(
            FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)
            if config.ENABLE_FACE_DETECTION else None
        ),
        recorder=build_recorder(shared, camera),
        direction_tracker=DirectionTracker() if config.ENABLE_DIRECTION else None,
        vlm_par=shared.vlm_par,
        vlm_pool=shared.vlm_pool,
    )


def main() -> None:
    shared = SharedResources()
    quit_event = threading.Event()
    workers = []
    for name, source in config.CAMERAS.items():
        print(f"Loading models for {name} ...")
        workers.append(CameraWorker(
            name, source, build_processor(shared, name),
            config.FRAME_SKIP, quit_event))
    MultiCameraApp(workers, quit_event).run()
