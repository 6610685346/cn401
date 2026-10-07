"""Wire the pieces together: stream -> preprocess -> detect -> record -> draw -> show."""

import time
import threading

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

from .core.geometry import Frame, FramePreprocessor

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
                    seen = self.vlm_frames_seen.get(track_id, 0) + 1
                    self.vlm_frames_seen[track_id] = seen
                    if seen >= config.VLM_PAR_MIN_FRAMES:
                        attributes = self.vlm_par.read(frame, detection.box)
                        if attributes is not None:
                            self.vlm_attributes[track_id] = attributes

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


# ------------------------------------------------------------------ builders

def build_face_identifier() -> FaceIdentifier | None:
    if not config.ENABLE_FACE_RECOGNITION:
        return None
    recognizer = FaceRecognizer(
        config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD)
    people_db = PeopleDB(config.PEOPLE_DB_PATH)
    print(
        f"Loaded {len(people_db.names())} enrolled people: {people_db.names()}")
    return FaceIdentifier(recognizer, people_db)


def build_recorder() -> PersonRecorder | None:
    """Build the CSV recorder, or None when SAVE_CSV is off."""
    if not config.SAVE_CSV:
        return None
    csv_path = config.CSV_PATH or (
        config.OUTPUT_DIR / f"person_log_{time.strftime('%Y-%m-%d')}.csv"
    )
    print(f"Logging people to: {csv_path}")
    return PersonRecorder(
        CsvLog(csv_path),
        CropSaver(config.CROPS_DIR) if config.SAVE_CROPS else None,
        TrackGate(config.MIN_FRAMES_BEFORE_LOG),
        face_identifier=build_face_identifier(),
        wardrobe=WardrobeLog(
            config.WARDROBE_LOG_PATH) if config.ENABLE_FACE_RECOGNITION else None,
    )


def build_processor() -> FrameProcessor:
    """Build a frame processor from the settings in config.py."""
    return FrameProcessor(
        preprocess=FramePreprocessor(
            config.ROTATE, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
        ),
        detector=PersonDetector(config.MODEL_PATH, config.CONFIDENCE),
        face_detector=(
            FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)
            if config.ENABLE_FACE_DETECTION else None
        ),
        recorder=build_recorder(),
        direction_tracker=DirectionTracker() if config.ENABLE_DIRECTION else None,
        vlm_par=(
            VLMParReader(config.VLM_PAR_MODEL_PATH,
                         config.VLM_PAR_COLOR_LABELS)
            if config.ENABLE_VLM_PAR else None
        ),
    )


def main() -> None:
    PersonDetectionApp(config.SOURCE, build_processor(),
                       config.FRAME_SKIP).run()
