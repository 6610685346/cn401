"""Log each tracked person once: a CSV row plus an optional photo."""

import csv
import time
import numpy as np
import cv2

from pathlib import Path

from ..vision.colors import Appearance, AppearanceReader
from ..vision.detector import Detection
from ..vision.detector import Detection
from ..vision.face import FaceRecognizer

from ..core.geometry import Frame, crop_box
from ..core.geometry import Box

from .people_db import PeopleDB
from .wardrobe import WardrobeLog

CSV_COLUMNS = ["timestamp", "track_id", "name",
               "shirt_color", "pants_color", "confidence"]


class TrackGate:
    """Holds a track back until it has been seen for enough frames."""

    def __init__(self, min_frames: int):
        self.min_frames = min_frames
        self.frames_seen = {}  # track_id -> frames seen so far

    def passes(self, track_id: int) -> bool:
        """Count one more sighting; True once the track is trustworthy."""
        self.frames_seen[track_id] = self.frames_seen.get(track_id, 0) + 1
        return self.frames_seen[track_id] >= self.min_frames

    def forget(self, track_id: int) -> None:
        self.frames_seen.pop(track_id, None)


class CsvLog:
    """Appends one row per person to a CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_row(CSV_COLUMNS)

    def write(self, when: time.struct_time, detection: Detection,
              appearance: Appearance, name: str | None) -> None:
        self._write_row(
            [
                time.strftime("%Y-%m-%d %H:%M:%S", when),
                detection.track_id,
                name or "unknown",
                appearance.shirt_color,
                appearance.pants_color,
                f"{detection.confidence:.2f}",
            ]
        )

    def _write_row(self, row) -> None:
        # utf-8-sig so Excel picks the encoding up correctly.
        with open(self.path, "a", newline="", encoding="utf-8-sig") as csv_file:
            csv.writer(csv_file).writerow(row)


class CropSaver:
    """Saves a photo of each logged person."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, frame: Frame, detection: Detection, when: time.struct_time) -> None:
        crop = crop_box(frame, detection.box)
        if not crop.size:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S", when)
        path = self.directory / f"person_{detection.track_id}_{stamp}.jpg"
        cv2.imwrite(str(path), crop)


def _center_inside(face_box: Box, person_box: Box) -> bool:
    """True if a face box's center sits inside a person box."""
    fx = (face_box[0] + face_box[2]) / 2
    fy = (face_box[1] + face_box[3]) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= fx <= px2 and py1 <= fy <= py2


class FaceIdentifier:
    """Matches faces to person boxes and recognizes a name for each track, once."""

    def __init__(self, recognizer: FaceRecognizer, people_db: PeopleDB):
        self.recognizer = recognizer
        self.people_db = people_db
        # track_id -> name, once resolved
        self.names: dict[int, str | None] = {}

    def identify(
        self, frame, detection: Detection,
        faces_raw: list[tuple[Box, np.ndarray]],
    ) -> str | None:
        """Return the name for this track, recognizing it the first time a
        face lands inside the person's box. None until then, "unknown" if
        a face was seen but matched nobody in the gallery."""
        track_id = detection.track_id
        if track_id in self.names:
            return self.names[track_id]

        for face_box, raw_row in faces_raw:
            if not _center_inside(face_box, detection.box):
                continue
            embedding = self.recognizer.embed(frame, raw_row)
            name, score = self.recognizer.best_match(
                embedding, self.people_db.gallery())
            self.names[track_id] = name or "unknown"
            return self.names[track_id]

        return None

    def forget(self, track_id: int) -> None:
        self.names.pop(track_id, None)


class PersonRecorder:
    """Decides when a person is logged and remembers what they wore."""

    def __init__(
        self,
        csv_log: CsvLog,
        crop_saver: CropSaver | None = None,
        gate: TrackGate | None = None,
        reader: AppearanceReader | None = None,
        face_identifier: FaceIdentifier | None = None,
        wardrobe: WardrobeLog | None = None,
    ):
        self.csv_log = csv_log
        self.crop_saver = crop_saver
        self.gate = gate or TrackGate(min_frames=5)
        self.reader = reader or AppearanceReader()
        self.face_identifier = face_identifier
        self.wardrobe = wardrobe
        self.appearances = {}
        self.names = {}

    def observe(self, frame: Frame, detection: Detection,
                faces_raw: list[tuple[Box, np.ndarray]] | None = None) -> Appearance | None:
        """Return the person's appearance once known, logging them the first time."""

        if not detection.is_tracked:
            return None
        track_id = detection.track_id

        if self.face_identifier and track_id not in self.names:
            name = self.face_identifier.identify(
                frame, detection, faces_raw or [])
            if name is not None:
                self.names[track_id] = name

        if track_id in self.appearances:
            return self.appearances[track_id]

        if not self.gate.passes(track_id):
            return None

        appearance = self.reader.read(frame, detection.box)
        if appearance is None:
            return None  # too small to read; retry on a later frame

        self.appearances[track_id] = appearance
        self.gate.forget(track_id)
        self._record(frame, detection, appearance)
        return appearance

    def name_for(self, track_id: int) -> str | None:
        return self.names.get(track_id)

    def _record(self, frame: Frame, detection: Detection, appearance: Appearance) -> None:
        now = time.localtime()
        name = self.names.get(detection.track_id)
        self.csv_log.write(now, detection, appearance, name)
        print(
            f"Logged ID {detection.track_id} ({name or 'unknown'}) : {appearance}")
        if self.crop_saver:
            self.crop_saver.save(frame, detection, now)
        if self.wardrobe and name and name != "unknown":
            self.wardrobe.record(name, appearance, now)
