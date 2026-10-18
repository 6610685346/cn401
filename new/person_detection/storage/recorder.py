"""Log each tracked person once: a CSV row plus an optional photo."""

import csv
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..vision.colors import Appearance, AppearanceReader
from ..vision.detector import Detection
from ..vision.face import FaceRecognizer

from ..core.geometry import Box, Frame, crop_box

from .people_db import PeopleDB
from .wardrobe import WardrobeLog

CSV_COLUMNS = ["timestamp", "camera", "track_id", "name",
               "shirt_color", "pants_color", "confidence", "global_id"]


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


# One lock per CSV file, shared by every camera thread in this process.
_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(path.resolve(), threading.Lock())


class CsvLog:
    """Appends one row per person to a CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def path_for(self, when: time.struct_time) -> Path:
        return self.path

    def write(self, when: time.struct_time, detection: Detection,
              appearance: Appearance, name: str | None, global_id: int | None = None,
              camera: str = "") -> None:
        self._write_row(
            self.path_for(when),
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", when),
                "camera": camera,
                "track_id": detection.track_id,
                "name": name or "unknown",
                "shirt_color": appearance.shirt_color,
                "pants_color": appearance.pants_color,
                "confidence": f"{detection.confidence:.2f}",
                "global_id": "" if global_id is None else f"G{global_id}",
            },
        )

    @staticmethod
    def _write_row(path: Path, row: dict) -> None:
        with _lock_for(path):
            columns = CSV_COLUMNS
            if path.exists():
                # A file started by an older version keeps its own columns, so
                # every row in one file always lines up with its header.
                with open(path, newline="", encoding="utf-8-sig") as existing:
                    header = next(csv.reader(existing), None)
                if header:
                    columns = header
            new_file = not path.exists()
            # utf-8-sig so Excel picks the encoding up correctly.
            with open(path, "a", newline="", encoding="utf-8-sig") as csv_file:
                writer = csv.writer(csv_file)
                if new_file:
                    writer.writerow(columns)
                writer.writerow([row.get(column, "") for column in columns])


def safe_slug(text: str) -> str:
    """Make a name safe to use inside a file name."""
    slug = re.sub(r"[^A-Za-z0-9ก-๙_-]+", "-", text).strip("-")
    return slug[:40] or "camera"


class DailyCsvLog(CsvLog):
    """Starts a new file every day: <dir>/person_log_<date>[_<tag>].csv.

    Needed for cameras that run for days: the old CLI picked the file name
    once at start-up, so after midnight rows kept going into yesterday's file.
    `tag` keeps cameras apart (one file per camera per day) so their rows and
    track IDs do not get mixed up.
    """

    def __init__(self, directory: str | Path, tag: str | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tag = safe_slug(tag) if tag else None
        self.path = self.path_for(time.localtime())

    def path_for(self, when: time.struct_time) -> Path:
        date = time.strftime("%Y-%m-%d", when)
        suffix = f"_{self.tag}" if self.tag else ""
        return self.directory / f"person_log_{date}{suffix}.csv"


class CropSaver:
    """Saves a photo of each logged person."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, frame: Frame, detection: Detection, when: time.struct_time,
             camera: str = "") -> None:
        crop = crop_box(frame, detection.box)
        if not crop.size:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S", when)
        prefix = f"{safe_slug(camera)}_" if camera else ""
        path = self.directory / f"{prefix}person_{detection.track_id}_{stamp}.jpg"
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
        on_record: Callable[[dict], None] | None = None,
        identity_lookup: Callable[[int], object] | None = None,
        camera: str = "",
    ):
        self.csv_log = csv_log
        self.crop_saver = crop_saver
        self.gate = gate or TrackGate(min_frames=5)
        self.reader = reader or AppearanceReader()
        self.face_identifier = face_identifier
        self.wardrobe = wardrobe
        self.on_record = on_record  # e.g. the web page's "recent people" list
        # track_id -> GlobalInfo from the multi-camera hub (global id + a name
        # another camera recognised), or None when cameras are not joined.
        self.identity_lookup = identity_lookup
        self.camera = camera  # written in the CSV "camera" column and crop names
        self.appearances = {}
        self.names = {}
        self._name_wait = {}  # track_id -> frames waited for another camera's name

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
        if self._should_wait_for_name(track_id):
            return None  # joined cameras may still be matching this person

        self.appearances[track_id] = appearance
        self.gate.forget(track_id)
        self._name_wait.pop(track_id, None)
        self._record(frame, detection, appearance)
        return appearance

    NAME_WAIT_FRAMES = 10

    def _should_wait_for_name(self, track_id: int) -> bool:
        """With several cameras joined, give another camera a moment to supply
        the name (it takes a few frames to be sure two views are one person)."""
        if not self.identity_lookup:
            return False
        name = self.names.get(track_id)
        if name and name != "unknown":
            return False
        info = self.identity_lookup(track_id)
        if info is not None and getattr(info, "name", None):
            return False
        waited = self._name_wait.get(track_id, 0)
        if waited >= self.NAME_WAIT_FRAMES:
            return False
        self._name_wait[track_id] = waited + 1
        return True

    def name_for(self, track_id: int) -> str | None:
        return self.names.get(track_id)

    def forget(self, track_id: int) -> None:
        """Drop everything kept for a track that has left the picture."""
        self.appearances.pop(track_id, None)
        self.names.pop(track_id, None)
        self._name_wait.pop(track_id, None)
        self.gate.forget(track_id)
        if self.face_identifier:
            self.face_identifier.forget(track_id)

    def _record(self, frame: Frame, detection: Detection, appearance: Appearance) -> None:
        now = time.localtime()
        name = self.names.get(detection.track_id)
        info = self.identity_lookup(detection.track_id) if self.identity_lookup else None
        global_id = getattr(info, "gid", None)
        if (not name or name == "unknown") and getattr(info, "name", None):
            name = info.name
        self.csv_log.write(now, detection, appearance, name, global_id, self.camera)
        where = f"[{self.camera}] " if self.camera else ""
        print(f"{where}Logged ID {detection.track_id} ({name or 'unknown'}) : {appearance}")
        if self.crop_saver:
            self.crop_saver.save(frame, detection, now, self.camera)
        if self.wardrobe and name and name != "unknown":
            self.wardrobe.record(name, appearance, now)
        if self.on_record:
            self.on_record({
                "time": time.strftime("%Y-%m-%d %H:%M:%S", now),
                "camera": self.camera,
                "track_id": detection.track_id,
                "name": name or "unknown",
                "shirt_color": appearance.shirt_color,
                "pants_color": appearance.pants_color,
                "confidence": round(detection.confidence, 2),
                "global_id": global_id,
            })
