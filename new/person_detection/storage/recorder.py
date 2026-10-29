"""Log each tracked person once: a CSV row and/or a database row, plus an optional photo."""

import csv
import datetime as dt
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
               "shirt_color", "pants_color", "confidence", "global_id",
               "similarity", "direction", "snapshot"]


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
              appearance: Appearance | None, name: str | None, global_id: int | None = None,
              camera: str = "", similarity: float | None = None,
              direction: str | None = None, snapshot: str = "") -> None:
        self._write_row(
            self.path_for(when),
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", when),
                "camera": camera,
                "track_id": detection.track_id,
                "name": name or "unknown",
                "shirt_color": appearance.shirt_color if appearance else "",
                "pants_color": appearance.pants_color if appearance else "",
                "confidence": f"{detection.confidence:.2f}",
                "global_id": "" if global_id is None else f"G{global_id}",
                "similarity": "" if similarity is None else f"{similarity:.2f}",
                "direction": direction or "",
                "snapshot": snapshot,
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
    """Saves a photo of each logged row: <dir>/<date>/<camera>_person_<id>_<time>[_in|_out].jpg"""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, frame: Frame, detection: Detection, when: time.struct_time,
             camera: str = "", tag: str = "") -> Path | None:
        crop = crop_box(frame, detection.box)
        if not crop.size:
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S", when)
        prefix = f"{safe_slug(camera)}_" if camera else ""
        folder = self.directory / time.strftime("%Y-%m-%d", when)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        path = folder / f"{prefix}person_{detection.track_id}_{stamp}{suffix}.jpg"
        return path if cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]) else None

    def relative(self, path: Path | None) -> str:
        """Path as written in the CSV: relative to the crops folder, with / separators."""
        if not path:
            return ""
        try:
            return path.resolve().relative_to(self.directory.resolve()).as_posix()
        except ValueError:
            return ""


def _center_inside(face_box: Box, person_box: Box) -> bool:
    """True if a face box's center sits inside a person box."""
    fx = (face_box[0] + face_box[2]) / 2
    fy = (face_box[1] + face_box[3]) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= fx <= px2 and py1 <= fy <= py2


class FaceIdentifier:
    """Decides a name for each track from the faces seen on it.

    votes=1, min_face_px=0 behaves like the first version: the first face
    inside the person's box decides. With the defaults (see PERFORMANCE.md):

    * faces smaller than `min_face_px` are not used (SFace is unreliable on
      tiny faces; they would mostly come out "unknown" or, worse, a wrong name);
    * a very clear match (score >= threshold + strong_margin) decides at once;
    * otherwise up to `votes` faces from different frames are averaged and
      the average decides - one blurry or turned face cannot pick the name.

    The best candidate and its score are kept for every decided track, so
    the video test can recompute FAR / FRR for other thresholds.
    """

    def __init__(self, recognizer: FaceRecognizer, people_db: PeopleDB,
                 votes: int = 1, min_face_px: int = 0, strong_margin: float = 0.12):
        self.recognizer = recognizer
        self.people_db = people_db
        self.votes = max(1, int(votes))
        self.min_face_px = int(min_face_px)
        self.strong_margin = strong_margin
        # track_id -> name, once decided ("unknown" if the face matched nobody)
        self.names: dict[int, str | None] = {}
        # track_id -> best cosine similarity of that decision (stored in the database)
        self.scores: dict[int, float] = {}
        # track_id -> (best gallery name, its score) whatever the threshold
        self.candidates: dict[int, tuple[str | None, float]] = {}
        self._samples: dict[int, list[np.ndarray]] = {}
        self.embeddings_made = 0  # face-model runs, for the statistics

    # ---- decisions -----------------------------------------------------------
    def _best(self, embedding: np.ndarray) -> tuple[str | None, float]:
        best, score = None, -1.0
        for name, reference in self.people_db.gallery().items():
            similarity = self.recognizer.similarity(embedding, reference)
            if similarity > score:
                best, score = name, similarity
        return best, float(score)

    def _decide(self, track_id: int, best: str | None, score: float) -> str:
        self.candidates[track_id] = (best, score)
        name = best if best is not None and score >= self.recognizer.match_threshold else "unknown"
        self.names[track_id] = name
        self.scores[track_id] = score
        self._samples.pop(track_id, None)
        return name

    def identify(self, frame, detection: Detection,
                 faces_raw: list[tuple[Box, np.ndarray]]) -> str | None:
        """The name for this track: None while still collecting faces."""
        track_id = detection.track_id
        if track_id in self.names:
            return self.names[track_id]
        for hit in faces_raw:
            face_box, raw_row = hit
            if not _center_inside(face_box, detection.box):
                continue
            # FACE_FULL_RES: the face cut from the camera's full-size picture
            source = getattr(hit, "source", None)
            width = face_box[2] - face_box[0]
            if source is not None:
                frame, raw_row, width = source, hit.source_row, float(hit.source_row[2])
            if width < self.min_face_px:
                continue  # too small to trust; wait for the person to come closer
            embedding = self.recognizer.embed(frame, raw_row)
            self.embeddings_made += 1
            samples = self._samples.setdefault(track_id, [])
            samples.append(embedding / (np.linalg.norm(embedding) or 1.0))
            best, score = self._best(embedding)
            if self.votes == 1 or (best is not None and score >= self.recognizer.match_threshold + self.strong_margin):
                return self._decide(track_id, best, score)
            if len(samples) >= self.votes:
                return self.finalize(track_id)
            return None  # one face per frame per track
        return None

    def pending(self, track_id: int) -> bool:
        """Faces collected but no decision yet."""
        return track_id not in self.names and bool(self._samples.get(track_id))

    def wants_face(self, track_id: int) -> bool:
        return track_id not in self.names

    def finalize(self, track_id: int) -> str | None:
        """Decide now from the faces collected so far (None if there are none)."""
        if track_id in self.names:
            return self.names[track_id]
        samples = self._samples.get(track_id)
        if not samples:
            return None
        mean = np.mean(np.stack(samples), axis=0).astype(np.float32)
        best, score = self._best(mean / (np.linalg.norm(mean) or 1.0))
        return self._decide(track_id, best, score)

    def score_for(self, track_id: int) -> float | None:
        return self.scores.get(track_id)

    def forget(self, track_id: int) -> None:
        self.names.pop(track_id, None)
        self.scores.pop(track_id, None)
        self.candidates.pop(track_id, None)
        self._samples.pop(track_id, None)


class PersonRecorder:
    """Decides when a person is logged and remembers what they wore."""

    def __init__(
        self,
        csv_log: CsvLog | None,
        crop_saver: CropSaver | None = None,
        gate: TrackGate | None = None,
        reader: AppearanceReader | None = None,
        face_identifier: FaceIdentifier | None = None,
        wardrobe: WardrobeLog | None = None,
        on_record: Callable[[dict], None] | None = None,
        identity_lookup: Callable[[int], object] | None = None,
        camera: str = "",
        db_log=None,
        person_ids: Callable[[str | None], int | None] | None = None,
        log_cooldown_sec: float = 0.0,
        face_wait_frames: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.csv_log = csv_log  # None when SAVE_CSV is off but the database is on
        # storage.database.DetectionLogWriter, or None without a database
        self.db_log = db_log
        self.person_ids = person_ids  # name -> persons.id
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
        # A named person already logged by this camera less than this many
        # seconds ago is not logged again (the tracker gave them a new ID after
        # being hidden for a moment). In/out rows are never skipped.
        self.log_cooldown_sec = log_cooldown_sec
        self.face_wait_frames = face_wait_frames
        self.clock = clock
        self._last_logged: dict[str, float] = {}
        self._face_wait = {}  # track_id -> frames waited for the face vote
        self.suppressed = 0  # rows skipped by the cooldown, for the statistics

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
        if self._should_wait_for_face(track_id):
            return None  # a few more faces will make the name more certain
        if self._should_wait_for_name(track_id):
            return None  # joined cameras may still be matching this person

        self.appearances[track_id] = appearance
        self.gate.forget(track_id)
        self._name_wait.pop(track_id, None)
        self._record(frame, detection, appearance)
        return appearance

    NAME_WAIT_FRAMES = 10

    def wants_face(self, track_id: int) -> bool:
        """Is a face still useful for this track? (else face detection can skip it)"""
        return (self.face_identifier is not None and track_id not in self.names
                and self.face_identifier.wants_face(track_id))

    def _settle_face(self, track_id: int) -> None:
        if self.face_identifier and track_id not in self.names:
            name = self.face_identifier.finalize(track_id)
            if name is not None:
                self.names[track_id] = name

    def _should_wait_for_face(self, track_id: int) -> bool:
        if not self.face_identifier or not self.face_identifier.pending(track_id):
            return False
        waited = self._face_wait.get(track_id, 0)
        if waited < self.face_wait_frames:
            self._face_wait[track_id] = waited + 1
            return True
        self._settle_face(track_id)  # decide from the faces collected so far
        return False

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
        self._face_wait.pop(track_id, None)
        self.gate.forget(track_id)
        if self.face_identifier:
            self.face_identifier.forget(track_id)

    def _identity(self, detection: Detection) -> tuple[str | None, int | None, float | None, str]:
        """(name, global id, similarity, where the name came from) for a track."""
        name = self.names.get(detection.track_id)
        info = self.identity_lookup(detection.track_id) if self.identity_lookup else None
        global_id = getattr(info, "gid", None)
        similarity = None
        if self.face_identifier:
            similarity = self.face_identifier.score_for(detection.track_id)
        source = "face" if similarity is not None else "none"
        if (not name or name == "unknown") and getattr(info, "name", None):
            name = info.name
            source = "other_camera"  # this camera did not see the face itself
        return name, global_id, similarity, source

    def _save(self, frame, detection: Detection, appearance: Appearance | None,
              direction: str | None = None) -> tuple[str | None, int | None, time.struct_time]:
        now_epoch = time.time()
        now = time.localtime(now_epoch)
        name, global_id, similarity, source = self._identity(detection)
        snapshot = None
        if self.crop_saver:  # one photo per row, crossings too (shown in the history)
            snapshot = self.crop_saver.save(frame, detection, now, self.camera, tag=direction or "")
        if self.csv_log:
            self.csv_log.write(now, detection, appearance, name, global_id, self.camera,
                               similarity=similarity, direction=direction,
                               snapshot=self.crop_saver.relative(snapshot) if self.crop_saver else "")
        if self.db_log:
            attributes = {"track_id": detection.track_id,
                          "confidence": round(float(detection.confidence), 2),
                          "name_source": source}
            if appearance:
                attributes["shirt_color"] = appearance.shirt_color
                attributes["pants_color"] = appearance.pants_color
            if global_id is not None:
                attributes["global_id"] = f"G{global_id}"
            known = name if name and name != "unknown" else None
            self.db_log.add(
                camera=self.camera or "default",
                person_id=self.person_ids(known) if self.person_ids and known else None,
                similarity=similarity, direction=direction, attributes=attributes,
                snapshot_path=snapshot,
                detected_at=dt.datetime.fromtimestamp(now_epoch, dt.timezone.utc))
        return name, global_id, now

    def _record(self, frame: Frame, detection: Detection, appearance: Appearance) -> None:
        known = self.names.get(detection.track_id)
        if self.log_cooldown_sec > 0 and known and known != "unknown":
            now_s = self.clock()
            last = self._last_logged.get(known)
            if last is not None and now_s - last < self.log_cooldown_sec:
                self.suppressed += 1
                return  # same person, same camera, moments ago: not a new visit
            self._last_logged[known] = now_s
        name, global_id, now = self._save(frame, detection, appearance)
        where = f"[{self.camera}] " if self.camera else ""
        print(f"{where}Logged ID {detection.track_id} ({name or 'unknown'}) : {appearance}")
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

    def record_direction(self, frame: Frame, detection: Detection, direction: str) -> None:
        """Log one crossing ("in" / "out") as its own row."""
        if not detection.is_tracked:
            return
        self._settle_face(detection.track_id)  # use the best name we have right now
        appearance = self.appearances.get(detection.track_id)
        name, _, _ = self._save(frame, detection, appearance, direction=direction)
        where = f"[{self.camera}] " if self.camera else ""
        print(f"{where}ID {detection.track_id} ({name or 'unknown'}) walked {direction}")
