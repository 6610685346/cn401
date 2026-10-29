"""Stores enrolled people as name -> face embedding, on disk as JSON."""

import json
import threading
import time
from pathlib import Path

import numpy as np


class PeopleDB:
    """Name-to-embedding gallery, persisted as JSON.

    Safe to share between camera threads. If people.json is changed by another
    program (e.g. enroll.py run while the web portal is up), the new gallery is
    picked up automatically within a few seconds.
    """

    RELOAD_CHECK_SEC = 3.0

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._people: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self._mtime = None
        self._last_check = 0.0
        self._load()

    def _file_mtime(self):
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _load(self) -> None:
        with self._lock:
            self._mtime = self._file_mtime()
            if self._mtime is None:
                self._people = {}
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._people = {name: np.array(vec, dtype=np.float32)
                            for name, vec in raw.items()}

    def reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self._last_check < self.RELOAD_CHECK_SEC:
            return
        self._last_check = now
        if self._file_mtime() != self._mtime:
            try:
                self._load()
            except (json.JSONDecodeError, OSError):
                pass  # file is mid-write; try again on the next check

    def save(self) -> None:
        with self._lock:
            raw = {name: vec.tolist() for name, vec in self._people.items()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw), encoding="utf-8")
            tmp.replace(self.path)  # atomic: readers never see half a file
            self._mtime = self._file_mtime()

    def enroll(self, name: str, embeddings: list[np.ndarray], image_paths=None,
               surname: str | None = None, id_number: str | None = None) -> None:
        # image_paths / surname / id_number are only kept by the SQL version.
        if not embeddings:
            raise ValueError(f"No embeddings to enroll for '{name}'")
        mean = np.mean(np.stack(embeddings), axis=0)
        mean = mean / np.linalg.norm(mean)
        with self._lock:
            self._people[name] = mean.astype(np.float32)
            self.save()

    def remove(self, name: str) -> None:
        with self._lock:
            self._people.pop(name, None)
            self.save()

    def names(self) -> list[str]:
        self.reload_if_changed()
        return list(self._people.keys())

    def gallery(self) -> dict[str, np.ndarray]:
        self.reload_if_changed()
        return self._people

    def person_id(self, name: str | None) -> int | None:
        return None  # people.json has no ids


def people_db_path(model: str | None = None) -> Path:
    """people.json for SFace (as before), people_<model>.json for other face models."""
    from .. import config

    model = model or config.FACE_RECOGNIZER
    base = Path(config.PEOPLE_DB_PATH)
    return base if model == "sface" else base.with_name(f"{base.stem}_{model}{base.suffix}")


def open_people_db(model: str | None = None):
    """The face gallery the settings ask for: PostgreSQL when DATABASE_URL is
    set, otherwise people.json. For command-line tools (the web portal and
    connect.py share one through PipelineFactory)."""
    from .. import config

    if config.DATABASE_URL:
        from .database import Database, SqlPeopleDB
        return SqlPeopleDB(Database(config.DATABASE_URL, config.DB_TIMEZONE), model or config.FACE_RECOGNIZER)
    return PeopleDB(people_db_path(model))
