"""Tally each named person's clothing colours over time, keyed by day."""

import json
import threading
import time
from pathlib import Path

from ..vision.colors import Appearance


class WardrobeLog:
    """Counts how often each name is seen in each shirt/pants combo, per day.

    Safe to share between camera threads. Each write re-reads the file first,
    so counts added by another program (connect.py) are not overwritten.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, dict[str, int]]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # keep what we had rather than crash on a half-written file

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, name: str, appearance: Appearance, when: time.struct_time | None = None) -> None:
        date = time.strftime("%Y-%m-%d", when or time.localtime())
        combo = str(appearance)
        with self._lock:
            self._load()
            day = self._data.setdefault(date, {})
            person = day.setdefault(name, {})
            person[combo] = person.get(combo, 0) + 1
            self._save()

    def daily_wear(self, name: str, date: str | None = None) -> str | None:
        date = date or time.strftime("%Y-%m-%d")
        with self._lock:
            self._load()
            person = self._data.get(date, {}).get(name)
        if not person:
            return None
        return max(person, key=person.get)

    def history(self, name: str) -> dict[str, str]:
        with self._lock:
            self._load()
            data = self._data
        out = {}
        for date, people in data.items():
            if name in people:
                out[date] = max(people[name], key=people[name].get)
        return out
