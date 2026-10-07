"""Tally each named person's clothing colours over time, keyed by day."""

import json
import time
from pathlib import Path

from ..vision.colors import Appearance


class WardrobeLog:
    """Counts how often each name is seen in each shirt/pants combo, per day."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict[str, dict[str, int]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            self._data, indent=2), encoding="utf-8")

    def record(self, name: str, appearance: Appearance, when: time.struct_time | None = None) -> None:
        date = time.strftime("%Y-%m-%d", when or time.localtime())
        combo = str(appearance)
        day = self._data.setdefault(date, {})
        person = day.setdefault(name, {})
        person[combo] = person.get(combo, 0) + 1
        self._save()

    def daily_wear(self, name: str, date: str | None = None) -> str | None:
        date = date or time.strftime("%Y-%m-%d")
        person = self._data.get(date, {}).get(name)
        if not person:
            return None
        return max(person, key=person.get)

    def history(self, name: str) -> dict[str, str]:
        out = {}
        for date, people in self._data.items():
            if name in people:
                out[date] = max(people[name], key=people[name].get)
        return out
