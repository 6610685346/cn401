"""Stores enrolled people as name -> face embedding, on disk as JSON."""

import json
from pathlib import Path

import numpy as np


class PeopleDB:
    """Name-to-embedding gallery, persisted as JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._people: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._people = {name: np.array(vec, dtype=np.float32)
                        for name, vec in raw.items()}

    def save(self) -> None:
        raw = {name: vec.tolist() for name, vec in self._people.items()}
        self.path.write_text(json.dumps(raw), encoding="utf-8")

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            raise ValueError(f"No embeddings to enroll for '{name}'")
        mean = np.mean(np.stack(embeddings), axis=0)
        mean = mean / np.linalg.norm(mean)
        self._people[name] = mean.astype(np.float32)
        self.save()

    def remove(self, name: str) -> None:
        self._people.pop(name, None)
        self.save()

    def names(self) -> list[str]:
        return list(self._people.keys())

    def gallery(self) -> dict[str, np.ndarray]:
        return self._people
