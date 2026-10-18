"""Join people seen by several cameras into one global person.

Every camera reports, each frame, the people it tracks: its own track id,
where their feet are on the floor (metres, from the camera's calibration) and
the name face recognition gave them, if any.

* Every new camera track starts as its own global person (G1, G2 ...).
* When a person from camera A and a person from camera B stand on the same
  spot (within MATCH_RADIUS_M) at the same moment (within MAX_TIME_GAP_S) for
  MERGE_AFTER updates in a row, they are merged into one global person.
* A name known to any camera is then shown by every camera for that person:
  camera 1 sees the face, camera 2 only sees the back, both say "Arm".

Safety rules: never merge two people that the same camera sees separately at
the same time, never merge two different known names, and require the match
to be mutual (each is the other's nearest) so crowds do not get mixed up.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass, field

MATCH_RADIUS_M = 0.8  # feet within this distance = same spot
MAX_TIME_GAP_S = 1.0  # cameras' frames arrive a little apart
MERGE_AFTER = 6  # agreeing updates in a row (from either camera) before merging
FORGET_AFTER_S = 20.0  # a global person unseen this long is dropped
ACTIVE_S = 1.0  # a camera track counts as "on screen now" within this


def _known(name: str | None) -> str | None:
    return name if name and name != "unknown" else None


@dataclass
class Observation:
    track_id: int
    floor: tuple[float, float] | None  # metres, None if the camera is not calibrated
    name: str | None = None  # from face recognition on this camera


@dataclass
class Sighting:
    camera: str
    track_id: int
    floor: tuple[float, float] | None
    time: float


@dataclass
class GlobalPerson:
    gid: int
    name: str | None = None
    name_camera: str | None = None  # which camera recognised the face
    sightings: dict[tuple[str, int], Sighting] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def active(self, now: float) -> list[Sighting]:
        return [s for s in self.sightings.values() if now - s.time <= ACTIVE_S]

    def position(self, now: float) -> tuple[float, float] | None:
        points = [s.floor for s in self.active(now) if s.floor is not None]
        if not points:
            return None
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


@dataclass(frozen=True)
class GlobalInfo:
    gid: int
    name: str | None
    cameras: tuple[str, ...]


class IdentityHub:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._people: dict[int, GlobalPerson] = {}
        self._links: dict[tuple[str, int], int] = {}  # (camera, track) -> gid
        self._votes: dict[tuple[int, int], int] = {}  # (gid a, gid b), a < b -> agreeing updates

    # ---- main entry ------------------------------------------------------------
    def update(self, camera: str, observations: list[Observation]) -> dict[int, GlobalInfo]:
        """Report one frame from one camera; returns track_id -> global person."""
        now = self._clock()
        with self._lock:
            for obs in observations:
                if obs.track_id < 0:
                    continue
                person = self._person_for(camera, obs.track_id, now)
                person.sightings[(camera, obs.track_id)] = Sighting(camera, obs.track_id, obs.floor, now)
                person.last_seen = now
                name = _known(obs.name)
                if name and person.name is None:
                    person.name, person.name_camera = name, camera
            self._match(camera, observations, now)
            self._forget_old(now)
            return {obs.track_id: self._info(self._links[(camera, obs.track_id)], now)
                    for obs in observations if (camera, obs.track_id) in self._links}

    def lookup(self, camera: str, track_id: int) -> GlobalInfo | None:
        with self._lock:
            gid = self._links.get((camera, track_id))
            return self._info(gid, self._clock()) if gid is not None else None

    def snapshot(self) -> list[dict]:
        """Everyone currently on screen somewhere, for the floor map."""
        now = self._clock()
        with self._lock:
            out = []
            for person in self._people.values():
                active = person.active(now)
                if not active:
                    continue
                position = person.position(now)
                out.append({
                    "gid": person.gid,
                    "name": person.name,
                    "name_camera": person.name_camera,
                    "x": None if position is None else round(position[0], 2),
                    "y": None if position is None else round(position[1], 2),
                    "cameras": sorted({s.camera for s in active}),
                })
            return sorted(out, key=lambda p: p["gid"])

    def forget_camera(self, camera: str) -> None:
        """Drop a stopped/restarted camera's tracks (its track ids restart)."""
        with self._lock:
            for key in [k for k in self._links if k[0] == camera]:
                gid = self._links.pop(key)
                person = self._people.get(gid)
                if person:
                    person.sightings.pop(key, None)
                    if not person.sightings:
                        del self._people[gid]

    # ---- internals -------------------------------------------------------------
    def _person_for(self, camera: str, track_id: int, now: float) -> GlobalPerson:
        gid = self._links.get((camera, track_id))
        if gid is None:
            gid = next(self._ids)
            self._people[gid] = GlobalPerson(gid, first_seen=now, last_seen=now)
            self._links[(camera, track_id)] = gid
        return self._people[gid]

    def _info(self, gid: int, now: float) -> GlobalInfo:
        person = self._people[gid]
        return GlobalInfo(gid, person.name, tuple(sorted({s.camera for s in person.active(now)})))

    def _candidates(self, camera: str, now: float):
        """Positions of people that OTHER cameras see right now: gid -> floor point."""
        out = {}
        for person in self._people.values():
            points = [s.floor for s in person.sightings.values()
                      if s.camera != camera and s.floor is not None and now - s.time <= MAX_TIME_GAP_S]
            if points:
                out[person.gid] = (sum(p[0] for p in points) / len(points),
                                   sum(p[1] for p in points) / len(points))
        return out

    def _match(self, camera: str, observations: list[Observation], now: float) -> None:
        mine = {self._links[(camera, o.track_id)]: o.floor for o in observations
                if o.track_id >= 0 and o.floor is not None}
        others = {gid: p for gid, p in self._candidates(camera, now).items() if gid not in mine}
        if not mine or not others:
            return
        distance = {(a, b): math.dist(pa, pb) for a, pa in mine.items() for b, pb in others.items()}
        nearest_other = {a: min(others, key=lambda b: distance[(a, b)]) for a in mine}
        nearest_mine = {b: min(mine, key=lambda a: distance[(a, b)]) for b in others}

        agreeing = set()
        for a, b in nearest_other.items():
            if nearest_mine[b] == a and distance[(a, b)] <= MATCH_RADIUS_M and self._may_merge(a, b, now):
                key = (min(a, b), max(a, b))
                self._votes[key] = self._votes.get(key, 0) + 1
                agreeing.add(key)
                if self._votes[key] >= MERGE_AFTER:
                    self._merge(*key)
        # A pair must agree in a row: forget votes of pairs involving this
        # camera's people that did not agree this time.
        for key in list(self._votes):
            if key not in agreeing and (key[0] in mine or key[1] in mine):
                del self._votes[key]

    def _may_merge(self, a: int, b: int, now: float) -> bool:
        pa, pb = self._people.get(a), self._people.get(b)
        if pa is None or pb is None:
            return False
        if _known(pa.name) and _known(pb.name) and pa.name != pb.name:
            return False  # two different recognised faces
        cams_a = {s.camera for s in pa.active(now)}
        cams_b = {s.camera for s in pb.active(now)}
        return not (cams_a & cams_b)  # one camera sees them as two people right now

    def _merge(self, keep: int, drop: int) -> None:
        kept, dropped = self._people[keep], self._people.pop(drop)
        for key, sighting in dropped.sightings.items():
            kept.sightings[key] = sighting
            self._links[key] = keep
        if kept.name is None and dropped.name:
            kept.name, kept.name_camera = dropped.name, dropped.name_camera
        kept.first_seen = min(kept.first_seen, dropped.first_seen)
        kept.last_seen = max(kept.last_seen, dropped.last_seen)
        self._votes = {k: v for k, v in self._votes.items() if drop not in k}

    def _forget_old(self, now: float) -> None:
        for gid in [g for g, p in self._people.items() if now - p.last_seen > FORGET_AFTER_S]:
            person = self._people.pop(gid)
            for key in person.sightings:
                self._links.pop(key, None)
            self._votes = {k: v for k, v in self._votes.items() if gid not in k}
