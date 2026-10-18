"""Connects one camera's FrameProcessor to the shared IdentityHub."""

from __future__ import annotations

from .floor import FloorCalibration
from .hub import GlobalInfo, IdentityHub, Observation, _known


class CameraLink:
    """Called by the FrameProcessor after each frame (see pipeline.py).

    Converts each person's foot point to floor metres, reports to the hub,
    and writes back the global id and any name another camera recognised.
    """

    def __init__(self, hub: IdentityHub, camera: str, calibration: FloorCalibration | None = None):
        self.hub = hub
        self.camera = camera
        self.calibration = calibration  # may be replaced while running (re-calibrated)

    def __call__(self, result) -> None:
        calibration = self.calibration
        observations = []
        for person in result.people:
            if person.track_id < 0:
                continue
            floor = None
            if calibration is not None and person.foot is not None:
                floor = calibration.to_floor(person.foot)
                person.floor = (round(floor[0], 2), round(floor[1], 2))
            observations.append(Observation(person.track_id, floor, person.name))
        infos = self.hub.update(self.camera, observations)
        for person in result.people:
            info = infos.get(person.track_id)
            if info is None:
                continue
            person.global_id = info.gid
            person.cameras = list(info.cameras)
            if not _known(person.name) and info.name:
                person.name = info.name
                person.name_from_other_camera = True

    def lookup(self, track_id: int) -> GlobalInfo | None:
        return self.hub.lookup(self.camera, track_id)

    def close(self) -> None:
        self.hub.forget_camera(self.camera)
