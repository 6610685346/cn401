"""Camera picture <-> floor plan, using a homography (a flat-floor mapping).

Calibration: put 4 or more marks on the floor (e.g. tape at the corners of a
2 m x 3 m rectangle), measure where they are in metres from one corner of the
room, click them in each camera's picture and type the same metres. Cameras
that use the same marks / same room corner share one floor coordinate system,
which is what lets the hub say "these two cameras see the same person".

Picture points are 0..1 fractions of the processed frame, so the calibration
keeps working whatever the camera resolution (but not if the camera is moved
or the picture rotation setting is changed).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_POINTS = 4
MAX_ERROR_M = 0.5  # average error above this is almost surely a typing/click mistake


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class FloorCalibration:
    image_points: tuple[tuple[float, float], ...]  # 0..1 in the picture
    floor_points: tuple[tuple[float, float], ...]  # metres on the floor
    matrix: tuple[tuple[float, ...], ...]  # 3x3, picture (0..1) -> floor (m)
    error_m: float  # average distance between typed and fitted floor points
    point_errors_m: tuple[float, ...]

    # ---- building ------------------------------------------------------------
    @classmethod
    def fit(cls, image_points, floor_points) -> "FloorCalibration":
        image = np.asarray(image_points, dtype=np.float64)
        floor = np.asarray(floor_points, dtype=np.float64)
        if image.ndim != 2 or image.shape[1] != 2 or floor.shape != image.shape:
            raise CalibrationError("จำนวนจุดในภาพกับจุดบนพื้นต้องเท่ากัน")
        if len(image) < MIN_POINTS:
            raise CalibrationError(f"ต้องมีอย่างน้อย {MIN_POINTS} จุด")
        if not np.isfinite(image).all() or not np.isfinite(floor).all():
            raise CalibrationError("มีค่าที่ไม่ใช่ตัวเลข")
        if (image < -0.01).any() or (image > 1.01).any():
            raise CalibrationError("จุดในภาพต้องอยู่ในภาพ")
        for name, pts, min_area in (("ในภาพ", image, 1e-3), ("บนพื้น", floor, 0.05)):
            if _spread_area(pts) < min_area:
                raise CalibrationError(f"จุด{name}อยู่ชิดกันหรือเรียงเป็นเส้นตรงเกินไป ให้กระจายเป็นรูปสี่เหลี่ยม")

        matrix, _ = cv2.findHomography(image, floor, 0)
        if matrix is None or not np.isfinite(matrix).all():
            raise CalibrationError("คำนวณไม่ได้ ตรวจลำดับจุดกับค่าเมตรอีกครั้ง")
        fitted = _apply(matrix, image)
        errors = np.linalg.norm(fitted - floor, axis=1)
        if errors.mean() > MAX_ERROR_M:
            raise CalibrationError(
                f"จุดไม่สอดคล้องกัน (คลาดเฉลี่ย {errors.mean():.2f} ม.) "
                "มักเกิดจากคลิกจุดผิดลำดับ หรือพิมพ์เมตรผิด")
        if not _points_consistent(matrix, image, floor):
            raise CalibrationError("ลำดับจุดไม่ตรงกัน: จุดที่ 1 ในภาพต้องเป็นจุดเดียวกับเมตรของจุดที่ 1 "
                                   "(มักเกิดจากสลับลำดับตอนพิมพ์เมตร)")
        return cls(
            tuple(map(tuple, image.round(5).tolist())),
            tuple(map(tuple, floor.round(4).tolist())),
            tuple(map(tuple, matrix.tolist())),
            round(float(errors.mean()), 3),
            tuple(round(float(e), 3) for e in errors),
        )

    # ---- storing -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"image": self.image_points, "floor": self.floor_points, "matrix": self.matrix,
                "error_m": self.error_m, "point_errors_m": self.point_errors_m}

    @classmethod
    def from_dict(cls, data: dict | None) -> "FloorCalibration | None":
        if not data:
            return None
        try:
            return cls(
                tuple(map(tuple, data["image"])), tuple(map(tuple, data["floor"])),
                tuple(map(tuple, data["matrix"])), float(data["error_m"]),
                tuple(data.get("point_errors_m", ())),
            )
        except (KeyError, TypeError, ValueError):
            return None

    # ---- using ---------------------------------------------------------------
    def to_floor(self, point: tuple[float, float]) -> tuple[float, float]:
        """0..1 picture point -> (x, y) metres."""
        x, y = _apply(np.asarray(self.matrix), np.asarray([point], dtype=np.float64))[0]
        return float(x), float(y)

    def to_image(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """(x, y) metres -> 0..1 picture point (None if behind the camera)."""
        inverse = np.linalg.inv(np.asarray(self.matrix))
        x, y, w = inverse @ np.array([point[0], point[1], 1.0])
        if w <= 1e-9:
            return None
        return float(x / w), float(y / w)

    def grid_lines(self, step_m: float = 1.0, margin_m: float = 2.0) -> list[list[tuple[float, float]]]:
        """Floor grid drawn back onto the picture, to check the calibration by eye."""
        floor = np.asarray(self.floor_points)
        x0, y0 = np.floor(floor.min(axis=0) - margin_m)
        x1, y1 = np.ceil(floor.max(axis=0) + margin_m)
        lines = []
        for x in np.arange(x0, x1 + 1e-9, step_m):
            lines.append(self._polyline([(x, y) for y in np.linspace(y0, y1, 24)]))
        for y in np.arange(y0, y1 + 1e-9, step_m):
            lines.append(self._polyline([(x, y) for x in np.linspace(x0, x1, 24)]))
        return [line for line in lines if len(line) >= 2]

    def _polyline(self, floor_points) -> list[tuple[float, float]]:
        out = []
        for point in floor_points:
            image = self.to_image(point)
            if image is not None and -0.5 <= image[0] <= 1.5 and -0.5 <= image[1] <= 1.5:
                out.append((round(image[0], 4), round(image[1], 4)))
        return out


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), np.asarray(matrix)).reshape(-1, 2)


def _spread_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def _points_consistent(matrix: np.ndarray, image: np.ndarray, floor: np.ndarray) -> bool:
    """Catch points typed in the wrong order.

    With exactly 4 points any order "fits" perfectly, so the error alone can't
    tell. But a real camera keeps the shape: going round the picture points in
    order must also go round the floor points in order (a convex shape stays a
    convex shape, not a bow-tie), and no point may fall "behind" the camera.
    """
    w = (np.c_[image, np.ones(len(image))] @ np.asarray(matrix).T)[:, 2]
    if not (np.all(w > 0) or np.all(w < 0)):
        return False
    hull = cv2.convexHull(image.astype(np.float32), returnPoints=False).flatten()
    if len(hull) < 3:
        return False
    ring = floor[hull]
    turns = []
    for i in range(len(ring)):
        a, b, c = ring[i], ring[(i + 1) % len(ring)], ring[(i + 2) % len(ring)]
        turns.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(t > 0 for t in turns) or all(t < 0 for t in turns)
