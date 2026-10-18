"""Enroll people for face recognition from a few photos.

Used by the web portal ("คนที่ลงทะเบียน" page). Same idea as enroll.py, with
extra checks because the photos come from anywhere (phones, any resolution):

* phone photos are turned upright using their EXIF orientation;
* very large photos are scaled down (long side 1600 px) before use;
* in each photo the biggest confident face is used, and a small crop of it is
  shown back so the user can see which face was picked;
* photos are compared pair by pair; only the biggest group that looks like
  one person is used (catches a wrong face being picked, or someone else);
* a warning is given if the new person looks like someone already enrolled.

Photos are kept in models/photos/<name>/ (the folder capture_faces.py uses),
so adding photos later re-computes the face from all of them.
"""

from __future__ import annotations

import io
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .core.geometry import Box, clip_box, scale_to_long_side
from .storage.people_db import PeopleDB
from .vision.face import FaceDetector, FaceRecognizer

MAX_PHOTO_SIDE = 1600  # photos are stored and processed at most this big
MIN_FACE_PX = 40  # faces narrower than this are too small to recognise well
GOOD_FACE_SCORE = 0.8  # prefer faces the detector is this sure about
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_NAME_LEN = 40

_NAME_CHARS = re.compile(r"^[A-Za-z0-9ก-๙ _.\-]+$")


class EnrollError(ValueError):
    """Something the user can fix (bad name, no usable photo...)."""


def clean_name(name: str) -> str:
    """Check a person's name; it is also used as a folder name."""
    name = " ".join((name or "").split())
    if not name:
        raise EnrollError("กรุณาใส่ชื่อ")
    if len(name) > MAX_NAME_LEN:
        raise EnrollError(f"ชื่อยาวเกิน {MAX_NAME_LEN} ตัวอักษร")
    if not _NAME_CHARS.match(name) or name.strip(".") == "" or name[0] in ".-" or name[-1] == ".":
        raise EnrollError("ชื่อใช้ได้เฉพาะตัวอักษรไทย/อังกฤษ ตัวเลข เว้นวรรค และ _ - .")
    return name


def load_photo(data: bytes) -> np.ndarray | None:
    """Bytes of a JPG/PNG/WebP/BMP -> upright BGR array, or None if unreadable."""
    # OpenCV first: it rejects non-images quietly. (ultralytics patches PIL so
    # that a failed open tries to pip-install a HEIC plugin at run time.)
    array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        return None
    try:
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)  # phone photos taken sideways
            rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return array


def _thumbnail(image: np.ndarray, box: Box, height: int = 112) -> bytes:
    """Small JPEG of the face (with some margin) to show the user."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) // 4, (y2 - y1) // 4
    x1, y1, x2, y2 = clip_box(image, (x1 - mx, y1 - my, x2 + mx, y2 + my))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return b""
    scale = height / crop.shape[0]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


@dataclass
class PhotoReport:
    """What happened to one photo."""

    filename: str
    ok: bool
    message: str
    is_new: bool = True  # False = a photo that was already in the folder
    thumbnail: bytes = b""  # the face that was used
    similarity: float | None = None  # how alike this photo is to the others
    image: np.ndarray | None = field(default=None, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class EnrollResult:
    name: str
    reports: list[PhotoReport]
    used: int
    saved: list[Path]
    looks_like: tuple[str, float] | None = None  # (other person, similarity)


class FaceEnroller:
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer,
                 people_db: PeopleDB, photos_root: str | Path):
        self.detector = detector
        self.recognizer = recognizer
        self.people_db = people_db
        self.photos_root = Path(photos_root)
        self._lock = threading.Lock()  # one enrollment at a time

    # ---- photos on disk ------------------------------------------------------
    def person_dir(self, name: str) -> Path:
        folder = (self.photos_root / clean_name(name)).resolve()
        if folder.parent != self.photos_root.resolve():
            raise EnrollError("ชื่อไม่ถูกต้อง")
        return folder

    def photos_of(self, name: str) -> list[Path]:
        try:
            folder = self.person_dir(name)
        except EnrollError:
            return []
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES)

    # ---- one photo -------------------------------------------------------------
    def _read_face(self, report: PhotoReport, data: bytes) -> PhotoReport:
        image = load_photo(data)
        if image is None:
            report.ok, report.message = False, "เปิดไฟล์รูปไม่ได้ (ใช้ JPG หรือ PNG)"
            return report
        if max(image.shape[:2]) > MAX_PHOTO_SIDE:
            image = scale_to_long_side(image, MAX_PHOTO_SIDE)
        report.image = image

        faces = self.detector.detect_raw(image)
        if not faces:
            report.ok, report.message = False, "ไม่พบใบหน้าในรูปนี้"
            return report
        confident = [f for f in faces if float(f[1][14]) >= GOOD_FACE_SCORE] or faces
        box, raw_row = max(confident, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        report.thumbnail = _thumbnail(image, box)
        if box[2] - box[0] < MIN_FACE_PX:
            report.ok, report.message = False, "ใบหน้าเล็กเกินไป ใช้รูปที่ถ่ายใกล้กว่านี้"
            return report

        report.embedding = self.recognizer.embed(image, raw_row)
        report.ok = True
        report.message = "ใช้ได้" if len(faces) == 1 else f"พบ {len(faces)} ใบหน้า ใช้ใบที่ใหญ่ที่สุด (ดูรูปเล็ก)"
        return report

    # ---- enroll / remove -----------------------------------------------------------
    def enroll(self, name: str, uploads: list[tuple[str, bytes]]) -> EnrollResult:
        """Add photos for `name` (new or existing) and update people.json."""
        name = clean_name(name)
        if not uploads:
            raise EnrollError("กรุณาเลือกรูปอย่างน้อย 1 รูป")
        return self._enroll(name, uploads)

    def enroll_folder(self, name: str) -> EnrollResult:
        """Re-compute a person from the photos already in their folder."""
        name = clean_name(name)
        if not self.photos_of(name):
            raise EnrollError(f"ไม่มีรูปใน {self.person_dir(name)}")
        return self._enroll(name, [])

    def _enroll(self, name: str, uploads: list[tuple[str, bytes]]) -> EnrollResult:
        with self._lock:
            reports = [self._read_face(PhotoReport(filename, False, ""), data) for filename, data in uploads]
            for path in self.photos_of(name):  # photos added earlier
                reports.append(self._read_face(PhotoReport(path.name, False, "", is_new=False), path.read_bytes()))

            usable = [r for r in reports if r.ok]
            if not usable:
                raise EnrollError("ไม่มีรูปที่ใช้ได้เลย: " + "; ".join(f"{r.filename}: {r.message}" for r in reports))
            self._drop_odd_ones_out(usable)
            usable = [r for r in usable if r.ok]
            if not usable:
                raise EnrollError("รูปที่ส่งมาดูไม่ใช่คนเดียวกัน กรุณาใช้รูปของคนคนเดียว")

            embeddings = [r.embedding for r in usable]
            looks_like = self._closest_other_person(name, embeddings)
            saved = self._save_new_photos(name, [r for r in usable if r.is_new])
            self.people_db.enroll(name, embeddings)
            return EnrollResult(name, reports, len(usable), saved, looks_like)

    def _drop_odd_ones_out(self, usable: list[PhotoReport]) -> None:
        """Keep only the biggest group of photos that look like the same person.

        Every pair of photos is compared (the same 1:1 test the cameras use).
        The photo that matches the most others is the anchor; photos that do
        not match the anchor are left out. If the left-out photos are not
        clearly fewer than the kept ones, the set is too mixed to trust.
        """
        threshold = self.recognizer.match_threshold
        count = len(usable)
        if count < 2:
            return
        sims = np.ones((count, count), np.float32)
        for i in range(count):
            for j in range(i + 1, count):
                sims[i, j] = sims[j, i] = self.recognizer.similarity(usable[i].embedding, usable[j].embedding)

        if count == 2:
            usable[0].similarity = usable[1].similarity = round(float(sims[0, 1]), 2)
            if sims[0, 1] < threshold:
                raise EnrollError("สองรูปนี้ดูไม่ใช่คนเดียวกัน กรุณาใช้รูปของคนคนเดียว")
            return

        matches = (sims >= threshold).sum(axis=1) - 1  # minus itself
        anchor = max(range(count), key=lambda i: (matches[i], sims[i].sum()))
        keep = [i for i in range(count) if i == anchor or sims[anchor, i] >= threshold]
        if len(keep) * 2 <= count:  # no clear majority
            raise EnrollError("รูปที่ส่งมามีหน้าหลายคนปนกัน (ดูไม่ออกว่าคนไหนคือ "
                              "คนที่จะลงทะเบียน) กรุณาใช้รูปที่มีคนเดียว")
        for i, report in enumerate(usable):
            others = [sims[i, j] for j in keep if j != i]
            report.similarity = round(float(np.mean(others)), 2) if others else None
            if i not in keep:
                report.ok = False
                report.message = "ไม่เหมือนรูปอื่น อาจเป็นคนละคนหรือเลือกหน้าผิด — ไม่ได้ใช้"

    def _closest_other_person(self, name: str, embeddings) -> tuple[str, float] | None:
        mean = np.mean(np.stack(embeddings), axis=0)
        mean = (mean / np.linalg.norm(mean)).astype(np.float32)
        best = None
        for other, reference in self.people_db.gallery().items():
            if other == name:
                continue
            score = self.recognizer.similarity(mean, reference)
            if score >= self.recognizer.match_threshold and (best is None or score > best[1]):
                best = (other, round(score, 2))
        return best

    def _save_new_photos(self, name: str, reports: list[PhotoReport]) -> list[Path]:
        folder = self.person_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = []
        for index, report in enumerate(reports, start=1):
            path = folder / f"web_{stamp}_{index}.jpg"
            while path.exists():
                index += 100
                path = folder / f"web_{stamp}_{index}.jpg"
            cv2.imwrite(str(path), report.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(path)
        return saved

    def remove(self, name: str) -> None:
        """Forget a person: remove from people.json and delete their photos."""
        with self._lock:
            self.people_db.remove(name)
            try:
                folder = self.person_dir(name)
            except EnrollError:
                return
            if folder.is_dir():
                shutil.rmtree(folder)
