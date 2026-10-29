"""PostgreSQL storage (see DATABASE.md). Used only when config.DATABASE_URL is set.

Four tables:
    persons          registered people (soft delete with deleted_at)
    face_embeddings  one face vector per enrollment photo and face model
                     (128 numbers for SFace, 512 for ArcFace; column "model")
    cameras          one row per camera name
    detection_logs   one row per person logged, plus one per in/out crossing

Without DATABASE_URL the program keeps using people.json and the CSV files,
so nothing changes for anyone who has not set up PostgreSQL.
"""

from __future__ import annotations

import atexit
import datetime as dt
import queue
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy import (BigInteger, CheckConstraint, DateTime, ForeignKey, Index, Integer,
                        String, Text, create_engine, delete, func, insert, select, text, update)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

EMBEDDING_SIZE = 128  # SFace output
EMBEDDING_SIZES = (128, 512)  # SFace, ArcFace (vision/face_models.py)
DIRECTIONS = ("in", "out")


# ===================================================================== tables
class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # The name the cameras show and match on (e.g. "arm"); unique among active people.
    name: Mapped[str] = mapped_column(String(100))
    surname: Mapped[str] = mapped_column(String(100), default="", server_default="")
    # Student / staff / national ID. Optional: the web form may leave it empty.
    id_number: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("uq_persons_id_number_active", "id_number", unique=True,
              postgresql_where=text("deleted_at IS NULL AND id_number IS NOT NULL")),
        Index("uq_persons_name_active", "name", unique=True,
              postgresql_where=text("deleted_at IS NULL")),
    )


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(REAL))
    # Which face model made the vector: vectors of different models can't be compared.
    model: Mapped[str] = mapped_column(String(32), default="sface", server_default="sface")
    image_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(f"cardinality(embedding) IN {EMBEDDING_SIZES}", name="ck_face_embeddings_size"),
        Index("ix_face_embeddings_model_person", "model", "person_id"),
    )


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_name: Mapped[str] = mapped_column(String(100), unique=True)


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    person_id: Mapped[int | None] = mapped_column(ForeignKey("persons.id"))  # NULL = unknown
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"))
    # Best cosine similarity of this camera's face match. NULL = no face was
    # seen (e.g. back to the camera) or the name came from another camera.
    similarity: Mapped[float | None] = mapped_column(REAL)
    # NULL = a sighting; 'in' / 'out' = the person crossed the counting zones.
    direction: Mapped[str | None] = mapped_column(String(3))
    # Clothing colours, detection confidence, global id, where the name came from.
    attributes: Mapped[dict | None] = mapped_column(JSONB)
    snapshot_path: Mapped[str | None] = mapped_column(Text)
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("direction IN ('in', 'out')", name="ck_detection_logs_direction"),
        Index("ix_detection_logs_detected_at", "detected_at"),
        Index("ix_detection_logs_person_time", "person_id", "detected_at"),
        Index("ix_detection_logs_camera_time", "camera_id", "detected_at"),
    )


# =================================================================== database
class Database:
    """Engine + helpers shared by every camera thread."""

    def __init__(self, url: str, timezone: str = "Asia/Bangkok", echo: bool = False):
        self.url = url
        self.timezone = timezone
        # Every connection uses Thai time, so date_trunc('day', now()) and
        # the history page's "date" mean a Thai calendar day.
        self.engine = create_engine(url, echo=echo, pool_pre_ping=True,
                                    connect_args={"options": f"-c timezone={timezone}"})
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self._camera_ids: dict[str, int] = {}
        self._lock = threading.Lock()

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        self.upgrade()

    # Changes to tables made by older versions (create_all never alters a table).
    UPGRADES = (
        "ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS model varchar(32) NOT NULL DEFAULT 'sface'",
        "ALTER TABLE face_embeddings DROP CONSTRAINT IF EXISTS ck_face_embeddings_size",
        f"ALTER TABLE face_embeddings ADD CONSTRAINT ck_face_embeddings_size CHECK (cardinality(embedding) IN {EMBEDDING_SIZES})",
        "CREATE INDEX IF NOT EXISTS ix_face_embeddings_model_person ON face_embeddings (model, person_id)",
    )

    def upgrade(self) -> None:
        """Bring an older database up to date (safe to run again)."""
        with self.engine.begin() as conn:
            if conn.execute(text("SELECT to_regclass('face_embeddings')")).scalar() is None:
                return  # no tables yet: create_tables() makes them up to date
            has_model = conn.execute(text(
                "SELECT 1 FROM information_schema.columns WHERE table_name = 'face_embeddings'"
                " AND column_name = 'model' AND table_schema = current_schema()")).first()
            check = conn.execute(text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'ck_face_embeddings_size'")).scalar()
            if has_model and check and "512" in check:
                return
            for statement in self.UPGRADES:
                conn.execute(text(statement))

    def set_database_timezone(self) -> None:
        """Make psql and other tools use Thai time too (needs owner rights)."""
        if not re.fullmatch(r"[A-Za-z0-9_/+-]+", self.timezone):
            raise ValueError(f"bad time zone name: {self.timezone!r}")
        with self.engine.begin() as conn:
            name = conn.execute(text("SELECT current_database()")).scalar_one()
            quoted = name.replace('"', '""')
            conn.execute(text(f"ALTER DATABASE \"{quoted}\" SET timezone TO '{self.timezone}'"))

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def camera_id(self, name: str) -> int:
        """Id of a camera, inserting the row the first time a name is used."""
        name = (name or "default")[:100]
        with self._lock:
            if name in self._camera_ids:
                return self._camera_ids[name]
        with self.engine.begin() as conn:
            conn.execute(pg_insert(Camera).values(camera_name=name).on_conflict_do_nothing())
            camera_id = conn.execute(select(Camera.id).where(Camera.camera_name == name)).scalar_one()
        with self._lock:
            self._camera_ids[name] = camera_id
        return camera_id

    def dispose(self) -> None:
        self.engine.dispose()


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


# ================================================================== people
class SqlPeopleDB:
    """Same interface as PeopleDB (people.json), stored in PostgreSQL.

    Each enrollment photo is one face_embeddings row. Matching uses the mean
    of a person's photos, exactly like people.json did, so recognition gives
    the same results; the database just keeps the photos apart.
    The gallery is cached in memory and re-read when another program
    (enroll.py, the web page) changes it.
    """

    RELOAD_CHECK_SEC = 3.0

    def __init__(self, db: Database, model: str = "sface"):
        self.db = db
        self.model = model  # face model whose vectors this gallery holds
        self._lock = threading.RLock()
        self._people: dict[str, np.ndarray] = {}
        self._ids: dict[str, int] = {}
        self._version = None
        self._last_check = 0.0
        self._warned = False
        try:
            db.upgrade()  # an old database without face_embeddings.model
        except SQLAlchemyError as exc:
            print(f"Could not upgrade the face_embeddings table: {str(exc).splitlines()[0]}")
        self._load()

    # ---- cache ---------------------------------------------------------------
    _VERSION_SQL = text(
        "SELECT (SELECT count(*) FROM persons WHERE deleted_at IS NULL),"
        " (SELECT coalesce(max(id), 0) FROM persons),"
        " (SELECT max(deleted_at) FROM persons),"
        " (SELECT count(*) FROM face_embeddings),"
        " (SELECT coalesce(max(id), 0) FROM face_embeddings)")

    def _load(self) -> None:
        with self.db.engine.connect() as conn:
            version = tuple(conn.execute(self._VERSION_SQL).one())
            rows = conn.execute(
                select(Person.id, Person.name, FaceEmbedding.embedding)
                .join(FaceEmbedding, (FaceEmbedding.person_id == Person.id) & (FaceEmbedding.model == self.model),
                      isouter=True)
                .where(Person.deleted_at.is_(None))).all()
        vectors = defaultdict(list)
        ids = {}
        for person_id, name, embedding in rows:
            ids[name] = person_id
            if embedding is not None:
                vectors[name].append(embedding)
        people = {name: _unit(np.mean(np.asarray(v, dtype=np.float32), axis=0))
                  for name, v in vectors.items()}
        with self._lock:
            self._people, self._ids, self._version = people, ids, version

    def reload_if_changed(self) -> None:
        now = time.monotonic()
        if now - self._last_check < self.RELOAD_CHECK_SEC:
            return
        self._last_check = now
        try:
            with self.db.engine.connect() as conn:
                version = tuple(conn.execute(self._VERSION_SQL).one())
            if version != self._version:
                self._load()
            self._warned = False
        except SQLAlchemyError as exc:
            if not self._warned:  # keep matching with the last gallery
                print(f"Database not reachable, using the last known faces: {str(exc).splitlines()[0]}")
                self._warned = True

    # ---- PeopleDB interface ----------------------------------------------------
    def names(self) -> list[str]:
        self.reload_if_changed()
        return list(self._people.keys())

    def gallery(self) -> dict[str, np.ndarray]:
        self.reload_if_changed()
        return self._people

    def all_names(self) -> list[str]:
        """Every active person, also those without a vector for this face model."""
        self.reload_if_changed()
        return list(self._ids)

    def person_id(self, name: str | None) -> int | None:
        """persons.id of an active person, from the cache (no database call)."""
        if not name or name == "unknown":
            return None
        return self._ids.get(name)

    def enroll(self, name: str, embeddings: list[np.ndarray], image_paths=None,
               surname: str | None = None, id_number: str | None = None) -> None:
        """Create or update `name`; its photos are replaced by `embeddings`.

        The enroller always passes every photo of the person (old + new), so
        replacing keeps one row per photo with no duplicates.
        """
        if not embeddings:
            raise ValueError(f"No embeddings to enroll for '{name}'")
        paths = list(image_paths or [])
        paths += [None] * (len(embeddings) - len(paths))
        id_number = (id_number or "").strip() or None
        try:
            with self.db.Session.begin() as session:
                person = session.scalars(
                    select(Person).where(Person.name == name, Person.deleted_at.is_(None))
                    .with_for_update()).first()
                if person is None:
                    person = Person(name=name, surname=(surname or "").strip(), id_number=id_number)
                    session.add(person)
                    session.flush()
                else:
                    if surname is not None and surname.strip():
                        person.surname = surname.strip()
                    if id_number:
                        person.id_number = id_number
                session.flush()
                session.execute(delete(FaceEmbedding).where(FaceEmbedding.person_id == person.id,
                                                            FaceEmbedding.model == self.model))
                session.execute(insert(FaceEmbedding), [
                    {"person_id": person.id, "model": self.model, "embedding": [float(x) for x in _unit(e)],
                     "image_path": None if p is None else str(p)}
                    for e, p in zip(embeddings, paths)])
        except IntegrityError as exc:
            if "uq_persons_id_number_active" in str(exc):
                raise ValueError(f"รหัส {id_number} มีคนอื่นใช้อยู่แล้ว") from None
            raise
        self._last_check = 0.0
        self._load()

    def remove(self, name: str) -> None:
        """Soft delete: hide the person, keep their old log rows, and delete
        their face vectors (biometric data, PDPA section 26)."""
        with self.db.Session.begin() as session:
            ids = session.scalars(select(Person.id).where(
                Person.name == name, Person.deleted_at.is_(None))).all()
            if ids:
                session.execute(delete(FaceEmbedding).where(FaceEmbedding.person_id.in_(ids)))
                session.execute(update(Person).where(Person.id.in_(ids)).values(deleted_at=func.now()))
        self._load()

    def details(self) -> list[dict]:
        """Active people with surname, id number and photo count (for lists)."""
        with self.db.engine.connect() as conn:
            rows = conn.execute(
                select(Person.id, Person.name, Person.surname, Person.id_number, Person.created_at,
                       func.count(FaceEmbedding.id))
                .join(FaceEmbedding, (FaceEmbedding.person_id == Person.id) & (FaceEmbedding.model == self.model),
                      isouter=True)
                .where(Person.deleted_at.is_(None))
                .group_by(Person.id).order_by(Person.name)).all()
        return [{"id": r[0], "name": r[1], "surname": r[2], "id_number": r[3],
                 "created_at": r[4], "photos": r[5]} for r in rows]


# =============================================================== log writer
class DetectionLogWriter:
    """Writes detection_logs rows in batches from a background thread.

    add() only puts a dict on a queue, so a slow or stopped database never
    slows the video. If the database is down, rows wait (up to MAX_PENDING)
    and are written when it comes back; the CSV file is still written as
    a backup when SAVE_CSV is on.
    """

    BATCH_SIZE = 200
    FLUSH_EVERY_SEC = 1.0
    MAX_PENDING = 20000
    RETRY_SEC = 5.0

    def __init__(self, db: Database):
        self.db = db
        self._queue: queue.Queue = queue.Queue()
        self._pending: list[dict] = []
        self._idle = threading.Event()
        self._idle.set()
        self._stop = threading.Event()
        self._warned = False
        self.written = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="db-log-writer", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def add(self, *, camera: str, person_id: int | None, similarity: float | None = None,
            direction: str | None = None, attributes: dict | None = None,
            snapshot_path: str | None = None, detected_at: dt.datetime | None = None) -> None:
        if direction is not None and direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        self._idle.clear()
        self._queue.put({
            "camera": camera, "person_id": person_id,
            "similarity": None if similarity is None else float(similarity),
            "direction": direction, "attributes": attributes,
            "snapshot_path": None if snapshot_path is None else str(snapshot_path),
            "detected_at": detected_at or now_utc(),
        })

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until everything queued so far is written (for tests / exit)."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        if not self._stop.is_set():
            self.flush(5.0)
            self._stop.set()

    # ---- background thread --------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._pending.append(self._queue.get(timeout=self.FLUSH_EVERY_SEC))
                deadline = time.monotonic() + 0.2  # gather what arrives together
                while len(self._pending) < self.BATCH_SIZE and time.monotonic() < deadline:
                    try:
                        self._pending.append(self._queue.get(timeout=0.05))
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if self._pending and not self._write():
                time.sleep(self.RETRY_SEC)
                continue
            if not self._pending and self._queue.empty():
                self._idle.set()

    def _write(self) -> bool:
        batch = self._pending[: self.BATCH_SIZE]
        try:
            rows = []
            for row in batch:
                row = dict(row)
                row["camera_id"] = self.db.camera_id(row.pop("camera"))
                rows.append(row)
            with self.db.engine.begin() as conn:
                conn.execute(insert(DetectionLog), rows)
        except SQLAlchemyError as exc:
            if not self._warned:
                print(f"Database write failed, will retry: {str(exc).splitlines()[0]}")
                self._warned = True
            if len(self._pending) > self.MAX_PENDING:
                extra = len(self._pending) - self.MAX_PENDING
                del self._pending[:extra]  # oldest first; the CSV still has them
                self.dropped += extra
            return False
        del self._pending[: len(batch)]
        self.written += len(batch)
        self._warned = False
        return True


# ================================================================== queries
def log_dates(db: Database, limit: int = 120) -> list[str]:
    """Days (Thai time) that have log rows, newest first."""
    with db.engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT to_char(detected_at, 'YYYY-MM-DD') AS day FROM detection_logs"
            " ORDER BY day DESC LIMIT :limit"), {"limit": limit}).all()
    return [r[0] for r in rows]


def log_rows(db: Database, date: str, camera: str = "", name_query: str = "",
             limit: int | None = 500) -> tuple[list[dict], int, list[str]]:
    """Rows of one day: (rows newest first, total, camera names that day)."""
    day = dt.date.fromisoformat(date)
    zone = _zone(db.timezone)
    start = dt.datetime(day.year, day.month, day.day, tzinfo=zone)
    end = start + dt.timedelta(days=1)
    in_day = (DetectionLog.detected_at >= start, DetectionLog.detected_at < end)
    query = (select(DetectionLog, Camera.camera_name, Person.name, Person.surname, Person.id_number)
             .join(Camera, Camera.id == DetectionLog.camera_id)
             .join(Person, Person.id == DetectionLog.person_id, isouter=True)
             .where(*in_day))
    with db.Session() as session:
        cameras = list(session.scalars(
            select(Camera.camera_name).join(DetectionLog, DetectionLog.camera_id == Camera.id)
            .where(*in_day).distinct().order_by(Camera.camera_name)).all())
        if camera:
            query = query.where(Camera.camera_name == camera)
        if name_query:
            pattern = f"%{name_query}%"
            query = query.where(func.coalesce(Person.name + " " + Person.surname, "unknown").ilike(pattern))
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        query = query.order_by(DetectionLog.detected_at.desc())
        if limit:
            query = query.limit(limit)
        rows = []
        for log, camera_name, name, surname, id_number in session.execute(query).all():
            attrs = log.attributes or {}
            local = log.detected_at.astimezone(zone)
            rows.append({
                "id": log.id,
                "timestamp": local.strftime("%Y-%m-%d %H:%M:%S"),
                "camera": camera_name,
                "name": name or "unknown",
                "surname": surname or "",
                "id_number": id_number or "",
                "similarity": "" if log.similarity is None else f"{log.similarity:.2f}",
                "direction": log.direction or "",
                "shirt_color": attrs.get("shirt_color", ""),
                "pants_color": attrs.get("pants_color", ""),
                "confidence": attrs.get("confidence", ""),
                "global_id": attrs.get("global_id", ""),
                "track_id": attrs.get("track_id", ""),
                "snapshot_path": log.snapshot_path or "",
            })
    return rows, total, cameras


def _zone(name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # no tz database on this machine: Thailand is UTC+7 all year
        return dt.timezone(dt.timedelta(hours=7))


def purge_older_than(db: Database, days: int, delete_snapshots: bool = True) -> tuple[int, int]:
    """Delete log rows (and their snapshot files) older than `days` days."""
    cutoff = now_utc() - dt.timedelta(days=days)
    with db.engine.begin() as conn:
        paths = conn.execute(select(DetectionLog.snapshot_path).where(
            DetectionLog.detected_at < cutoff, DetectionLog.snapshot_path.is_not(None))).scalars().all()
        count = conn.execute(delete(DetectionLog).where(DetectionLog.detected_at < cutoff)).rowcount
    removed = 0
    if delete_snapshots:
        for path in paths:
            try:
                Path(path).unlink()
                removed += 1
            except OSError:
                pass
    return count, removed
