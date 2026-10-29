"""PostgreSQL storage (DATABASE.md). Needs a throw-away database:

    set TEST_DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/person_detection_test
    python -m unittest tests.test_database -v

Every test drops and re-creates the tables, so never point it at the real database.
Skipped when TEST_DATABASE_URL is not set.
"""

import csv
import datetime as dt
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection import config  # noqa: E402
from person_detection.storage.people_db import PeopleDB  # noqa: E402
from person_detection.storage.recorder import FaceIdentifier, PersonRecorder, TrackGate  # noqa: E402
from person_detection.vision.colors import Appearance  # noqa: E402
from person_detection.vision.detector import Detection  # noqa: E402

URL = os.environ.get("TEST_DATABASE_URL", "")
if URL:
    from sqlalchemy import func, select, text

    from person_detection.storage import database as dbm


def vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return v / np.linalg.norm(v)


@unittest.skipUnless(URL, "set TEST_DATABASE_URL to run the PostgreSQL tests")
class DbTest(unittest.TestCase):
    def setUp(self):
        self.db = dbm.Database(URL)
        dbm.Base.metadata.drop_all(self.db.engine)
        self.db.create_tables()
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.db.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def count(self, model, *where):
        with self.db.Session() as s:
            return s.scalar(select(func.count()).select_from(model).where(*where))


class PeopleTest(DbTest):
    def test_same_gallery_as_people_json(self):
        photos = [vec(1), vec(2), vec(3)]
        sql = dbm.SqlPeopleDB(self.db)
        sql.enroll("arm", photos, image_paths=["a.jpg", "b.jpg", "c.jpg"], surname="Penwong", id_number="6610685015")
        js = PeopleDB(self.tmp / "people.json")
        js.enroll("arm", photos)
        np.testing.assert_allclose(sql.gallery()["arm"], js.gallery()["arm"], atol=1e-5)
        self.assertEqual(self.count(dbm.FaceEmbedding), 3)  # one row per photo
        self.assertEqual(sql.details()[0]["id_number"], "6610685015")
        self.assertIsNotNone(sql.person_id("arm"))
        self.assertIsNone(sql.person_id("unknown"))

    def test_re_enroll_replaces_photos(self):
        sql = dbm.SqlPeopleDB(self.db)
        sql.enroll("arm", [vec(1), vec(2)])
        first_id = sql.person_id("arm")
        sql.enroll("arm", [vec(1), vec(2), vec(3)])  # enroller always passes every photo
        self.assertEqual(sql.person_id("arm"), first_id)
        self.assertEqual(self.count(dbm.FaceEmbedding), 3)
        self.assertEqual(self.count(dbm.Person), 1)

    def test_id_number_must_be_unique(self):
        sql = dbm.SqlPeopleDB(self.db)
        sql.enroll("arm", [vec(1)], id_number="111")
        with self.assertRaises(ValueError):
            sql.enroll("oung", [vec(2)], id_number="111")
        self.assertEqual(sql.names(), ["arm"])  # nothing half-written

    def test_soft_delete_keeps_logs_and_drops_vectors(self):
        sql = dbm.SqlPeopleDB(self.db)
        sql.enroll("arm", [vec(1), vec(2)], id_number="111")
        old_id = sql.person_id("arm")
        writer = dbm.DetectionLogWriter(self.db)
        writer.add(camera="cam1", person_id=old_id, similarity=0.8)
        self.assertTrue(writer.flush())
        sql.remove("arm")
        self.assertEqual(sql.names(), [])
        self.assertEqual(self.count(dbm.FaceEmbedding), 0)  # biometric data gone
        self.assertEqual(self.count(dbm.DetectionLog, dbm.DetectionLog.person_id == old_id), 1)
        sql.enroll("arm", [vec(3)], id_number="111")  # same name and ID may come back
        self.assertNotEqual(sql.person_id("arm"), old_id)
        writer.close()

    def test_other_program_changes_are_picked_up(self):
        camera_side = dbm.SqlPeopleDB(self.db)
        dbm.SqlPeopleDB(dbm.Database(URL)).enroll("arm", [vec(1)])  # e.g. enroll.py
        camera_side._last_check = 0
        self.assertEqual(camera_side.names(), ["arm"])


class FaceModelTest(DbTest):
    """One person, a vector per face model (SFace 128 / ArcFace 512)."""

    def test_galleries_per_model(self):
        sface = dbm.SqlPeopleDB(self.db)
        arc = dbm.SqlPeopleDB(self.db, "arcface_mbf")
        sface.enroll("arm", [vec(1), vec(2)], surname="Penwong", id_number="111")
        arc._last_check = 0
        self.assertEqual(arc.names(), [])  # no ArcFace vector yet...
        self.assertEqual(arc.all_names(), ["arm"])  # ...but the person exists
        arc.enroll("arm", [np.ones(512, np.float32)] * 2)  # e.g. GalleryBuilder from the photos
        self.assertEqual(arc.person_id("arm"), sface.person_id("arm"))  # same person row
        self.assertEqual(self.count(dbm.Person), 1)
        self.assertEqual(arc.details()[0]["surname"], "Penwong")  # kept
        self.assertEqual(arc.details()[0]["photos"], 2)
        sface._last_check = 0
        self.assertEqual(sface.gallery()["arm"].shape, (128,))
        self.assertEqual(arc.gallery()["arm"].shape, (512,))
        sface.enroll("arm", [vec(3)])  # re-enrolling SFace leaves the ArcFace rows alone
        self.assertEqual(self.count(dbm.FaceEmbedding, dbm.FaceEmbedding.model == "arcface_mbf"), 2)
        with self.assertRaises(Exception):  # a vector of a size no model makes
            arc.enroll("oung", [np.ones(300, np.float32)])
        arc.remove("arm")  # removing deletes every model's vectors
        self.assertEqual(self.count(dbm.FaceEmbedding), 0)

    def test_upgrade_an_old_database(self):
        with self.db.engine.begin() as conn:  # the tables as the first version made them
            conn.execute(text("ALTER TABLE face_embeddings DROP COLUMN model"))
            conn.execute(text("ALTER TABLE face_embeddings DROP CONSTRAINT ck_face_embeddings_size"))
            conn.execute(text("ALTER TABLE face_embeddings ADD CONSTRAINT ck_face_embeddings_size"
                              " CHECK (cardinality(embedding) = 128)"))
            conn.execute(text("INSERT INTO persons (name) VALUES ('arm')"))
            conn.execute(text("INSERT INTO face_embeddings (person_id, embedding)"
                              " SELECT id, array_fill(0.1::real, ARRAY[128]) FROM persons"))
        arc = dbm.SqlPeopleDB(self.db, "arcface_r50")  # upgrades on start
        self.assertEqual(dbm.SqlPeopleDB(self.db).names(), ["arm"])  # old rows became "sface"
        arc.enroll("arm", [np.ones(512, np.float32)])
        self.assertEqual(arc.names(), ["arm"])
        self.db.upgrade()  # again: nothing to do


class LogTest(DbTest):
    def test_writer_batches_rows_and_creates_cameras(self):
        writer = dbm.DetectionLogWriter(self.db)
        for i in range(250):
            writer.add(camera="Gate A" if i % 2 else "cam2", person_id=None, attributes={"i": i})
        self.assertTrue(writer.flush())
        self.assertEqual(self.count(dbm.DetectionLog), 250)
        self.assertEqual(self.count(dbm.Camera), 2)
        with self.assertRaises(ValueError):
            writer.add(camera="cam1", person_id=None, direction="up")
        writer.close()

    def test_database_rejects_bad_direction(self):
        camera_id = self.db.camera_id("cam1")
        with self.assertRaises(Exception):
            with self.db.engine.begin() as conn:
                conn.execute(dbm.insert(dbm.DetectionLog).values(
                    camera_id=camera_id, direction="up", detected_at=dbm.now_utc()))

    def test_writer_waits_while_database_is_down(self):
        down = dbm.Database("postgresql+psycopg://nobody@127.0.0.1:1/none?connect_timeout=1")
        writer = dbm.DetectionLogWriter(down)
        writer.RETRY_SEC = 0.1
        writer.add(camera="cam1", person_id=None)
        self.assertFalse(writer.flush(1.5))  # not written, but nothing crashed
        self.assertEqual(writer.written, 0)
        writer.db = self.db  # database is back
        self.assertTrue(writer.flush(10))
        self.assertEqual(self.count(dbm.DetectionLog), 1)
        writer.close()

    def test_thai_calendar_day(self):
        writer = dbm.DetectionLogWriter(self.db)
        bangkok = dt.timezone(dt.timedelta(hours=7))
        # 23:30 Thai time on the 24th is 16:30 UTC; 00:30 on the 25th is 17:30 UTC on the 24th
        writer.add(camera="cam1", person_id=None, detected_at=dt.datetime(2026, 10, 24, 23, 30, tzinfo=bangkok))
        writer.add(camera="cam1", person_id=None, detected_at=dt.datetime(2026, 10, 25, 0, 30, tzinfo=bangkok))
        writer.flush()
        self.assertEqual(dbm.log_dates(self.db), ["2026-10-25", "2026-10-24"])
        rows, total, cameras = dbm.log_rows(self.db, "2026-10-24")
        self.assertEqual((total, rows[0]["timestamp"], cameras), (1, "2026-10-24 23:30:00", ["cam1"]))
        with self.db.engine.connect() as conn:  # SQL in reports uses Thai days too
            self.assertEqual(conn.execute(text("SELECT current_setting('TimeZone')")).scalar(), "Asia/Bangkok")
        writer.close()

    def test_purge(self):
        writer = dbm.DetectionLogWriter(self.db)
        snap = self.tmp / "old.jpg"
        snap.write_bytes(b"x")
        writer.add(camera="cam1", person_id=None, snapshot_path=str(snap),
                   detected_at=dbm.now_utc() - dt.timedelta(days=100))
        writer.add(camera="cam1", person_id=None)
        writer.flush()
        self.assertEqual(dbm.purge_older_than(self.db, 90), (1, 1))
        self.assertFalse(snap.exists())
        self.assertEqual(self.count(dbm.DetectionLog), 1)
        writer.close()


class FakeRecognizer:
    match_threshold = 0.363

    def embed(self, frame, raw_row):
        return vec(1)

    def similarity(self, a, b):
        return 0.81  # every registered face looks like this one

    def best_match(self, embedding, gallery):
        return ("arm", 0.81) if "arm" in gallery else (None, 0.12)


class FakeReader:
    def read(self, frame, box):
        return Appearance("red", "blue")


class RecorderTest(DbTest):
    def recorder(self, people, writer, identity_lookup=None):
        return PersonRecorder(
            None, gate=TrackGate(1), reader=FakeReader(),
            face_identifier=FaceIdentifier(FakeRecognizer(), people),
            camera="cam1", db_log=writer, person_ids=people.person_id,
            identity_lookup=identity_lookup)

    def test_sighting_and_crossing_rows(self):
        people = dbm.SqlPeopleDB(self.db)
        people.enroll("arm", [vec(1)])
        writer = dbm.DetectionLogWriter(self.db)
        recorder = self.recorder(people, writer)
        frame = np.zeros((480, 640, 3), np.uint8)
        person = Detection((100, 50, 300, 470), 4, 0.9)
        recorder.observe(frame, person, [((150, 60, 250, 160), np.zeros(15, np.float32))])
        recorder.record_direction(frame, person, "in")
        stranger = Detection((350, 50, 550, 470), 5, 0.8)  # back to the camera: no face
        recorder.observe(frame, stranger, [])
        writer.flush()
        rows, total, _ = dbm.log_rows(self.db, dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date().isoformat())
        self.assertEqual(total, 3)
        by_kind = {(r["track_id"], r["direction"]): r for r in rows}
        sighting = by_kind[(4, "")]
        self.assertEqual((sighting["name"], sighting["similarity"], sighting["shirt_color"]), ("arm", "0.81", "red"))
        self.assertEqual(by_kind[(4, "in")]["name"], "arm")
        unknown = by_kind[(5, "")]
        self.assertEqual((unknown["name"], unknown["similarity"]), ("unknown", ""))  # NULL: no face seen
        writer.close()

    def test_name_from_other_camera_has_no_similarity(self):
        people = dbm.SqlPeopleDB(self.db)
        people.enroll("arm", [vec(1)])
        writer = dbm.DetectionLogWriter(self.db)

        class Info:
            gid, name = 7, "arm"
        recorder = self.recorder(people, writer, identity_lookup=lambda track_id: Info)
        recorder.observe(np.zeros((480, 640, 3), np.uint8), Detection((100, 50, 300, 470), 9, 0.9), [])
        writer.flush()
        with self.db.Session() as s:
            log = s.scalars(select(dbm.DetectionLog)).one()
        self.assertEqual(log.person_id, people.person_id("arm"))
        self.assertIsNone(log.similarity)
        self.assertEqual((log.attributes["name_source"], log.attributes["global_id"]), ("other_camera", "G7"))
        writer.close()


class RegisterCliTest(DbTest):
    def setUp(self):
        super().setUp()
        self._url = config.DATABASE_URL
        config.DATABASE_URL = URL

    def tearDown(self):
        config.DATABASE_URL = self._url
        super().tearDown()

    def test_import_people_and_csv_twice(self):
        import json

        import register
        (self.tmp / "people.json").write_text(json.dumps({"arm": vec(1).tolist()}), encoding="utf-8")
        register.main(["import-people", "--file", str(self.tmp / "people.json")])
        log = self.tmp / "person_log_2026-10-20_cam-Gate.csv"
        with open(log, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "camera", "track_id", "name", "shirt_color", "pants_color", "confidence", "global_id"])
            w.writerow(["2026-10-20 08:00:01", "Gate", "3", "arm", "red", "blue", "0.91", ""])
            w.writerow(["2026-10-20 08:00:05", "Gate", "4", "unknown", "black", "black", "0.70", ""])
        old = self.tmp / "person_log_2026-10-19.csv"  # first version: no camera / name columns
        old.write_text("timestamp,track_id,shirt_color,pants_color,confidence\n2026-10-19 09:00:00,1,red,blue,0.5\n",
                       encoding="utf-8-sig")
        register.main(["import-csv", "--dir", str(self.tmp)])
        register.main(["import-csv", "--dir", str(self.tmp)])  # second run adds nothing
        self.assertEqual(self.count(dbm.DetectionLog), 3)
        rows, _, cameras = dbm.log_rows(self.db, "2026-10-20")
        self.assertEqual(cameras, ["Gate"])
        self.assertEqual({r["name"] for r in rows}, {"arm", "unknown"})
        self.assertEqual(dbm.log_rows(self.db, "2026-10-19")[2], ["connect.py"])


if __name__ == "__main__":
    unittest.main()
