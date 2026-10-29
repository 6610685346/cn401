"""Portal with the PostgreSQL database (DATABASE_URL). Needs a throw-away database:

    set TEST_DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/person_detection_test
    python manage.py test cameras.tests_db

Skipped when TEST_DATABASE_URL is not set. The tables are dropped and re-created.
"""

import datetime as dt
import os
import unittest
from pathlib import Path

from person_detection import config

from . import services
from . import tests as base  # not `import EnrollViewTest`: it would run twice

URL = os.environ.get("TEST_DATABASE_URL", "")


@unittest.skipUnless(URL, "set TEST_DATABASE_URL to run the PostgreSQL tests")
class DatabasePortalTest(base.EnrollViewTest):
    """Runs the enrollment page tests again on PostgreSQL, plus history from the database."""

    def setUp(self):
        from person_detection.storage import database as dbm
        self.dbm = dbm
        self._saved_url = config.DATABASE_URL
        config.DATABASE_URL = URL
        db = dbm.Database(URL)
        dbm.Base.metadata.drop_all(db.engine)
        db.create_tables()
        db.dispose()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.DATABASE_URL = self._saved_url
        services.reset_shared_state()

    def db(self):
        return services.factory.database()

    def test_surname_id_and_photo_paths_stored(self):
        self.client.force_login(self.admin)
        self.client.post("/people/enroll/", {"name": "Arm", "surname": "Penwong", "id_number": "6610685015",
                                             "photos": self.photos(3)})
        people = services.factory.people_db()
        detail = people.details()[0]
        self.assertEqual((detail["name"], detail["surname"], detail["id_number"], detail["photos"]),
                         ("Arm", "Penwong", "6610685015", 3))
        with self.db().Session() as s:
            paths = s.scalars(self.dbm.select(self.dbm.FaceEmbedding.image_path)).all()
        self.assertTrue(all(p and Path(p).is_file() for p in paths))
        html = self.client.get("/people/").content.decode()
        self.assertIn("Penwong", html)
        self.assertIn("6610685015", html)
        # the same ID for someone else is refused and their photo is not kept
        response = self.client.post("/people/enroll/", {"name": "Other", "id_number": "6610685015",
                                                        "photos": self.photos(1)}, follow=True)
        self.assertIn("มีคนอื่นใช้อยู่แล้ว", response.content.decode())
        self.assertEqual(people.names(), ["Arm"])
        self.assertFalse(list((Path(self.tmp) / "photos" / "Other").glob("*.jpg")))

    def test_recognised_person_logged_and_shown_in_history(self):
        self.client.force_login(self.admin)
        self.client.post("/people/enroll/", {"name": "Arm", "photos": self.photos()})
        frame = base.jpeg(base.ASSETS / "zidane.jpg", 1920, 1080)
        for _ in range(10):
            self.client.post("/api/browser/frame/?session=db-check-1&vlm=0", data=frame, content_type="image/jpeg")
        self.assertTrue(services.factory.log_writer().flush())
        today = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date().isoformat()
        rows, total, cameras = self.dbm.log_rows(self.db(), today)
        arm = [r for r in rows if r["name"] == "Arm"]
        self.assertTrue(arm)
        self.assertGreater(float(arm[0]["similarity"]), config.FACE_MATCH_THRESHOLD)
        html = self.client.get("/history/").content.decode()
        self.assertIn("อ่านจากฐานข้อมูล PostgreSQL", html)
        self.assertIn(f"{total} รายการ", html)
        self.assertIn("<b>Arm</b>", html)
        export = self.client.get(f"/history/export.csv?date={today}&name=arm")
        self.assertEqual(export.status_code, 200)
        lines = export.content.decode("utf-8-sig").strip().splitlines()
        self.assertTrue(lines[0].startswith("timestamp,camera,name"))
        self.assertEqual(len(lines) - 1, len(arm))
        self.assertIn("PostgreSQL", self.client.get("/").content.decode())

    def test_delete_removes_person_and_photos(self):
        super().test_delete_removes_person_and_photos()
        with self.db().Session() as s:
            self.assertEqual(s.scalar(self.dbm.select(self.dbm.func.count()).select_from(self.dbm.FaceEmbedding)), 0)
            person = s.scalars(self.dbm.select(self.dbm.Person)).one()
        self.assertIsNotNone(person.deleted_at)  # soft delete: old log rows still point to it
