"""Several cameras seeing the same people.  Run from `new`:  python -m unittest discover -s tests"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from person_detection import config  # noqa: E402
from person_detection.multicam import hub as hub_module  # noqa: E402
from person_detection.multicam.floor import CalibrationError, FloorCalibration  # noqa: E402
from person_detection.multicam.hub import IdentityHub, Observation  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FloorTest(unittest.TestCase):
    IMAGE = [(0.2, 0.8), (0.8, 0.8), (0.7, 0.4), (0.3, 0.4)]
    FLOOR = [(0, 0), (3, 0), (3, 4), (0, 4)]

    def test_fit_and_round_trip(self):
        cal = FloorCalibration.fit(self.IMAGE, self.FLOOR)
        for image, floor in zip(self.IMAGE, self.FLOOR):
            np.testing.assert_allclose(cal.to_floor(image), floor, atol=1e-6)
            np.testing.assert_allclose(cal.to_image(floor), image, atol=1e-6)
        self.assertEqual(FloorCalibration.from_dict(cal.to_dict()), cal)
        self.assertTrue(cal.grid_lines())

    def test_more_points_least_squares_and_error(self):
        image = self.IMAGE + [(0.5, 0.6)]
        cal = FloorCalibration.fit(image, self.FLOOR + [(1.5, 1.7)])  # true value ~ (1.5, 1.6)
        self.assertGreater(cal.error_m, 0.0)
        self.assertLess(cal.error_m, 0.1)

    def test_mistakes_are_caught(self):
        swapped = [self.FLOOR[0], self.FLOOR[1], self.FLOOR[3], self.FLOOR[2]]
        cases = {
            "three points": (self.IMAGE[:3], self.FLOOR[:3]),
            "points swapped": (self.IMAGE, swapped),
            "all on a line": ([(0.1, 0.5), (0.3, 0.5), (0.6, 0.5), (0.9, 0.5)], self.FLOOR),
            "outside picture": ([(0.2, 0.8), (1.8, 0.8), (0.7, 0.4), (0.3, 0.4)], self.FLOOR),
            "count mismatch": (self.IMAGE, self.FLOOR[:3]),
        }
        for label, (image, floor) in cases.items():
            with self.subTest(label), self.assertRaises(CalibrationError):
                FloorCalibration.fit(image, floor)

    def test_two_cameras_share_the_floor(self):
        # Camera 2 looks at the same floor from another angle (a perspective warp).
        warp = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.float32([[0.1, 0.05], [0.95, 0.15], [0.85, 0.95], [0.05, 0.9]]))
        image2 = cv2.perspectiveTransform(np.float32([self.IMAGE]), warp)[0].tolist()
        cam1 = FloorCalibration.fit(self.IMAGE, self.FLOOR)
        cam2 = FloorCalibration.fit(image2, self.FLOOR)
        spot1 = (0.55, 0.62)
        spot2 = tuple(cv2.perspectiveTransform(np.float32([[spot1]]), warp)[0][0])
        np.testing.assert_allclose(cam1.to_floor(spot1), cam2.to_floor(spot2), atol=1e-4)


class HubTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.hub = IdentityHub(clock=self.clock)

    def frames(self, n, cam1, cam2, step=0.1):
        """n frames from both cameras; cam1/cam2 = lists of Observations (or callables)."""
        out = None
        for i in range(n):
            self.clock.now += step
            a = self.hub.update("cam1", cam1(i) if callable(cam1) else cam1)
            self.clock.now += 0.02
            b = self.hub.update("cam2", cam2(i) if callable(cam2) else cam2)
            out = (a, b)
        return out

    def test_face_on_one_camera_back_on_the_other(self):
        a, b = self.frames(6, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (2.2, 3.1), None)])
        self.assertEqual(a[5].gid, b[12].gid)
        self.assertEqual(b[12].name, "Arm")  # camera 2 never saw the face
        self.assertEqual(set(b[12].cameras), {"cam1", "cam2"})

    def test_needs_several_agreeing_frames(self):
        a, b = self.frames(1, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (2.1, 3.0), None)])
        self.assertNotEqual(a[5].gid, b[12].gid)  # one frame is not enough

    def test_far_apart_people_stay_separate(self):
        a, b = self.frames(10, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (4.0, 3.0), None)])
        self.assertNotEqual(a[5].gid, b[12].gid)
        self.assertIsNone(b[12].name)

    def test_two_people_matched_to_the_right_partner(self):
        cam1 = [Observation(1, (1.0, 1.0), "Arm"), Observation(2, (1.6, 1.0), "Oung")]
        cam2 = [Observation(7, (1.65, 1.1), None), Observation(8, (1.05, 0.9), None)]
        a, b = self.frames(8, cam1, cam2)
        self.assertEqual(b[8].name, "Arm")
        self.assertEqual(b[7].name, "Oung")

    def test_different_names_never_merge(self):
        a, b = self.frames(10, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (2.0, 3.0), "Oung")])
        self.assertNotEqual(a[5].gid, b[12].gid)

    def test_one_camera_seeing_two_people_keeps_them_apart(self):
        # cam1 sees two people standing close together; they must not merge with each other
        cam1 = [Observation(1, (2.0, 3.0), None), Observation(2, (2.3, 3.0), None)]
        a, _ = self.frames(10, cam1, [])
        self.assertNotEqual(a[1].gid, a[2].gid)

    def test_uncalibrated_camera_is_never_joined(self):
        a, b = self.frames(10, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, None, None)])
        self.assertNotEqual(a[5].gid, b[12].gid)
        self.assertIsNone(b[12].name)

    def test_walking_together_between_cameras(self):
        walk = lambda i: [Observation(5, (1.0 + 0.1 * i, 2.0), "Arm")]  # noqa: E731
        walk2 = lambda i: [Observation(9, (1.05 + 0.1 * i, 2.05), None)]  # noqa: E731
        _, b = self.frames(8, walk, walk2)
        self.assertEqual(b[9].name, "Arm")

    def test_lookup_snapshot_and_forget(self):
        self.frames(6, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (2.1, 3.0), None)])
        info = self.hub.lookup("cam2", 12)
        self.assertEqual(info.name, "Arm")
        snap = self.hub.snapshot()
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0]["cameras"], ["cam1", "cam2"])
        self.assertAlmostEqual(snap[0]["x"], 2.05, places=2)
        self.clock.now += hub_module.FORGET_AFTER_S + 1
        self.hub.update("cam1", [])
        self.assertEqual(self.hub.snapshot(), [])
        self.assertIsNone(self.hub.lookup("cam2", 12))

    def test_camera_restart_forgets_its_tracks(self):
        self.frames(6, [Observation(5, (2.0, 3.0), "Arm")], [Observation(12, (2.1, 3.0), None)])
        self.hub.forget_camera("cam2")
        self.assertIsNone(self.hub.lookup("cam2", 12))
        self.assertEqual(self.hub.lookup("cam1", 5).name, "Arm")


@unittest.skipUnless(Path(config.MODEL_PATH).exists(), "models not downloaded")
class FootPointTest(unittest.TestCase):
    def test_pose_model_gives_ankles(self):
        import ultralytics
        from person_detection.vision.detector import PersonDetector
        image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        people = PersonDetector(config.MODEL_PATH, 0.4).detect(image)
        self.assertTrue(people)
        for person in people:
            x1, y1, x2, y2 = person.box
            self.assertIsNotNone(person.foot)
            fx, fy = person.foot
            self.assertTrue(x1 <= fx <= x2)
            self.assertTrue(y1 + 0.8 * (y2 - y1) <= fy <= y2)  # near the bottom of the person


if __name__ == "__main__":
    unittest.main()
