"""Video-test metrics (person_detection/evaluation/metrics.py) and the runner.

Run from the `new` folder:   python -m unittest tests.test_evaluation -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection.evaluation import metrics as M  # noqa: E402


def sighting(gt_status="ok", gt_identity="", face=True, candidate="Arm", score=0.8, latency=0.5):
    return {"kind": "sighting", "face_seen": face, "candidate": candidate if face else None,
            "score": score if face else None, "gt_status": gt_status, "gt_identity": gt_identity, "latency_s": latency}


def crossing(kind, gt="ok"):
    return {"kind": kind, "gt_direction": gt}


class DetectionTest(unittest.TestCase):
    def test_precision_recall_f1(self):
        events = [sighting(), sighting(), sighting(), sighting("false"), sighting("duplicate"), sighting("")]
        m = M.compute([(events, {"people": 1})], 0.363)
        d = m["detection"]
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (3, 2, 1))
        self.assertAlmostEqual(d["precision"], 3 / 5)
        self.assertAlmostEqual(d["recall"], 3 / 4)
        self.assertAlmostEqual(d["f1"], 2 * 0.6 * 0.75 / 1.35)
        self.assertEqual((m["events"], m["reviewed"]), (6, 5))  # unreviewed rows are left out

    def test_nothing_to_divide(self):
        m = M.compute([([], {})], 0.363)
        self.assertIsNone(m["detection"]["precision"])
        self.assertIsNone(m["face"]["far"])
        self.assertIsNone(m["direction"]["occupancy_error"])


class FaceTest(unittest.TestCase):
    def events(self):
        return [
            sighting(gt_identity="Arm", candidate="Arm", score=0.80),   # correct
            sighting(gt_identity="Arm", candidate="Arm", score=0.30),   # false reject at 0.363
            sighting(gt_identity="Arm", candidate="Oung", score=0.50),  # misidentified
            sighting(gt_identity=M.NONE, candidate="Arm", score=0.40),  # false accept
            sighting(gt_identity=M.NONE, candidate="Arm", score=0.10),  # correct reject
            sighting(gt_identity="Arm", face=False),                    # no face: coverage only
            sighting(gt_identity=M.SKIP, candidate="Arm", score=0.9),   # cannot tell: left out
            sighting("false", gt_identity="", candidate="Arm", score=0.9),  # not a person
        ]

    def test_far_frr_at_threshold(self):
        f = M.compute([(self.events(), {})], 0.363)["face"]
        self.assertEqual((f["genuine"], f["impostor"]), (3, 2))
        self.assertEqual((f["correct"], f["false_reject"], f["misidentified"], f["false_accept"]), (1, 1, 1, 1))
        self.assertAlmostEqual(f["far"], 1 / 2)
        self.assertAlmostEqual(f["frr"], 1 / 3)
        self.assertAlmostEqual(f["accuracy"], 1 / 3)
        self.assertAlmostEqual(f["coverage"], 6 / 7)  # 7 real people, 6 with a face

    def test_threshold_changes_the_answer(self):
        f = M.compute([(self.events(), {})], 0.45)["face"]
        self.assertEqual(f["false_accept"], 0)  # 0.40 no longer accepted
        self.assertEqual(f["false_reject"], 1)
        self.assertEqual(f["misidentified"], 1)  # Oung at 0.50 still wins

    def test_sweep_suggests_fewest_errors(self):
        s = M.sweep(self.events())
        self.assertEqual(len(s["rows"]), 11)
        self.assertEqual(s["rows"][0]["threshold"], 0.30)
        self.assertEqual(s["rows"][-1]["threshold"], 0.50)
        # at 0.30: FA=1/2, FR=0 -> 0.5 ; from 0.42: FA=0, FR=1/3 -> 0.33 (highest of the ties wins)
        self.assertEqual(s["best"]["threshold"], 0.50)
        self.assertAlmostEqual(s["best"]["far"], 0)

    def test_predicted_name(self):
        self.assertIsNone(M.predicted_name(sighting(face=False), 0.3))
        self.assertEqual(M.predicted_name(sighting(score=0.2), 0.3), M.UNKNOWN)
        self.assertEqual(M.predicted_name(sighting(score=0.3), 0.3), "Arm")


class DirectionTest(unittest.TestCase):
    def test_in_out(self):
        events = [crossing("in"), crossing("in"), crossing("in", "wrong"), crossing("out"), crossing("out", "false")]
        d = M.compute([(events, {"in": 1, "out": 2})], 0.363)["direction"]
        self.assertEqual((d["in"]["tp"], d["in"]["fp"], d["in"]["fn"]), (2, 1, 1))
        # the "in" that was really an "out" counts as a missed out
        self.assertEqual((d["out"]["tp"], d["out"]["fp"], d["out"]["fn"]), (1, 1, 3))
        self.assertAlmostEqual(d["in"]["precision"], 2 / 3)
        self.assertAlmostEqual(d["out"]["recall"], 1 / 4)
        # system: 3 in - 2 out = 1 ; truth: (2+1) in - (1+3) out = -1
        self.assertEqual(d["occupancy_error"], 2)

    def test_videos_are_summed(self):
        a = ([sighting()], {"people": 1})
        b = ([sighting(), sighting("false")], {"people": 0})
        d = M.compute([a, b], 0.363)["detection"]
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (2, 1, 1))


class RunnerTest(unittest.TestCase):
    """The real pipeline on a short synthetic video (needs the models in new/models)."""

    @classmethod
    def setUpClass(cls):
        from person_detection import config
        if not (Path(config.MODEL_PATH).exists() and Path(config.FACE_RECOGNITION_MODEL_PATH).exists()):
            raise unittest.SkipTest("models not downloaded")

    def test_realtime_drops_frames_but_finds_the_same_people(self):
        import cv2
        import numpy as np
        import ultralytics

        from person_detection import config
        from person_detection.evaluation.runner import EvalOptions, VideoEvaluator
        from person_detection.pipeline import PipelineFactory

        tmp = Path(tempfile.mkdtemp())
        saved = (config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH, config.OUTPUT_DIR)
        config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH, config.OUTPUT_DIR = tmp / "p.json", tmp / "w.json", tmp
        try:
            scene = cv2.resize(cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg")), (960, 540))
            path = tmp / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
            for i in range(45):
                writer.write(cv2.warpAffine(scene, np.float32([[1, 0, i - 20], [0, 1, 0]]), (960, 540)))
            writer.release()
            factory = PipelineFactory()
            results = {}
            for realtime in (False, True):
                events, crops = [], {}
                stats = VideoEvaluator(factory, EvalOptions(realtime=realtime, frame_skip=1)).run(
                    path, events.append, lambda t, c: crops.__setitem__(t, c))
                results[realtime] = (events, stats, crops)
            events, stats, crops = results[False]
            self.assertEqual(stats.frames_dropped, 0)
            self.assertEqual(stats.frames_processed, stats.frames_read)
            self.assertGreaterEqual(len([e for e in events if e["kind"] == "sighting"]), 2)
            self.assertTrue(all(e["snapshot"] for e in events))
            self.assertTrue(any(e["face_seen"] for e in events))
            self.assertTrue(crops)
            rt_events, rt_stats, _ = results[True]
            self.assertEqual(rt_stats.frames_processed + rt_stats.frames_dropped, rt_stats.frames_read)
            self.assertGreater(rt_stats.processing_fps, 0)
        finally:
            config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH, config.OUTPUT_DIR = saved


if __name__ == "__main__":
    unittest.main()
