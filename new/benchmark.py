"""Compare the pipeline with and without the work-savers, or compare models (PERFORMANCE.md).

    python benchmark.py                      a built-in test clip (empty room / people / empty ...)
    python benchmark.py myvideo.mp4          your own recording from the real camera
    python benchmark.py --models [video]     the same, but each person / face model in turn

Runs the same video several times with different settings and prints how much
work each one did and how long it took. (Speed only: for accuracy, upload the
video on the "ทดสอบด้วยวิดีโอ" page and check the rows.) Nothing is written to the history or the face gallery
(a temporary folder is used; with the built-in clip one test face is enrolled there).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from person_detection import config

PRESETS = {
    "เดิม (ก่อนปรับ)": dict(YOLO_BACKEND="pytorch", MOTION_GATE=False, SMART_FACES=False, FACE_VOTES=1,
                         FACE_MIN_SIZE=0, LOG_COOLDOWN_SEC=0, TRACKER_STATIC=False, TRACK_BUFFER=30,
                         FACE_FULL_RES=False),
    "ระบบเสริม (PyTorch)": dict(YOLO_BACKEND="pytorch", MOTION_GATE=True, SMART_FACES=True, FACE_VOTES=3,
                              FACE_MIN_SIZE=24, LOG_COOLDOWN_SEC=30, TRACKER_STATIC=True, TRACK_BUFFER=60),
    "ระบบเสริม + OpenVINO": dict(YOLO_BACKEND="auto", MOTION_GATE=True, SMART_FACES=True, FACE_VOTES=3,
                               FACE_MIN_SIZE=24, LOG_COOLDOWN_SEC=30, TRACKER_STATIC=True, TRACK_BUFFER=60),
}

# --models: everything else as the default settings (work-savers on, OpenVINO on CPU)
MODEL_PRESETS = {
    "YOLO11n + SFace": dict(PERSON_MODEL="yolo11n-pose", FACE_RECOGNIZER="sface"),
    "YOLO11n + ArcFace-MBF": dict(PERSON_MODEL="yolo11n-pose", FACE_RECOGNIZER="arcface_mbf"),
    "YOLO11n + ArcFace-R50": dict(PERSON_MODEL="yolo11n-pose", FACE_RECOGNIZER="arcface_r50"),
    "YOLO26n + SFace": dict(PERSON_MODEL="yolo26n-pose", FACE_RECOGNIZER="sface"),
    "YOLO11s + SFace": dict(PERSON_MODEL="yolo11s-pose", FACE_RECOGNIZER="sface"),
    "YOLO11n + SFace + ReID": dict(PERSON_MODEL="yolo11n-pose", FACE_RECOGNIZER="sface", TRACKER_REID=True),
}


def demo_clip(path: Path, fps: int = 15) -> Path:
    import ultralytics
    assets = Path(ultralytics.__file__).parent / "assets"
    size = (960, 540)
    empty = cv2.resize(cv2.imread(str(assets / "bus.jpg"))[0:300, :], size)  # building, nobody there
    two = cv2.resize(cv2.imread(str(assets / "zidane.jpg")), size)
    bus = cv2.resize(cv2.imread(str(assets / "bus.jpg")), size)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)

    def still(image, seconds):
        for _ in range(seconds * fps):  # camera noise only
            writer.write(cv2.add(image, rng.integers(0, 3, image.shape, dtype=np.uint8)))

    def slide(image, seconds):
        for i in range(seconds * fps):
            writer.write(cv2.warpAffine(image, np.float32([[1, 0, i - 30], [0, 1, 0]]), size))

    for part in (lambda: still(empty, 4), lambda: slide(two, 4), lambda: still(empty, 6),
                 lambda: slide(bus, 4), lambda: still(empty, 6)):
        part()
    writer.release()
    return path


def enroll_demo_face(factory) -> None:
    import ultralytics

    from person_detection.enrollment import FaceEnroller
    from person_detection.vision import face_models
    from person_detection.vision.face import FaceDetector
    scene = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    face = scene[120:720, 380:860]
    recognizer = face_models.create_recognizer()
    FaceEnroller(FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE), recognizer,
                 factory.people_db(recognizer.model_key), Path(config.OUTPUT_DIR) / "photos").enroll(
        "Demo", [("demo.jpg", cv2.imencode(".jpg", face)[1].tobytes())])


def run(video: Path, preset: dict, demo: bool) -> dict:
    from person_detection.evaluation.runner import EvalOptions, VideoEvaluator
    from person_detection.pipeline import PipelineFactory

    saved = {k: getattr(config, k) for k in preset}
    for key, value in preset.items():
        setattr(config, key, value)
    try:
        factory = PipelineFactory()
        if demo:
            enroll_demo_face(factory)
        events = []
        evaluator = VideoEvaluator(factory, EvalOptions(realtime=False, enable_vlm_par=False, frame_skip=1))
        started = time.perf_counter()
        stats = evaluator.run(video, events.append)
        wall = time.perf_counter() - started
        perf = evaluator.processor.perf()
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
    sightings = [e for e in events if e["kind"] == "sighting"]
    return {
        "ms/เฟรม": round(stats.ms_per_frame, 1),
        "fps": round(stats.processing_fps, 1),
        "เวลารวม (s)": round(wall, 1),
        "YOLO (ครั้ง)": perf["detect_runs"],
        "YuNet (ครั้ง)": perf["face_runs"],
        "จำหน้า (ครั้ง)": perf["face_embeddings"],
        "แถวที่บันทึก": len(sightings),
        "ชื่อที่ได้": ", ".join(sorted({e["name"] for e in sightings})) or "-",
        "backend": perf["backend"],
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--models"]
    presets = MODEL_PRESETS if "--models" in sys.argv else PRESETS
    tmp = Path(tempfile.mkdtemp(prefix="cn401-bench-"))
    config.OUTPUT_DIR = tmp
    config.PEOPLE_DB_PATH = tmp / "people.json"
    config.WARDROBE_LOG_PATH = tmp / "wardrobe.json"
    config.DATABASE_URL = ""  # never touch the real database
    demo = not args
    video = demo_clip(tmp / "demo.mp4") if demo else Path(args[0])
    frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"วิดีโอ: {video.name} ({frames} เฟรม) · ทุกเฟรมถูกประมวลผล (ไม่จำลองเวลาจริง)\n")
    results = {name: run(video, preset, demo) for name, preset in presets.items()}
    columns = list(next(iter(results.values())))
    width = max(len(n) for n in results) + 2
    print("".ljust(width) + "".join(c.rjust(14) for c in columns[:-2]))
    for name, r in results.items():
        print(name.ljust(width) + "".join(str(r[c]).rjust(14) for c in columns[:-2]) + f"   [{r['backend']}] ชื่อ: {r['ชื่อที่ได้']}")
    base = next(iter(results.values()))
    best = list(results.values())[-1]
    if presets is PRESETS and base["ms/เฟรม"]:
        print(f"\nเร็วขึ้น {base['ms/เฟรม'] / max(best['ms/เฟรม'], 0.1):.1f} เท่าต่อเฟรม · "
              f"YOLO ลดลง {100 - 100 * best['YOLO (ครั้ง)'] / max(base['YOLO (ครั้ง)'], 1):.0f}% · "
              f"YuNet ลดลง {100 - 100 * best['YuNet (ครั้ง)'] / max(base['YuNet (ครั้ง)'], 1):.0f}% · "
              f"จำหน้า {base['จำหน้า (ครั้ง)']} -> {best['จำหน้า (ครั้ง)']} ครั้ง "
              f"(ระบบเสริมเก็บหลายใบหน้าต่อคนเพื่อโหวตชื่อ จึงอาจมากกว่าเดิมเล็กน้อย)")


if __name__ == "__main__":
    main()
