"""Enroll a person's face.  Usage:  python enroll.py "Alice" models/photos/Alice/

Uses the same checks as the web portal (person_detection/enrollment.py):
the most confident big face in each photo, and photos that do not look like
the others are left out. Photos from another folder are copied into
models/photos/<name>/ so the web page can show them and add more later.
"""

import sys
from pathlib import Path

from person_detection import config
from person_detection.enrollment import PHOTO_TYPES, EnrollError, FaceEnroller
from person_detection.storage.people_db import PeopleDB
from person_detection.vision.face import FaceDetector, FaceRecognizer


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python enroll.py <name> <photos_dir>")
        sys.exit(1)

    name, photos_dir = sys.argv[1], Path(sys.argv[2])
    enroller = FaceEnroller(
        FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE),
        FaceRecognizer(config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD),
        PeopleDB(config.PEOPLE_DB_PATH),
        config.PHOTOS_DIR,
    )
    try:
        if photos_dir.resolve() == enroller.person_dir(name):
            result = enroller.enroll_folder(name)
        else:
            files = sorted(p for p in photos_dir.glob("*") if p.suffix.lower() in PHOTO_TYPES)
            result = enroller.enroll(name, [(p.name, p.read_bytes()) for p in files])
    except EnrollError as exc:
        print(f"Nothing enrolled: {exc}")
        sys.exit(1)

    for report in result.reports:
        mark = "used   " if report.ok else "skipped"
        print(f"  {mark} {report.filename}: {report.message}")
    if result.looks_like:
        print(f"WARNING: looks a lot like '{result.looks_like[0]}' (similarity {result.looks_like[1]})")
    print(f"Enrolled '{result.name}' from {result.used} photo(s) -> {config.PEOPLE_DB_PATH}")


if __name__ == "__main__":
    main()
