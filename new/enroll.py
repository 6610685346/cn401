"""Enroll a person's face.  Usage:  python enroll.py "Alice" models/photos/Alice/
                                   [--surname Smith] [--id 6610685346]

Uses the same checks as the web portal (person_detection/enrollment.py):
the most confident big face in each photo, and photos that do not look like
the others are left out. Photos from another folder are copied into
models/photos/<name>/ so the web page can show them and add more later.
"""

import argparse
import sys
from pathlib import Path

from person_detection import config
from person_detection.enrollment import PHOTO_TYPES, EnrollError, FaceEnroller
from person_detection.storage.people_db import open_people_db, people_db_path
from person_detection.vision import face_models
from person_detection.vision.face import FaceDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a person's face from a folder of photos.")
    parser.add_argument("name")
    parser.add_argument("photos_dir", type=Path)
    parser.add_argument("--surname", help="kept in the database (DATABASE_URL) only")
    parser.add_argument("--id", dest="id_number", help="student / staff ID, database only")
    parser.add_argument("--model", choices=list(face_models.FACE_MODELS),
                        help="face model (default: FACE_RECOGNIZER in config / settings.json)")
    args = parser.parse_args()
    name, photos_dir = args.name, args.photos_dir
    model = args.model or config.FACE_RECOGNIZER
    enroller = FaceEnroller(
        FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE),
        face_models.create_recognizer(model),
        open_people_db(model),
        config.PHOTOS_DIR,
    )
    try:
        if photos_dir.resolve() == enroller.person_dir(name):
            result = enroller.enroll_folder(name, args.surname, args.id_number)
        else:
            files = sorted(p for p in photos_dir.glob("*") if p.suffix.lower() in PHOTO_TYPES)
            result = enroller.enroll(name, [(p.name, p.read_bytes()) for p in files],
                                     args.surname, args.id_number)
    except EnrollError as exc:
        print(f"Nothing enrolled: {exc}")
        sys.exit(1)

    for report in result.reports:
        mark = "used   " if report.ok else "skipped"
        print(f"  {mark} {report.filename}: {report.message}")
    if result.looks_like:
        print(f"WARNING: looks a lot like '{result.looks_like[0]}' (similarity {result.looks_like[1]})")
    where = "the database" if config.DATABASE_URL else people_db_path(model)
    print(f"Enrolled '{result.name}' from {result.used} photo(s) with {model} -> {where}")


if __name__ == "__main__":
    main()
