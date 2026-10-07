"""Person detection from a phone video stream, webcam, or video file.

Edit person_detection/config.py, then run:  python connect.py

Models are read from models/. Each tracked person is logged once to
output/person_log_<date>.csv with their shirt and trouser colours, plus a
crop under output/cropped_persons/. Press "q" to quit.
"""

from person_detection.app import main

if __name__ == "__main__":
    main()
