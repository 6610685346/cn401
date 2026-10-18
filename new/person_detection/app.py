"""Command-line app: camera -> pipeline -> OpenCV window.  Run: python connect.py

The frame-by-frame work lives in pipeline.py so the web portal can reuse it.
"""

import threading

from . import config
from .pipeline import CameraOptions, FrameProcessor, PipelineFactory
from .sources.stream import open_stream
from .ui.display import DisplayWindow


class PersonDetectionApp:
    """Reads frames from a stream, processes them, and shows the result."""

    def __init__(self, source, processor: FrameProcessor, frame_skip: int = 1):
        self.source = source
        self.processor = processor
        self.frame_skip = max(1, frame_skip)
        self._quit = threading.Event()

    def _watch_terminal_quit(self) -> None:
        """Lets 'q' + Enter in the terminal also stop the app."""
        while not self._quit.is_set():
            try:
                if input().strip().lower() == "q":
                    self._quit.set()
                    break
            except (EOFError, OSError):
                break

    def run(self) -> None:
        threading.Thread(target=self._watch_terminal_quit, daemon=True).start()
        with open_stream(
            self.source, config.MAX_READ_FAILURES, config.RECONNECT_DELAY_SEC
        ) as stream, DisplayWindow(
            config.WINDOW_NAME, config.PRINT_MOUSE_POSITION,
            max_size=(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
        ) as window:
            print(
                f"Reading stream from: {self.source}  (press 'q' in the window or type 'q' + Enter here to quit)")
            for frame_count, frame in enumerate(stream.frames(), start=1):
                if self._quit.is_set():
                    print("Exiting...")
                    break
                if frame_count % self.frame_skip != 0:
                    continue
                if not window.show(self.processor(frame)):
                    print("Exiting...")
                    break


# ------------------------------------------------------------------ builders
def build_processor() -> FrameProcessor:
    """Build a frame processor from the settings in config.py."""
    factory = PipelineFactory()
    if config.ENABLE_FACE_RECOGNITION:
        names = factory.people_db().names()
        print(f"Loaded {len(names)} enrolled people: {names}")
    if config.SAVE_CSV:
        print(f"Logging people to: {config.CSV_PATH or config.OUTPUT_DIR / 'person_log_<date>.csv'}")
    return factory.create(CameraOptions())


def main() -> None:
    PersonDetectionApp(config.SOURCE, build_processor(), config.FRAME_SKIP).run()
