from typing import Optional
import time


class StopWatch:
    """
    A simple stop watch to be used as a context manager
    """
    def __init__(self):
        self._start: Optional[float] = None
        self._stop: Optional[float] = None

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def duration(self):
        return self._stop - self._start

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop = time.time()