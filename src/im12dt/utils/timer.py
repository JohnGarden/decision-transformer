from __future__ import annotations
import time
from contextlib import contextmanager


@contextmanager
def chrono(label: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[TIMER] {label}: {dt:.3f}s")