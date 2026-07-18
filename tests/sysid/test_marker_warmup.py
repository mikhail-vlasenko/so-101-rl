"""Gate tests for CameraMarkerSource.warmup() — the pre-episode wait that blocks
until the table anchor tag is being detected before the real-arm lift rollout
starts moving. Pure timing/state logic, exercised on a bare instance so no
camera or detector is touched: warmup must return once the table tag is fresh,
fail loud on timeout, and re-raise a dead capture-thread error.
"""

import threading
import time

import numpy as np
import pytest

from real.rollout.marker_obs import CameraMarkerSource
from real.marker_spec import TABLE_TAG_ID


def bare_source(table_last_capture_t: float = -np.inf,
                error: Exception | None = None) -> CameraMarkerSource:
    """A CameraMarkerSource with only the fields warmup()/table_age() read set,
    skipping __init__'s extrinsic/detector/intrinsic loading (needs no hardware,
    but is irrelevant to the gate logic under test)."""
    src = object.__new__(CameraMarkerSource)
    src._lock = threading.Lock()
    src._table_last_capture_t = table_last_capture_t
    src.error = error
    return src


def test_table_age_inf_when_never_seen():
    assert bare_source(table_last_capture_t=-np.inf).table_age() == np.inf


def test_warmup_returns_immediately_when_table_fresh():
    src = bare_source(table_last_capture_t=time.monotonic())
    waited = src.warmup(timeout_s=2.0)
    assert waited < 0.1


def test_warmup_times_out_when_table_never_detected():
    src = bare_source(table_last_capture_t=-np.inf)
    with pytest.raises(RuntimeError, match=f"table anchor tag \\(id {TABLE_TAG_ID}\\)"):
        src.warmup(timeout_s=0.1)


def test_warmup_reraises_dead_thread_error():
    boom = RuntimeError("camera read failed mid-session")
    src = bare_source(error=boom)
    with pytest.raises(RuntimeError, match="camera read failed mid-session"):
        src.warmup(timeout_s=2.0)


def test_warmup_waits_until_table_becomes_fresh():
    src = bare_source(table_last_capture_t=-np.inf)

    def detect_later():
        time.sleep(0.15)
        with src._lock:
            src._table_last_capture_t = time.monotonic()

    threading.Thread(target=detect_later, daemon=True).start()
    t0 = time.monotonic()
    waited = src.warmup(timeout_s=2.0)
    elapsed = time.monotonic() - t0
    assert 0.1 < waited <= elapsed + 1e-3
    assert elapsed < 1.0
