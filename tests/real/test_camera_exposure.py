"""Manual exposure has to be the value we asked for, not the one the camera
felt like applying (real/vision/camera.set_exposure).

The main C922 sat at power_line_frequency=1, where it clamped every exposure
written before streaming to 83 (1/120 s) instead of the requested 100 (10 ms).
Under 50 Hz mains that is a fractional flicker period, so the rolling shutter
recorded ~3 dark bands per frame — silently, because nobody read the control
back. set_exposure now pins the anti-flicker mode and verifies the readback.
"""

import pytest

from real.vision import camera


class FakeV4l2:
    """Stands in for v4l2-ctl: records writes, answers reads from its state.

    `applied` is what the camera claims it actually set; None = it obeys.
    """

    def __init__(self, applied=None):
        self.writes = []
        self._applied = applied

    def set(self, ctrl, value, device=None):
        self.writes.append((ctrl, value))

    def get(self, ctrl, device=None):
        assert ctrl == "exposure_time_absolute"
        written = dict(self.writes)["exposure_time_absolute"]
        return written if self._applied is None else self._applied


def patch_v4l2(monkeypatch, applied=None):
    fake = FakeV4l2(applied)
    monkeypatch.setattr(camera, "v4l2_set", fake.set)
    monkeypatch.setattr(camera, "v4l2_get", fake.get)
    monkeypatch.setattr(camera, "resolve_device", lambda device: 0)
    return fake


def test_anti_flicker_is_pinned_before_the_exposure_write(monkeypatch):
    fake = patch_v4l2(monkeypatch)
    camera.set_exposure(100)
    ctrls = [ctrl for ctrl, _ in fake.writes]
    assert ("power_line_frequency", camera.POWER_LINE_FREQUENCY) in fake.writes
    assert ("exposure_time_absolute", 100) in fake.writes
    # The clamp happens at write time, so the mode has to be set first.
    assert ctrls.index("power_line_frequency") < ctrls.index("exposure_time_absolute")


def test_manual_mode_and_fixed_framerate_still_set(monkeypatch):
    fake = patch_v4l2(monkeypatch)
    camera.set_exposure(100)
    assert ("auto_exposure", 1) in fake.writes
    assert ("exposure_dynamic_framerate", 0) in fake.writes


def test_a_camera_that_overrides_the_exposure_fails_loud(monkeypatch):
    patch_v4l2(monkeypatch, applied=83)      # the observed 1/120 s clamp
    with pytest.raises(RuntimeError, match="applied exposure 83"):
        camera.set_exposure(100)


def test_the_cameras_own_quantisation_is_tolerated(monkeypatch):
    patch_v4l2(monkeypatch, applied=99)      # what the C922 reports for 100
    camera.set_exposure(100)
