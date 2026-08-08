"""Contracts for the shared camera annotation and stereo canvas."""
import numpy as np
import pytest

from real.vision.detect import Detection
from real.vision.overlay import (
    RED,
    OverlayLine,
    TagStyle,
    annotate_tags,
    compose_stereo_view,
)


def test_annotate_tags_applies_explicit_style_in_place():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detection = Detection(7, np.array([
        [20.0, 30.0], [60.0, 30.0], [60.0, 60.0], [20.0, 60.0]
    ], dtype=np.float32))
    result = annotate_tags(frame, [detection], {7: TagStyle("rejected", RED, 3)})
    assert result is frame
    assert np.any(np.all(frame == RED, axis=2))


def test_compose_stereo_view_preserves_inputs_and_layout():
    frames = {
        "main": np.full((20, 30, 3), (10, 20, 30), dtype=np.uint8),
        "aux": np.full((20, 30, 3), (40, 50, 60), dtype=np.uint8),
    }
    before = {camera: frame.copy() for camera, frame in frames.items()}
    canvas = compose_stereo_view(
        frames,
        camera_lines={"main": [OverlayLine("main status")]},
        header_lines=[OverlayLine("recording"), OverlayLine("coverage")],
        camera_names=("main", "aux"),
        view_size=(30, 20),
    )
    assert canvas.shape == (84, 60, 3)
    for camera in frames:
        np.testing.assert_array_equal(frames[camera], before[camera])
    np.testing.assert_array_equal(canvas[-1, 0], [10, 20, 30])
    np.testing.assert_array_equal(canvas[-1, -1], [40, 50, 60])


def test_compose_stereo_view_requires_every_configured_camera():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        compose_stereo_view({"main": frame}, camera_names=("main", "aux"))
