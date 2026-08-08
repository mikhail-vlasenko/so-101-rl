"""Contracts for the shared camera annotation and stereo canvas."""
import cv2
import numpy as np
import pytest

from real.vision.detect import Detection
from real.vision.overlay import (
    RED,
    OverlayLine,
    OverlaySpan,
    TagStyle,
    StereoViewer,
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


def test_stereo_viewer_sizes_normal_window_once(monkeypatch):
    calls = {"named": [], "resized": [], "shown": [], "destroyed": []}
    monkeypatch.setattr("real.vision.overlay.cv2.namedWindow",
                        lambda title, flags: calls["named"].append((title, flags)))
    monkeypatch.setattr("real.vision.overlay.cv2.resizeWindow",
                        lambda title, width, height:
                        calls["resized"].append((title, width, height)))
    monkeypatch.setattr("real.vision.overlay.cv2.imshow",
                        lambda title, canvas:
                        calls["shown"].append((title, canvas.shape)))
    monkeypatch.setattr("real.vision.overlay.cv2.waitKey", lambda _delay: -1)
    monkeypatch.setattr("real.vision.overlay.cv2.destroyWindow",
                        lambda title: calls["destroyed"].append(title))

    frames = {
        "main": np.zeros((20, 30, 3), dtype=np.uint8),
        "aux": np.zeros((20, 30, 3), dtype=np.uint8),
    }
    viewer = StereoViewer("stable", camera_names=("main", "aux"))
    viewer.show(frames, header_lines=[OverlayLine("one")])
    viewer.show(frames, header_lines=[OverlayLine("one"), OverlayLine("two")])
    viewer.close()

    assert calls["named"] == [("stable", cv2.WINDOW_NORMAL)]
    assert calls["resized"] == [("stable", 1280, 396)]
    assert len(calls["shown"]) == 2
    assert calls["shown"][0][1] != calls["shown"][1][1]
    assert calls["destroyed"] == ["stable"]


def test_compose_stereo_view_draws_each_span_in_its_own_color(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "real.vision.overlay.cv2.putText",
        lambda _frame, text, origin, _font, _scale, color, _thickness:
        calls.append((text, origin, color)))
    monkeypatch.setattr(
        "real.vision.overlay.cv2.getTextSize",
        lambda text, _font, _scale, _thickness: ((10 * len(text), 12), 0))
    frames = {
        "main": np.zeros((20, 30, 3), dtype=np.uint8),
        "aux": np.zeros((20, 30, 3), dtype=np.uint8),
    }
    compose_stereo_view(
        frames,
        header_lines=[OverlayLine(spans=(
            OverlaySpan("pass ", (0, 255, 0)),
            OverlaySpan("fail", (0, 0, 255)),
        ))],
        camera_names=("main", "aux"),
        view_size=(30, 20),
    )
    assert calls == [
        ("pass ", (12, 26), (0, 255, 0)),
        ("fail", (62, 26), (0, 0, 255)),
    ]
