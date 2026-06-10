"""Contract tests for panel.streamer — hardware-free, synthetic frames only."""

import http.client
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from panel.streamer import STREAM_BOUNDARY, FrameBox, JpegStreamer


def encode_frame(value: int) -> bytes:
    img = np.full((48, 64, 3), value, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", img)
    assert ok
    return jpeg.tobytes()


@pytest.fixture
def streamer():
    box = FrameBox()
    s = JpegStreamer(port=0, box=box, host="127.0.0.1")
    s.start()
    yield s
    s.close()


def url(streamer: JpegStreamer, path: str) -> str:
    return f"http://127.0.0.1:{streamer.port}{path}"


def test_snapshot_503_before_first_frame(streamer):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url(streamer, "/snapshot"), timeout=5)
    assert e.value.code == 503


def test_stream_503_before_first_frame(streamer):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url(streamer, "/stream"), timeout=5)
    assert e.value.code == 503


def test_unknown_path_404(streamer):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url(streamer, "/nope"), timeout=5)
    assert e.value.code == 404


def test_snapshot_returns_decodable_latest_frame(streamer):
    streamer.box.publish(encode_frame(10))
    streamer.box.publish(encode_frame(200))
    with urllib.request.urlopen(url(streamer, "/snapshot"), timeout=5) as resp:
        assert resp.headers["Content-Type"] == "image/jpeg"
        body = resp.read()
    img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (48, 64, 3)
    assert abs(int(img.mean()) - 200) < 10  # latest frame wins, not the first


def test_stream_delivers_successive_frames(streamer):
    streamer.box.publish(encode_frame(50))
    conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
    try:
        conn.request("GET", "/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        assert STREAM_BOUNDARY in resp.headers["Content-Type"]

        def read_part():
            assert resp.fp.readline().strip() == f"--{STREAM_BOUNDARY}".encode()
            headers = {}
            while True:
                line = resp.fp.readline().strip()
                if not line:
                    break
                k, v = line.decode().split(": ", 1)
                headers[k] = v
            body = resp.fp.read(int(headers["Content-Length"]))
            assert resp.fp.readline() == b"\r\n"
            return body

        first = read_part()
        streamer.box.publish(encode_frame(180))
        second = read_part()
    finally:
        conn.close()

    for body, value in [(first, 50), (second, 180)]:
        img = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        assert abs(int(img.mean()) - value) < 10


def test_query_string_is_ignored(streamer):
    # The panel's <img> retry appends ?t=<ms> cache-busters; they must not 404.
    streamer.box.publish(encode_frame(99))
    with urllib.request.urlopen(url(streamer, "/snapshot?t=12345"), timeout=5) as resp:
        assert resp.status == 200
    conn = http.client.HTTPConnection("127.0.0.1", streamer.port, timeout=5)
    try:
        conn.request("GET", "/stream?t=12345")
        assert conn.getresponse().status == 200
    finally:
        conn.close()


def test_busy_port_raises_oserror(streamer):
    box = FrameBox()
    with pytest.raises(OSError):
        JpegStreamer(port=streamer.port, box=box, host="127.0.0.1")


def test_framebox_wait_newer_times_out():
    box = FrameBox()
    assert box.wait_newer(0, timeout=0.05) is None
    box.publish(b"x")
    assert box.wait_newer(0, timeout=0.05) == (b"x", 1)
    assert box.wait_newer(1, timeout=0.05) is None
