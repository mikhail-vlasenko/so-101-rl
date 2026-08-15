"""Latest-frame JPEG mailbox + minimal MJPEG-over-HTTP server.

Stdlib-only (`http.server` + `threading`) so any script in the repo can serve
its frames without depending on the panel's FastAPI stack. A `FrameBox` holds
exactly one frame — the latest — and readers that fall behind simply skip
ahead; there is no queue and no backlog.

`JpegStreamer` binds its port at construction time, so a busy port fails loud
with `OSError` before the owning script does any real work.

Endpoints:
    GET /stream    multipart/x-mixed-replace MJPEG (one part per published frame)
    GET /snapshot  single image/jpeg of the latest frame
Both return 503 until the first frame is published.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

STREAM_BOUNDARY = "panelframe"
# How long a /stream reader waits for the next frame before re-checking that
# the server is still up (lets streams end cleanly on close()).
_STREAM_WAIT_S = 1.0
_SERVER_POLL_S = 0.05


class FrameBox:
    """Single-slot mailbox for the latest encoded JPEG frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def latest(self) -> tuple[bytes, int] | None:
        """Return (jpeg, seq) of the newest frame, or None before the first publish."""
        with self._cond:
            if self._jpeg is None:
                return None
            return self._jpeg, self._seq

    def wait_newer(self, seen_seq: int, timeout: float) -> tuple[bytes, int] | None:
        """Block until a frame newer than seen_seq exists; None on timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seen_seq, timeout=timeout)
            if self._seq > seen_seq:
                return self._jpeg, self._seq
            return None


class JpegStreamer:
    """Serve a FrameBox over HTTP. Binds the port immediately; call start()."""

    def __init__(self, port: int, box: FrameBox, host: str = "0.0.0.0") -> None:
        self.box = box

        streamer = self

        class Handler(BaseHTTPRequestHandler):
            # One persistent connection per /stream reader; default HTTP/1.0
            # would close after the first multipart chunk on some clients.
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:
                pass  # frame-rate request logging would drown the script's own output

            def do_GET(self) -> None:
                # Ignore any query string — browsers append cache-busters.
                path = urlparse(self.path).path
                if path == "/snapshot":
                    self._snapshot()
                elif path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _snapshot(self) -> None:
                latest = streamer.box.latest()
                if latest is None:
                    self.send_error(503, "no frame published yet")
                    return
                jpeg, _ = latest
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)

            def _stream(self) -> None:
                if streamer.box.latest() is None:
                    self.send_error(503, "no frame published yet")
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = 0
                try:
                    while not streamer._closed:
                        frame = streamer.box.wait_newer(seq, timeout=_STREAM_WAIT_S)
                        if frame is None:
                            continue
                        jpeg, seq = frame
                        self.wfile.write(
                            f"--{STREAM_BOUNDARY}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # reader closed the tab; not an error for the script

        self._closed = False
        # OSError (address in use) propagates to the caller — fail loud.
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]  # actual port (resolves port=0)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        args=(_SERVER_POLL_S,),
                                        name=f"jpeg-streamer-{port}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
