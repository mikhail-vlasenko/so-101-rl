"""Opt-in MJPEG streaming of a script's MuJoCo state.

A script that owns a (model, data) pair constructs a `SimStreamPublisher` when
the user passes `--stream-port N` / `stream_port=N`, then calls `publish(data)`
once per control tick. Frames are rendered offscreen, JPEG-encoded once, and
served at `http://<host>:<port>/stream` (see panel.streamer for the endpoints).

Headless processes need a GL backend: the panel's runner exports
`MUJOCO_GL=egl` for streamed launches; terminal users with a display need
nothing.
"""

from __future__ import annotations

import cv2
import mujoco

from panel.streamer import FrameBox, JpegStreamer

WIDTH, HEIGHT = 640, 480
JPEG_QUALITY = 80


class SimStreamPublisher:
    def __init__(self, model: mujoco.MjModel, port: int) -> None:
        self._renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._camera)
        # The default free camera frames the whole (huge) floor; frame the arm
        # workspace instead. Same framing for every scene — the arm is at origin.
        self._camera.lookat[:] = [0.0, 0.0, 0.15]
        self._camera.distance = 1.1
        self._camera.azimuth = 135.0
        self._camera.elevation = -25.0
        self._box = FrameBox()
        self._streamer = JpegStreamer(port, self._box)
        self._streamer.start()
        print(f"sim stream: http://0.0.0.0:{self._streamer.port}/stream")

    def publish(self, data: mujoco.MjData) -> None:
        self._renderer.update_scene(data, camera=self._camera)
        rgb = self._renderer.render()
        ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1],
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding of sim frame failed")
        self._box.publish(jpeg.tobytes())

    def close(self) -> None:
        self._streamer.close()
        self._renderer.close()
