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
import numpy as np

from panel.streamer import FrameBox, JpegStreamer

WIDTH, HEIGHT = 960, 720
JPEG_QUALITY = 92

# Detected-marker overlay (drawn on top of the arm's own marker sites so the
# camera's measured pose can be eyeballed against the FK geometry). Magenta, set
# apart from the sites' green/red detected/hidden tint. Box matches the sites'
# 20 mm tag half-edge (so101.xml marker_finger/marker_wrist size); the ball is a
# touch larger so a position-only marker still reads clearly.
DETECTED_MARKER_RGBA = (1.0, 0.0, 1.0, 0.9)
_MARKER_BOX_HALF = np.array([0.01, 0.01, 0.001])
_MARKER_BALL_RADIUS = np.array([0.012, 0.0, 0.0])


def _rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector -> row-major 9-vec for mjvGeom.mat."""
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec / angle if angle > 0.0 else np.array([0.0, 0.0, 1.0])
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat


def draw_detected_markers(scn: mujoco.MjvScene, marker_pos: np.ndarray,
                          marker_rot: np.ndarray, include_rot: bool) -> None:
    """Append a decorative geom per detected marker to a built scene.

    Used identically by the rollout's passive viewer (`viewer.user_scn`, whose
    ngeom the caller resets each tick) and the MJPEG stream (the offscreen
    Renderer's scene, reset every `update_scene`). Markers the camera has never
    placed are all-zero (real/rollout/marker_obs.py) and skipped; a currently-hidden tag
    draws at its held last pose, like the policy sees it. With orientation known
    (`include_rot`) each marker is an oriented flat box — the tag square — else a
    sphere, since only its position is known.
    """
    rgba = np.asarray(DETECTED_MARKER_RGBA, dtype=np.float32)
    for pos, rot in zip(marker_pos, marker_rot):
        if not np.any(pos):
            continue
        if scn.ngeom >= scn.maxgeom:
            return
        if include_rot:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                _MARKER_BOX_HALF, np.asarray(pos, np.float64),
                                _rotvec_to_mat(rot), rgba)
        else:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                _MARKER_BALL_RADIUS, np.asarray(pos, np.float64),
                                np.eye(3).flatten(), rgba)
        scn.ngeom += 1


class SimStreamPublisher:
    def __init__(self, model: mujoco.MjModel, port: int) -> None:
        # Scenes leave the offscreen framebuffer at MuJoCo's 640x480 default;
        # enlarge it to match our stream resolution or the Renderer rejects it.
        model.vis.global_.offwidth = WIDTH
        model.vis.global_.offheight = HEIGHT
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

    def publish(self, data: mujoco.MjData, marker_pos: np.ndarray | None = None,
                marker_rot: np.ndarray | None = None,
                marker_include_rot: bool = False) -> None:
        self._renderer.update_scene(data, camera=self._camera)
        if marker_pos is not None:
            draw_detected_markers(self._renderer.scene, marker_pos, marker_rot,
                                  marker_include_rot)
        rgb = self._renderer.render()
        ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1],
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding of sim frame failed")
        self._box.publish(jpeg.tobytes())

    def close(self) -> None:
        self._streamer.close()
        self._renderer.close()
