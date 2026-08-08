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
from src.shape_obs import sqrtm_from_upper

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


# Object-channel overlay (tag-free cube obs, src/shape_obs.py): the live
# triangulated centroid as a cyan ball, the precise channel as an orange
# ellipsoid ({d: d^T M^-1 d <= 3}, the box-equivalent spread) at the served
# center — the policy's two views of the sponge, drawn where it believes them.
LIVE_POINT_RGBA = (0.0, 0.9, 0.9, 0.9)
PRECISE_ELLIPSOID_RGBA = (1.0, 0.55, 0.1, 0.45)
_LIVE_BALL_RADIUS = np.array([0.008, 0.0, 0.0])


def draw_sqrtm_ellipsoid(scn: mujoco.MjvScene, center: np.ndarray,
                         sqrtm6: np.ndarray, rgba) -> None:
    """Append the box-equivalent ellipsoid represented by one √M channel.

    Its semi-axes are √3 times √M's eigenvalues, so a true box tensor touches
    the corresponding box at every face center. A caller-selected color lets
    diagnostics overlay ground truth and an observation using identical
    geometry; rollout views use the standard orange below.
    """
    if not np.any(sqrtm6) or scn.ngeom >= scn.maxgeom:
        return
    w, V = np.linalg.eigh(sqrtm_from_upper(sqrtm6))
    if np.linalg.det(V) < 0:  # mjvGeom.mat must be a proper rotation
        V[:, 0] = -V[:, 0]
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.sqrt(3.0) * np.clip(w, 1e-4, None),
                        np.asarray(center, np.float64), V.T.flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def draw_object_channels(scn: mujoco.MjvScene, live: np.ndarray,
                         center: np.ndarray, sqrtm6: np.ndarray) -> None:
    """Append the object channels' decorative geoms to a built scene: the live
    centroid ball and the √M ellipsoid (semi-axes √3·eigenvalues along the
    eigenvectors). Channels never measured are all-zero and skipped."""
    if np.any(live) and scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            _LIVE_BALL_RADIUS, np.asarray(live, np.float64),
                            np.eye(3).flatten(),
                            np.asarray(LIVE_POINT_RGBA, np.float32))
        scn.ngeom += 1
    draw_sqrtm_ellipsoid(scn, center, sqrtm6, PRECISE_ELLIPSOID_RGBA)


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
                marker_include_rot: bool = False,
                object_channels: tuple | None = None) -> None:
        """`object_channels` optionally draws the tag-free cube obs as
        (live (3,), center (3,), sqrtm6 (6,)) — see draw_object_channels."""
        self._renderer.update_scene(data, camera=self._camera)
        if marker_pos is not None:
            draw_detected_markers(self._renderer.scene, marker_pos, marker_rot,
                                  marker_include_rot)
        if object_channels is not None:
            draw_object_channels(self._renderer.scene, *object_channels)
        rgb = self._renderer.render()
        ok, jpeg = cv2.imencode(".jpg", rgb[:, :, ::-1],
                                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG encoding of sim frame failed")
        self._box.publish(jpeg.tobytes())

    def close(self) -> None:
        self._streamer.close()
        self._renderer.close()
