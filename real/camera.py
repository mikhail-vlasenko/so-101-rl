"""Shared webcam access for the real rig (Logitech C922).

Single source of truth for the capture device and the MJPG/resolution settings,
so the calibration and frame-grab scripts don't each hard-code them.
"""
import os
import subprocess

import cv2

# cv2's pip-wheel Qt5 highgui sets QT_QPA_FONTDIR / the platform plugin path to
# its own bundled dirs (the fonts dir doesn't exist), the session exports
# "wayland;xcb" (its Qt5 has no Wayland plugin), and the Kvantum style override
# can't load — so cv2.imshow spams qt.qpa warnings on this Hyprland session.
# Override the bad values here: after cv2's import (which sets them) and before
# the first highgui call. All GUI scripts import this module, so it's the one place.
os.environ["QT_QPA_PLATFORM"] = "xcb"               # force XWayland; no Wayland plugin in cv2's Qt5
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"   # cv2's bundled fonts dir is missing
os.environ.pop("QT_STYLE_OVERRIDE", None)           # cv2's Qt can't load the system Kvantum style

DEVICE = 0
WIDTH = 1280
HEIGHT = 720
WARMUP_FRAMES = 10


def v4l2_set(ctrl, value, device=DEVICE):
    """Set a V4L2/UVC control by name. Raises on failure — fail loud."""
    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}", "-c", f"{ctrl}={value}"],
                   check=True)


def open_camera(device=DEVICE, width=WIDTH, height=HEIGHT, warmup=WARMUP_FRAMES,
                focus=None):
    """Open the C922 in MJPG at the configured resolution and warm it up.

    Warmup discards the first few frames so auto-exposure/white-balance settle.
    `focus` controls the lens:
      - None      leave the camera's default (continuous autofocus).
      - int 0-250 disable autofocus and pin `focus_absolute` to this exact value
                  (step 5; 0=far, 250=near). Required for intrinsic calibration
                  and for any rig that consumes those intrinsics — calibration and
                  deployment MUST use the same value, or the focal length differs.
    Raises RuntimeError if the device can't be opened — fail loud, no silent None.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if focus is not None:
        # Drive focus through v4l2-ctl: OpenCV's CAP_PROP_FOCUS silently no-ops on
        # this camera, so the lens would stay at the wrong position otherwise.
        v4l2_set("focus_automatic_continuous", 0, device)
        v4l2_set("focus_absolute", focus, device)
    for _ in range(warmup):
        cap.read()
    return cap
