"""Shared webcam access for the real rig (two Logitech C922 units).

Single source of truth for capture-device identity and the MJPG/resolution
settings, so the calibration and frame-grab scripts don't each hard-code them.

With two identical C922s plugged in, /dev/videoN indices are assigned in USB
enumeration order and can swap across reboots — a raw index silently opens the
wrong camera. Cameras are therefore identified by their USB serial (`SERIALS`)
and resolved to the current index through the udev by-id symlink at open time.
Every device argument defaults to the "main" camera, so single-camera callers
need no changes.
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

WIDTH = 1280
HEIGHT = 720
WARMUP_FRAMES = 10

# USB serials of the two rig cameras. "main" is the original mounted camera all
# existing single-camera scripts mean by default; "aux" is the second unit for
# binocular triangulation.
SERIALS = {"main": "D12FB3AF", "aux": "0AB5B87F"}


def device_index_for_serial(serial):
    """Resolve a C922 USB serial to its current /dev/videoN index.

    Uses the udev by-id symlink, which is keyed by serial and always points at
    the device's current node — stable across reboots and replugs, unlike the
    raw index. Raises if that camera isn't connected.
    """
    link = f"/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_{serial}-video-index0"
    if not os.path.exists(link):
        raise RuntimeError(f"camera with serial {serial} not connected ({link} missing)")
    return int(os.path.realpath(link).removeprefix("/dev/video"))


def resolve_device(device):
    """None -> current index of the main camera; an explicit index passes through."""
    return device_index_for_serial(SERIALS["main"]) if device is None else device


def v4l2_set(ctrl, value, device=None):
    """Set a V4L2/UVC control by name. Raises on failure — fail loud."""
    device = resolve_device(device)
    subprocess.run(["v4l2-ctl", "-d", f"/dev/video{device}", "-c", f"{ctrl}={value}"],
                   check=True)


def set_exposure(value, device=None):
    """Switch to manual exposure and set exposure_time_absolute (100 us units).

    Also pins the framerate so the camera can't lengthen exposure by dropping FPS.
    Lower value = less motion blur but darker; raise gain to compensate.
    """
    v4l2_set("auto_exposure", 1, device)               # 1 = manual exposure mode
    v4l2_set("exposure_dynamic_framerate", 0, device)
    v4l2_set("exposure_time_absolute", int(value), device)


def set_auto_exposure(device=None):
    v4l2_set("auto_exposure", 3, device)               # 3 = aperture priority (auto)


def open_camera(device=None, width=WIDTH, height=HEIGHT, warmup=WARMUP_FRAMES,
                focus=None, exposure=None, gain=None):
    """Open a C922 in MJPG at the configured resolution and warm it up.

    `device` is a /dev/videoN index, normally obtained from
    `device_index_for_serial`; None opens the main camera.

    Warmup discards the first few frames so auto-exposure/white-balance settle.
    `focus` controls the lens:
      - None      leave the camera's default (continuous autofocus).
      - int 0-250 disable autofocus and pin `focus_absolute` to this exact value
                  (step 5; 0=far, 250=near). Required for intrinsic calibration
                  and for any rig that consumes those intrinsics — calibration and
                  deployment MUST use the same value, or the focal length differs.
    `exposure` controls motion blur:
      - None      leave auto-exposure on (long exposure indoors -> blur on motion).
      - int 3-2047 switch to manual exposure and set `exposure_time_absolute` (units
                  of 100 us; lower = sharper but darker). Also pins the framerate so
                  the camera can't lengthen exposure by dropping FPS.
    `gain` (0-255) brightens the manual-exposure image without lengthening it.
    Raises RuntimeError if the device can't be opened — fail loud, no silent None.
    """
    device = resolve_device(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open /dev/video{device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Keep only the newest frame in the V4L2 queue. The default depth (~4) means
    # a consumer slower than the capture rate — AprilTag detection on a 720p
    # frame is — always dequeues stale buffered frames, so measured poses lag
    # reality by several frame intervals. Depth 1 makes cap.read() hand back the
    # freshest frame instead of draining a backlog.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if focus is not None:
        # Drive focus through v4l2-ctl: OpenCV's CAP_PROP_FOCUS silently no-ops on
        # this camera, so the lens would stay at the wrong position otherwise.
        v4l2_set("focus_automatic_continuous", 0, device)
        v4l2_set("focus_absolute", focus, device)
    if exposure is not None:
        set_exposure(exposure, device)
    if gain is not None:
        v4l2_set("gain", gain, device)
    for _ in range(warmup):
        cap.read()
    return cap
