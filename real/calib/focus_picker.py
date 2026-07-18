"""Find the focus_absolute value that makes the working scene sharpest.

The C922's `focus_absolute` is a MANUAL target only — with continuous autofocus
on, firmware drives the lens internally and never reports that position back, so
you can't read the AF-chosen value. Instead we do what AF does, explicitly:
autofocus OFF, sweep focus_absolute, and pick the value that maximises image
sharpness (variance of the Laplacian over a centre ROI). The winning number is
what you pin via FOCUS_ABSOLUTE in calibrate_camera.

Focus is driven through v4l2-ctl (the authoritative UVC control); OpenCV's
CAP_PROP_FOCUS silently no-ops on some builds.

Controls (focus the preview window):
  - / =   step focus down / up by FOCUS_STEP
  a       auto-sweep the full range and jump to the sharpest value
  q / ESC quit

Run:
    conda run -n mujoco_env python -m real.calib.focus_picker
"""
import cv2
import numpy as np

from real.vision.camera import open_camera, v4l2_set, HEIGHT

FOCUS_MIN, FOCUS_MAX, FOCUS_STEP = 0, 250, 5
ROI_FRAC = 0.4                 # centre fraction of the frame used for the sharpness metric
SETTLE_FRAMES = 6              # frames to discard after moving focus (motor + buffer lag)


def center_roi(frame):
    h, w = frame.shape[:2]
    rh, rw = int(h * ROI_FRAC), int(w * ROI_FRAC)
    y, x = (h - rh) // 2, (w - rw) // 2
    return frame[y:y + rh, x:x + rw]


def sharpness(cap):
    """Mean variance-of-Laplacian over a couple of fresh centre-ROI frames."""
    vals = []
    for _ in range(3):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        gray = cv2.cvtColor(center_roi(frame), cv2.COLOR_BGR2GRAY)
        vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.mean(vals))


def set_focus(cap, value):
    value = int(np.clip(value, FOCUS_MIN, FOCUS_MAX))
    v4l2_set("focus_absolute", value)
    for _ in range(SETTLE_FRAMES):
        cap.read()
    return value


def auto_sweep(cap):
    print("sweeping focus_absolute ...")
    best_val, best_sharp = FOCUS_MIN, -1.0
    for v in range(FOCUS_MIN, FOCUS_MAX + 1, FOCUS_STEP):
        set_focus(cap, v)
        s = sharpness(cap)
        print(f"  focus={v:3d}  sharpness={s:9.1f}")
        if s > best_sharp:
            best_val, best_sharp = v, s
    print(f"sharpest: focus_absolute = {best_val}  (sharpness {best_sharp:.1f})")
    set_focus(cap, best_val)
    return best_val


def main():
    cap = open_camera(focus=None)
    v4l2_set("focus_automatic_continuous", 0)     # AF off so focus_absolute actually drives the lens
    focus = set_focus(cap, FOCUS_MIN)
    print("autofocus OFF — use -/= to step, 'a' to auto-sweep, q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        s = cv2.Laplacian(cv2.cvtColor(center_roi(frame), cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var()

        h, w = frame.shape[:2]
        rh, rw = int(h * ROI_FRAC), int(w * ROI_FRAC)
        cv2.rectangle(frame, ((w - rw) // 2, (h - rh) // 2),
                      ((w + rw) // 2, (h + rh) // 2), (0, 200, 0), 1)
        cv2.putText(frame, f"focus={focus}  sharpness={s:.0f}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "-/= step   a auto-sweep   q quit", (12, HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("focus picker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("-"):
            focus = set_focus(cap, focus - FOCUS_STEP)
            print(f"focus_absolute = {focus}")
        elif key in (ord("="), ord("+")):
            focus = set_focus(cap, focus + FOCUS_STEP)
            print(f"focus_absolute = {focus}")
        elif key == ord("a"):
            focus = auto_sweep(cap)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nfinal focus_absolute = {focus}  -> set FOCUS_ABSOLUTE in calibrate_camera.py")


if __name__ == "__main__":
    main()
