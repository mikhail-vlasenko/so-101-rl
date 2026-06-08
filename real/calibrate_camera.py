"""Camera intrinsic calibration from a plain checkerboard.

HOLD the board in front of the camera and move it around — do NOT lay it flat on
the table and shoot once. Intrinsics (focal length, principal point, lens
distortion) are solved from many views at different orientations. A single
coplanar shot is degenerate and gives garbage distortion. Aim for ~20 views that:
  - tilt the board at various angles (not parallel to the sensor),
  - vary distance (near and far), and
  - push the board into every corner of the frame (distortion lives at the edges).

Checkerboard geometry: OpenCV counts INNER corners, not squares. A board with
N×M squares has (N-1)×(M-1) inner corners. We auto-detect across a small set of
candidate inner-corner sizes and lock onto the first that matches, so a hand
miscount can't silently wreck the run — the matched size is printed and drawn.

Controls (focus the preview window):
  SPACE      force-capture the current frame
  BACKSPACE  drop the last captured view
  ENTER / q  finish and solve (needs >= MIN_VIEWS)
  ESC        abort without solving

Run:
    conda run -n mujoco_env python -m real.calibrate_camera
"""
import os
import time

import cv2
import numpy as np
import yaml

from real.camera import open_camera, WIDTH, HEIGHT

SQUARE_SIZE_M = 0.02                      # printed checkerboard square edge, metres
FOCUS_ABSOLUTE = 30                       # pinned lens focus (0-250, step 5); rig MUST reuse this
# Candidate INNER-corner dimensions (squares - 1 per axis). 8x10 squares -> 7x9.
CANDIDATE_PATTERNS = [(7, 9), (6, 8), (8, 10), (5, 7)]
TARGET_VIEWS = 20
MIN_VIEWS = 12
AUTO_MIN_SHIFT_PX = 60.0                  # min mean-corner move from last capture to auto-grab
AUTO_MIN_GAP_S = 0.8                      # min time between auto-captures
STILL_PX = 1.5                            # max frame-to-frame corner motion to count as "still"
PRUNE_FACTOR = 2.0                        # drop views with per-view RMS > factor * median, then refit
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OUT_PATH = os.path.join(os.path.dirname(__file__), "camera_intrinsics.yaml")


def detect(gray, pattern):
    """Find + subpixel-refine inner corners for one pattern size. Returns Nx1x2 or None."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


def object_points(pattern):
    """Metric 3D coordinates of the inner corners on the board plane (z=0)."""
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return pts * SQUARE_SIZE_M


def solve(objpoints, imgpoints, image_size, pattern):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        # Per-view RMS in pixels, same convention as the overall `rms` so the two
        # are directly comparable: sqrt(sum_sq / n_points) = L2norm / sqrt(n).
        per_view.append(cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj)))
    return rms, K, dist, per_view


def save(K, dist, rms, pattern, n_views, image_size):
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "reprojection_error_px": float(rms),
        "pattern_inner_corners": list(pattern),
        "square_size_m": SQUARE_SIZE_M,
        "focus_absolute": FOCUS_ABSOLUTE,
        "num_views": n_views,
    }
    with open(OUT_PATH, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)
    print(f"\nwrote {OUT_PATH}")


def main():
    cap = open_camera(focus=FOCUS_ABSOLUTE)   # pin focus: drifting focal length ruins intrinsics
    image_size = (WIDTH, HEIGHT)

    locked = None                  # inner-corner pattern, fixed after first detection
    objpoints, imgpoints = [], []
    centroids = []                 # captured-board centroids, for coverage overlay
    prev_corners = None            # previous frame's corners, for stillness check
    last_capture = None            # corners at last capture, for spacing
    last_auto_t = 0.0

    print(__doc__)
    print("Move the board around; auto-capture is on. Press ENTER/q when done.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed mid-session")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = None
        if locked is not None:
            corners = detect(gray, locked)
        else:
            for pat in CANDIDATE_PATTERNS:
                corners = detect(gray, pat)
                if corners is not None:
                    locked = pat
                    print(f"locked checkerboard inner corners = {pat} "
                          f"({pat[0] + 1}x{pat[1] + 1} squares)")
                    break

        view = frame.copy()
        still = False
        if corners is not None:
            cv2.drawChessboardCorners(view, locked, corners, True)
            motion = (np.inf if prev_corners is None or prev_corners.shape != corners.shape
                      else float(np.linalg.norm(corners - prev_corners, axis=2).mean()))
            still = motion < STILL_PX
            shift = (np.inf if last_capture is None
                     else float(np.linalg.norm(corners - last_capture, axis=2).mean()))
            now = time.time()
            # Auto-grab only when the board has settled (sharp, no motion blur) and
            # has moved enough since the last capture to add a genuinely new view.
            if still and shift > AUTO_MIN_SHIFT_PX and now - last_auto_t > AUTO_MIN_GAP_S:
                objpoints.append(object_points(locked))
                imgpoints.append(corners)
                centroids.append(corners.reshape(-1, 2).mean(axis=0))
                last_capture, last_auto_t = corners, now
            prev_corners = corners

        for c in centroids:
            cv2.circle(view, (int(c[0]), int(c[1])), 4, (0, 200, 0), -1)

        board_state = "STILL-ok" if still else ("hold still" if corners is not None else "no board")
        status = (f"views {len(imgpoints)}/{TARGET_VIEWS}   {board_state}   "
                  f"pattern={locked if locked else '?'}")
        cv2.putText(view, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if corners is not None else (0, 0, 255), 2)
        cv2.putText(view, "SPACE grab  BACKSPACE undo  ENTER/q solve  ESC abort",
                    (12, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 32 and corners is not None:          # SPACE
            objpoints.append(object_points(locked))
            imgpoints.append(corners)
            centroids.append(corners.reshape(-1, 2).mean(axis=0))
            last_capture, last_auto_t = corners, time.time()
        elif key == 8 and imgpoints:                   # BACKSPACE
            objpoints.pop(); imgpoints.pop(); centroids.pop()
        elif key in (13, ord("q")):                    # ENTER / q
            break
        elif key == 27:                                # ESC
            cap.release(); cv2.destroyAllWindows(); print("aborted"); return

    cap.release()
    cv2.destroyAllWindows()

    if len(imgpoints) < MIN_VIEWS:
        raise RuntimeError(f"only {len(imgpoints)} views; need >= {MIN_VIEWS}")

    rms, K, dist, per_view = solve(objpoints, imgpoints, image_size, locked)
    print(f"\nRMS reprojection error: {rms:.3f} px  (good < 0.5, acceptable < 1.0)")
    order = np.argsort(per_view)[::-1]
    print("worst views (per-view RMS, px):  " +
          "  ".join(f"#{i}={per_view[i]:.2f}" for i in order[:5]))

    # Drop views well above the median and refit; high-error views are usually
    # blurred or warped frames that drag the whole solve up.
    keep = [i for i in range(len(per_view)) if per_view[i] <= PRUNE_FACTOR * np.median(per_view)]
    if MIN_VIEWS <= len(keep) < len(per_view):
        rms2, K2, dist2, pv2 = solve([objpoints[i] for i in keep],
                                     [imgpoints[i] for i in keep], image_size, locked)
        print(f"after pruning {len(per_view) - len(keep)} outlier view(s): "
              f"RMS {rms2:.3f} px over {len(keep)} views")
        rms, K, dist = rms2, K2, dist2
        objpoints = [objpoints[i] for i in keep]

    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    save(K, dist, rms, locked, len(objpoints), image_size)


if __name__ == "__main__":
    main()
