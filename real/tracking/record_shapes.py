"""Record the shape-tracking dataset: raw dual-camera frames + tag GT.

Captures both rig cameras into `datasets/sponge_<stamp>/` — per-frame JPEGs,
all sponge/table tag detections, the EMA'd per-camera `T_base_cam`, the GT
sponge body pose (from any solved sponge tag, real/tracking/tag_body_calib.py)
and timestamps — one JSON line per frame pair in `index.jsonl`. Masks are
deliberately NOT recorded: they are computed offline by the estimator eval
(real/tracking/eval_estimator.py), so the dataset stays estimator-agnostic and
one recording session serves every candidate.

Static/moving is auto-labeled from the GT body track with the same
`is_static` gate the online pipeline uses (src/shape_obs.py). A live coverage
checklist (resting-face class x yaw bin x workspace region, static/moving,
settle events) prints as you go; occlusion coverage (park the arm over the
sponge at roughly 0/25/50/75 percent per camera) is a manual protocol step —
follow the printed reminder.

Run:
    conda run -n mujoco_env python -m real.tracking.record_shapes --minutes 10
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from real.calib.extrinsics import (
    PoseEMA,
    average_transforms,
    base_cam_from_table,
    load_extrinsics,
    mat_to_pos_quat,
    rt_to_mat,
)
from real.marker_spec import SPONGE_TAG_IDS, TABLE_TAG_ID
from real.tracking.tag_body_calib import SPONGE_TAGS_PATH, body_pose_from_tag, load_sponge_tags
from real.vision.detect import make_detector
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera
from src.shape_obs import STATIC_DWELL_S, is_static

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Same table-anchor smoothing as the rollout pipeline (real/rollout/marker_obs.py).
CAM_EMA_ALPHA = 0.05

# Coverage bins: workspace xy box (the cube spawn region, conf/env/lift.yaml)
# split 2x2, body yaw in 4 bins, resting face from the most vertical body axis.
WORKSPACE_LOW = np.array([0.15, -0.15])
WORKSPACE_HIGH = np.array([0.30, 0.15])


class Coverage:
    """Live tallies of what the dataset has covered so far."""

    def __init__(self):
        self.counts = {}          # (face, yaw_bin, region) -> frames
        self.static_frames = 0
        self.moving_frames = 0
        self.settle_events = 0
        self._prev_static = None

    def add(self, T_base_body, static):
        if static is None:
            return
        if static:
            self.static_frames += 1
        else:
            self.moving_frames += 1
        if self._prev_static is False and static:
            self.settle_events += 1
        self._prev_static = static
        R = T_base_body[:3, :3]
        face = "xyz"[int(np.argmax(np.abs(R[2, :])))]
        # Yaw of the body x-axis's horizontal projection (falls back to y when
        # x is the vertical axis).
        axis = R[:, 0] if face != "x" else R[:, 1]
        yaw_bin = int(((np.arctan2(axis[1], axis[0]) + np.pi) / (np.pi / 2))) % 4
        xy = T_base_body[:2, 3]
        cell = np.clip(((xy - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) * 2)
                       .astype(int), 0, 1)
        key = (face, yaw_bin, f"{cell[0]}{cell[1]}")
        self.counts[key] = self.counts.get(key, 0) + 1

    def report(self):
        lines = [f"coverage: static={self.static_frames} moving={self.moving_frames} "
                 f"settles={self.settle_events}"]
        for face in "xyz":
            cells = {k: v for k, v in self.counts.items() if k[0] == face}
            total = sum(cells.values())
            yaws = [sum(v for k, v in cells.items() if k[1] == b) for b in range(4)]
            regions = sorted({k[2] for k in cells})
            lines.append(f"  face {face}-up: {total:5d} frames | "
                         f"yaw bins {yaws} | regions {regions}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Record the sponge shape dataset")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--out", default=None,
                        help="dataset directory (default: datasets/sponge_<stamp>)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "datasets" / f"sponge_{stamp}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    T_base_table, _, _, _ = load_extrinsics()
    half_extents, tag_transforms = load_sponge_tags()
    detector = make_detector(args.family)
    wanted = set(SPONGE_TAG_IDS) | {TABLE_TAG_ID}

    caps, mats, dists, estimators, cam_ema = {}, {}, {}, {}, {}
    for name in CAMERA_NAMES:
        caps[name], mats[name], dists[name] = open_rig_camera(name)
        estimators[name] = PoseEstimator(mats[name], dists[name])
        cam_ema[name] = PoseEMA(CAM_EMA_ALPHA)

    with open(out_dir / "meta.yaml", "w") as f:
        yaml.safe_dump({
            "stamp": stamp,
            "family": args.family,
            "cameras": list(CAMERA_NAMES),
            "half_extents": [float(h) for h in half_extents],
            "sponge_tags": str(SPONGE_TAGS_PATH),
            "camera_matrix": {n: mats[n].tolist() for n in CAMERA_NAMES},
            "dist_coeffs": {n: dists[n].tolist() for n in CAMERA_NAMES},
        }, f, default_flow_style=None, sort_keys=False)

    print("protocol: cover 3 resting-face classes x yaws x workspace positions;\n"
          "park the arm over the sponge at ~0/25/50/75% occlusion per camera;\n"
          "add moving segments (hand-carried + in-gripper) and let it settle\n"
          "between placements. Ctrl-C or --minutes ends the session.")

    coverage = Coverage()
    body_hist_t: list[float] = []
    body_hist_p: list[np.ndarray] = []
    deadline = time.monotonic() + args.minutes * 60.0
    k = 0
    index = open(out_dir / "index.jsonl", "w")
    try:
        while time.monotonic() < deadline:
            k += 1
            record = {"k": k, "t": {}, "frame": {}, "T_base_cam": {},
                      "tags": {}, "body": {}}
            T_base_body_views = []
            for name in CAMERA_NAMES:
                ok, frame = caps[name].read()
                t_recv = time.monotonic()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{name}'")
                rel = f"frames/{name}_{k:06d}.jpg"
                cv2.imwrite(str(out_dir / rel), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = {d.id: d for d in detector.detect(gray) if d.id in wanted}
                record["t"][name] = t_recv
                record["frame"][name] = rel

                tags = {}
                for tag, det in dets.items():
                    rvec, tvec = estimators[name].estimate(det)
                    tags[tag] = {"corners": det.corners.tolist(),
                                 "rvec": rvec.tolist(), "tvec": tvec.tolist()}
                record["tags"][name] = tags

                T_base_cam = None
                if TABLE_TAG_ID in dets:
                    T_base_cam = cam_ema[name].update(base_cam_from_table(
                        T_base_table,
                        np.array(tags[TABLE_TAG_ID]["rvec"]),
                        np.array(tags[TABLE_TAG_ID]["tvec"])))
                if T_base_cam is None:
                    record["T_base_cam"][name] = None
                    record["body"][name] = None
                    continue
                pos, quat = mat_to_pos_quat(T_base_cam)
                record["T_base_cam"][name] = {"pos": pos.tolist(),
                                              "quat": quat.tolist()}
                # GT body pose from every solved sponge tag in this view.
                candidates = [body_pose_from_tag(np.array(t["rvec"]),
                                                 np.array(t["tvec"]),
                                                 tag_transforms[tag])
                              for tag, t in tags.items() if tag in tag_transforms]
                if not candidates:
                    record["body"][name] = None
                    continue
                T_base_body = T_base_cam @ average_transforms(candidates)
                T_base_body_views.append(T_base_body)
                pos, quat = mat_to_pos_quat(T_base_body)
                record["body"][name] = {"pos": pos.tolist(), "quat": quat.tolist(),
                                        "n_tags": len(candidates)}

            static = None
            if T_base_body_views:
                fused = average_transforms(T_base_body_views)
                pos, quat = mat_to_pos_quat(fused)
                record["body"]["fused"] = {"pos": pos.tolist(), "quat": quat.tolist()}
                t_now = float(np.mean([record["t"][n] for n in CAMERA_NAMES]))
                body_hist_t.append(t_now)
                body_hist_p.append(fused[:3, 3])
                cutoff = t_now - 2.0 * STATIC_DWELL_S
                while len(body_hist_t) > 2 and body_hist_t[0] < cutoff:
                    body_hist_t.pop(0)
                    body_hist_p.pop(0)
                static = bool(is_static(body_hist_t, body_hist_p))
                coverage.add(fused, static)
            else:
                record["body"]["fused"] = None
            record["static"] = static
            index.write(json.dumps(record) + "\n")

            if k % 60 == 0:
                index.flush()
                print(f"[{k} frame pairs]")
                print(coverage.report(), flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        index.close()
        for cap in caps.values():
            cap.release()

    print(f"\nrecorded {k} frame pairs into {out_dir}")
    print(coverage.report())


if __name__ == "__main__":
    main()
