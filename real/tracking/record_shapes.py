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
follow the printed reminder. The shared stereo viewer outlines table/sponge
tags and shows anchor status, the current static label, GT center and coverage;
press q/Esc to stop cleanly. Use `--no-gui` for unattended capture.

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

from real.calib.extrinsics import (
    average_transforms,
    mat_to_pos_quat,
)
from real.calib.table_anchor import TableAnchorTracker
from real.marker_spec import SPONGE_TAG_IDS, TABLE_TAG_IDS
from real.tracking.tag_body_calib import SPONGE_TAGS_PATH, body_pose_from_tag, load_sponge_tags
from real.vision.detect import make_detector
from real.vision.overlay import (
    GREEN,
    RED,
    TABLE_BLUE,
    WHITE,
    YELLOW,
    OverlayLine,
    StereoViewer,
    TagStyle,
    annotate_tags,
)
from real.vision.pose import PoseEstimator
from real.vision.stereo_rig import CAMERA_NAMES, open_rig_camera
from src.shape_obs import STATIC_DWELL_S, is_static

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LIFT_CONFIG_PATH = REPO_ROOT / "conf" / "env" / "lift.yaml"
CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"


def load_workspace_bounds():
    """Cube spawn xy bounds used to bin dataset coverage."""
    with LIFT_CONFIG_PATH.open() as stream:
        task = yaml.safe_load(stream)["lift_env"]
    return (np.asarray(task["cube_low"], dtype=np.float64),
            np.asarray(task["cube_high"], dtype=np.float64))


def load_static_mean_window_s():
    """Causal denoising window used before applying the shared static gate."""
    with CONFIG_PATH.open() as stream:
        value = yaml.safe_load(stream)["dense_stereo_feasibility"][
            "static_mean_window_s"]
    value = float(value)
    assert 0.0 < value < STATIC_DWELL_S
    return value


class CausalMeanPosition:
    """Suppress pose-estimator jitter without looking into future frames."""

    def __init__(self, window_s):
        self.window_s = float(window_s)
        assert self.window_s > 0.0
        self._times = []
        self._positions = []

    def update(self, t, position):
        t = float(t)
        self._times.append(t)
        self._positions.append(np.asarray(position, dtype=np.float64).copy())
        while len(self._times) > 1 and self._times[0] < t - self.window_s:
            self._times.pop(0)
            self._positions.pop(0)
        return np.mean(np.stack(self._positions), axis=0)


class Coverage:
    """Live tallies of what the dataset has covered so far."""

    def __init__(self, workspace_low, workspace_high):
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)
        self.counts = {}          # (face, yaw_bin, region) -> frames
        self.static_frames = 0
        self.moving_frames = 0
        self.outside_workspace_frames = 0
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
        if np.any(xy < self.workspace_low) or np.any(xy > self.workspace_high):
            self.outside_workspace_frames += 1
            return
        cell = np.clip(((xy - self.workspace_low)
                        / (self.workspace_high - self.workspace_low) * 2)
                       .astype(int), 0, 1)
        key = (face, yaw_bin, f"{cell[0]}{cell[1]}")
        self.counts[key] = self.counts.get(key, 0) + 1

    def report(self):
        lines = [f"coverage: static={self.static_frames} moving={self.moving_frames} "
                 f"settles={self.settle_events} "
                 f"outside={self.outside_workspace_frames}"]
        for face in "xyz":
            cells = {k: v for k, v in self.counts.items() if k[0] == face}
            total = sum(cells.values())
            yaws = [sum(v for k, v in cells.items() if k[1] == b) for b in range(4)]
            regions = sorted({k[2] for k in cells})
            lines.append(f"  face {face}-up: {total:5d} frames | "
                         f"yaw bins {yaws} | regions {regions}")
        return "\n".join(lines)

    def preview_summary(self):
        totals = {face: sum(value for (seen_face, _, _), value in self.counts.items()
                            if seen_face == face)
                  for face in "xyz"}
        return (f"coverage static={self.static_frames} moving={self.moving_frames} "
                f"settles={self.settle_events} "
                f"outside={self.outside_workspace_frames}  face frames "
                + " ".join(f"{face}={totals[face]}" for face in "xyz"))


def main():
    parser = argparse.ArgumentParser(description="Record the sponge shape dataset")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--family", choices=("apriltag", "aruco"), default="apriltag")
    parser.add_argument("--out", default=None,
                        help="dataset directory (default: datasets/sponge_<stamp>)")
    parser.add_argument("--no-gui", action="store_true",
                        help="disable the annotated stereo preview")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else REPO_ROOT / "datasets" / f"sponge_{stamp}"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)

    half_extents, tag_transforms = load_sponge_tags()
    detector = make_detector(args.family)
    wanted = set(SPONGE_TAG_IDS) | set(TABLE_TAG_IDS)

    caps, mats, dists, estimators, anchors = {}, {}, {}, {}, {}
    for name in CAMERA_NAMES:
        caps[name], mats[name], dists[name] = open_rig_camera(name)
        estimators[name] = PoseEstimator(mats[name], dists[name])
        anchors[name] = TableAnchorTracker(mats[name], dists[name])
    viewer = None if args.no_gui else StereoViewer("sponge shape dataset recorder")
    tag_styles = {tag: TagStyle(f"table {tag}", TABLE_BLUE, 3)
                  for tag in TABLE_TAG_IDS}
    tag_styles.update({tag: TagStyle(f"sponge tag {tag}", GREEN, 3)
                       for tag in tag_transforms})

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

    coverage = Coverage(*load_workspace_bounds())
    body_hist_t: list[float] = []
    body_hist_p: list[np.ndarray] = []
    static_filter = CausalMeanPosition(load_static_mean_window_s())
    started = time.monotonic()
    deadline = started + args.minutes * 60.0
    k = 0
    index = open(out_dir / "index.jsonl", "w")
    try:
        while time.monotonic() < deadline:
            k += 1
            record = {"k": k, "t": {}, "frame": {}, "T_base_cam": {},
                      "tags": {}, "body": {}}
            T_base_body_views = []
            frames = {}
            frame_dets = {}
            for name in CAMERA_NAMES:
                ok, frame = caps[name].read()
                t_recv = time.monotonic()
                if not ok:
                    raise RuntimeError(f"camera read failed on '{name}'")
                frames[name] = frame
                rel = f"frames/{name}_{k:06d}.jpg"
                cv2.imwrite(str(out_dir / rel), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dets = {d.id: d for d in detector.detect(gray) if d.id in wanted}
                frame_dets[name] = dets
                record["t"][name] = t_recv
                record["frame"][name] = rel

                tags = {}
                for tag, det in dets.items():
                    rvec, tvec = estimators[name].estimate(det)
                    tags[tag] = {"corners": det.corners.tolist(),
                                 "rvec": rvec.tolist(), "tvec": tvec.tolist()}
                record["tags"][name] = tags

                anchors[name].observe(dets)
                T_base_cam = anchors[name].value()
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
            fused = None
            if T_base_body_views:
                fused = average_transforms(T_base_body_views)
                pos, quat = mat_to_pos_quat(fused)
                record["body"]["fused"] = {"pos": pos.tolist(), "quat": quat.tolist()}
                t_now = float(np.mean([record["t"][n] for n in CAMERA_NAMES]))
                body_hist_t.append(t_now)
                body_hist_p.append(static_filter.update(t_now, fused[:3, 3]))
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

            if viewer is not None:
                annotated = {name: annotate_tags(
                    frames[name].copy(), frame_dets[name].values(), tag_styles)
                    for name in CAMERA_NAMES}
                camera_lines = {}
                for name in CAMERA_NAMES:
                    anchor_ok = record["T_base_cam"][name] is not None
                    sponge_seen = sorted(set(frame_dets[name]) & set(tag_transforms))
                    ok = anchor_ok and bool(sponge_seen)
                    camera_lines[name] = [OverlayLine(
                        f"{name}  table={'OK' if anchor_ok else 'MISSING'}  "
                        f"sponge tags={sponge_seen}", GREEN if ok else RED)]
                if static is True:
                    state, state_color = "STATIC", GREEN
                elif static is False:
                    state, state_color = "MOVING", YELLOW
                else:
                    state, state_color = "NO GT", RED
                elapsed = time.monotonic() - started
                gt_text = ("GT center unavailable" if fused is None else
                           "GT center " + np.array2string(
                               fused[:3, 3] * 100.0, precision=1,
                               suppress_small=True) + " cm")
                header_lines = [
                    OverlayLine(
                        f"RECORDING {elapsed:5.1f}/{args.minutes * 60.0:.0f}s  "
                        f"frame {k}  {state}", state_color),
                    OverlayLine(gt_text, WHITE),
                    OverlayLine(coverage.preview_summary(), WHITE),
                    OverlayLine("Move, release, hold  |  q/ESC stop safely", WHITE),
                ]
                key = viewer.show(annotated, camera_lines, header_lines)
                if key in (27, ord("q")):
                    break

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
        detector.close()
        if viewer is not None:
            viewer.close()

    print(f"\nrecorded {k} frame pairs into {out_dir}")
    print(coverage.report())


if __name__ == "__main__":
    main()
