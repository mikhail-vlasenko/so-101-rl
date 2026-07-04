"""Per-pose residual map for the encoder-bias calibration (real/calibrate_qpos.py).

For every captured pose this shows, per arm tag, the gap between where the
*calibrated model* predicts the tag (`FK(theta_enc - bias)`, base frame) and where
the *camera* actually saw it (`T_base_cam . tvec`, base frame). The arrow is planted
at the model prediction and points to reality, so its length is the leftover error
after calibration and its direction tells you *which way* the model is wrong there.

The committed artifacts are evaluated as-is — bias from calibration.yaml, camera
from extrinsics.yaml, marker sites (solved mounts included) from the XML — with the
same gravity-settled FK the solve used. No re-solving, so the plot can never drift
from the model deployment actually uses. Detections the calibration rejected as
outliers are still plotted; they stick out, which is the point of a residual map.

Two base-frame views (top X-Y and side X-Z) together carry all three error
components; arrow colour is the full 3-D magnitude so hotspots show in either view.
Residuals are tiny next to the workspace, so arrows are drawn exaggerated (see the
reference arrow in the legend).

Run:
    conda run -n mujoco_env python -m real.plot_calib_residuals
    conda run -n mujoco_env python -m real.plot_calib_residuals --from-samples <json>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import seaborn as sns

from real.calib_solve import load_samples
from real.calibrate_qpos import DEFAULT_CAL, DEFAULT_XML, SAMPLES_PATH, _true_poses, paired_points
from real.calibration import load_calibration, load_compliance
from real.extrinsics import load_extrinsics
from real.marker_spec import ARM_TAG_TO_SITE
from real.twin.mapping import JOINT_NAMES, load_joint_maps
from src.base_env import tag_cam_world_pos

DEFAULT_OUT = Path(__file__).resolve().parent / "calib_residuals.png"
TAG_COLORS = {0: "#1f77b4", 2: "#d62728"}   # marker_finger, marker_wrist
EXAGGERATE = 10.0       # arrow length = residual * this, in metres of data space
REF_MM = 5.0            # reference-arrow length drawn in the legend
DPI = 220


def compute_residuals(samples, model, data, qposadr, site_ids):
    """Return (p_model, resid, tags, b_full): per detection, the committed-model
    base position, the (camera - model) error vector, and the tag id. Pure
    evaluation of the saved artifacts (calibration.yaml bias + compliance,
    extrinsics.yaml camera, XML sites) through the settled FK — no re-solving."""
    b_full = load_calibration()
    compliance = load_compliance()
    _, T_base_cam, _, _ = load_extrinsics()
    corrected = _true_poses(samples, model, data, qposadr, b_full, compliance)
    src, dst, tags = paired_points(corrected, model, data, qposadr, site_ids)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p_model = dst                       # settled FK of the bias+compliance-corrected pose
    p_obs = src @ R.T + t               # camera tag centre mapped into base frame
    return p_model, p_obs - p_model, tags, b_full


def _quiver_panel(ax, p_model, resid, tags, mag_mm, ai, bi, cam_pos, labels):
    """One 2-D projection (axes ai, bi of the base frame). Arrows model->reality,
    exaggerated; coloured by full 3-D magnitude. Returns the quiver for the colorbar."""
    norm = plt.Normalize(0.0, mag_mm.max())
    q = None
    for tag in sorted(set(tags)):
        m = tags == tag
        ax.scatter(p_model[m, ai], p_model[m, bi], s=12, facecolors="none",
                   edgecolors=TAG_COLORS[tag], linewidths=0.8,
                   label=f"tag {tag} ({ARM_TAG_TO_SITE[tag]})")
        q = ax.quiver(p_model[m, ai], p_model[m, bi],
                      resid[m, ai] * EXAGGERATE, resid[m, bi] * EXAGGERATE,
                      mag_mm[m], cmap="viridis", norm=norm,
                      angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ax.scatter(cam_pos[ai], cam_pos[bi], marker="*", s=180, color="goldenrod",
               edgecolors="k", linewidths=0.5, zorder=5, label="camera")
    ax.set_xlabel(f"{labels[ai]} (m)")
    ax.set_ylabel(f"{labels[bi]} (m)")
    ax.set_aspect("equal", adjustable="datalim")
    return q


def _reference_arrow(ax):
    """Scale key: a horizontal arrow REF_MM long (same EXAGGERATE as the data
    arrows), drawn in the empty band of the top panel. Manual rather than
    `quiverkey`, which mis-scales against an angles='xy' quiver."""
    length = REF_MM / 1000.0 * EXAGGERATE
    x0, y0 = 0.09, -0.25
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.6))
    ax.text(x0 + length / 2, y0 + 0.018, f"{REF_MM:.0f} mm error",
            ha="center", va="bottom", color="0.25", fontsize=13)


def plot(p_model, resid, tags, cam_pos, out_path, rms_mm):
    sns.set_theme(style="whitegrid", context="talk")
    mag_mm = np.linalg.norm(resid, axis=1) * 1000.0
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    _quiver_panel(axes[0], p_model, resid, tags, mag_mm, 0, 1, cam_pos, labels)
    axes[0].set_title("top view (base X-Y)")
    q = _quiver_panel(axes[1], p_model, resid, tags, mag_mm, 0, 2, cam_pos, labels)
    axes[1].set_title("side view (base X-Z)")

    _reference_arrow(axes[0])
    fig.colorbar(q, ax=axes, shrink=0.8, label="error magnitude (mm)", pad=0.02)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"calibrated model vs. reality — {len(tags)} tag detections, "
                 f"RMS {rms_mm:.1f} mm  (arrows ×{EXAGGERATE:.0f})", fontsize=13)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-samples", default=str(SAMPLES_PATH))
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--cal", default=str(DEFAULT_CAL))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    jm = load_joint_maps(model, Path(args.cal))
    site_ids = {}
    for tag_id, site_name in ARM_TAG_TO_SITE.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert sid >= 0, f"site {site_name!r} not found in model"
        site_ids[tag_id] = sid

    samples = load_samples(args.from_samples)
    print(f"loaded {len(samples)} samples from {args.from_samples}")
    p_model, resid, tags, b_full = compute_residuals(
        samples, model, data, jm.qposadr(), site_ids)
    rms_mm = np.sqrt(np.mean(np.sum(resid ** 2, axis=1))) * 1000.0
    print("committed bias (deg): " + ", ".join(
        f"{n}={np.degrees(b):+.2f}" for n, b in zip(JOINT_NAMES, b_full)))
    print(f"residual RMS {rms_mm:.2f} mm over {len(tags)} detections "
          "(all detections incl. any the solve rejected)")
    plot(p_model, resid, tags, tag_cam_world_pos(model, data), args.out, rms_mm)


if __name__ == "__main__":
    main()
