"""Validate the fixed BPS contract on accepted Stage 2 real clouds.

The command reads the frozen SGBM winner and disparity interval from the Stage
2 report, then processes every cached tag-inpainted development and held-out
cloud for that exact backend.  It never reruns stereo and never tunes a BPS
parameter.  The emitted report records clipping margin, exact point-order
invariance, finiteness/range, and sensitivity to bounded sub-millimetre point
jitter.

Run::

    conda run --no-capture-output -n mujoco_env python -m \
        real.tracking.eval_bps --dataset datasets/sponge_20260808_203620
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from real.tracking.dense_stereo import (
    CALIBRATION_PATH,
    SGBMCandidate,
    calibration_fingerprint,
    load_config,
    preprocessing_key,
    sgbm_backend_key,
)
from src.bps import bps_fingerprint, encode_bps, load_bps_config


DEFAULT_JITTER_M = 0.0005


def validate_cached_clouds(paths: list[Path], jitter_m: float = DEFAULT_JITTER_M,
                           seed: int = 0) -> dict:
    """Return deterministic acceptance metrics for non-empty cached clouds."""
    if not paths:
        raise ValueError("no cached clouds to validate")
    if not 0.0 < jitter_m < 0.001:
        raise ValueError("validation jitter must be sub-millimetre and positive")
    config = load_bps_config()
    rng = np.random.default_rng(seed)
    maxima = []
    jitter_changes = []
    point_counts = []
    for path in sorted(paths):
        with np.load(path) as cached:
            points = cached["points"].astype(np.float64)
            left_mask_pixels = int(cached["left_mask_pixels"])
        if points.shape[0] == 0:
            raise ValueError(f"accepted cache contains an empty cloud: {path}")
        # Cached clouds are post-voxel, so their point/mask ratio is a stable
        # metadata proxy for contract validation, not the deployment worker's
        # eventual pre-voxel valid_fraction calculation.
        valid_fraction = float(np.clip(
            points.shape[0] / max(left_mask_pixels, 1), 0.0, 1.0))
        encoded = encode_bps(points, valid_fraction, config)
        permuted = encode_bps(points[::-1], valid_fraction, config)
        if not np.array_equal(encoded.distances, permuted.distances):
            raise AssertionError(f"point-order invariance failed: {path}")
        if not np.array_equal(encoded.center_base, permuted.center_base):
            raise AssertionError(f"point-order center invariance failed: {path}")
        if not np.all(np.isfinite(encoded.distances)):
            raise AssertionError(f"non-finite BPS distance: {path}")
        if np.any((encoded.distances < 0.0) | (encoded.distances > 1.0)):
            raise AssertionError(f"BPS distance outside [0, 1]: {path}")
        maxima.append(float(encoded.distances.max()))
        point_counts.append(points.shape[0])

        directions = rng.normal(size=points.shape)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.where(norms == 0.0, 1.0, norms)
        radii = rng.uniform(0.0, jitter_m, size=(points.shape[0], 1))
        jittered = encode_bps(points + directions * radii, valid_fraction, config)
        jitter_changes.append(float(np.max(
            np.abs(jittered.distances - encoded.distances))))

    max_normalized = float(np.max(maxima))
    max_jitter_change = float(np.max(jitter_changes))
    # Centering can move the cloud by at most the point perturbation, so the
    # normalized nearest-distance change is bounded by 2*jitter/cap.
    jitter_bound = 2.0 * jitter_m / config.distance_cap_m
    gates = {
        "finite_and_in_range": True,
        "point_order_invariant": True,
        "no_distance_clipping": max_normalized < 1.0,
        "submillimetre_jitter_is_lipschitz": max_jitter_change <= jitter_bound + 1e-7,
    }
    return {
        "clouds": len(paths),
        "points": {
            "min": int(np.min(point_counts)),
            "median": float(np.median(point_counts)),
            "max": int(np.max(point_counts)),
        },
        "largest_raw_basis_distance_mm": (
            max_normalized * config.distance_cap_m * 1000.0),
        "distance_cap_mm": config.distance_cap_m * 1000.0,
        "jitter_input_bound_mm": jitter_m * 1000.0,
        "jitter_max_normalized_change": max_jitter_change,
        "jitter_lipschitz_bound": jitter_bound,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }


def _write_yaml(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed BPS on Stage 2 clouds")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--sam2-model", choices=("tiny", "base+"), default="tiny")
    args = parser.parse_args()

    dense_config = load_config()
    root = (
        args.dataset
        / "dense_stereo"
        / calibration_fingerprint(args.calibration)[:16]
        / f"pre_{preprocessing_key(dense_config)}"
        / f"sam_{args.sam2_model}"
    )
    report_path = root / "sgbm_report.yaml"
    with report_path.open() as stream:
        stereo_report = yaml.safe_load(stream)
    if not stereo_report["passes_all_gates"]:
        raise RuntimeError("Stage 2 SGBM report did not pass all gates")
    candidate = SGBMCandidate(**stereo_report["frozen_candidate"])
    disparity = stereo_report["disparity"]
    backend = (
        root
        / "backends"
        / (f"sgbm_{sgbm_backend_key(dense_config, candidate)}"
           f"_d{disparity['min']}_n{disparity['num']}")
        / "inpainted"
    )
    paths = sorted(backend.glob("*.npz"))
    result = validate_cached_clouds(paths)
    bps_config = load_bps_config()
    report = {
        "schema_version": 1,
        "source": "accepted_tag_inpainted_stage2_clouds",
        "backend": str(backend.relative_to(root)),
        "frozen_bps": asdict(bps_config),
        "bps_fingerprint": bps_fingerprint(bps_config),
        **result,
    }
    output = root / "bps_report.yaml"
    _write_yaml(output, report)
    print(yaml.safe_dump(report, sort_keys=False).rstrip())
    print(f"report: {output}")
    if not result["passes_all_gates"]:
        raise RuntimeError("BPS contract validation failed")


if __name__ == "__main__":
    main()
