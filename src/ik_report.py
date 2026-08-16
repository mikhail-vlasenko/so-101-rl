"""Evaluate and report the supervised IK capacity experiment.

All architectures are compared on validation. The depth/width with the lowest
mean validation end-effector error across seeds is selected, and only that
architecture is evaluated on test. MuJoCo is used here solely to forward the
saved predicted joint configurations through the static SO-101 kinematic tree.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig

from src.ik_common import (
    ARM_JOINT_NAMES,
    TEST_CELL,
    TEST_SAMPLE,
    VAL_CELL,
    VAL_SAMPLE,
)
from src.robot_spec import EE_SITE_NAME, JOINT_NAMES


class ForwardEvaluator:
    """Static batched-through-a-loop FK for saved network predictions."""

    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ])
        self.joint_qposadr = self.model.jnt_qposadr[joint_ids]
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

    def positions(self, q: np.ndarray) -> np.ndarray:
        result = np.empty((len(q), 3), dtype=np.float64)
        for sample_id, pose in enumerate(q):
            self.data.qpos[self.joint_qposadr] = pose
            mujoco.mj_kinematics(self.model, self.data)
            result[sample_id] = self.data.site_xpos[self.ee_site_id]
        return result


def _load_dataset(path: Path) -> dict[str, np.ndarray | dict]:
    with np.load(path, allow_pickle=False) as dataset:
        arrays = {name: dataset[name].copy() for name in dataset.files
                  if name != "metadata_json"}
        arrays["metadata"] = json.loads(str(dataset["metadata_json"]))
    return arrays


def _task_arrays(dataset: dict, indices: np.ndarray, prediction: np.ndarray,
                 evaluator: ForwardEvaluator) -> dict[str, np.ndarray]:
    arm_count = len(ARM_JOINT_NAMES)
    joint_limits = dataset["joint_limits"]
    arm_range = joint_limits[:arm_count, 1] - joint_limits[:arm_count, 0]
    current_q = dataset["current_q"][indices]
    predicted_q = current_q.copy()
    predicted_q[:, :arm_count] += prediction * arm_range
    desired_xyz = dataset["sponge_xyz"][indices] + np.asarray(
        dataset["metadata"]["target_offset_m"])
    predicted_xyz = evaluator.positions(predicted_q)
    error_m = np.linalg.norm(predicted_xyz - desired_xyz, axis=1)

    teacher_delta = ((dataset["target_q"][indices, :arm_count]
                      - current_q[:, :arm_count]) / arm_range)
    teacher_travel = np.linalg.norm(teacher_delta, axis=1)
    predicted_travel = np.linalg.norm(prediction, axis=1)
    violation = np.any(
        np.logical_or(
            predicted_q[:, :arm_count] < joint_limits[:arm_count, 0],
            predicted_q[:, :arm_count] > joint_limits[:arm_count, 1],
        ),
        axis=1,
    )
    return {
        "error_m": error_m,
        "teacher_travel": teacher_travel,
        "predicted_travel": predicted_travel,
        "joint_limit_violation": violation,
    }


def _task_metrics(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    error_mm = 1000.0 * values["error_m"][mask]
    travel_gap = (values["predicted_travel"] - values["teacher_travel"])[mask]
    violation = values["joint_limit_violation"][mask]
    return {
        "count": int(len(error_mm)),
        "position_mean_mm": float(np.mean(error_mm)),
        "position_rmse_mm": float(np.sqrt(np.mean(error_mm ** 2))),
        "position_median_mm": float(np.median(error_mm)),
        "position_p95_mm": float(np.percentile(error_mm, 95)),
        "success_5mm": float(np.mean(error_mm <= 5.0)),
        "success_10mm": float(np.mean(error_mm <= 10.0)),
        "success_20mm": float(np.mean(error_mm <= 20.0)),
        "normalized_travel_gap_mean": float(np.mean(travel_gap)),
        "joint_limit_violation_fraction": float(np.mean(violation)),
    }


def _evaluate_prediction_file(dataset: dict, evaluator: ForwardEvaluator,
                              prediction_path: Path, partition: str) -> tuple[dict, dict]:
    with np.load(prediction_path, allow_pickle=False) as predictions:
        indices = predictions[f"{partition}_indices"]
        delta = predictions[f"{partition}_delta_normalized"]
    values = _task_arrays(dataset, indices, delta, evaluator)
    split = dataset["split"][indices]
    if partition == "val":
        sample_code, cell_code = VAL_SAMPLE, VAL_CELL
    elif partition == "test":
        sample_code, cell_code = TEST_SAMPLE, TEST_CELL
    else:
        raise ValueError(f"Unknown partition: {partition!r}")
    metrics = {
        "combined": _task_metrics(values, np.ones(len(indices), dtype=bool)),
        "sample": _task_metrics(values, split == sample_code),
        "cell": _task_metrics(values, split == cell_code),
    }
    raw = {**values, "indices": indices, "split": split}
    return metrics, raw


def _flatten_row(run: dict, metrics: dict) -> dict:
    row = {
        "run_name": run["run_name"],
        "depth": run["depth"],
        "width": run["width"],
        "seed": run["seed"],
        "baseline": run["baseline"],
        "parameter_count": run["parameter_count"],
    }
    for component, component_metrics in metrics.items():
        for name, value in component_metrics.items():
            row[f"{component}_{name}"] = value
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _architecture_aggregates(rows: list[dict]) -> list[dict]:
    architectures = sorted({
        (row["depth"], row["width"])
        for row in rows if not row["baseline"]
    })
    aggregates = []
    for depth, width in architectures:
        selected = [row for row in rows
                    if row["depth"] == depth and row["width"] == width]
        values = np.asarray([row["combined_position_mean_mm"] for row in selected])
        aggregates.append({
            "depth": depth,
            "width": width,
            "seeds": len(selected),
            "parameter_count": selected[0]["parameter_count"],
            "val_position_mean_mm_mean": float(np.mean(values)),
            "val_position_mean_mm_std": float(np.std(
                values, ddof=1 if len(values) > 1 else 0)),
            "val_sample_position_mean_mm_mean": float(np.mean([
                row["sample_position_mean_mm"] for row in selected])),
            "val_cell_position_mean_mm_mean": float(np.mean([
                row["cell_position_mean_mm"] for row in selected])),
            "val_success_10mm_mean": float(np.mean([
                row["combined_success_10mm"] for row in selected])),
        })
    return aggregates


def _make_plots(results_dir: Path, aggregates: list[dict], cell_rows: list[dict],
                spatial_cells: tuple[int, int, int]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ik-capacity")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = sorted({row["depth"] for row in aggregates})
    widths = sorted({row["width"] for row in aggregates})
    matrix = np.full((len(depths), len(widths)), np.nan)
    for row in aggregates:
        matrix[depths.index(row["depth"]), widths.index(row["width"])] = row[
            "val_position_mean_mm_mean"]
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(widths)), widths)
    axis.set_yticks(range(len(depths)), depths)
    axis.set_xlabel("Hidden width")
    axis.set_ylabel("Hidden depth")
    axis.set_title("Validation mean end-effector error (mm; lower is better)")
    for row_id in range(len(depths)):
        for column_id in range(len(widths)):
            axis.text(column_id, row_id, f"{matrix[row_id, column_id]:.2f}",
                      ha="center", va="center", color="black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Mean error (mm)")
    figure.tight_layout()
    figure.savefig(results_dir / "validation_depth_width.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for depth in depths:
        selected = sorted(
            (row for row in aggregates if row["depth"] == depth),
            key=lambda row: row["parameter_count"],
        )
        axis.plot(
            [row["parameter_count"] for row in selected],
            [row["val_position_mean_mm_mean"] for row in selected],
            marker="o",
            label=f"depth {depth}",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters")
    axis.set_ylabel("Validation mean end-effector error (mm)")
    axis.set_title("IK approximation error versus model size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "validation_parameter_scaling.png", dpi=180)
    plt.close(figure)

    nx, ny, nz = spatial_cells
    figure, axes = plt.subplots(1, nz, figsize=(4 * nz + 1, 4.5), squeeze=False)
    cell_values = {row["cell_id"]: row["position_mean_mm"] for row in cell_rows}
    for z_index in range(nz):
        grid = np.full((ny, nx), np.nan)
        for x_index in range(nx):
            for y_index in range(ny):
                cell_id = np.ravel_multi_index(
                    (x_index, y_index, z_index), spatial_cells)
                if cell_id in cell_values:
                    grid[y_index, x_index] = cell_values[cell_id]
        image = axes[0, z_index].imshow(
            grid, origin="lower", vmin=0.0,
            vmax=np.nanpercentile([row["position_mean_mm"] for row in cell_rows], 95),
            cmap="magma",
        )
        axes[0, z_index].set_title(f"z cell {z_index}")
        axes[0, z_index].set_xlabel("x cell")
        axes[0, z_index].set_ylabel("y cell")
        axes[0, z_index].set_facecolor("#eeeeee")
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.18, top=0.82, wspace=0.32)
    color_axis = figure.add_axes([0.90, 0.22, 0.012, 0.54])
    figure.colorbar(image, cax=color_axis, label="Test mean error (mm)")
    figure.suptitle("Selected architecture: test spatial error")
    figure.text(
        0.47, 0.06,
        "Blank cells have no test examples (validation-only or unreachable cell).",
        ha="center",
    )
    figure.savefig(results_dir / "selected_test_spatial_error.png", dpi=180)
    plt.close(figure)


def _resolved_path(original_cwd: str, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(original_cwd) / value


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()
    experiment_cfg = cfg.ik_capacity
    dataset_path = _resolved_path(original_cwd, str(experiment_cfg.dataset_path))
    results_dir = _resolved_path(original_cwd, str(experiment_cfg.results_dir))
    runs_dir = results_dir / "runs"
    dataset = _load_dataset(dataset_path)
    model_path = _resolved_path(
        original_cwd, str(experiment_cfg.dataset.model_path))
    evaluator = ForwardEvaluator(str(model_path))
    run_paths = sorted(runs_dir.glob("*/metrics.json"))
    assert run_paths, f"No trained IK runs found under {runs_dir}"

    desired_xyz = dataset["sponge_xyz"] + np.asarray(
        dataset["metadata"]["target_offset_m"])
    teacher_xyz = evaluator.positions(dataset["target_q"])
    teacher_error_mm = 1000.0 * np.linalg.norm(teacher_xyz - desired_xyz, axis=1)
    dataset_quality = {
        "samples": int(len(teacher_error_mm)),
        "occupied_spatial_cells": int(len(np.unique(dataset["cell_id"]))),
        "teacher_position_mean_mm": float(np.mean(teacher_error_mm)),
        "teacher_position_p95_mm": float(np.percentile(teacher_error_mm, 95)),
        "teacher_position_max_mm": float(np.max(teacher_error_mm)),
        "rejected_target_attempts": int(
            dataset["metadata"]["rejected_target_attempts"]),
    }
    (results_dir / "dataset_quality.json").write_text(
        json.dumps(dataset_quality, indent=2, sort_keys=True) + "\n")

    validation_rows = []
    print(f"Evaluating validation predictions for {len(run_paths)} runs", flush=True)
    for run_id, metrics_path in enumerate(run_paths, start=1):
        run = json.loads(metrics_path.read_text())
        task_metrics, _ = _evaluate_prediction_file(
            dataset, evaluator, metrics_path.parent / "predictions.npz", "val")
        validation_rows.append(_flatten_row(run, task_metrics))
        if run_id % 10 == 0 or run_id == len(run_paths):
            print(f"  evaluated {run_id}/{len(run_paths)}", flush=True)
    _write_csv(results_dir / "validation_task_metrics.csv", validation_rows)
    aggregates = _architecture_aggregates(validation_rows)
    _write_csv(results_dir / "validation_architectures.csv", aggregates)
    best = min(aggregates, key=lambda row: row["val_position_mean_mm_mean"])

    selected_validation = [
        row for row in validation_rows
        if row["depth"] == best["depth"] and row["width"] == best["width"]
    ]
    test_rows = []
    selected_raw = []
    for row in selected_validation:
        run_dir = runs_dir / row["run_name"]
        run = json.loads((run_dir / "metrics.json").read_text())
        task_metrics, raw = _evaluate_prediction_file(
            dataset, evaluator, run_dir / "predictions.npz", "test")
        test_rows.append(_flatten_row(run, task_metrics))
        selected_raw.append(raw)
    _write_csv(results_dir / "selected_test_metrics.csv", test_rows)

    all_errors = np.concatenate([raw["error_m"] for raw in selected_raw])
    all_indices = np.concatenate([raw["indices"] for raw in selected_raw])
    cell_rows = []
    for cell_id in np.unique(dataset["cell_id"][all_indices]):
        mask = dataset["cell_id"][all_indices] == cell_id
        errors_mm = 1000.0 * all_errors[mask]
        cell_rows.append({
            "cell_id": int(cell_id),
            "count_across_seeds": int(np.count_nonzero(mask)),
            "position_mean_mm": float(np.mean(errors_mm)),
            "position_p95_mm": float(np.percentile(errors_mm, 95)),
        })
    _write_csv(results_dir / "selected_test_cell_metrics.csv", cell_rows)

    linear_row = next(row for row in validation_rows if row["baseline"])
    test_combined_mean = float(np.mean([
        row["combined_position_mean_mm"] for row in test_rows]))
    test_values = [row["combined_position_mean_mm"] for row in test_rows]
    test_combined_std = float(np.std(
        test_values, ddof=1 if len(test_values) > 1 else 0))
    test_sample_mean = float(np.mean([
        row["sample_position_mean_mm"] for row in test_rows]))
    test_cell_mean = float(np.mean([
        row["cell_position_mean_mm"] for row in test_rows]))
    selection = {
        "depth": best["depth"],
        "width": best["width"],
        "parameter_count": best["parameter_count"],
        "selection_metric": "mean validation end-effector error across seeds",
        "validation_position_mean_mm": best["val_position_mean_mm_mean"],
        "validation_position_std_mm": best["val_position_mean_mm_std"],
        "test_position_mean_mm": test_combined_mean,
        "test_position_std_mm": test_combined_std,
        "test_sample_position_mean_mm": test_sample_mean,
        "test_cell_position_mean_mm": test_cell_mean,
    }
    (results_dir / "selected_architecture.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n")

    top_architectures = sorted(
        aggregates, key=lambda row: row["val_position_mean_mm_mean"])[:5]
    lines = [
        "# Offline IK capacity experiment",
        "",
        (f"Selected **depth {best['depth']}, width {best['width']}** "
         f"({best['parameter_count']:,} parameters) using validation only."),
        "",
        (f"Validation mean end-effector error: "
         f"{best['val_position_mean_mm_mean']:.3f} ± "
         f"{best['val_position_mean_mm_std']:.3f} mm across seeds."),
        (f"Test mean end-effector error: {test_combined_mean:.3f} ± "
         f"{test_combined_std:.3f} mm across seeds; held-out samples "
         f"{test_sample_mean:.3f} mm, held-out cells {test_cell_mean:.3f} mm."),
        (f"Linear baseline validation error: "
         f"{linear_row['combined_position_mean_mm']:.3f} mm."),
        (f"Teacher-label FK residual: mean "
         f"{dataset_quality['teacher_position_mean_mm']:.4f} mm, maximum "
         f"{dataset_quality['teacher_position_max_mm']:.4f} mm."),
        "",
        "## Five best architectures on validation",
        "",
        "| Depth | Width | Parameters | Mean error (mm) | 10 mm success |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in top_architectures:
        lines.append(
            f"| {row['depth']} | {row['width']} | {row['parameter_count']} | "
            f"{row['val_position_mean_mm_mean']:.3f} | "
            f"{100.0 * row['val_success_10mm_mean']:.1f}% |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n")

    _make_plots(
        results_dir,
        aggregates,
        cell_rows,
        tuple(int(v) for v in dataset["metadata"]["spatial_cells"]),
    )
    print(json.dumps(selection, indent=2), flush=True)
    print(f"Saved report to {results_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
