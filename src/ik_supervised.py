"""Train the offline SO-101 inverse-kinematics capacity sweep.

The process reads ``datasets/ik_capacity.npz`` and has no MuJoCo dependency.
Each MLP receives normalized ``[current_q(6), sponge_xyz(3)]`` and predicts the
joint-range-normalized displacement of the five joints that move
``gripperframe``. A linear baseline is trained alongside the configured
depth/width grid. Validation combines held-out samples and held-out spatial
cells; test predictions are saved but never consulted for training, early
stopping, or architecture selection.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from src.ik_common import (
    ARM_JOINT_NAMES,
    SPLIT_NAMES,
    TEST_CELL,
    TEST_SAMPLE,
    TRAIN,
    VAL_CELL,
    VAL_SAMPLE,
)


@dataclass(frozen=True)
class ArchitectureSpec:
    depth: int
    width: int
    seed: int
    baseline: bool = False

    @property
    def name(self) -> str:
        if self.baseline:
            return f"linear_seed{self.seed}"
        return f"depth{self.depth}_width{self.width}_seed{self.seed}"


class IKMLP(nn.Module):
    """Plain fixed-width MLP used to isolate depth and width effects."""

    def __init__(self, input_dim: int, output_dim: int, depth: int,
                 width: int, activation: str):
        super().__init__()
        if depth == 0:
            assert width == 0
            self.network = nn.Linear(input_dim, output_dim)
            return
        assert depth > 0 and width > 0
        if activation == "silu":
            activation_type = nn.SiLU
        elif activation == "relu":
            activation_type = nn.ReLU
        elif activation == "tanh":
            activation_type = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation!r}")
        layers: list[nn.Module] = []
        layer_input = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(layer_input, width))
            layers.append(activation_type())
            layer_input = width
        layers.append(nn.Linear(layer_input, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arrays(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        required = {
            "current_q", "sponge_xyz", "target_q", "split", "joint_limits",
            "target_low_m", "target_high_m", "metadata_json",
        }
        missing = required - set(dataset.files)
        assert not missing, f"IK dataset is missing arrays: {sorted(missing)}"
        arrays = {name: dataset[name].copy() for name in required if name != "metadata_json"}
        metadata = json.loads(str(dataset["metadata_json"]))
    assert tuple(metadata["arm_joint_names"]) == ARM_JOINT_NAMES
    assert tuple(metadata["split_names"]) == SPLIT_NAMES
    return arrays


def normalized_supervised_arrays(
        arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized features and labels using dataset/model bounds only."""
    joint_limits = arrays["joint_limits"]
    joint_center = joint_limits.mean(axis=1)
    joint_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
    target_center = 0.5 * (arrays["target_low_m"] + arrays["target_high_m"])
    target_half_range = 0.5 * (arrays["target_high_m"] - arrays["target_low_m"])
    q_features = (arrays["current_q"] - joint_center) / joint_half_range
    xyz_features = (arrays["sponge_xyz"] - target_center) / target_half_range
    features = np.concatenate([q_features, xyz_features], axis=1).astype(np.float32)

    arm_count = len(ARM_JOINT_NAMES)
    arm_range = joint_limits[:arm_count, 1] - joint_limits[:arm_count, 0]
    labels = ((arrays["target_q"][:, :arm_count]
               - arrays["current_q"][:, :arm_count]) / arm_range).astype(np.float32)
    return features, labels


def _predict(model: nn.Module, features: torch.Tensor,
             batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(
                model(features[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _regression_metrics(prediction: np.ndarray, label: np.ndarray,
                        current_q: np.ndarray, joint_limits: np.ndarray) -> dict[str, float]:
    arm_count = len(ARM_JOINT_NAMES)
    arm_range = joint_limits[:arm_count, 1] - joint_limits[:arm_count, 0]
    normalized_error = prediction - label
    radian_error = normalized_error * arm_range
    predicted_q = current_q[:, :arm_count] + prediction * arm_range
    violations = np.logical_or(
        predicted_q < joint_limits[:arm_count, 0],
        predicted_q > joint_limits[:arm_count, 1],
    )
    return {
        "normalized_mse": float(np.mean(normalized_error ** 2)),
        "normalized_rmse": float(np.sqrt(np.mean(normalized_error ** 2))),
        "joint_rmse_rad": float(np.sqrt(np.mean(radian_error ** 2))),
        "joint_mae_rad": float(np.mean(np.abs(radian_error))),
        "joint_limit_violation_fraction": float(np.mean(np.any(violations, axis=1))),
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _initialize_training_worker(threads_per_job: int) -> None:
    """Set process-wide Torch pools once before a worker handles many runs."""
    torch.set_num_threads(threads_per_job)
    torch.set_num_interop_threads(1)


def _train_one(spec: ArchitectureSpec, dataset_path: str, results_dir: str,
               sweep_config: dict, fingerprint: str) -> dict:
    device = torch.device(str(sweep_config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "ik_capacity.sweep.device=cuda but CUDA is unavailable. The Codex "
            "filesystem sandbox hides /dev/nvidia*; launch this command with "
            "host/GPU access instead of falling back to CPU.")
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
    np.random.seed(spec.seed)
    torch.use_deterministic_algorithms(True)

    run_dir = Path(results_dir) / "runs" / spec.name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if (existing["dataset_sha256"] == fingerprint
                and existing["sweep_config"] == sweep_config):
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays = _load_arrays(dataset_path)
    features_np, labels_np = normalized_supervised_arrays(arrays)
    split = arrays["split"]
    train_indices = np.flatnonzero(split == TRAIN)
    val_indices = np.flatnonzero(
        np.logical_or(split == VAL_SAMPLE, split == VAL_CELL))
    test_indices = np.flatnonzero(
        np.logical_or(split == TEST_SAMPLE, split == TEST_CELL))
    features = torch.from_numpy(features_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)

    depth = 0 if spec.baseline else spec.depth
    width = 0 if spec.baseline else spec.width
    model = IKMLP(
        features.shape[1], labels.shape[1], depth, width,
        str(sweep_config["activation"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(sweep_config["learning_rate"]))
    batch_size = int(sweep_config["batch_size"])
    max_epochs = int(sweep_config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=float(sweep_config["learning_rate_min"]),
    )
    patience = int(sweep_config["early_stopping_patience"])
    min_delta = float(sweep_config["early_stopping_min_delta"])
    generator = torch.Generator(device=device).manual_seed(spec.seed)
    train_index_tensor = torch.from_numpy(train_indices).to(device)
    val_index_tensor = torch.from_numpy(val_indices).to(device)
    test_index_tensor = torch.from_numpy(test_indices).to(device)
    val_features = features[val_index_tensor]
    val_labels = labels[val_index_tensor]

    best_val = np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []
    started = time.monotonic()
    for epoch in range(max_epochs):
        model.train()
        permutation = train_index_tensor[
            torch.randperm(len(train_index_tensor), generator=generator, device=device)]
        train_squared_error = 0.0
        train_values = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            prediction = model(features[indices])
            loss = torch.mean((prediction - labels[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_squared_error += float(loss.detach()) * prediction.numel()
            train_values += prediction.numel()
        train_mse = train_squared_error / train_values

        model.eval()
        with torch.inference_mode():
            val_mse = float(torch.mean((model(val_features) - val_labels) ** 2))
        history.append({"epoch": epoch + 1, "train_mse": train_mse,
                        "val_mse": val_mse,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_mse < best_val - min_delta:
            best_val = val_mse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        scheduler.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed_seconds = time.monotonic() - started

    train_prediction = _predict(model, features[train_index_tensor], batch_size)
    val_prediction = _predict(model, val_features, batch_size)
    test_prediction = _predict(model, features[test_index_tensor], batch_size)
    metrics = {
        "run_name": spec.name,
        "depth": depth,
        "width": width,
        "seed": spec.seed,
        "baseline": spec.baseline,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": elapsed_seconds,
        "dataset_sha256": fingerprint,
        "sweep_config": sweep_config,
        "train": _regression_metrics(
            train_prediction, labels_np[train_indices],
            arrays["current_q"][train_indices], arrays["joint_limits"]),
        "validation": _regression_metrics(
            val_prediction, labels_np[val_indices],
            arrays["current_q"][val_indices], arrays["joint_limits"]),
        "validation_sample": _regression_metrics(
            val_prediction[split[val_indices] == VAL_SAMPLE],
            labels_np[val_indices][split[val_indices] == VAL_SAMPLE],
            arrays["current_q"][val_indices][split[val_indices] == VAL_SAMPLE],
            arrays["joint_limits"]),
        "validation_cell": _regression_metrics(
            val_prediction[split[val_indices] == VAL_CELL],
            labels_np[val_indices][split[val_indices] == VAL_CELL],
            arrays["current_q"][val_indices][split[val_indices] == VAL_CELL],
            arrays["joint_limits"]),
        "history": history,
    }
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": features.shape[1],
            "output_dim": labels.shape[1],
            "depth": depth,
            "width": width,
            "activation": sweep_config["activation"],
            "dataset_sha256": fingerprint,
        },
        run_dir / "model.pt",
    )
    prediction_path = run_dir / "predictions.tmp.npz"
    np.savez_compressed(
        prediction_path,
        val_indices=val_indices.astype(np.int32),
        val_delta_normalized=val_prediction,
        test_indices=test_indices.astype(np.int32),
        test_delta_normalized=test_prediction,
    )
    os.replace(prediction_path, run_dir / "predictions.npz")
    _atomic_json(metrics_path, metrics)
    return metrics


def architecture_specs(cfg: dict) -> list[ArchitectureSpec]:
    specs = [
        ArchitectureSpec(int(depth), int(width), int(seed))
        for depth in cfg["depths"]
        for width in cfg["widths"]
        for seed in cfg["seeds"]
    ]
    specs.append(ArchitectureSpec(0, 0, int(cfg["seeds"][0]), baseline=True))
    return specs


def _write_metrics_csv(results_dir: Path, metrics: list[dict]) -> None:
    rows = []
    for item in metrics:
        rows.append({
            "run_name": item["run_name"],
            "depth": item["depth"],
            "width": item["width"],
            "seed": item["seed"],
            "baseline": item["baseline"],
            "parameter_count": item["parameter_count"],
            "best_epoch": item["best_epoch"],
            "epochs_ran": item["epochs_ran"],
            "elapsed_seconds": item["elapsed_seconds"],
            "train_normalized_rmse": item["train"]["normalized_rmse"],
            "val_normalized_rmse": item["validation"]["normalized_rmse"],
            "val_sample_normalized_rmse": item[
                "validation_sample"]["normalized_rmse"],
            "val_cell_normalized_rmse": item[
                "validation_cell"]["normalized_rmse"],
            "val_joint_rmse_rad": item["validation"]["joint_rmse_rad"],
            "val_joint_limit_violation_fraction": item[
                "validation"]["joint_limit_violation_fraction"],
        })
    path = results_dir / "supervised_metrics.csv"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["run_name"]))
    os.replace(temporary, path)


def _resolved_path(original_cwd: str, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(original_cwd) / value


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()
    experiment_cfg = cfg.ik_capacity
    dataset_path = _resolved_path(original_cwd, str(experiment_cfg.dataset_path))
    assert dataset_path.exists(), (
        f"IK dataset not found: {dataset_path}; run python -m src.ik_dataset first")
    results_dir = _resolved_path(original_cwd, str(experiment_cfg.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_sha256(dataset_path)
    sweep_config = OmegaConf.to_container(experiment_cfg.sweep, resolve=True)
    assert isinstance(sweep_config, dict)
    specs = architecture_specs(sweep_config)
    device = torch.device(str(sweep_config["device"]))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not visible. Run outside the restricted "
                "filesystem sandbox so the process can access /dev/nvidia*.")
        device_description = torch.cuda.get_device_name(device)
    else:
        device_description = str(device)
    print(f"Training {len(specs) - 1} MLP runs plus one linear baseline on "
          f"{device_description}", flush=True)

    metrics = []
    completed = 0
    parallel_jobs = int(sweep_config["parallel_jobs"])

    def record(item: dict) -> None:
        nonlocal completed
        metrics.append(item)
        completed += 1
        print(
            f"  {completed:3d}/{len(specs)} {item['run_name']}: "
            f"val normalized RMSE={item['validation']['normalized_rmse']:.6f} "
            f"epoch={item['best_epoch']}",
            flush=True,
        )

    if parallel_jobs == 1:
        _initialize_training_worker(int(sweep_config["threads_per_job"]))
        for spec in specs:
            item = _train_one(
                spec, str(dataset_path), str(results_dir), sweep_config, fingerprint)
            record(item)
    else:
        assert device.type == "cpu", (
            "parallel_jobs > 1 is CPU-only; a single process must own the GPU")
        with ProcessPoolExecutor(
                max_workers=parallel_jobs,
                initializer=_initialize_training_worker,
                initargs=(int(sweep_config["threads_per_job"]),)) as executor:
            futures = {
                executor.submit(
                    _train_one,
                    spec,
                    str(dataset_path),
                    str(results_dir),
                    sweep_config,
                    fingerprint,
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                item = future.result()
                record(item)
    _write_metrics_csv(results_dir, metrics)
    print(f"Saved supervised results to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
