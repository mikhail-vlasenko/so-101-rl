"""Generate the offline supervised inverse-kinematics benchmark dataset.

This module is the only training-side component that uses MuJoCo. It loads the
SO-101 MJCF as a static kinematic tree: there are no environment instances,
physics steps, cameras, rewards, or rollouts. For every example it samples a
collision-free current pose and a sponge point in the configured 3D box, then
solves

    min_q sum_i ((q_i - q_current_i) / joint_range_i) ** 2
    subject to gripperframe_xyz(q) = sponge_xyz + target_offset

over the five joints that can move ``gripperframe``. The gripper joint remains
at its current value. Multiple SLSQP starts are tried and the nearest valid,
collision-free terminal solution is retained.

Split codes retain evaluation provenance. Validation and test each contain the
same two components: random samples held out within cells otherwise present in
training, and all samples from a disjoint set of spatial cells. Architecture
selection must use validation; test is reserved for the selected depth/width.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Each generator process solves tiny 5-variable problems. Letting BLAS create a
# full thread pool per process only oversubscribes the 16-core host.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import Bounds, minimize

from src.ik_common import (
    ARM_JOINT_NAMES,
    SPLIT_NAMES,
    TEST_CELL,
    TEST_SAMPLE,
    TRAIN,
    VAL_CELL,
    VAL_SAMPLE,
)
from src.robot_spec import EE_SITE_NAME, JOINT_NAMES


@dataclass(frozen=True)
class WorkerConfig:
    model_path: str
    count: int
    seed: int
    target_low_m: tuple[float, float, float]
    target_high_m: tuple[float, float, float]
    target_offset_m: tuple[float, float, float]
    position_tolerance_m: float
    joint_limit_margin_fraction: float
    ik_restarts: int
    ik_max_iterations: int
    max_sample_attempts: int


class IKTeacher:
    """Static MuJoCo FK/Jacobian model plus closest-terminal-pose IK."""

    def __init__(self, model_path: str, position_tolerance_m: float,
                 joint_limit_margin_fraction: float, ik_max_iterations: int):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.asarray([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ])
        assert np.all(self.joint_ids >= 0), "SO-101 joint missing from IK model"
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
        assert self.ee_site_id >= 0, f"site {EE_SITE_NAME!r} missing from IK model"

        joint_range = self.model.jnt_range[self.joint_ids].copy()
        self.joint_low = joint_range[:, 0]
        self.joint_high = joint_range[:, 1]
        self.joint_range = self.joint_high - self.joint_low
        margin = joint_limit_margin_fraction * self.joint_range
        self.sample_low = self.joint_low + margin
        self.sample_high = self.joint_high - margin
        assert np.all(self.sample_low < self.sample_high)

        self.arm_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if self.model.geom_group[geom_id] == 3
        }
        self.position_tolerance_m = position_tolerance_m
        self.ik_max_iterations = ik_max_iterations
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._cached_x: np.ndarray | None = None
        self._cached_position = np.zeros(3, dtype=np.float64)
        self._cached_jacobian = np.zeros((3, len(ARM_JOINT_NAMES)), dtype=np.float64)

    def _write_pose(self, q: np.ndarray) -> None:
        self.data.qpos[self.joint_qposadr] = q

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Return gripperframe xyz using kinematics only."""
        self._write_pose(q)
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.ee_site_id].copy()

    def collision_free(self, q: np.ndarray) -> bool:
        """Check terminal arm/environment and self collisions without stepping."""
        self._write_pose(q)
        mujoco.mj_forward(self.model, self.data)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if contact.geom1 in self.arm_geom_ids or contact.geom2 in self.arm_geom_ids:
                return False
        return True

    def sample_current(self, rng: np.random.Generator,
                       max_attempts: int) -> np.ndarray:
        """Sample one collision-free current pose inside the configured margin."""
        for _ in range(max_attempts):
            q = rng.uniform(self.sample_low, self.sample_high)
            if self.collision_free(q):
                return q
        raise RuntimeError(
            f"Could not sample a collision-free current pose in {max_attempts} attempts")

    def _position_and_jacobian(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return self._cached_position, self._cached_jacobian
        self.data.qpos[self.joint_qposadr[:len(ARM_JOINT_NAMES)]] = x
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._jacp.fill(0.0)
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, None, self.ee_site_id)
        self._cached_x = x.copy()
        self._cached_position = self.data.site_xpos[self.ee_site_id].copy()
        self._cached_jacobian = self._jacp[
            :, self.joint_dofadr[:len(ARM_JOINT_NAMES)]].copy()
        return self._cached_position, self._cached_jacobian

    def solve(self, current_q: np.ndarray, target_xyz: np.ndarray,
              rng: np.random.Generator, restarts: int) -> np.ndarray | None:
        """Return the nearest valid local IK optimum across deterministic starts."""
        assert current_q.shape == (len(JOINT_NAMES),)
        assert target_xyz.shape == (3,)
        n_arm = len(ARM_JOINT_NAMES)
        arm_range = self.joint_range[:n_arm]
        arm_low = self.sample_low[:n_arm]
        arm_high = self.sample_high[:n_arm]
        current_arm = current_q[:n_arm]
        self._write_pose(current_q)

        def objective(x: np.ndarray) -> float:
            delta = (x - current_arm) / arm_range
            return float(delta @ delta)

        def objective_jacobian(x: np.ndarray) -> np.ndarray:
            return 2.0 * (x - current_arm) / (arm_range * arm_range)

        def constraint(x: np.ndarray) -> np.ndarray:
            position, _ = self._position_and_jacobian(x)
            return position - target_xyz

        def constraint_jacobian(x: np.ndarray) -> np.ndarray:
            _, jacobian = self._position_and_jacobian(x)
            return jacobian

        constraints = {
            "type": "eq",
            "fun": constraint,
            "jac": constraint_jacobian,
        }
        starts = [current_arm]
        starts.extend(
            rng.uniform(arm_low, arm_high) for _ in range(restarts - 1))

        best_q = None
        best_distance = np.inf
        for start in starts:
            self._cached_x = None
            result = minimize(
                objective,
                start,
                method="SLSQP",
                jac=objective_jacobian,
                bounds=Bounds(arm_low, arm_high),
                constraints=constraints,
                options={"maxiter": self.ik_max_iterations, "ftol": 1e-10,
                         "disp": False},
            )
            q = current_q.copy()
            q[:n_arm] = result.x
            residual = float(np.linalg.norm(self.forward(q) - target_xyz))
            if residual > self.position_tolerance_m:
                continue
            if not self.collision_free(q):
                continue
            distance = objective(result.x)
            if distance < best_distance:
                best_q = q
                best_distance = distance
        return best_q


def spatial_cell_ids(points: np.ndarray, low: np.ndarray, high: np.ndarray,
                     cells: np.ndarray) -> np.ndarray:
    """Map XYZ points to flat cell ids in a fixed workspace grid."""
    assert points.ndim == 2 and points.shape[1] == 3
    assert low.shape == high.shape == cells.shape == (3,)
    assert np.all(high > low) and np.all(cells > 0)
    normalized = (points - low) / (high - low)
    indices = np.floor(normalized * cells).astype(np.int64)
    indices = np.clip(indices, 0, cells - 1)
    return np.ravel_multi_index(indices.T, tuple(int(v) for v in cells))


def assign_structured_splits(
        points: np.ndarray, low: np.ndarray, high: np.ndarray, cells: np.ndarray,
        validation_cell_fraction: float, test_cell_fraction: float,
        validation_sample_fraction: float, test_sample_fraction: float,
        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Assign matched sample/cell validation and test components.

    Held-out-cell membership is exclusive. Inside every sufficiently populated
    remaining cell, samples are independently partitioned into train,
    validation-sample, and test-sample groups with equal split logic for
    validation and test. Cells too sparse to contain all three stay in training.
    """
    assert 0.0 < validation_cell_fraction < 1.0
    assert 0.0 < test_cell_fraction < 1.0
    assert validation_cell_fraction + test_cell_fraction < 1.0
    assert 0.0 < validation_sample_fraction < 1.0
    assert 0.0 < test_sample_fraction < 1.0
    assert validation_sample_fraction + test_sample_fraction < 1.0

    cell_ids = spatial_cell_ids(points, low, high, cells)
    occupied = np.unique(cell_ids)
    shuffled_cells = rng.permutation(occupied)
    n_val_cells = max(1, int(round(validation_cell_fraction * len(occupied))))
    n_test_cells = max(1, int(round(test_cell_fraction * len(occupied))))
    assert n_val_cells + n_test_cells < len(occupied), (
        "spatial grid has too few occupied cells for train/validation/test")
    heldout_cells = shuffled_cells[:n_val_cells + n_test_cells]
    counts = {
        int(cell_id): int(np.count_nonzero(cell_ids == cell_id))
        for cell_id in heldout_cells
    }
    # Equal cell counts alone can yield very different component sizes because
    # reachability makes occupancy nonuniform. Greedily balance sample mass while
    # preserving the requested number of disjoint cells in each partition.
    val_cells_list = []
    test_cells_list = []
    val_samples = 0
    test_samples = 0
    for cell_id in sorted(heldout_cells, key=lambda value: counts[int(value)],
                          reverse=True):
        if len(val_cells_list) == n_val_cells:
            destination = test_cells_list
        elif len(test_cells_list) == n_test_cells:
            destination = val_cells_list
        elif val_samples <= test_samples:
            destination = val_cells_list
        else:
            destination = test_cells_list
        destination.append(cell_id)
        if destination is val_cells_list:
            val_samples += counts[int(cell_id)]
        else:
            test_samples += counts[int(cell_id)]
    val_cells = np.asarray(val_cells_list)
    test_cells = np.asarray(test_cells_list)

    split = np.full(len(points), TRAIN, dtype=np.uint8)
    split[np.isin(cell_ids, val_cells)] = VAL_CELL
    split[np.isin(cell_ids, test_cells)] = TEST_CELL

    shared_cells = shuffled_cells[n_val_cells + n_test_cells:]
    for cell_id in shared_cells:
        indices = np.flatnonzero(cell_ids == cell_id)
        indices = rng.permutation(indices)
        n_val = max(1, int(round(validation_sample_fraction * len(indices))))
        n_test = max(1, int(round(test_sample_fraction * len(indices))))
        if n_val + n_test >= len(indices):
            continue
        split[indices[:n_val]] = VAL_SAMPLE
        split[indices[n_val:n_val + n_test]] = TEST_SAMPLE

    assert np.any(split == TRAIN)
    for code in (VAL_SAMPLE, VAL_CELL, TEST_SAMPLE, TEST_CELL):
        assert np.any(split == code), f"empty dataset split component: {SPLIT_NAMES[code]}"
    return split, cell_ids


def _generate_shard(config: WorkerConfig) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(config.seed)
    teacher = IKTeacher(
        config.model_path,
        config.position_tolerance_m,
        config.joint_limit_margin_fraction,
        config.ik_max_iterations,
    )
    current_q = np.empty((config.count, len(JOINT_NAMES)), dtype=np.float32)
    sponge_xyz = np.empty((config.count, 3), dtype=np.float32)
    target_q = np.empty((config.count, len(JOINT_NAMES)), dtype=np.float32)
    low = np.asarray(config.target_low_m)
    high = np.asarray(config.target_high_m)
    offset = np.asarray(config.target_offset_m)
    rejected_targets = 0

    for sample_id in range(config.count):
        solved = None
        for _ in range(config.max_sample_attempts):
            current = teacher.sample_current(rng, config.max_sample_attempts)
            sponge = rng.uniform(low, high)
            solved = teacher.solve(
                current, sponge + offset, rng, config.ik_restarts)
            if solved is not None:
                current_q[sample_id] = current
                sponge_xyz[sample_id] = sponge
                target_q[sample_id] = solved
                break
            rejected_targets += 1
        if solved is None:
            raise RuntimeError(
                f"Worker seed {config.seed} could not generate sample {sample_id} "
                f"in {config.max_sample_attempts} attempts")
    return {
        "current_q": current_q,
        "sponge_xyz": sponge_xyz,
        "target_q": target_q,
        "rejected_targets": rejected_targets,
    }


def _resolved_path(original_cwd: str, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(original_cwd) / value


def _resplit_existing(output_path: Path, dataset_cfg: DictConfig,
                      seed: int) -> None:
    assert output_path.exists(), f"Cannot resplit missing dataset: {output_path}"
    with np.load(output_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files
                  if name not in {"split", "cell_id", "metadata_json"}}
        metadata = json.loads(str(source["metadata_json"]))
    target_low = np.asarray(dataset_cfg.target_low_m, dtype=np.float64)
    target_high = np.asarray(dataset_cfg.target_high_m, dtype=np.float64)
    cells = np.asarray(dataset_cfg.spatial_cells, dtype=np.int64)
    split, cell_ids = assign_structured_splits(
        arrays["sponge_xyz"], target_low, target_high, cells,
        float(dataset_cfg.validation_cell_fraction),
        float(dataset_cfg.test_cell_fraction),
        float(dataset_cfg.validation_sample_fraction),
        float(dataset_cfg.test_sample_fraction),
        np.random.default_rng(seed + 1),
    )
    metadata["split_config"] = {
        "spatial_cells": [int(value) for value in cells],
        "validation_cell_fraction": float(dataset_cfg.validation_cell_fraction),
        "test_cell_fraction": float(dataset_cfg.test_cell_fraction),
        "validation_sample_fraction": float(dataset_cfg.validation_sample_fraction),
        "test_sample_fraction": float(dataset_cfg.test_sample_fraction),
        "seed": seed + 1,
    }
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        **arrays,
        split=split,
        cell_id=cell_ids.astype(np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary_path, output_path)
    counts_by_split = {
        SPLIT_NAMES[code]: int(np.count_nonzero(split == code))
        for code in range(len(SPLIT_NAMES))
    }
    print(f"Re-split {output_path}", flush=True)
    print(json.dumps(counts_by_split, indent=2), flush=True)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    original_cwd = hydra.utils.get_original_cwd()
    experiment_cfg = cfg.ik_capacity
    dataset_cfg = experiment_cfg.dataset
    output_path = _resolved_path(original_cwd, str(experiment_cfg.dataset_path))
    model_path = _resolved_path(original_cwd, str(dataset_cfg.model_path))
    mode = str(dataset_cfg.mode)
    if mode == "resplit":
        _resplit_existing(output_path, dataset_cfg, int(experiment_cfg.seed))
        return
    if mode != "generate":
        raise ValueError(f"Unknown ik_capacity.dataset.mode: {mode!r}")
    sample_count = int(dataset_cfg.samples)
    worker_count = min(int(dataset_cfg.workers), sample_count)
    counts = np.full(worker_count, sample_count // worker_count, dtype=np.int64)
    counts[:sample_count % worker_count] += 1

    shared = {
        "model_path": str(model_path),
        "target_low_m": tuple(float(v) for v in dataset_cfg.target_low_m),
        "target_high_m": tuple(float(v) for v in dataset_cfg.target_high_m),
        "target_offset_m": tuple(float(v) for v in dataset_cfg.target_offset_m),
        "position_tolerance_m": float(dataset_cfg.position_tolerance_m),
        "joint_limit_margin_fraction": float(
            dataset_cfg.joint_limit_margin_fraction),
        "ik_restarts": int(dataset_cfg.ik_restarts),
        "ik_max_iterations": int(dataset_cfg.ik_max_iterations),
        "max_sample_attempts": int(dataset_cfg.max_sample_attempts),
    }
    worker_configs = [
        WorkerConfig(
            count=int(count),
            seed=int(experiment_cfg.seed) + worker_id,
            **shared,
        )
        for worker_id, count in enumerate(counts)
    ]

    shards = [None] * worker_count
    completed = 0
    rejected_targets = 0
    print(f"Generating {sample_count:,} IK examples with {worker_count} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_generate_shard, item): worker_id
            for worker_id, item in enumerate(worker_configs)
        }
        for future in as_completed(futures):
            shard = future.result()
            shards[futures[future]] = shard
            completed += len(shard["current_q"])
            rejected_targets += int(shard["rejected_targets"])
            print(f"  accepted {completed:,}/{sample_count:,} "
                  f"(rejected target attempts: {rejected_targets:,})", flush=True)

    assert all(shard is not None for shard in shards)
    current_q = np.concatenate([shard["current_q"] for shard in shards])
    sponge_xyz = np.concatenate([shard["sponge_xyz"] for shard in shards])
    target_q = np.concatenate([shard["target_q"] for shard in shards])
    rng = np.random.default_rng(int(experiment_cfg.seed))
    order = rng.permutation(sample_count)
    current_q = current_q[order]
    sponge_xyz = sponge_xyz[order]
    target_q = target_q[order]

    target_low = np.asarray(dataset_cfg.target_low_m, dtype=np.float64)
    target_high = np.asarray(dataset_cfg.target_high_m, dtype=np.float64)
    cells = np.asarray(dataset_cfg.spatial_cells, dtype=np.int64)
    split, cell_ids = assign_structured_splits(
        sponge_xyz,
        target_low,
        target_high,
        cells,
        float(dataset_cfg.validation_cell_fraction),
        float(dataset_cfg.test_cell_fraction),
        float(dataset_cfg.validation_sample_fraction),
        float(dataset_cfg.test_sample_fraction),
        np.random.default_rng(int(experiment_cfg.seed) + 1),
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    joint_limits = model.jnt_range[joint_ids].astype(np.float32)
    metadata = {
        "schema_version": 1,
        "joint_names": JOINT_NAMES,
        "arm_joint_names": ARM_JOINT_NAMES,
        "ee_site_name": EE_SITE_NAME,
        "split_names": SPLIT_NAMES,
        "target_offset_m": [float(v) for v in dataset_cfg.target_offset_m],
        "spatial_cells": [int(v) for v in cells],
        "generator_config": OmegaConf.to_container(dataset_cfg, resolve=True),
        "seed": int(experiment_cfg.seed),
        "rejected_target_attempts": rejected_targets,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        current_q=current_q,
        sponge_xyz=sponge_xyz,
        target_q=target_q,
        split=split,
        cell_id=cell_ids.astype(np.int32),
        joint_limits=joint_limits,
        target_low_m=target_low.astype(np.float32),
        target_high_m=target_high.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary_path, output_path)
    counts_by_split = {
        SPLIT_NAMES[code]: int(np.count_nonzero(split == code))
        for code in range(len(SPLIT_NAMES))
    }
    print(f"Saved {output_path}", flush=True)
    print(json.dumps(counts_by_split, indent=2), flush=True)


if __name__ == "__main__":
    main()
