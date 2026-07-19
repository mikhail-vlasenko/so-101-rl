"""Curated registry of every script the panel can launch.

This is deliberately an explicit, hand-maintained list — no argparse
introspection, no dynamic discovery. Each `ScriptSpec` names the module, the
panel page it appears on, the arguments its form exposes, the hardware it
claims, and whether it can stream its sim view. `validate_registry()` runs at
panel startup (and in tests) so an inconsistent spec fails loud before anyone
clicks a button.

Form semantics: an empty form value omits the argument entirely, so the
script's own default remains the single source of truth. `ArgSpec.default`
is a placeholder shown in the UI, never injected into the command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAGES = ("train", "sim", "real", "camera", "sysid")
ARG_KINDS = ("str", "int", "float", "flag", "choice")
ARG_STYLES = ("argparse", "hydra")

FLAG_TRUE = ("1", "true", "on", "yes")
FLAG_FALSE = ("", "0", "false", "off", "no")


class Resource(Enum):
    SERIAL = "serial"
    CAMERA = "camera"


@dataclass(frozen=True)
class ArgSpec:
    # argparse style: the literal flag ("--seed"); hydra style: the override
    # key, including any append prefix ("train.total_timesteps", "+episodes").
    name: str
    kind: str
    label: str
    default: str = ""           # UI placeholder only
    choices: tuple[str, ...] = ()
    # Log-dir template for the checkpoint dropdown next to this field, e.g.
    # "ppo_lift" or "{algorithm}_{env}" ({x} is replaced client-side with the
    # form's current value for field x). None = plain text input.
    checkpoint_picker: str | None = None

    @property
    def field_name(self) -> str:
        """Form-field key: the name without CLI/hydra prefixes."""
        return self.name.lstrip("+-")


@dataclass(frozen=True)
class ScriptSpec:
    id: str
    title: str
    page: str
    module: str
    arg_style: str
    description: str
    args: tuple[ArgSpec, ...] = ()
    resources: tuple[Resource, ...] = ()
    supports_stream: bool = False
    stream_extra_args: tuple[str, ...] = ()  # appended verbatim on streamed launches
    native_gui: bool = False                 # opens windows on the host machine itself


def _arm_loop_args() -> tuple[ArgSpec, ...]:
    """The rollout args shared via real.rollout.rollout_common.add_common_args."""
    return (
        ArgSpec("--execute", "flag", "Execute on real servos (default: dry-run)"),
        ArgSpec("--max-steps", "int", "Max policy steps"),
        ArgSpec("--ema-alpha", "float", "Action EMA alpha (1.0 = off)", default="1.0"),
        ArgSpec("--slow", "float", "Time dilation factor (>= 1)", default="1.0"),
        ArgSpec("--interp-hz", "float", "Sub-target streaming rate"),
        ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
    )


SCRIPTS: tuple[ScriptSpec, ...] = (
    # ---- Train ----
    ScriptSpec(
        id="train", title="Train policy", page="train",
        module="src.train", arg_style="hydra",
        description="PPO/SAC training with Hydra config and W&B logging.",
        args=(
            ArgSpec("env", "choice", "Environment", default="pickplace",
                    choices=("lift", "pickplace", "multitask", "reach")),
            ArgSpec("algorithm", "choice", "Algorithm", default="ppo",
                    choices=("ppo", "sac")),
            ArgSpec("train.total_timesteps", "int", "Total timesteps"),
            ArgSpec("train.time_limit_minutes", "int", "Wall-clock limit (min)"),
            ArgSpec("resume", "str", "Resume from checkpoint (.zip)",
                    checkpoint_picker="{algorithm}_{env}"),
            ArgSpec("seed", "int", "Seed"),
            ArgSpec("wandb.enabled", "choice", "W&B logging",
                    choices=("true", "false")),
        ),
    ),
    ScriptSpec(
        id="distill", title="Distill policy", page="train",
        module="src.distill", arg_style="hydra",
        description="DAgger-distill a trained teacher onto a fresh student across "
                    "an architecture or obs-layout change (identical or privileged "
                    "teacher view), then fine-tune the output via Train's resume.",
        args=(
            ArgSpec("env", "choice", "Environment", default="lift",
                    choices=("lift", "pickplace", "multitask", "reach")),
            ArgSpec("algorithm", "choice", "Algorithm", default="ppo",
                    choices=("ppo", "sac")),
            ArgSpec("distill.teacher", "str", "Teacher checkpoint (.zip)",
                    checkpoint_picker="{algorithm}_{env}"),
            ArgSpec("distill.teacher_obs", "choice", "Teacher obs view",
                    default="identical", choices=("identical", "privileged")),
            ArgSpec("distill.net_arch", "str", "Student net_arch (e.g. [512,512,512,512])"),
            ArgSpec("distill.iterations", "int", "DAgger iterations"),
            ArgSpec("distill.out", "str", "Output checkpoint (.zip)", default="distilled.zip"),
            ArgSpec("seed", "int", "Seed"),
            ArgSpec("wandb.enabled", "choice", "W&B logging",
                    choices=("true", "false")),
        ),
    ),
    # ---- Sim ----
    ScriptSpec(
        id="eval", title="Evaluate checkpoint", page="sim",
        module="src.eval", arg_style="hydra",
        description="Roll out a trained checkpoint in sim and print per-episode returns.",
        args=(
            ArgSpec("env", "choice", "Environment", default="pickplace",
                    choices=("lift", "pickplace", "multitask", "reach")),
            ArgSpec("algorithm", "choice", "Algorithm", default="ppo",
                    choices=("ppo", "sac")),
            ArgSpec("+model", "str", "Model (latest / best / path)", default="latest",
                    checkpoint_picker="{algorithm}_{env}"),
            ArgSpec("+episodes", "int", "Episodes", default="10"),
            ArgSpec("seed", "int", "Seed"),
            ArgSpec("slow_factor", "float", "Slowdown factor", default="2"),
            ArgSpec("render", "choice", "Native render window", default="true",
                    choices=("true", "false")),
        ),
        supports_stream=True,
        stream_extra_args=("render=false",),
    ),
    ScriptSpec(
        id="show_starts", title="Show spawn positions", page="sim",
        module="src.show_starts", arg_style="hydra",
        description="Reset the pickplace env every few seconds to inspect spawn variety.",
        supports_stream=True,  # streaming itself switches off the native viewer
    ),
    ScriptSpec(
        id="scripted_lift", title="Scripted lift probe", page="sim",
        module="scripts.scripted_lift", arg_style="argparse",
        description="Hand-coded grasp-and-lift in sim; sanity-checks the scene physics.",
        args=(
            ArgSpec("--n", "int", "Episodes", default="5"),
            ArgSpec("--render", "flag", "Native render window"),
        ),
    ),
    # ---- Real arm ----
    ScriptSpec(
        id="rollout_lift", title="Rollout: lift", page="real",
        module="real.rollout.rollout_lift", arg_style="argparse",
        description="Lift policy on the real arm. Marker source 'camera' tracks "
                    "the real sponge tag-free (SAM stereo dual channels); 'fk' "
                    "uses a lockstep sim cube through the same channel logic. "
                    "Dry-run unless Execute is checked.",
        args=(
            ArgSpec("--model", "str", "Model (latest / best / path)", default="latest",
                    checkpoint_picker="ppo_lift"),
            *_arm_loop_args(),
            ArgSpec("--seed", "int", "Cube spawn seed"),
            ArgSpec("--no-view", "flag", "Disable native MuJoCo viewer"),
            ArgSpec("--marker-source", "choice", "Marker obs source", default="fk",
                    choices=("fk", "camera")),
            ArgSpec("--family", "choice", "Marker family (camera source)", default="apriltag",
                    choices=("apriltag", "aruco")),
            ArgSpec("--prompt", "str", "SAM3 text prompt (camera source)",
                    default="sponge"),
            ArgSpec("--sam2-model", "choice", "SAM2 tracker size (camera source)",
                    default="tiny", choices=("tiny", "base+")),
            ArgSpec("--gui", "flag", "Mask/ellipse overlay windows (camera source)"),
        ),
        resources=(Resource.SERIAL, Resource.CAMERA),
        supports_stream=True,
        stream_extra_args=("--no-view",),
    ),
    ScriptSpec(
        id="rollout_reach", title="Rollout: reach", page="real",
        module="real.rollout.rollout_real", arg_style="argparse",
        description="Reach policy on the real arm toward a fixed waypoint. "
                    "Dry-run unless Execute is checked.",
        args=(
            ArgSpec("--waypoint", "int", "Waypoint index", default="0"),
            ArgSpec("--model", "str", "Model (latest / best / path)", default="latest",
                    checkpoint_picker="ppo_reach"),
            *_arm_loop_args(),
        ),
        resources=(Resource.SERIAL,),
        supports_stream=True,
    ),
    ScriptSpec(
        id="digital_twin", title="Digital twin", page="real",
        module="real.twin.digital_twin", arg_style="argparse",
        description="MuJoCo viewer + Tk panel mirroring the real arm's encoders "
                    "(native windows on the host).",
        args=(
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
            ArgSpec("--allow-missing", "flag", "Boot even if some servos don't ping"),
        ),
        resources=(Resource.SERIAL,),
        native_gui=True,
    ),
    ScriptSpec(
        id="shake_probe", title="Shake probe", page="real",
        module="scripts.shake_probe", arg_style="argparse",
        description="High-rate single-servo recording to isolate joint chatter: "
                    "hold (PID limit cycle / tap ring-down) or slow triangle sweep. "
                    "Torque ON immediately; restores standard gains on exit.",
        args=(
            ArgSpec("--joint", "choice", "Joint", default="shoulder_pan",
                    choices=("shoulder_pan", "shoulder_lift", "elbow_flex",
                             "wrist_flex", "wrist_roll", "gripper")),
            ArgSpec("--mode", "choice", "Mode", default="hold",
                    choices=("hold", "sweep")),
            ArgSpec("--watch-joint", "choice", "Record this joint instead (sweep only)",
                    choices=("shoulder_pan", "shoulder_lift", "elbow_flex",
                             "wrist_flex", "wrist_roll", "gripper")),
            ArgSpec("--duration", "float", "Recording length (s)", default="10"),
            ArgSpec("--kp", "int", "Probed-servo Kp override"),
            ArgSpec("--deadzone", "int", "Probed-servo deadzone override"),
            ArgSpec("--speed", "int", "Servo speed argument"),
            ArgSpec("--accel", "int", "Servo accel argument"),
            ArgSpec("--sweep-range", "int", "Sweep amplitude (raw units)", default="150"),
            ArgSpec("--sweep-raw-per-s", "float", "Sweep velocity (raw/s)", default="100"),
            ArgSpec("--dwell-s", "float", "Dwell at sweep extremes (s)", default="2"),
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
        ),
        resources=(Resource.SERIAL,),
    ),
    ScriptSpec(
        id="read_kp", title="Read servo gains", page="real",
        module="real.diagnostics.read_kp", arg_style="argparse",
        description="Dump position-loop gains and related registers from every servo.",
        args=(ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),),
        resources=(Resource.SERIAL,),
    ),
    # ---- Camera (interactive cv2 tools stay native windows) ----
    ScriptSpec(
        id="calibrate_camera", title="Calibrate intrinsics", page="camera",
        module="real.calib.calibrate_camera", arg_style="argparse",
        description="Checkerboard intrinsic calibration (native cv2 window).",
        args=(
            ArgSpec("--camera", "choice", "C922 unit (per-lens intrinsics)",
                    default="main", choices=("main", "aux")),
        ),
        resources=(Resource.CAMERA,),
        native_gui=True,
    ),
    ScriptSpec(
        id="sam_track", title="SAM stereo tracking", page="camera",
        module="real.tracking.sam_track", arg_style="argparse",
        description="Text-prompt SAM3 once per view, track with real-time SAM2, "
                    "triangulate the mask centroids and compare against the sponge "
                    "tag's triangulated center (estimator characterization).",
        args=(
            ArgSpec("--prompt", "str", "SAM3 text prompt", default="sponge"),
            ArgSpec("--model", "choice", "SAM2 tracker size", default="tiny",
                    choices=("tiny", "base+")),
            ArgSpec("--frames", "int", "Frames to accumulate", default="300"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
            ArgSpec("--gui", "flag", "Live overlay window (native cv2)"),
            ArgSpec("--save-frames", "str", "Directory for annotated frame pair"),
        ),
        resources=(Resource.CAMERA,),
        native_gui=True,
    ),
    ScriptSpec(
        id="stereo_check", title="Stereo triangulation check", page="camera",
        module="real.tracking.stereo_check", arg_style="argparse",
        description="Triangulate the sponge tag from both table-anchored cameras "
                    "and report metric accuracy (recovered tag edge vs printed size) "
                    "and cross-view ray consistency.",
        args=(
            ArgSpec("--frames", "int", "Frames to accumulate", default="100"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
            ArgSpec("--save-frames", "str", "Directory for annotated frame pair"),
        ),
        resources=(Resource.CAMERA,),
    ),
    ScriptSpec(
        id="tag_body_calib", title="Sponge tag placement calib", page="camera",
        module="real.tracking.tag_body_calib", arg_style="argparse",
        description="Solve each glued sponge tag's in-plane offset + yaw on its "
                    "declared face from co-visible pairs -> sponge_tags.yaml "
                    "(GT body pose for the shape dataset).",
        args=(
            ArgSpec("--frames", "int", "Frame pairs to accumulate", default="60"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
        ),
        resources=(Resource.CAMERA,),
    ),
    ScriptSpec(
        id="record_shapes", title="Record shape dataset", page="camera",
        module="real.tracking.record_shapes", arg_style="argparse",
        description="Record dual-camera frames + tag GT + static labels into "
                    "datasets/sponge_<stamp>/ for offline estimator evaluation "
                    "(masks computed later by eval_estimator).",
        args=(
            ArgSpec("--minutes", "float", "Recording length (minutes)", default="10"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
            ArgSpec("--out", "str", "Dataset directory (default: datasets/sponge_<stamp>)"),
        ),
        resources=(Resource.CAMERA,),
    ),
    ScriptSpec(
        id="focus_picker", title="Focus picker", page="camera",
        module="real.calib.focus_picker", arg_style="argparse",
        description="Interactively pick the pinned focus_absolute value (native cv2 window).",
        resources=(Resource.CAMERA,),
        native_gui=True,
    ),
    # ---- Sysid ----
    ScriptSpec(
        id="calibrate_qpos", title="Calibrate encoder bias", page="sysid",
        module="real.calib.calibrate_qpos", arg_style="argparse",
        description="Self-drive the arm through a Cartesian sweep and jointly solve "
                    "encoder zero-offsets + camera extrinsics from the arm tags (writes "
                    "calibration.yaml + extrinsics.yaml). The sim view shows the sweep "
                    "preview on dry-run, and the live annotated camera feed while executing.",
        args=(
            ArgSpec("--execute", "flag", "Drive the arm and capture (default: preview sweep)"),
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
        ),
        resources=(Resource.CAMERA, Resource.SERIAL),
        supports_stream=True,
    ),
    ScriptSpec(
        id="sysid_record", title="Record real trajectories", page="sysid",
        module="sysid.record_real", arg_style="argparse",
        description="Drive sysid trajectories on the real arm and record encoders. "
                    "Dry-run unless Execute is checked.",
        args=(
            ArgSpec("--traj", "str", "Trajectory name (omit with All)"),
            ArgSpec("--all", "flag", "Record every trajectory"),
            ArgSpec("--execute", "flag", "Execute on real servos (default: dry-run)"),
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
        ),
        resources=(Resource.SERIAL,),
    ),
    ScriptSpec(
        id="probe_backlash", title="Probe joint backlash", page="sysid",
        module="sysid.probe_backlash", arg_style="argparse",
        description="Approach identical targets from opposite directions and compare "
                    "camera-measured link motion vs encoder motion: per-joint gear "
                    "play in degrees. Dry-run previews the drive plan in the sim view; "
                    "executing streams the annotated camera feed.",
        args=(
            ArgSpec("--execute", "flag", "Drive the arm and capture (default: preview plan)"),
            ArgSpec("--n-poses", "int", "Base poses from the calibration sweep", default="3"),
            ArgSpec("--approach-deg", "float", "Approach offset per joint (deg)", default="6"),
            ArgSpec("--joints", "str", "Comma-separated joints (default: all but gripper)"),
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
        ),
        resources=(Resource.CAMERA, Resource.SERIAL),
        supports_stream=True,
    ),
    ScriptSpec(
        id="probe_cam_latency", title="Probe camera latency", page="sysid",
        module="sysid.probe_cam_latency", arg_style="argparse",
        description="Cross-correlate encoder-FK tag positions against camera-measured "
                    "ones to recover the camera pipeline delay (feeds cam_latency.delay_ms "
                    "in conf/dr). Execute drives a sine on the chosen joint — the reliable "
                    "mode; dry-run records a hand-wiggle, a rough check only (back-driving "
                    "defeats the encoder ground truth).",
        args=(
            ArgSpec("--execute", "flag", "Drive the sine on real servos (default: hand-wiggle)"),
            ArgSpec("--joint", "choice", "Joint to drive", default="wrist_flex",
                    choices=("shoulder_pan", "shoulder_lift", "elbow_flex",
                             "wrist_flex", "wrist_roll", "gripper")),
            ArgSpec("--amp", "float", "Sine amplitude (rad)", default="0.25"),
            ArgSpec("--freq", "float", "Sine frequency (Hz)", default="0.4"),
            ArgSpec("--seconds", "float", "Recording length (s)", default="30"),
            ArgSpec("--port", "str", "Serial port", default="/dev/ttyACM0"),
            ArgSpec("--family", "choice", "Marker family", default="apriltag",
                    choices=("apriltag", "aruco")),
        ),
        resources=(Resource.CAMERA, Resource.SERIAL),
    ),
    ScriptSpec(
        id="sysid_replay", title="Replay in sim", page="sysid",
        module="sysid.replay_sim", arg_style="argparse",
        description="Replay recorded trajectories through the sim model.",
        args=(
            ArgSpec("--traj", "str", "Trajectory name (omit with All)"),
            ArgSpec("--all", "flag", "Replay every recording"),
        ),
    ),
    ScriptSpec(
        id="sysid_fit", title="Fit sim parameters", page="sysid",
        module="sysid.fit_params", arg_style="argparse",
        description="Optimize actuator/friction params to match the recordings (CMA-style search).",
        args=(
            ArgSpec("--maxiter", "int", "Max iterations", default="25"),
            ArgSpec("--popsize", "int", "Population size", default="12"),
            ArgSpec("--seed", "int", "Seed", default="0"),
        ),
    ),
    ScriptSpec(
        id="sysid_analyze", title="Analyze recordings", page="sysid",
        module="sysid.analyze", arg_style="argparse",
        description="Per-joint real-vs-sim error report (report.tsv + plots).",
        args=(ArgSpec("--no-plots", "flag", "Skip plot generation"),),
    ),
    ScriptSpec(
        id="make_markers", title="Generate marker PDFs", page="sysid",
        module="scripts.make_markers", arg_style="argparse",
        description="Render the ArUco/AprilTag sheets defined in real.marker_spec.",
    ),
)


def get_spec(spec_id: str) -> ScriptSpec:
    for spec in SCRIPTS:
        if spec.id == spec_id:
            return spec
    raise KeyError(f"unknown script id: {spec_id}")


def build_command(spec: ScriptSpec, values: dict[str, str],
                  stream_port: int | None = None) -> list[str]:
    """Build the `python` argv (starting at -m) for a launch request.

    `values` maps field_name -> raw form string. Empty values omit the
    argument. Unknown keys, bad choices, and unparseable numbers raise
    ValueError — the panel must never silently launch something other than
    what the form said.
    """
    by_field = {a.field_name: a for a in spec.args}
    unknown = set(values) - set(by_field)
    if unknown:
        raise ValueError(f"{spec.id}: unknown form fields {sorted(unknown)}")

    argv = ["-m", spec.module]
    for arg in spec.args:
        raw = values.get(arg.field_name, "").strip()
        if arg.kind == "flag":
            if raw.lower() in FLAG_FALSE:
                continue
            if raw.lower() not in FLAG_TRUE:
                raise ValueError(f"{spec.id}: bad flag value {arg.name}={raw!r}")
            argv.append(arg.name)
            continue
        if raw == "":
            continue
        if arg.kind == "int":
            int(raw)
        elif arg.kind == "float":
            float(raw)
        elif arg.kind == "choice":
            if raw not in arg.choices:
                raise ValueError(
                    f"{spec.id}: {arg.name}={raw!r} not in {arg.choices}")
        if spec.arg_style == "argparse":
            argv += [arg.name, raw]
        else:
            argv.append(f"{arg.name}={raw}")

    if stream_port is not None:
        if not spec.supports_stream:
            raise ValueError(f"{spec.id} does not support streaming")
        argv += list(spec.stream_extra_args)
        if spec.arg_style == "argparse":
            argv += ["--stream-port", str(stream_port)]
        else:
            argv.append(f"stream_port={stream_port}")
    return argv


def validate_registry(scripts: tuple[ScriptSpec, ...] = SCRIPTS) -> None:
    """Fail loud on any internally inconsistent spec. Run at panel startup."""
    ids = [s.id for s in scripts]
    assert len(ids) == len(set(ids)), f"duplicate script ids: {ids}"
    for s in scripts:
        ctx = f"spec {s.id!r}"
        assert s.page in PAGES, f"{ctx}: unknown page {s.page!r}"
        assert s.arg_style in ARG_STYLES, f"{ctx}: bad arg_style {s.arg_style!r}"
        assert s.title and s.description, f"{ctx}: title/description required"
        module_path = REPO_ROOT / (s.module.replace(".", "/") + ".py")
        assert module_path.is_file(), f"{ctx}: module file missing: {module_path}"
        assert not (s.native_gui and s.supports_stream), \
            f"{ctx}: native_gui scripts cannot stream"
        assert s.supports_stream or not s.stream_extra_args, \
            f"{ctx}: stream_extra_args without supports_stream"

        fields = [a.field_name for a in s.args]
        assert len(fields) == len(set(fields)), f"{ctx}: duplicate arg fields"
        for a in s.args:
            actx = f"{ctx} arg {a.name!r}"
            assert a.kind in ARG_KINDS, f"{actx}: bad kind {a.kind!r}"
            assert a.label, f"{actx}: label required"
            assert (a.kind == "choice") == bool(a.choices), \
                f"{actx}: choices iff kind=choice"
            if a.kind == "choice" and a.default:
                assert a.default in a.choices, f"{actx}: default not in choices"
            if a.checkpoint_picker is not None:
                assert a.kind == "str", f"{actx}: checkpoint_picker needs kind=str"
            if s.arg_style == "argparse":
                assert a.name.startswith("--"), f"{actx}: argparse args start with --"
            else:
                assert not a.name.startswith("-"), f"{actx}: hydra keys take no dashes"
                assert a.kind != "flag", f"{actx}: hydra has no flags; use a true/false choice"
