"""Manually drive the SO-101 arm in the lift scene, no time limit.

Builds a real SO101LiftEnv so the sponge spawns camera-visible and the arm
starts in a valid pose (exactly like training reset), then opens MuJoCo's
passive viewer. Open the left panel ("Control" group) to get one slider per
joint — the position actuators drive the arm toward whatever you set, so you
can reach in and pick up the sponge by hand. Nothing calls env.step(), so the
episode never terminates or times out; press R in the viewer to re-sample a
fresh sponge/arm layout (Backspace is MuJoCo's built-in reset to the XML
default and zeroes the joint targets).

Every tag site is tinted live by the env's own camera-visibility pipeline —
green when the tag camera can currently see it, red when it can't (grazing
angle, out of frame, or occluded by the arm) — for arm markers and the sponge
tag alike. Because DR (marker_dropout) is left off here, that visibility is the
deterministic geometric test, so the colors track exactly what the camera
would detect this instant with no random flicker.

    conda run -n mujoco_env python manual_lift.py
"""

import time

import mujoco
import mujoco.viewer
from hydra import compose, initialize_config_dir
import os

from src.train import _resolve_env, make_env, runtime_cfg_from_hydra


def build_lift_env():
    orig_dir = os.path.dirname(os.path.abspath(__file__))
    with initialize_config_dir(config_dir=os.path.join(orig_dir, "conf"),
                               version_base=None):
        cfg = compose(config_name="config", overrides=["env=lift"])
    env_cls, env_cfg, xml_path = _resolve_env(cfg, orig_dir, "lift")
    runtime_cfg = runtime_cfg_from_hydra(cfg)
    # render_mode=None: we own the viewer. DR params left at defaults — they
    # only touch observations, and here the physics is all that matters.
    return make_env(env_cls, env_cfg, xml_path, cfg=runtime_cfg)


def refresh_tag_colors(env):
    """Recolor every tag site green/red for its current camera visibility,
    reusing the env's own detection pipeline (no duplicated visibility math).
    Runs the capture -> detect -> ingest path once at the present instant, so
    _set_marker_render_colors and the cube-tag recolor in _ingest_frame fire.
    Side effects on held obs state are irrelevant here — we never read obs."""
    state = env._capture_camera_state()
    frame = env._process_frame(state)
    env._ingest_frame(env.data.time, frame)


def main():
    env = build_lift_env()
    env.reset()  # places sponge (camera-visible) + arm; ctrl set to hold pose
    model, data = env.model, env.data
    dt = model.opt.timestep
    # Refresh tag colors at the real camera's 30 fps rather than every physics
    # step — matches the rig and keeps the occlusion ray-casts cheap.
    refresh_every = max(1, round((1.0 / 30.0) / dt))

    print("Open the left panel and expand 'Control' to move each joint.")
    print("R = re-sample sponge/arm layout. Close window to exit.")

    def on_key(keycode):
        if keycode == ord("R"):
            env.reset()

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        step = 0
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            if step % refresh_every == 0:
                refresh_tag_colors(env)
            viewer.sync()
            step += 1
            # Real-time pacing so the arm moves at a natural speed.
            time.sleep(max(0.0, dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
