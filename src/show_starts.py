"""Visualize random starting positions for the pick-and-place environment.

Resets the environment every 10 seconds so you can inspect spawn variety.
"""

import time

import hydra
from omegaconf import DictConfig

from src.pickplace_env import SO101PickPlaceEnv
from src.train import runtime_cfg_from_hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # When streaming (web panel), render offscreen instead of the native viewer.
    stream = cfg.stream_port is not None
    runtime_cfg = runtime_cfg_from_hydra(cfg)
    env = SO101PickPlaceEnv(render_mode=None if stream else "human",
                            env_cfg=cfg["pickplace_env"],
                            cfg=runtime_cfg)
    publisher = None
    if stream:
        from panel.sim_stream import SimStreamPublisher
        publisher = SimStreamPublisher(env.model, int(cfg.stream_port))
    try:
        i = 0
        while True:
            env.reset()
            i += 1
            joints = env._get_joint_pos()
            print(f"Reset #{i} — cube={env._get_cube_pos()}  ring_h={env.ring_height:.3f}  joints={joints.round(3)}")
            t0 = time.time()
            while time.time() - t0 < 5:
                if publisher is not None:
                    publisher.publish(env.data)
                else:
                    env._render_human()
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.close()
        env.close()


if __name__ == "__main__":
    main()
