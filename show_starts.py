"""Visualize random starting positions for the pick-and-place environment.

Resets the environment every 10 seconds so you can inspect spawn variety.
"""

import time

import hydra
from omegaconf import DictConfig

from pickplace_env import SO101PickPlaceEnv


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    env = SO101PickPlaceEnv(render_mode="human", env_cfg=cfg["pickplace_env"])
    try:
        i = 0
        while True:
            env.reset()
            i += 1
            print(f"Reset #{i} — cube at {env._get_cube_pos()}")
            t0 = time.time()
            while time.time() - t0 < 5:
                env._render_human()
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
