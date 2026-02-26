"""Take top-down screenshot of the hexagonal wall around the target area."""

import mujoco
from pathlib import Path
import PIL.Image

model = mujoco.MjModel.from_xml_path("so101/scene_pickplace.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, width=480, height=480)

out = Path("screenshots")
out.mkdir(exist_ok=True)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.18, 0.0, 0.015]

# Top-down view
cam.distance = 0.18
cam.azimuth = 0
cam.elevation = -90
renderer.update_scene(data, cam)
PIL.Image.fromarray(renderer.render()).save(out / "hex_wall_top.png")
print("Saved hex_wall_top.png")
