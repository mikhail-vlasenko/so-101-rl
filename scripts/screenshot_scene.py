"""Screenshot the lift scene to verify the sponge box dimensions."""
from pathlib import Path

import mujoco
import numpy as np
import PIL.Image

model = mujoco.MjModel.from_xml_path("so101/scene_lift.xml")
data = mujoco.MjData(model)

pose = {"shoulder_pan": 0.0, "shoulder_lift": -0.5, "elbow_flex": 0.5,
        "wrist_flex": 0.5, "wrist_roll": 0.0, "gripper": 0.5}
for name, val in pose.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = val

# Sponge standing on its 4 x 2.5 cm face (6 cm tall) near the gripper.
cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
adr = model.jnt_qposadr[cube_jid]
data.qpos[adr:adr + 3] = [0.25, -0.05, 0.03]
data.qpos[adr + 3:adr + 7] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]  # x-axis up
mujoco.mj_forward(model, data)

cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_center = data.xpos[cube_body_id].copy()
cube_half = model.geom_size[cube_geom_id].copy()
print(f"cube center: {cube_center.round(4)}")
print(f"cube half extents (m): {cube_half.round(4)}")

renderer = mujoco.Renderer(model, height=480, width=640)
out = Path("screenshots")
out.mkdir(exist_ok=True)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
opt.sitegroup[:] = 1

# View 1: side close-up on the sponge.
cam.lookat[:] = cube_center
cam.distance = 0.14
cam.azimuth = 40
cam.elevation = -8
renderer.update_scene(data, camera=cam, scene_option=opt)
PIL.Image.fromarray(renderer.render()).save(out / "sponge_close_side.png")

# View 2: angled view to show sponge proportions in context.
cam.lookat[:] = cube_center
cam.distance = 0.20
cam.azimuth = 130
cam.elevation = -25
renderer.update_scene(data, camera=cam, scene_option=opt)
PIL.Image.fromarray(renderer.render()).save(out / "sponge_angled_context.png")
print("saved screenshots/sponge_close_side.png screenshots/sponge_angled_context.png")
