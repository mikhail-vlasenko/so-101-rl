"""Screenshot the lift scene to verify marker sites and the sponge box.

Renders two views, each aimed at one marker site (lookat taken from the site's
FK pose so the views track placement edits).
"""
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

# Sponge standing on its 2 x 1.5 cm face (3 cm tall) near the gripper.
cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
adr = model.jnt_qposadr[cube_jid]
data.qpos[adr:adr + 3] = [0.25, -0.05, 0.015]
data.qpos[adr + 3:adr + 7] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]  # x-axis up
mujoco.mj_forward(model, data)

site_pos = {}
for sname in ["marker_finger", "marker_wrist"]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
    site_pos[sname] = data.site_xpos[sid].copy()
    xmat = data.site_xmat[sid].reshape(3, 3)
    print(f"{sname}: world pos {site_pos[sname].round(4)}  "
          f"face normal (site z) -> world {xmat[:, 2].round(3)}")

renderer = mujoco.Renderer(model, height=480, width=640)
out = Path("screenshots")
out.mkdir(exist_ok=True)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
opt.sitegroup[:] = 1

# View 1: at the wrist marker from -y (its face normal direction).
cam.lookat[:] = site_pos["marker_wrist"]
cam.distance = 0.18
cam.azimuth = 90   # camera on -y, looking toward +y: the arm's right side
cam.elevation = 0
renderer.update_scene(data, camera=cam, scene_option=opt)
PIL.Image.fromarray(renderer.render()).save(out / "markers_right_side.png")

# View 2: face-on at the finger marker — camera below, looking up its normal
# (gripper-body -x maps to world (-0.479, 0.049, -0.877) at this pose).
cam.lookat[:] = site_pos["marker_finger"]
cam.distance = 0.18
cam.azimuth = -6
cam.elevation = 61
renderer.update_scene(data, camera=cam, scene_option=opt)
PIL.Image.fromarray(renderer.render()).save(out / "markers_finger_sponge.png")
print("saved screenshots/markers_right_side.png screenshots/markers_finger_sponge.png")
