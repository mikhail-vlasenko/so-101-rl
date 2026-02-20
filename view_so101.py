import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("so101/scene.xml")
data = mujoco.MjData(model)

print("Joints:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)])
print("Launching viewer — drag to rotate, scroll to zoom, close to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
