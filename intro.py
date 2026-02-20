import mujoco
import mujoco.viewer
import numpy as np

# Load the model
model = mujoco.MjModel.from_xml_path("pendulum.xml")
data = mujoco.MjData(model)

print(f"Model: {model.nq} DoF, {model.nu} actuator(s)")
print(f"Timestep: {model.opt.timestep}s")
print(f"Joint name: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)}")

# Give it a nudge — set initial angle (radians)
data.qpos[0] = 0.5  # ~28 degrees from vertical

# Run a quick headless sim and print state every 0.5s
print("\nHeadless simulation (2 seconds):")
print(f"{'time':>6}  {'angle (deg)':>12}  {'velocity':>10}")
print("-" * 35)

while data.time < 2.0:
    mujoco.mj_step(model, data)
    if round(data.time % 0.5, 3) == 0.0:
        angle_deg = np.degrees(data.qpos[0])
        print(f"{data.time:6.2f}  {angle_deg:12.2f}  {data.qvel[0]:10.4f}")

print("\nLaunching interactive viewer — close window to exit.")
print("  Drag to rotate, scroll to zoom, double-click to track a body.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset for the viewer
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.5

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
