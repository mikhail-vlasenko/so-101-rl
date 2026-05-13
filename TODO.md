# TODO

Long-term tasks and ideas. Not a changelog.

## Sim-to-real fidelity (reach env)

Current symptom: with sim/real kp aligned at 64, real arm has trouble holding/lifting itself even though sim trains fine. Also worth noting: experiments so far have been on the **leader** arm, not the follower the policy will ultimately deploy on. Leader has different mechanical load (no payload, no gripper jaws engaged, possibly different calibration), so some of this gap may collapse once we move to the follower. Independently though, sim has several optimistic assumptions:

- **`forcerange` is stall torque, not continuous.** `so101.xml` actuators use `forcerange="-2.94 2.94"` ≈ Feetech STS3215 stall (~30 kg·cm). Continuous holding torque is roughly 1/3 of stall (~1 N·m). Sim lets the actuator pin 2.94 indefinitely; real will current-limit, heat-throttle, or sag. Lower `forcerange` to ~1.0 and retrain — if the policy still solves the task we've removed a hidden "infinite peak torque" cheat from sim.

- **Link masses / inertias.** Menagerie's `so101.xml` derives inertials from STL meshes at an assumed density (likely plastic). Real arm is metal servos + steel screws. Sum body masses in the XML, compare against the weighed arm. If sim is lighter, real torque demand exceeds what the policy learned.

- **Joint friction / damping.** Check `<joint frictionloss damping>` values in `so101.xml`. Real servos have stiction and gearbox friction that's commonly modeled at zero. Add a reasonable fixed value or domain randomize.

- **Internal servo dynamics.** Real Feetech has internal PID + current loop + comms latency; sim's `<position>` actuator is instantaneous torque ∝ err. Affects transients more than holding ability — lower priority than the three above.
