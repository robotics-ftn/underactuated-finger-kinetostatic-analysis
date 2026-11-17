#!/usr/bin/env python3
"""
Underactuated finger — contact force test.

Pose the finger from the three phalanx angles, park an object against the middle
of each phalanx, apply a motor torque, print the contact forces, then open the
viewer.  Requires mujoco >= 3.1.4 (mj_geomDistance).
"""

import numpy as np
import mujoco
from mujoco import viewer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
MODEL_PATH   = str(ROOT / "models" / "finger.xml")

THETA_DEG    = (-0.0, -10.0, -10.0)   # joint_A, joint_B, joint_C (negative = flexion)
MOTOR_TORQUE = 0.0                     # N m on actuator "motor_input"
SETTLE_TIME  = 2.0                     # s simulated before the forces are read
PENETRATION  = 2e-5                    # m of interference, so contacts are live at t=0
SHOW_VIEWER  = True

# phalanx body, phalanx geom, object body, object geom, phalanx length [m]
PARTS = [("proximal", "proximal_geom", "object1", "object_geom1", 0.051),
         ("middle",   "middle_geom",   "object2", "object_geom2", 0.051),
         ("distal",   "distal_geom",   "object3", "object_geom3", 0.040)]

POSE_JOINTS = ("joint_A", "joint_B", "joint_C")
LOOP_JOINTS = ("joint_input", "joint_b1", "joint_triangle", "joint_b2")
LOOP_SITES  = (("b1_end", "P2_prime"), ("b2_end", "P3"))


def closure_error(m, d, q_loop, loop_adr):
    """Planar (x, z) error of the two `connect` constraints."""
    d.qpos[loop_adr] = q_loop
    mujoco.mj_kinematics(m, d)
    return np.concatenate([(d.site(a).xpos - d.site(b).xpos)[[0, 2]] for a, b in LOOP_SITES])


def pose_finger(m, d, theta_deg):
    """Set joint_A/B/C and solve the 4 passive joints so both loops stay closed.
    Ramped from the home pose in small steps, so Gauss-Newton never branch-flips."""
    pose_adr = [m.joint(j).qposadr[0] for j in POSE_JOINTS]
    loop_adr = [m.joint(j).qposadr[0] for j in LOOP_JOINTS]
    q, target = d.qpos[loop_adr].copy(), np.deg2rad(theta_deg)

    for s in np.linspace(0, 1, 41)[1:]:
        d.qpos[pose_adr] = s * target
        for _ in range(30):
            r = closure_error(m, d, q, loop_adr)
            if r @ r < 1e-20:
                break
            J = np.column_stack([(closure_error(m, d, q + 1e-7 * e, loop_adr) - r) / 1e-7
                                 for e in np.eye(4)])
            q = q + np.linalg.lstsq(J, -r, rcond=None)[0]

    d.qpos[pose_adr], d.qpos[loop_adr], d.qvel[:] = target, q, 0
    mujoco.mj_forward(m, d)
    return np.linalg.norm(closure_error(m, d, q, loop_adr))


def place_objects(m, d):
    """Slide each object along the phalanx normal until it just touches the phalanx
    midpoint.  mj_geomDistance measures the real gap, so the mesh surface and the
    cylinder radius are both accounted for — no offsets are hardcoded."""
    Rb, pb = d.body("base").xmat.reshape(3, 3).copy(), d.body("base").xpos.copy()
    for ph, ph_geom, ob, ob_geom, length in PARTS:
        R, p = d.body(ph).xmat.reshape(3, 3), d.body(ph).xpos
        y = d.body(ob).xpos[1] - p[1]          # keep the authored lateral offset (y is pose-invariant)
        m.body(ob).pos[:] = Rb.T @ (p + R @ [-0.030, y, 0.5 * length] - pb)
        mujoco.mj_forward(m, d)

        for _ in range(6):                     # converges in 1-2 passes on a flat face
            gap = mujoco.mj_geomDistance(m, d, m.geom(ph_geom).id, m.geom(ob_geom).id, 0.1, None)
            m.body(ob).pos[:] += (gap + PENETRATION) * (Rb.T @ R[:, 0])
            mujoco.mj_forward(m, d)
            if abs(gap) < 1e-6:
                break
        print(f"  {ob:<8} placed at {np.array2string(m.body(ob).pos, precision=6)} (base frame)")


def contact_forces(m, d):
    """Resultant world force on each object, summed over its contact points."""
    out, f6 = {ob: np.zeros(3) for _, _, ob, _, _ in PARTS}, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        for _, ph_geom, ob, ob_geom, _ in PARTS:
            og, pg = m.geom(ob_geom).id, m.geom(ph_geom).id
            if {int(c.geom1), int(c.geom2)} == {og, pg}:
                f = c.frame.reshape(3, 3).T @ f6[:3]
                out[ob] += f if int(c.geom2) == og else -f
    return out


def main():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"loop closure residual : {pose_finger(m, d, THETA_DEG):.2e} m")
    place_objects(m, d)

    d.ctrl[m.actuator("motor_input").id] = MOTOR_TORQUE
    for _ in range(int(SETTLE_TIME / m.opt.timestep)):
        mujoco.mj_step(m, d)

    print(f"\ntorque {MOTOR_TORQUE} N m, max joint speed {np.abs(d.qvel).max():.1e} rad/s")
    print("angle [deg]   commanded -> settled")
    for j, cmd in zip(POSE_JOINTS, THETA_DEG):
        print(f"  {j:<12} {cmd:8.2f} -> {np.rad2deg(d.qpos[m.joint(j).qposadr[0]]):8.2f}")

    print("\ncontact forces:")
    for _, _, ob, _, _ in PARTS:
        f = contact_forces(m, d)[ob]
        note = "" if np.linalg.norm(f) > 1e-6 else "   <- separated, pose not an equilibrium"
        print(f"  {ob:<8} |F| = {np.linalg.norm(f):8.4f} N   "
              f"F = [{f[0]: .4f} {f[1]: .4f} {f[2]: .4f}] N{note}")

    if SHOW_VIEWER:
        act = m.actuator("motor_input").id
        lo, hi = m.actuator_ctrlrange[act]
        state = {"torque": MOTOR_TORQUE}
        STEP = 0.5                                   # N m per key press

        def on_key(keycode):
            if keycode == 265:                       # GLFW Up arrow
                state["torque"] = min(state["torque"] + STEP, hi)
            elif keycode == 264:                     # GLFW Down arrow
                state["torque"] = max(state["torque"] - STEP, lo)
            else:
                return
            print(f"torque = {state['torque']:.2f} N m")

        with viewer.launch_passive(m, d, key_callback=on_key) as v:
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            while v.is_running():
                with v.lock():
                    d.ctrl[act] = state["torque"]
                    mujoco.mj_step(m, d)
                v.sync()


if __name__ == "__main__":
    main()