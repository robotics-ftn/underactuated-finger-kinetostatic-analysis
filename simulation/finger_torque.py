#!/usr/bin/env python3
"""
Underactuated finger — torque ramp videos.

Pose the finger, park an object against the middle of each phalanx, let it settle
at zero torque, then ramp the motor torque 0 -> TORQUE_MAX and record two videos
with identical length and frame rate, so they stay in sync side by side:

    FINGER_OUT : the finger, on a plain white background (no wall, no shadows)
    TORQUE_OUT : the input torque over time, drawn up to the current instant

Requires mujoco >= 3.1.4 (mj_geomDistance), imageio + imageio-ffmpeg, matplotlib.
Paths are resolved from the repo root, so it can be run from any directory.
"""

import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
MODEL_PATH   = ROOT / "models" / "finger.xml"
VIDEO_DIR    = ROOT / "results" / "videos"
FINGER_OUT   = VIDEO_DIR / "finger.mp4"
TORQUE_OUT   = VIDEO_DIR / "torque.mp4"

THETA_DEG    = (-0.0, -10.0, -10.0)   # joint_A, joint_B, joint_C (negative = flexion)
TORQUE_MAX   = 20.0                    # N m on actuator "motor_input"
RAMP_TIME    = 10.0                    # s, linear ramp 0 -> TORQUE_MAX
HOLD_TIME    = 2.0                     # s held at TORQUE_MAX; video = RAMP + HOLD
SETTLE_TIME  = 2.0                     # s simulated at zero torque before recording
PENETRATION  = 2e-5                    # m of interference, so contacts are live at t=0

FPS          = 30
WIDTH, HEIGHT = 1280, 720              # keep both multiples of 16 for the encoder

CAM_AZIMUTH, CAM_ELEVATION = 90, 0     # side view, along the joint axes
CAM_DISTANCE = 0.30                    # m
CAM_LOOKAT   = None                    # None = middle of the finger, else (x, y, z)
SHOW_CONTACT_FORCES = False
FORCE_ARROW_SCALE = 0.1                # x the arrow length set in finger.xml (<map force>)

FONT_SIZE    = 18
LINE_COLOR   = "#31688e"

# phalanx body, phalanx geom, object body, object geom, phalanx length [m]
PARTS = [("proximal", "proximal_geom", "object1", "object_geom1", 0.051),
         ("middle",   "middle_geom",   "object2", "object_geom2", 0.051),
         ("distal",   "distal_geom",   "object3", "object_geom3", 0.040)]

POSE_JOINTS = ("joint_A", "joint_B", "joint_C")
LOOP_JOINTS = ("joint_input", "joint_b1", "joint_triangle", "joint_b2")
LOOP_SITES  = (("b1_end", "P2_prime"), ("b2_end", "P3"))

WHITE_SKY = ('<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" '
             'width="8" height="48"/></asset></mujoco>')


def load_model():
    """finger.xml plus a flat white skybox; the wall plane is made invisible."""
    xml = MODEL_PATH.read_text().replace("</mujoco>", WHITE_SKY)
    # from_xml_string resolves mesh files against the CWD, so hand them over explicitly
    assets = {p.relative_to(MODEL_PATH.parent).as_posix(): p.read_bytes()
              for p in (MODEL_PATH.parent / "assets").glob("*.stl")}
    m = mujoco.MjModel.from_xml_string(xml, assets)
    m.geom_rgba[m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, 3] = 0
    m.vis.global_.offwidth, m.vis.global_.offheight = WIDTH, HEIGHT
    m.vis.map.force *= FORCE_ARROW_SCALE
    return m


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


def contact_forces(m, d):
    """Magnitude of the resultant contact force on each object."""
    out, f6 = {ob: np.zeros(3) for _, _, ob, _, _ in PARTS}, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        for _, ph_geom, ob, ob_geom, _ in PARTS:
            if {int(c.geom1), int(c.geom2)} == {m.geom(ob_geom).id, m.geom(ph_geom).id}:
                out[ob] += c.frame.reshape(3, 3).T @ f6[:3]
    return [np.linalg.norm(f) for f in out.values()]


def record_finger(m, d):
    """Simulate the ramp in real time, one rendered frame every 1/FPS s.
    Returns the (time, torque) sample of every frame."""
    act = m.actuator("motor_input").id

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = CAM_AZIMUTH, CAM_ELEVATION, CAM_DISTANCE
    tip = d.body("distal").xpos + d.body("distal").xmat.reshape(3, 3) @ [0, 0, PARTS[2][4]]
    cam.lookat[:] = CAM_LOOKAT if CAM_LOOKAT is not None else 0.5 * (d.body("base").xpos + tip)

    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = SHOW_CONTACT_FORCES

    renderer = mujoco.Renderer(m, HEIGHT, WIDTH)
    for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                 mujoco.mjtRndFlag.mjRND_HAZE):
        renderer.scene.flags[flag] = False

    samples, t0 = [], d.time
    print("\n  t [s]   torque [N m]   |F1|      |F2|      |F3|  [N]")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(FINGER_OUT, fps=FPS, quality=8) as video:
        for k in range(int(round((RAMP_TIME + HOLD_TIME) * FPS)) + 1):
            while d.time - t0 < k / FPS:
                d.ctrl[act] = TORQUE_MAX * min((d.time - t0) / RAMP_TIME, 1.0)
                mujoco.mj_step(m, d)
            renderer.update_scene(d, cam, vopt)
            video.append_data(renderer.render())
            samples.append((k / FPS, d.ctrl[act]))
            if k % FPS == 0:
                print(f"  {k / FPS:5.1f}   {d.ctrl[act]:8.2f}    "
                      + "  ".join(f"{f:8.2f}" for f in contact_forces(m, d)))
    renderer.close()
    print(f"wrote {FINGER_OUT}")
    return np.array(samples)


def record_torque(samples):
    """Same frames as the finger video: the torque trace grows with time."""
    t, tau = samples.T
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)

    ax.plot(t, tau, color="0.85", linewidth=2)                    # the whole profile, faint
    line, = ax.plot([], [], color=LINE_COLOR, linewidth=3)
    dot,  = ax.plot([], [], "o", color=LINE_COLOR, markersize=12, clip_on=False,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
    readout = ax.text(0.04, 0.90, "", transform=ax.transAxes, fontsize=FONT_SIZE * 1.6)

    ax.set_xlim(0, t[-1]), ax.set_ylim(0, TORQUE_MAX * 1.1)
    ax.set_xlabel("time [s]"), ax.set_ylabel(r"input torque $\tau_{\alpha_1}$ [Nm]")
    ax.grid(color="0.92"), ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    with imageio.get_writer(TORQUE_OUT, fps=FPS, quality=8) as video:
        for k in range(len(t)):
            line.set_data(t[:k + 1], tau[:k + 1])
            dot.set_data([t[k]], [tau[k]])
            readout.set_text(rf"$\tau_{{\alpha_1}}$ = {tau[k]:.1f} Nm")
            fig.canvas.draw()
            video.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print(f"wrote {TORQUE_OUT}")


def main():
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"loop closure residual : {pose_finger(m, d, THETA_DEG):.2e} m")
    place_objects(m, d)

    for _ in range(int(SETTLE_TIME / m.opt.timestep)):            # zero torque
        mujoco.mj_step(m, d)
    print("angle [deg]   commanded -> settled at zero torque")
    for j, cmd in zip(POSE_JOINTS, THETA_DEG):
        print(f"  {j:<12} {cmd:8.2f} -> {np.rad2deg(d.qpos[m.joint(j).qposadr[0]]):8.2f}")

    record_torque(record_finger(m, d))


if __name__ == "__main__":
    main()