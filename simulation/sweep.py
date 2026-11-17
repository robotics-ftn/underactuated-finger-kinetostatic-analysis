import numpy as np
import pandas as pd
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
MODEL_PATH  = str(ROOT / "models" / "finger.xml")
CSV_PATH    = ROOT / "output_data" / "output.csv"
OUT_DIR     = ROOT / "results" / "sweep"
ROW_STRIDE  = 1       # 1 = every row; raise it to decimate
LAMBDA      = 0.5     # keep only rows with lambda1 = lambda2 = lambda3 = this
Q1          = 0.0     # keep only this q1 slice -> exactly one image per torque

SETTLE_TIME = 2.0     # s of closing motion per row
AVG_TIME    = 0.1     # s at the end over which the forces are averaged
DAMPING     = 0.1     # N m s/rad added to every joint so the motion settles
F_MIN       = 1e-9    # N; above this a phalanx counts as loaded

# contact point on each phalanx, measured along its axis from its own joint.
# Half of the phalanx lengths in finger.xml: 51, 51 and 40 mm.
MIDPOINTS = (0.0255, 0.0255, 0.020)

PARTS = [("proximal", "proximal_geom", "object1", "object_geom1"),
         ("middle",   "middle_geom",   "object2", "object_geom2"),
         ("distal",   "distal_geom",   "object3", "object_geom3")]
POSE_JOINTS = ("joint_A", "joint_B", "joint_C")
LOOP_JOINTS = ("joint_input", "joint_b1", "joint_triangle", "joint_b2")
LOOP_SITES  = (("b1_end", "P2_prime"), ("b2_end", "P3"))


def closure_error(m, d, q_loop, loop_adr):
    d.qpos[loop_adr] = q_loop
    mujoco.mj_kinematics(m, d)
    return np.concatenate([(d.site(a).xpos - d.site(b).xpos)[[0, 2]] for a, b in LOOP_SITES])

def pose_finger(m, d, theta_deg):
    """Set joint_A/B/C, solve the 4 passive joints so both loops stay closed."""
    pose_adr = [m.joint(j).qposadr[0] for j in POSE_JOINTS]
    loop_adr = [m.joint(j).qposadr[0] for j in LOOP_JOINTS]
    q, target = d.qpos[loop_adr].copy(), np.deg2rad(theta_deg)

    for s in np.linspace(0, 1, 21)[1:]:
        d.qpos[pose_adr] = s * target
        r = closure_error(m, d, q, loop_adr)
        for _ in range(30):
            if r @ r < 1e-20:
                break
            J = np.column_stack([(closure_error(m, d, q + 1e-7 * e, loop_adr) - r) / 1e-7
                                 for e in np.eye(4)])
            dq, step = np.linalg.lstsq(J, -r, rcond=None)[0], 1.0
            for _ in range(20):                      # backtrack: keeps it off dead points
                r_try = closure_error(m, d, q + step * dq, loop_adr)
                if np.isfinite(r_try).all() and r_try @ r_try < r @ r:
                    q, r = q + step * dq, r_try
                    break
                step *= 0.5
            else:
                break

    d.qpos[pose_adr], d.qpos[loop_adr], d.qvel[:] = target, q, 0
    mujoco.mj_forward(m, d)
    return np.linalg.norm(closure_error(m, d, q, loop_adr))


def place_objects(m, d):
    for (ph, ph_geom, ob, ob_geom), ki in zip(PARTS, MIDPOINTS):
        R, p = d.body(ph).xmat.reshape(3, 3), d.body(ph).xpos
        y = d.body(ob).xpos[1] - p[1]        # authored lateral offset (y is pose-invariant)
        m.body(ob).pos[:] = p + R @ [-0.030, y, ki] - d.body("base").xpos
        mujoco.mj_forward(m, d)
        for _ in range(6):
            gap = mujoco.mj_geomDistance(m, d, m.geom(ph_geom).id, m.geom(ob_geom).id, 0.1, None)
            m.body(ob).pos[:] += gap * d.body(ph).xmat.reshape(3, 3)[:, 0]
            mujoco.mj_forward(m, d)
            if abs(gap) < 1e-6:
                break


def contact_forces(m, d):
    """Force magnitude on each object right now, summed over its contact points."""
    f, f6 = np.zeros(3), np.zeros(6)
    for i in range(d.ncon):
        pair = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
        for k, (_, ph_geom, _, ob_geom) in enumerate(PARTS):
            if pair == {m.geom(ob_geom).id, m.geom(ph_geom).id}:
                mujoco.mj_contactForce(m, d, i, f6)
                f[k] += np.linalg.norm(f6[:3])
    return f


def simulate_row(m, d, row):
    """Place the objects for this pose, then close the finger onto them."""
    mujoco.mj_resetData(m, d)
    res = pose_finger(m, d, (-row.q1, -row.q2, -row.q3))   # CSV angles are flexion magnitudes
    if not np.isfinite(res) or res > 1e-6:
        return np.full(3, np.nan), f"loops not closed ({res:.1e} m)"
    place_objects(m, d)

    mujoco.mj_resetData(m, d)                              # finger back to fully open
    d.ctrl[m.actuator("motor_input").id] = row.M_mot
    n_avg = int(AVG_TIME / m.opt.timestep)
    for _ in range(int(SETTLE_TIME / m.opt.timestep) - n_avg):
        mujoco.mj_step(m, d)

    f = np.zeros(3)                                        # average out any residual ringing
    for _ in range(n_avg):
        mujoco.mj_step(m, d)
        f += contact_forces(m, d)
    f /= n_avg

    if not np.isfinite(f).all():
        return f, "diverged"
    return f, "ok" if np.abs(d.qvel).max() < 1e-3 else "still moving"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df = df[(df.lambda1 == LAMBDA) & (df.lambda2 == LAMBDA) & (df.lambda3 == LAMBDA)
            & (df.q1 == Q1)].iloc[::ROW_STRIDE].copy()
    if df.empty:
        raise SystemExit(f"no rows with lambda = {LAMBDA} and q1 = {Q1} in {CSV_PATH}")

    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    m.dof_damping[:] = DAMPING          # the XML has none; without it nothing ever settles
    d = mujoco.MjData(m)

    print(f"{len(df)} rows (lambda = {LAMBDA}, q1 = {Q1}, stride {ROW_STRIDE})")
    forces, status = [], []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        f, st = simulate_row(m, d, row)
        forces.append(f); status.append(st)
        if n % 100 == 0 or n == len(df):
            print(f"  {n}/{len(df)}")

    df[["F1", "F2", "F3"]] = np.asarray(forces, dtype=float)
    df["status"] = status
    df["contact"] = (df[["F1", "F2", "F3"]] > F_MIN).all(axis=1).astype(int)
    df.loc[~np.isfinite(df[["F1", "F2", "F3"]]).all(axis=1), "contact"] = np.nan
    df.to_csv(f"{OUT_DIR}/sim_results.csv", index=False)

    for st, c in df.status.value_counts().items():
        print(f"  {c:5d}  {st}")
    bad = df.contact.isna().sum()
    if bad:
        print(f"dropping {bad} unusable rows — see the status column of sim_results.csv")
        df = df[df.contact.notna()]

    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#2b7a3d"])
    legend = [Patch(facecolor="#2b7a3d", label="all three loaded (1)"),
              Patch(facecolor="#d9d9d9", label="at least one unloaded (0)")]

    for mot, g in df.groupby("M_mot"):
        if len(g) < 3:
            print(f"skipping M_mot={mot:g} — only {len(g)} usable rows")
            continue
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.tricontourf(g.q2, g.q3, g.contact, levels=[-0.5, 0.5, 1.5], cmap=cmap)
        ax.legend(handles=legend, loc="upper right", framealpha=0.9)
        ax.set_title(f"$M_{{mot}}$ = {mot:g},  $q_1$ = {Q1:g}°,  $\\lambda$ = {LAMBDA:g}   "
                     f"({int(g.contact.sum())}/{len(g)} rows fully loaded)", fontsize=11)
        ax.set_xlabel(r"$q_2$ ($\degree$)")
        ax.set_ylabel(r"$q_3$ ($\degree$)")
        ax.grid(alpha=0.3, ls="--")
        fig.tight_layout()
        name = f"lambda-{LAMBDA:g}_mot-{mot:g}_q1-{Q1:g}.png"
        fig.savefig(f"{OUT_DIR}/{name}")
        plt.close(fig)
        print(f"wrote {name}   {int(g.contact.sum())}/{len(g)} fully loaded")


if __name__ == "__main__":
    main()