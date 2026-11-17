"""Plot min-force difference (analytical output.csv minus simulation sim_results.csv)
over the q2/q3 grid for selected motor torques, plus a mean/std summary plot.

Edit the paths in the settings block below and run: python plotting/errors.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]                 # repo root
OUTPUT_CSV = ROOT / "output_data" / "output.csv"           # analytical results (F_N1..3)
SIM_CSV = ROOT / "results" / "sweep" / "sim_results.csv"   # simulation results (F1..3)
OUT_DIR = ROOT / "results" / "force_diff"                  # where plots are written
TORQUES = list(range(2, 21, 2))          # motor torques [Nm] to plot
FORMATS = ("png", "eps")                 # image formats to save
DPI = 150
# --------------------------------------------------------------------------


def save(fig, stem):
    for ext in FORMATS:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")

OUT_DIR.mkdir(parents=True, exist_ok=True)

out = pd.read_csv(OUTPUT_CSV)
sim = pd.read_csv(SIM_CSV)

# sim_results only covers q1 = 0; output.csv also carries F_N1..3 (analytical) rather than F1..3
out = out[out["q1"] == 0].copy()
# negative analytical forces are treated as 0 (no pushing contact)
out["Fmin_out"] = out[["F_N1", "F_N2", "F_N3"]].min(axis=1).clip(lower=0)
sim["Fmin_sim"] = sim[["F1", "F2", "F3"]].min(axis=1)

keys = ["M_mot", "q2", "q3"]
df = sim[keys + ["Fmin_sim"]].merge(out[keys + ["Fmin_out"]], on=keys, how="inner")
df["diff"] = df["Fmin_out"] - df["Fmin_sim"]

summary = []
for M in TORQUES:
    d = df[df["M_mot"] == M]
    if d.empty:
        print(f"no data for M_mot = {M}, skipping")
        continue

    grid = d.pivot(index="q3", columns="q2", values="diff").sort_index().sort_index(axis=1)
    Q2, Q3 = np.meshgrid(grid.columns.values, grid.index.values)
    Z = grid.values

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Q2, Q3, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("q2 [deg]")
    ax.set_ylabel("q3 [deg]")
    ax.set_zlabel("ΔF [N]")
    ax.set_title(rf"min(F_N) analytical − min(F) simulation, $\tau_{{\alpha_1}}$ = {M} Nm")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="force difference [N]")
    save(fig, f"force_diff_M{M}")
    plt.close(fig)

    summary.append({"M_mot": M, "mean": d["diff"].mean(), "std": d["diff"].std(),
                    "n": len(d)})
    print(f"M = {M:2d} Nm: mean = {summary[-1]['mean']:8.3f} N, std = {summary[-1]['std']:7.3f} N")

summ = pd.DataFrame(summary)
summ.to_csv(OUT_DIR / "force_diff_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(summ["M_mot"], summ["mean"], yerr=summ["std"], fmt="o-", capsize=4,
            label="mean ± std")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel(r"$\tau_{\alpha_1}$ [Nm]", fontsize=18)
ax.set_ylabel(r"$\Delta m_F = m_F - m_{F\_sim}$ [N]", fontsize=18)
ax.set_xticks(summ["M_mot"])
ax.grid(alpha=0.3)
ax.legend()
save(fig, "force_diff_summary")
plt.close(fig)

print(f"saved to {OUT_DIR.resolve()}")