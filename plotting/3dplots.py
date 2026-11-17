"""
Contact force surfaces vs. q2 / q3, one image per (torque, force) pair.

Produces 8 files instead of 2, each holding a single 3D plot, with no titles:
    forces_M5_F_N1 / F_N2 / F_N3 / F_min    at M_mot = 5
    forces_M20_F_N1 / F_N2 / F_N3 / F_min   at M_mot = 20
Each is written as PNG and EPS.

Colours are unchanged from the four-panel version: the surface is coloured by
height with CMAP, the zero plane is translucent grey, the zero crossing is a red
line on the floor, and each plot keeps its own colorbar labelled in newtons.

Reads the CSV in chunks, so the file size does not matter.

Rows are kept when ALL of these hold:
    M_mot   in TORQUES
    q1      == Q_FIXED_VALUE
    lambda1 == lambda2 == lambda3 == LAMBDA

Since lambda_i is the normalized contact position (k_i = lambda_i * L_i), fixing
LAMBDA also fixes k1/k2/k3. One row per (q2, q3) is therefore expected. The
script still counts rows per cell: a count above 1 means some OTHER swept
parameter is varying, which you probably want to know about.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed; comment out to use plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

# ----------------------------- settings -----------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
CSV = str(ROOT / "output_data" / "output.csv")

TORQUES = [5.0, 20.0]          # 2 torques x 4 forces = 8 images
LAMBDA = 0.5                   # required value of lambda1, lambda2, lambda3

Q_FIXED_COL = "q1"             # the joint held constant ("q0" in your message)
Q_FIXED_VALUE = 0.0
Q_X, Q_Y = "q2", "q3"          # the two swept joints

FORCE_COLS = ["F_N1", "F_N2", "F_N3"]
FORCE_LABELS = ["proximal  $F_{N1}$", "middle  $F_{N2}$", "distal  $F_{N3}$"]

# Fourth image per torque: the weakest of the three contacts at each pose. Where
# this is negative, at least one contact has gone invalid.
ADD_MIN_PANEL = True
MIN_COL = "F_min"
MIN_LABEL = "$\\min(F_{N1},F_{N2},F_{N3})$"

# Fallback if more than one row lands on a grid point. With lambda fixed this
# should not happen (k is determined by lambda), so if it does, check what else
# your sweep varies before trusting the surface. "mean", "min", "max", "median",
# "first".
DUPLICATE_AGG = "mean"

FEASIBLE_ONLY = False          # keep only IsPoseFeasible? == Yes
POSITIVE_ONLY = False          # keep only IsForcePositice? == Yes

ANGLE_UNITS = "deg"            # axis label only; no conversion is applied
SHARE_Z = False                # True -> identical z limits across the 4 images
                               #         of one torque (was: across the panels)
CMAP = "viridis"

ZERO_PLANE = True              # draw a translucent plane at z = 0
ZERO_PLANE_COLOR = "0.35"
ZERO_PLANE_ALPHA = 0.22
ZERO_CONTOUR = True            # red line where the surface crosses zero
CHUNKSIZE = 500_000
TOL = 1e-9                     # float comparison tolerance

OUT_DIR = str(ROOT / "results" / "figs")   # created if missing
OUT_PATTERN = "forces_M{t:g}_{col}.{ext}"
SAVE_FORMATS = ["png", "eps"]  # add "pdf" or "svg" for vector output WITH alpha
FIGSIZE = (6.8, 5.8)           # one plot per figure now, not a row of four
DPI = 150

# EPS/PostScript has no alpha channel, so the translucent zero plane would be
# dropped or force the whole page to raster. Rasterising only the 3D artists
# keeps it and leaves axes, ticks, labels and colorbar as true vectors.
RASTERIZE_3D = True
RASTER_DPI = 400
# --------------------------------------------------------------------

USECOLS = list(dict.fromkeys(
    ["M_mot", Q_FIXED_COL, Q_X, Q_Y, "lambda1", "lambda2", "lambda3"]
    + FORCE_COLS + ["IsPoseFeasible?", "IsForcePositice?"]
))


def close_to_any(series, values, tol=TOL):
    """Boolean mask: series is within tol of any value in `values`."""
    arr = series.to_numpy(dtype=float, copy=False)
    m = np.zeros(len(arr), dtype=bool)
    for v in values:
        m |= np.isclose(arr, v, atol=tol, rtol=0.0)
    return m


def save(fig, T, col):
    """Write one figure in every requested format."""
    for ext in SAVE_FORMATS:
        name = OUT_PATTERN.format(t=T, col=col, ext=ext)
        path = os.path.join(OUT_DIR, name) if OUT_DIR else name
        fig.savefig(path, format=ext, bbox_inches="tight",
                    dpi=RASTER_DPI if ext in ("eps", "ps") else DPI)
        print(f"    wrote {path}")


# ============================ load ==================================
if not os.path.exists(CSV):
    raise SystemExit(f"CSV not found: {os.path.abspath(CSV)}")

if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

print(f"reading {CSV} in chunks of {CHUNKSIZE:,} rows ...")

kept, total = [], 0
seen_torques, seen_lambdas, seen_qfixed = set(), set(), set()

reader = pd.read_csv(CSV, usecols=USECOLS, chunksize=CHUNKSIZE)
for n, chunk in enumerate(reader, 1):
    total += len(chunk)

    for c in ["M_mot", Q_FIXED_COL, Q_X, Q_Y, "lambda1", "lambda2", "lambda3"] + FORCE_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

    # remember what values actually exist, for a useful message if nothing matches
    seen_torques.update(chunk["M_mot"].dropna().unique()[:50])
    seen_lambdas.update(chunk["lambda1"].dropna().unique()[:50])
    seen_qfixed.update(chunk[Q_FIXED_COL].dropna().unique()[:50])

    m = close_to_any(chunk["M_mot"], TORQUES)
    m &= np.isclose(chunk[Q_FIXED_COL].to_numpy(float), Q_FIXED_VALUE, atol=TOL, rtol=0.0)
    for lam in ("lambda1", "lambda2", "lambda3"):
        m &= np.isclose(chunk[lam].to_numpy(float), LAMBDA, atol=TOL, rtol=0.0)

    if FEASIBLE_ONLY and "IsPoseFeasible?" in chunk:
        m &= chunk["IsPoseFeasible?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()
    if POSITIVE_ONLY and "IsForcePositice?" in chunk:
        m &= chunk["IsForcePositice?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()

    if m.any():
        kept.append(chunk.loc[m, ["M_mot", Q_X, Q_Y] + FORCE_COLS])

    print(f"  chunk {n}: {total:,} rows scanned, "
          f"{sum(len(k) for k in kept):,} kept", end="\r")

print()

if not kept:
    print("\nNo rows matched. Values present in the file:")
    print(f"  M_mot           : {sorted(seen_torques)[:20]}")
    print(f"  lambda1         : {sorted(seen_lambdas)[:20]}")
    print(f"  {Q_FIXED_COL:16s}: {sorted(seen_qfixed)[:20]}")
    raise SystemExit("Adjust TORQUES / LAMBDA / Q_FIXED_VALUE to match.")

df = pd.concat(kept, ignore_index=True)

if ADD_MIN_PANEL:
    # per row, before any aggregation, and NaN if any component is missing
    df[MIN_COL] = df[FORCE_COLS].min(axis=1)
    df.loc[df[FORCE_COLS].isna().any(axis=1), MIN_COL] = np.nan

PLOT_COLS = FORCE_COLS + ([MIN_COL] if ADD_MIN_PANEL else [])
PLOT_LABELS = FORCE_LABELS + ([MIN_LABEL] if ADD_MIN_PANEL else [])
print(f"scanned {total:,} rows, kept {len(df):,}")
print(f"-> up to {len(TORQUES) * len(PLOT_COLS)} figures "
      f"x {len(SAVE_FORMATS)} format(s)")


# ============================ plot ==================================
def build_grid(sub, col):
    """Pivot to a regular (q3 x q2) grid. Returns X, Y, Z, n_per_cell."""
    piv = sub.pivot_table(index=Q_Y, columns=Q_X, values=col,
                          aggfunc=DUPLICATE_AGG, dropna=False)
    cnt = sub.pivot_table(index=Q_Y, columns=Q_X, values=col,
                          aggfunc="count", dropna=False)
    X, Y = np.meshgrid(piv.columns.to_numpy(float), piv.index.to_numpy(float))
    return X, Y, piv.to_numpy(float), cnt.to_numpy(float)


for T in TORQUES:
    sub_all = df[np.isclose(df["M_mot"].to_numpy(float), T, atol=TOL, rtol=0.0)]
    if sub_all.empty:
        print(f"\nM_mot = {T:g}: no rows, skipping.")
        continue

    nx = sub_all[Q_X].nunique()
    ny = sub_all[Q_Y].nunique()
    print(f"\nM_mot = {T:g}: {len(sub_all):,} rows, "
          f"{nx} unique {Q_X} x {ny} unique {Q_Y}")

    per_cell = len(sub_all) / max(nx * ny, 1)
    if per_cell > 1.01:
        print(f"  ! {per_cell:.1f} rows per (q2,q3) cell. With lambda fixed this "
              f"should be 1.0 -- something else in the sweep varies.")
        print(f"  ! collapsing with '{DUPLICATE_AGG}'; check before trusting the surface.")

    grids = [build_grid(sub_all, c) for c in PLOT_COLS]

    if SHARE_Z:
        finite = np.concatenate([g[2][np.isfinite(g[2])].ravel() for g in grids])
        zlim = (finite.min(), finite.max()) if finite.size else None
    else:
        zlim = None

    for i, (col, label) in enumerate(zip(PLOT_COLS, PLOT_LABELS)):
        X, Y, Z, cnt = grids[i]

        filled = np.isfinite(Z)
        if not filled.any():
            print(f"  {col}: no data, skipping figure")
            continue

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(projection="3d")

        coverage = filled.sum() / filled.size
        if coverage < 0.5:
            # sparse / irregular sampling: a surface would be mostly holes
            pts = sub_all.dropna(subset=[col])
            ax.plot_trisurf(pts[Q_X].to_numpy(float), pts[Q_Y].to_numpy(float),
                            pts[col].to_numpy(float), cmap=CMAP,
                            linewidth=0.1, antialiased=True,
                            rasterized=RASTERIZE_3D)
            print(f"  {col}: scattered, {coverage:.0%} grid coverage")
        else:
            surf = ax.plot_surface(X, Y, Z, cmap=CMAP, linewidth=0,
                                   antialiased=True, rstride=1, cstride=1,
                                   rasterized=RASTERIZE_3D)
            fig.colorbar(surf, ax=ax, shrink=0.60, aspect=16, pad=0.13,
                         label="force [N]")

        ax.set_xlabel(f"${Q_X}$ [{ANGLE_UNITS}]", labelpad=8)
        ax.set_ylabel(f"${Q_Y}$ [{ANGLE_UNITS}]", labelpad=8)
        ax.set_zlabel("force [N]", labelpad=8)
        ax.view_init(elev=26, azim=-125)

        # z limits must bracket 0 so the plane is inside the box
        if zlim:
            lo, hi = zlim
        else:
            vals = Z[filled]
            lo, hi = float(vals.min()), float(vals.max())
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.set_zlim(lo - pad, hi + pad)

        if ZERO_PLANE:
            # 2x2 quad, not a full-resolution grid: renders far more cleanly
            px = np.array([[X.min(), X.max()], [X.min(), X.max()]])
            py = np.array([[Y.min(), Y.min()], [Y.max(), Y.max()]])
            ax.plot_surface(px, py, np.zeros((2, 2)),
                            color=ZERO_PLANE_COLOR, alpha=ZERO_PLANE_ALPHA,
                            shade=False, linewidth=0, antialiased=False,
                            rasterized=RASTERIZE_3D)

        if ZERO_CONTOUR and np.nanmin(Z) < 0 < np.nanmax(Z):
            # projected onto the floor, not onto z=0: matplotlib's 3d renderer
            # has no per-fragment depth sorting, so a line at z=0 is swallowed
            # by the opaque surface.
            try:
                ax.contour(X, Y, Z, levels=[0.0], zdir="z", offset=lo - pad,
                           colors="red", linewidths=2.0)
            except Exception:
                pass       # degenerate grid; the plane alone still shows the sign

        n_neg = int((Z[filled] < 0).sum())
        if n_neg:
            print(f"  {col}: {n_neg}/{filled.sum()} grid points below zero "
                  f"(min {np.nanmin(Z):.3f} N)")

        fig.tight_layout()
        save(fig, T, col)
        plt.close(fig)

print("\ndone.")