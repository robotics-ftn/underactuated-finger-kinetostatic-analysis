"""
Animated family of worst-case contact force surfaces, one frame per motor torque.

z = min(F_N1, F_N2, F_N3) at each (q2, q3). Frame 1 shows TORQUES[0] (2 Nm),
frame 2 TORQUES[1] (4 Nm), and so on. Axes, camera and colours are identical in
every frame, so the only thing that changes is the surface itself. A grey plane
sits at z = 0: wherever the surface dips below it, at least one contact has gone
negative and the grasp is infeasible.

Writes a GIF and an MP4. GIF needs only Pillow; MP4 needs ffmpeg on PATH (or
`pip install imageio-ffmpeg`) and is skipped with a message if neither exists.

Rows are kept when:
    M_mot   in TORQUES
    q1      == Q_FIXED_VALUE
    lambda1 == lambda2 == lambda3 == LAMBDA
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm, Normalize, to_rgba
from matplotlib.cm import ScalarMappable

# ----------------------------- settings -----------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
CSV = str(ROOT / "output_data" / "output.csv")
OUT = str(ROOT / "results" / "family" / "forces_min_family")   # stem; one file per entry in FORMATS
FORMATS = ["gif", "mp4"]

TORQUES = list(np.arange(2.0, 20.1, 2.0))   # 2, 4, 6, ... 20 Nm, in frame order
SECONDS_PER_TORQUE = 1.0
END_HOLD = 2.0                 # extra seconds on the last frame before the loop

# False: each frame shows only its own torque.
# True : surfaces accumulate, so the last frame is the whole family.
CUMULATIVE = False
ALPHA_SINGLE = 0.90
ALPHA_CUMULATIVE = 0.40        # alpha compounds across stacked surfaces

LAMBDA = 0.5
Q_FIXED_COL = "q1"
Q_FIXED_VALUE = 0.0
Q_X, Q_Y = "q2", "q3"
FORCE_COLS = ["F_N1", "F_N2", "F_N3"]

DUPLICATE_AGG = "mean"         # only fires if a cell holds >1 row
FEASIBLE_ONLY = False
POSITIVE_ONLY = False
NORMALIZE = False              # divide each surface by its own torque

# "force" : surface coloured by its own height, one colour scale shared by all
#           frames (so a low-torque surface stays dark and brightens as it grows)
# "torque": one flat colour per torque -- the only readable choice if CUMULATIVE
COLOR_BY = "force"
CMAP = "viridis"
EDGE_LINEWIDTH = 0.15
ZERO_PLANE = True
ZERO_PLANE_COLOR = "0.35"
ZERO_PLANE_ALPHA = 0.30
ZERO_CONTOURS = True           # zero crossing projected on the floor
ZERO_CONTOUR_COLOR = "red"     # used when COLOR_BY = "force"
ZERO_SHADE = True              # fill the floor where the force is negative
ZERO_SHADE_ALPHA = 0.25

FONT_SIZE = 22
CBAR_FONT_SIZE = 22            # the two end values on the colourbar
Z_AXIS_SIDE = "lower"          # needs matplotlib >= 3.8
ANGLE_UNITS = "deg"
ELEV, AZIM = 24, -128
FIGSIZE = (11.5, 9.0)
DPI = 120                      # 1380 x 1080 px; keep FIGSIZE * DPI even for mp4

CHUNKSIZE = 500_000
TOL = 1e-9
# --------------------------------------------------------------------

plt.rcParams.update({"font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
                     "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
                     "ytick.labelsize": FONT_SIZE})

NUMERIC = ["M_mot", Q_FIXED_COL, Q_X, Q_Y, "lambda1", "lambda2", "lambda3"] + FORCE_COLS
USECOLS = list(dict.fromkeys(NUMERIC + ["IsPoseFeasible?", "IsForcePositice?"]))


def close(series, value):
    return np.isclose(series.to_numpy(dtype=float), value, atol=TOL, rtol=0.0)


# ============================ load ==================================
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if not os.path.exists(CSV):
    raise SystemExit(f"CSV not found: {os.path.abspath(CSV)}")

print(f"reading {CSV} in chunks of {CHUNKSIZE:,} rows ...")
kept, total, seen_torques = [], 0, set()

for n, chunk in enumerate(pd.read_csv(CSV, usecols=USECOLS, chunksize=CHUNKSIZE), 1):
    total += len(chunk)
    for c in NUMERIC:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    seen_torques.update(chunk["M_mot"].dropna().unique()[:80])

    m = np.zeros(len(chunk), dtype=bool)
    for T in TORQUES:
        m |= close(chunk["M_mot"], T)
    m &= close(chunk[Q_FIXED_COL], Q_FIXED_VALUE)
    for lam in ("lambda1", "lambda2", "lambda3"):
        m &= close(chunk[lam], LAMBDA)
    if FEASIBLE_ONLY:
        m &= chunk["IsPoseFeasible?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()
    if POSITIVE_ONLY:
        m &= chunk["IsForcePositice?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()

    if m.any():
        kept.append(chunk.loc[m, ["M_mot", Q_X, Q_Y] + FORCE_COLS])
    print(f"  chunk {n}: {total:,} scanned, {sum(len(k) for k in kept):,} kept", end="\r")

print()
if not kept:
    print(f"\nNo rows matched. M_mot values present in the file:\n  {sorted(seen_torques)[:40]}")
    raise SystemExit("Adjust TORQUES / LAMBDA / Q_FIXED_VALUE.")

df = pd.concat(kept, ignore_index=True)
df["F_min"] = df[FORCE_COLS].min(axis=1)
df.loc[df[FORCE_COLS].isna().any(axis=1), "F_min"] = np.nan
print(f"scanned {total:,} rows, kept {len(df):,}")


# ============================ grids =================================
present, grids = [], []
for T in TORQUES:
    sub = df[close(df["M_mot"], T)]
    if sub.empty:
        print(f"  M_mot = {T:g}: no rows, skipped")
        continue
    piv = sub.pivot_table(index=Q_Y, columns=Q_X, values="F_min",
                          aggfunc=DUPLICATE_AGG, dropna=False)
    X, Y = np.meshgrid(piv.columns.to_numpy(float), piv.index.to_numpy(float))
    Z = piv.to_numpy(float) / (T if NORMALIZE else 1.0)
    print(f"  M_mot = {T:5g}: {Z.shape[1]}x{Z.shape[0]} grid, min {np.nanmin(Z):8.3f}, "
          f"max {np.nanmax(Z):8.3f}, {int((Z < 0).sum())}/{np.isfinite(Z).sum()} points below zero")
    present.append(T)
    grids.append((X, Y, Z))

if not grids:
    raise SystemExit("Nothing to plot.")


# ============================ figure ================================
# limits come from the whole family, so every frame shares one set of axes
xlim = (min(g[0].min() for g in grids), max(g[0].max() for g in grids))
ylim = (min(g[1].min() for g in grids), max(g[1].max() for g in grids))
zmin = min(0.0, min(np.nanmin(g[2]) for g in grids))
zmax = max(0.0, max(np.nanmax(g[2]) for g in grids))
pad = 0.06 * (zmax - zmin) if zmax > zmin else 1.0
zlim = (zmin - pad, zmax + pad)

zlabel = "$\\min(F_{N1}, F_{N2}, F_{N3})$" + (" $/\\, M_{mot}$  [N/Nm]" if NORMALIZE else "  [N]")
alpha = ALPHA_CUMULATIVE if CUMULATIVE else ALPHA_SINGLE

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(projection="3d")

if COLOR_BY == "force":
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(min(np.nanmin(g[2]) for g in grids), max(np.nanmax(g[2]) for g in grids))
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                      shrink=0.62, aspect=22, pad=0.11)
    cb.set_label("force / torque [N/Nm]" if NORMALIZE else "force [N]", fontsize=FONT_SIZE)
    ends = [norm.vmin, norm.vmax]
else:
    edges = np.array(present, dtype=float)
    step = np.diff(edges).min() if len(edges) > 1 else 1.0
    cmap = plt.get_cmap(CMAP, len(present))
    norm = BoundaryNorm(np.append(edges - step / 2, edges[-1] + step / 2), cmap.N)
    colors = [cmap(i) for i in range(len(present))]
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                      shrink=0.62, aspect=22, pad=0.11)
    cb.set_label(r"$\tau_{\alpha_1}$ [Nm]", fontsize=FONT_SIZE)
    ends = [present[0], present[-1]]

# only the smallest and largest value are labelled
cb.set_ticks(ends, labels=[f"{v:.3g}".replace("-", "\u2212") for v in ends])
cb.ax.tick_params(labelsize=CBAR_FONT_SIZE)


def draw(k):
    """Frame k: the surface for present[k] (plus all earlier ones if CUMULATIVE)."""
    ax.cla()
    shown = range(k + 1) if CUMULATIVE else [k]

    # lowest first: mplot3d depth-sorts whole collections, not individual faces
    for i in sorted(shown, key=lambda i: np.nanmean(grids[i][2])):
        X, Y, Z = grids[i]
        if COLOR_BY == "force":
            look = dict(cmap=cmap, norm=norm)
        else:
            look = dict(color=colors[i], edgecolors=to_rgba(colors[i], 0.55))
        surf = ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=EDGE_LINEWIDTH, shade=False,
                               antialiased=True, rstride=1, cstride=1, **look)
        surf._sort_zpos = float(np.nanmean(Z))
        if np.nanmin(Z) < 0 < np.nanmax(Z):
            c = ZERO_CONTOUR_COLOR if COLOR_BY == "force" else colors[i]
            if ZERO_SHADE:                     # the infeasible region, between the zero lines
                ax.contourf(X, Y, Z, levels=[np.nanmin(Z) - 1.0, 0.0], zdir="z",
                            offset=zlim[0], colors=[c], alpha=ZERO_SHADE_ALPHA)
            if ZERO_CONTOURS:
                ax.contour(X, Y, Z, levels=[0.0], zdir="z", offset=zlim[0],
                           linewidths=2.0, colors=[c])

    if ZERO_PLANE:
        px, py = np.meshgrid(xlim, ylim)
        plane = ax.plot_surface(px, py, np.zeros((2, 2)), color=ZERO_PLANE_COLOR,
                                alpha=ZERO_PLANE_ALPHA, shade=False, linewidth=0)
        plane._sort_zpos = 0.0

    ax.set_xlim(xlim), ax.set_ylim(ylim), ax.set_zlim(zlim)
    try:
        ax.zaxis.set_ticks_position(Z_AXIS_SIDE)
        ax.zaxis.set_label_position(Z_AXIS_SIDE)
    except AttributeError:
        pass                                   # matplotlib < 3.8
    ax.set_xlabel(f"${Q_X}$ [{ANGLE_UNITS}]", labelpad=10)
    ax.set_ylabel(f"${Q_Y}$ [{ANGLE_UNITS}]", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=12)
    ax.set_title(f"$\\lambda_1=\\lambda_2=\\lambda_3$ = {LAMBDA:g},   "
                 f"${Q_FIXED_COL}$ = {Q_FIXED_VALUE:g}\n"
                 f"$\\tau_{{\\alpha_1}}$ = {present[k]:g} Nm", pad=18)
    ax.view_init(elev=ELEV, azim=AZIM)


draw(len(present) - 1)
fig.tight_layout()             # once, so the layout cannot shift between frames


# ============================ write =================================
fps = 1.0 / SECONDS_PER_TORQUE
writers = {}
for ext in FORMATS:
    if ext == "gif":
        writers[ext] = animation.PillowWriter(fps=fps)
    elif ext == "mp4":
        if not animation.writers.is_available("ffmpeg"):
            try:
                import imageio_ffmpeg
                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                print("!! mp4 skipped: no ffmpeg on PATH (pip install imageio-ffmpeg)")
                continue
        # frames go in at `fps`; -r 30 duplicates them so every player copes
        writers[ext] = animation.FFMpegWriter(fps=fps, extra_args=["-r", "30"])
    else:
        print(f"!! unknown format '{ext}', skipped")

for ext, w in writers.items():
    w.setup(fig, f"{OUT}.{ext}", dpi=DPI)

hold = int(round(END_HOLD / SECONDS_PER_TORQUE))
for k, T in enumerate(present):
    draw(k)
    for w in writers.values():
        for _ in range(1 + hold * (k == len(present) - 1)):
            w.grab_frame()
    print(f"  frame {k + 1}/{len(present)}: {T:g} Nm")

for ext, w in writers.items():
    w.finish()
    print(f"wrote {OUT}.{ext}")
plt.close(fig)