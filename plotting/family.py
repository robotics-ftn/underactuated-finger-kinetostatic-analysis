"""
Family of worst-case contact force surfaces, one per motor torque.

Single 3D plot. z = min(F_N1, F_N2, F_N3) at each (q2, q3), drawn once for every
value of M_mot in TORQUES, semi-transparent so the whole family is visible at
once. A grey plane sits at z = 0: anywhere a surface dips below it, at least one
contact has gone negative and the grasp is infeasible.

Each surface is a single flat colour taken from a colormap indexed by torque, so
colour identifies torque. Three redundant cues are provided, since overlapping
translucent surfaces are hard to read:
    - a discrete colorbar with a tick per torque value
    - a legend
    - a text label floating at the corner of each surface (off by default)

Rows are kept when:
    M_mot   in TORQUES
    q1      == Q_FIXED_VALUE
    lambda1 == lambda2 == lambda3 == LAMBDA
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # comment out (and see SHOW below) for a live window
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------- settings -----------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
CSV = str(ROOT / "output_data" / "output.csv")
OUT = str(ROOT / "results" / "family" / "forces_min_family.png")

TORQUES = list(np.arange(2.0, 20.1, 2.0))   # 2, 4, 6, ... 20 Nm
LAMBDA = 0.5
Q_FIXED_COL = "q1"
Q_FIXED_VALUE = 0.0
Q_X, Q_Y = "q2", "q3"
FORCE_COLS = ["F_N1", "F_N2", "F_N3"]

DUPLICATE_AGG = "mean"         # only fires if a cell holds >1 row
FEASIBLE_ONLY = False
POSITIVE_ONLY = False

# "ribs"      : parallel curves at constant q3 -- clearest for many surfaces
# "wireframe" : full mesh, busier
# "surface"   : translucent filled, prettiest but muddiest with 10 levels
STYLE = "surface"
RIB_STRIDE = 5                 # draw every Nth rib ("ribs" mode)
RIB_LINEWIDTH = 1.6
# Alpha COMPOUNDS: N surfaces at alpha a transmit (1-a)^N. Ten at 0.40 leave
# 0.6% -- effectively opaque. AUTO_ALPHA picks alpha so the whole stack still
# transmits TARGET_TRANSMISSION, which is the only way "surface" mode stays
# readable past about four torques.
AUTO_ALPHA = False
TARGET_TRANSMISSION = 0.35
SURFACE_ALPHA = 0.40           # used when AUTO_ALPHA = False

# "mean"      paint by mean height (the original look: the widest surface lands
#             last and dominates, giving one solid outer shell)
# "amplitude" paint widest first so narrower surfaces sit on top -- same style,
#             a bit more of the family visible
DRAW_ORDER = "mean"
EDGE_LINEWIDTH = 0.15
WIRE_STRIDE = 3                # "wireframe" mode
CMAP = "plasma"                # "turbo" separates the levels more

# Divide each surface by its own torque. Removes the near-linear scaling with
# M_mot so the surfaces stop stacking far apart and their SHAPES can be
# compared. Sign is unchanged (M_mot > 0), so the z=0 plane stays meaningful.
NORMALIZE = False

# Purely visual vertical offset between consecutive surfaces, in z units. Any
# non-zero value makes the z axis unreadable as force -- use only to untangle a
# badly overlapping family, never for a figure someone will read numbers off.
EXPLODE = 0.0

ZERO_PLANE = True
ZERO_PLANE_COLOR = "0.35"
ZERO_PLANE_ALPHA = 0.30
ZERO_CONTOURS = True           # per-torque zero crossing, projected on the floor

LABEL_SURFACES = False         # floating "N Nm" text on each surface
LABEL_ANNOTATE_SPREAD = True   # labels go where the family fans out most
SHOW_LEGEND = True

# One size for every piece of text: title, axis labels, tick labels, legend,
# colorbar and surface labels.
FONT_SIZE = 18

# Which vertical edge of the box carries the z axis (ticks + label). With the
# default view (AZIM = -128) "upper" is the left edge, "lower" the right edge.
# Needs matplotlib >= 3.8; older versions silently keep the default.
Z_AXIS_SIDE = "lower"

ANGLE_UNITS = "deg"
ELEV, AZIM = 24, -128
FIGSIZE = (11.5, 9.0)
DPI = 160

# Extra formats written alongside OUT, e.g. ["eps"], ["eps", "pdf"], ["svg"].
SAVE_FORMATS = ["eps"]
# EPS/PostScript cannot do transparency: alpha is silently dropped and the
# surfaces come out opaque. Rasterizing just the surfaces keeps the look while
# text, axes, labels and the legend stay vector. Turn this off only if you want
# a fully vector EPS and accept losing the translucency.
RASTERIZE_SURFACES = True

SHOW = False                   # True -> plt.show() instead of / as well as saving
CHUNKSIZE = 500_000
TOL = 1e-9
# --------------------------------------------------------------------

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
})

USECOLS = list(dict.fromkeys(
    ["M_mot", Q_FIXED_COL, Q_X, Q_Y, "lambda1", "lambda2", "lambda3"]
    + FORCE_COLS + ["IsPoseFeasible?", "IsForcePositice?"]
))


def close_to_any(series, values, tol=TOL):
    arr = series.to_numpy(dtype=float, copy=False)
    m = np.zeros(len(arr), dtype=bool)
    for v in values:
        m |= np.isclose(arr, v, atol=tol, rtol=0.0)
    return m


# ============================ load ==================================
if not os.path.exists(CSV):
    raise SystemExit(f"CSV not found: {os.path.abspath(CSV)}")

print(f"reading {CSV} in chunks of {CHUNKSIZE:,} rows ...")
kept, total = [], 0
seen_torques = set()

for n, chunk in enumerate(pd.read_csv(CSV, usecols=USECOLS, chunksize=CHUNKSIZE), 1):
    total += len(chunk)
    for c in ["M_mot", Q_FIXED_COL, Q_X, Q_Y, "lambda1", "lambda2", "lambda3"] + FORCE_COLS:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    seen_torques.update(chunk["M_mot"].dropna().unique()[:80])

    m = close_to_any(chunk["M_mot"], TORQUES)
    m &= np.isclose(chunk[Q_FIXED_COL].to_numpy(float), Q_FIXED_VALUE, atol=TOL, rtol=0.0)
    for lam in ("lambda1", "lambda2", "lambda3"):
        m &= np.isclose(chunk[lam].to_numpy(float), LAMBDA, atol=TOL, rtol=0.0)
    if FEASIBLE_ONLY:
        m &= chunk["IsPoseFeasible?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()
    if POSITIVE_ONLY:
        m &= chunk["IsForcePositice?"].astype(str).str.strip().str.lower().eq("yes").to_numpy()

    if m.any():
        kept.append(chunk.loc[m, ["M_mot", Q_X, Q_Y] + FORCE_COLS])
    print(f"  chunk {n}: {total:,} scanned, {sum(len(k) for k in kept):,} kept", end="\r")

print()
if not kept:
    print("\nNo rows matched. M_mot values present in the file:")
    print(f"  {sorted(seen_torques)[:40]}")
    raise SystemExit("Adjust TORQUES / LAMBDA / Q_FIXED_VALUE.")

df = pd.concat(kept, ignore_index=True)
df["F_min"] = df[FORCE_COLS].min(axis=1)
df.loc[df[FORCE_COLS].isna().any(axis=1), "F_min"] = np.nan
print(f"scanned {total:,} rows, kept {len(df):,}")


# ============================ grids =================================
def build_grid(sub):
    piv = sub.pivot_table(index=Q_Y, columns=Q_X, values="F_min",
                          aggfunc=DUPLICATE_AGG, dropna=False)
    X, Y = np.meshgrid(piv.columns.to_numpy(float), piv.index.to_numpy(float))
    return X, Y, piv.to_numpy(float)


present, grids = [], []
for T in TORQUES:
    sub = df[np.isclose(df["M_mot"].to_numpy(float), T, atol=TOL, rtol=0.0)]
    if sub.empty:
        print(f"  M_mot = {T:g}: no rows, skipped")
        continue
    nx, ny = sub[Q_X].nunique(), sub[Q_Y].nunique()
    per_cell = len(sub) / max(nx * ny, 1)
    if per_cell > 1.01:
        print(f"  ! M_mot = {T:g}: {per_cell:.1f} rows per cell, collapsed with "
              f"'{DUPLICATE_AGG}'")
    X, Y, Z = build_grid(sub)
    if NORMALIZE:
        Z = Z / T
    finite = np.isfinite(Z)
    n_neg = int((Z[finite] < 0).sum())
    print(f"  M_mot = {T:5g}: {nx}x{ny} grid, min {np.nanmin(Z):8.3f} N, "
          f"max {np.nanmax(Z):8.3f} N, {n_neg}/{finite.sum()} points below zero")
    present.append(T)
    grids.append((X, Y, Z))

if not grids:
    raise SystemExit("Nothing to plot.")


# ============================ plot ==================================
edges = np.array(present, dtype=float)
step = np.diff(edges).min() if len(edges) > 1 else 1.0
bounds = np.append(edges - step / 2, edges[-1] + step / 2)
cmap = plt.get_cmap(CMAP, len(present))
norm = BoundaryNorm(bounds, cmap.N)
colors = [cmap(i) for i in range(len(present))]

alpha = SURFACE_ALPHA
if AUTO_ALPHA and STYLE == "surface":
    alpha = 1.0 - TARGET_TRANSMISSION ** (1.0 / max(len(present), 1))
    print(f"  AUTO_ALPHA: {len(present)} surfaces -> alpha = {alpha:.3f} "
          f"(stack transmits {TARGET_TRANSMISSION:.0%})")

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(projection="3d")

# draw lowest surfaces first: matplotlib sorts whole collections by mean depth,
# so painting bottom-up keeps the stack looking roughly correct
# nested families (each torque wrapping the previous) must be painted widest
# first, or the outermost surface is drawn last and hides everything inside it
if DRAW_ORDER == "amplitude":
    amp = [np.nanmax(g[2]) - np.nanmin(g[2]) for g in grids]
    order = np.argsort(amp)[::-1]
else:
    order = np.argsort([np.nanmean(g[2]) for g in grids])

if EXPLODE:
    grids = [(X, Y, Z + i * EXPLODE) for i, (X, Y, Z) in enumerate(grids)]
    print(f"  ! EXPLODE = {EXPLODE:g}: z axis is offset, not true force")

drawn_artists = []

for idx in order:
    T, (X, Y, Z) = present[idx], grids[idx]
    c = colors[idx]
    if STYLE == "ribs":
        # cstride=0 omits the column lines, leaving clean parallel curves
        drawn_artists.append(
            ax.plot_wireframe(X, Y, Z, color=c, linewidth=RIB_LINEWIDTH,
                              rstride=RIB_STRIDE, cstride=0))
    elif STYLE == "wireframe":
        drawn_artists.append(
            ax.plot_wireframe(X, Y, Z, color=c, linewidth=0.9,
                              rstride=WIRE_STRIDE, cstride=WIRE_STRIDE))
    else:
        surf = ax.plot_surface(X, Y, Z, color=c, alpha=alpha,
                               linewidth=EDGE_LINEWIDTH,
                               edgecolors=to_rgba(c, 0.55),
                               shade=False, antialiased=True,
                               rstride=1, cstride=1)
        surf._sort_zpos = float(np.nanmean(Z))
        drawn_artists.append(surf)

zmin = min(np.nanmin(g[2]) for g in grids)
zmax = max(np.nanmax(g[2]) for g in grids)
zmin, zmax = min(zmin, 0.0), max(zmax, 0.0)
pad = 0.06 * (zmax - zmin) if zmax > zmin else 1.0
ax.set_zlim(zmin - pad, zmax + pad)

X0, Y0 = grids[0][0], grids[0][1]

if ZERO_PLANE:
    px = np.array([[X0.min(), X0.max()], [X0.min(), X0.max()]])
    py = np.array([[Y0.min(), Y0.min()], [Y0.max(), Y0.max()]])
    plane = ax.plot_surface(px, py, np.zeros((2, 2)), color=ZERO_PLANE_COLOR,
                            alpha=ZERO_PLANE_ALPHA, shade=False,
                            linewidth=0, antialiased=False)
    plane._sort_zpos = 0.0
    drawn_artists.append(plane)

if ZERO_CONTOURS:
    for idx in order:
        X, Y, Z = grids[idx]
        if np.nanmin(Z) < 0 < np.nanmax(Z):
            try:
                ax.contour(X, Y, Z, levels=[0.0], zdir="z", offset=zmin - pad,
                           colors=[colors[idx]], linewidths=1.6)
            except Exception:
                pass

if LABEL_SURFACES:
    # Every label sits at the ONE (q2,q3) cell where the family fans out most,
    # so the labels separate in z instead of stacking. They are projected to 2D
    # and drawn as annotations: mplot3d depth-sorts Text3D like any other
    # artist, so a plain ax.text ends up hidden behind the surfaces.
    from mpl_toolkits.mplot3d import proj3d

    stack = np.stack([g[2] for g in grids])
    spread = np.nanstd(stack, axis=0)
    spread[~np.isfinite(stack).all(axis=0)] = -np.inf
    iy, ix = np.unravel_index(np.argmax(spread), spread.shape)
    lx, ly = grids[0][0][iy, ix], grids[0][1][iy, ix]

    fig.canvas.draw()          # projection matrix must exist before transforming
    for idx in order:
        z = grids[idx][2][iy, ix]
        if not np.isfinite(z):
            continue
        xp, yp, _ = proj3d.proj_transform(lx, ly, z, ax.get_proj())
        ax.annotate(f"{present[idx]:g}", xy=(xp, yp), xycoords="data",
                    xytext=(14, 0), textcoords="offset points",
                    color=colors[idx], fontsize=FONT_SIZE, fontweight="bold",
                    va="center", ha="left", zorder=100,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=colors[idx], lw=0.6, alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=colors[idx],
                                    lw=0.7, shrinkA=0, shrinkB=1))

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, ticks=present, shrink=0.62, aspect=22, pad=0.11)
cb.set_label(r"$\tau_{\alpha_1}$", fontsize=FONT_SIZE)
cb.ax.tick_params(labelsize=FONT_SIZE)

if SHOW_LEGEND:
    handles = [Patch(facecolor=colors[i], alpha=min(alpha * 2.5, 1.0),
                     edgecolor=colors[i], label=f"{present[i]:g} Nm")
               for i in range(len(present))]
    if ZERO_PLANE:
        handles.append(Patch(facecolor=ZERO_PLANE_COLOR,
                             alpha=ZERO_PLANE_ALPHA, label="$z = 0$"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.10, 0.94),
              fontsize=FONT_SIZE, title_fontsize=FONT_SIZE,
              framealpha=0.9, title="torque")

# Put the z axis (ticks and label) on the chosen vertical edge of the box.
try:
    ax.zaxis.set_ticks_position(Z_AXIS_SIDE)
    ax.zaxis.set_label_position(Z_AXIS_SIDE)
except AttributeError:
    print("  ! matplotlib < 3.8: Z_AXIS_SIDE ignored, z axis stays on default edge")

ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
ax.set_xlabel(f"${Q_X}$ [{ANGLE_UNITS}]", labelpad=10, fontsize=FONT_SIZE)
ax.set_ylabel(f"${Q_Y}$ [{ANGLE_UNITS}]", labelpad=10, fontsize=FONT_SIZE)
zlab = "$\\min(F_{N1}, F_{N2}, F_{N3})$"
zlab += " $/\\, M_{mot}$  [N/Nm]" if NORMALIZE else "  [N]"
if EXPLODE:
    zlab += "  (offset)"
ax.set_zlabel(zlab, labelpad=12, fontsize=FONT_SIZE)
ax.set_title(f"$\\lambda_1=\\lambda_2=\\lambda_3$ = {LAMBDA:g},   "
             f"${Q_FIXED_COL}$ = {Q_FIXED_VALUE:g}", fontsize=FONT_SIZE, pad=18)
ax.view_init(elev=ELEV, azim=AZIM)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
stem = os.path.splitext(OUT)[0]
first = os.path.splitext(OUT)[1].lstrip(".") or "png"
exts = list(dict.fromkeys([first] + [e.lower().lstrip(".") for e in SAVE_FORMATS]))
ps_exts = {"eps", "ps"}

# Write the non-PostScript formats first, at full fidelity.
for ext in [e for e in exts if e not in ps_exts]:
    path = f"{stem}.{ext}"
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"wrote {path}")
    except Exception as e:
        print(f"!! could not write {path}: {e}")

# Then adapt the figure for PostScript, which has no transparency at all.
if any(e in ps_exts for e in exts):
    if RASTERIZE_SURFACES:
        for a in drawn_artists:
            a.set_rasterized(True)
    # remaining translucent bits are 2d overlays; make them opaque so the
    # backend has nothing to warn about and they stay crisp vector art
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_alpha(1.0)
        for h in leg.get_patches():
            h.set_alpha(1.0)
    for child in ax.texts:
        bb = child.get_bbox_patch()
        if bb is not None:
            bb.set_alpha(1.0)

    for ext in [e for e in exts if e in ps_exts]:
        path = f"{stem}.{ext}"
        try:
            # the 3d axis panes carry alpha by default, so the backend always
            # warns; it is expected and only affects those panes
            ps_log = logging.getLogger("matplotlib.backends.backend_ps")
            prev = ps_log.level
            ps_log.setLevel(logging.ERROR)
            try:
                fig.savefig(path, dpi=DPI, bbox_inches="tight")
            finally:
                ps_log.setLevel(prev)
            print(f"wrote {path}"
                  + ("  (surfaces rasterized, text/axes vector)"
                     if RASTERIZE_SURFACES else "  (fully vector, alpha lost)"))
        except Exception as e:
            print(f"!! could not write {path}: {e}")

if SHOW:
    plt.show()
plt.close(fig)