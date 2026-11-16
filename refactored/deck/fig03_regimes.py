"""Slide 3 · FIG 3: atmospheric regimes, 100-700 km curved sector (schematic).

Altitude is linear along the radius. Band edges are where Kn = lambda_mfp / H crosses 0.1 and 10
for an isothermal column with H = 40 km and Kn = 1 at the 230 km exobase:
z = 230 +/- H ln 10 = 138 and 322 km. The sector's curvature and the dot density (which thins with
a 150 km e-folding so the upper bands are not empty) are illustrative, not to scale.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from _deck import ds, save

Z0, Z1 = 100.0, 700.0
H_KM = 40.0
EDGES = {"collisional": (Z0, ds.EXOBASE_KM - H_KM * np.log(10)),
         "transition": (ds.EXOBASE_KM - H_KM * np.log(10), ds.EXOBASE_KM + H_KM * np.log(10)),
         "exosphere": (ds.EXOBASE_KM + H_KM * np.log(10), Z1)}
KN_TEXT = {"collisional": r"$\mathrm{Kn} < 0.1$", "transition": r"$0.1 < \mathrm{Kn} < 10$",
           "exosphere": r"$\mathrm{Kn} > 10$"}
LABEL_Z = {"collisional": 119.0, "transition": 338.0, "exosphere": 520.0}

# sector geometry in inches on the 7.22 x 4.17 canvas
CX, CY = 3.12, -2.10          # centre of curvature
RHO0, THICK = 3.20, 2.75      # radius at 100 km, radial thickness of 100-700 km
ALPHA = np.deg2rad(22.0)
N_DOTS, DOT_EFOLD_KM, SEED = 2600, 150.0, 2026


def rho(z):
    return RHO0 + (np.asarray(z, float) - Z0) / (Z1 - Z0) * THICK


def xy(z, a):
    return CX + rho(z) * np.sin(a), CY + rho(z) * np.cos(a)


def arc(z, a0=-ALPHA, a1=ALPHA, n=120):
    return xy(z, np.linspace(a0, a1, n))


def main():
    fig, ax = ds.figure("evidence")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    w, h = ds.SIZES["evidence"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    for name, (za, zb) in EDGES.items():
        xa, ya = arc(za)
        xb, yb = arc(zb)
        ax.add_patch(Polygon(np.column_stack([np.r_[xa, xb[::-1]], np.r_[ya, yb[::-1]]]),
                             closed=True, fc=ds.REGIME[name], alpha=0.15, lw=0, zorder=0))

    # molecules: countable marks that thin with altitude
    rng = np.random.default_rng(SEED)
    z = Z0 + rng.exponential(DOT_EFOLD_KM, 4 * N_DOTS)
    z = z[z < Z1 - 4][:N_DOTS]
    a = rng.uniform(-ALPHA * 0.985, ALPHA * 0.985, z.size)
    ax.scatter(*xy(z, a), s=1.7, color=ds.MIDNIGHT, alpha=0.8, lw=0, zorder=2)

    hair = dict(color=ds.GRANITE, lw=0.8, zorder=3)
    ax.plot(*arc(Z0), **hair)                                   # lower boundary
    ax.plot(*arc(Z1), color=ds.GRANITE, lw=0.5, zorder=3)       # domain top (open)
    for s in (-1, 1):
        ax.plot(*xy(np.array([Z0, Z1]), s * ALPHA), color=ds.GRANITE, lw=0.5, zorder=3)

    # exobase
    ax.plot(*arc(ds.EXOBASE_KM), color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5, zorder=4)

    # altitude scale, left
    for zt in (Z0, ds.EXOBASE_KM, Z1):
        x, y = xy(zt, -ALPHA)
        ax.text(x - 0.10, y, f"{zt:.0f} km", ha="right", va="center", fontsize=12)

    # regime names, right
    for name, zt in LABEL_Z.items():
        x, y = xy(zt, ALPHA)
        ds.label(ax, name, (x + 0.16, y + 0.11), ha="left", va="center")
        ax.text(x + 0.16, y - 0.12, KN_TEXT[name], ha="left", va="center", fontsize=12)
    x, y = xy(ds.EXOBASE_KM, ALPHA)
    ax.text(x + 0.16, y, "EXOBASE", ha="left", va="center", fontsize=11, fontweight="semibold")

    # free escape through the top
    for ang in np.linspace(-0.72, 0.72, 5) * ALPHA:
        (x0, x1), (y0, y1) = xy(np.array([655.0, 748.0]), ang)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ds.GRANITE, lw=1.0, mutation_scale=9,
                                    shrinkA=0, shrinkB=0), zorder=5)
    x, y = xy(Z1, ALPHA)
    ds.label(ax, "free escape", (x + 0.16, y + 0.10), ha="left", va="center")

    # wave forcing at the lower boundary
    for ang in np.linspace(-0.62, 0.62, 9) * ALPHA:
        x0, y0 = CX + (RHO0 - 0.36) * np.sin(ang), CY + (RHO0 - 0.36) * np.cos(ang)
        x1, y1 = CX + (RHO0 - 0.04) * np.sin(ang), CY + (RHO0 - 0.04) * np.cos(ang)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ds.ROLE["forcing"], lw=1.6,
                                    mutation_scale=10, shrinkA=0, shrinkB=0), zorder=5)
    y_lab = CY + (RHO0 - 0.36) * np.cos(0.62 * ALPHA) - 0.10
    ds.label(ax, "wave input", (CX, y_lab), ha="center", va="top")
    ax.text(CX, y_lab - 0.24, r"imposed flux $\mathit{or}$ fluid-model stream", ha="center", va="top",
            fontsize=12)

    save(fig, "fig03_regime_schematic", "schematic: Kn = 0.1, 10 for H = 40 km, Kn = 1 at 230 km")
    plt.close(fig)


if __name__ == "__main__":
    main()
