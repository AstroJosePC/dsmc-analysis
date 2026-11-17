"""Slide 1 · title artwork: the rarefied curved wedge, for the Midnight Blue title field.

Procedural and seeded (no simulation data). A 100-700 km sector: hairline shells and spokes,
Cultured dots thinning with altitude, Tangelo dots gathered on wave fronts that rise from the
forced patch and spread sideways, and the Gold Foil exobase at 230 km.

The figure is transparent so it sits on the slide's Midnight field; a second PNG with the field
painted in is written for previews. Not one of the deckstyle slots: 7.6 x 6.4 in, the right
field of the title slide. The title text is set in Keynote, not here.
"""
import matplotlib.pyplot as plt
import numpy as np

from _deck import MONO, OUT, ds, save

SIZE_IN = ds.SIZES["title"]
Z0, Z1 = 100.0, 700.0
SEED = 20261116

# sector geometry in inches
CX, CY = 4.02, -9.02              # centre of curvature
RHO0, THICK = 9.70, 5.36          # radius at 100 km, radial thickness of 100-700 km
ALPHA = np.deg2rad(13.4)
PATCH = 0.42 * ALPHA              # half-width of the forced patch

N_GAS, GAS_EFOLD_KM = 9000, 135.0
FRONTS = [  # altitude (km), dots, thickness (km), lateral half-width (fraction of ALPHA)
    (138.0, 500, 15.0, 0.44),
    (205.0, 470, 16.0, 0.56),
    (285.0, 330, 21.0, 0.68),
    (378.0, 220, 27.0, 0.79),
    (482.0, 140, 33.0, 0.88),
    (596.0, 85, 38.0, 0.95),
]
FRONT_DROOP_KM = 46.0             # how far a front's edges trail its centre


def rho(z):
    return RHO0 + (np.asarray(z, float) - Z0) / (Z1 - Z0) * THICK


def xy(z, a):
    return CX + rho(z) * np.sin(a), CY + rho(z) * np.cos(a)


def main():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=SIZE_IN)
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, SIZE_IN[0])
    ax.set_ylim(0, SIZE_IN[1])
    aa = np.linspace(-ALPHA, ALPHA, 200)
    ink = ds.CULTURED

    # scaffold: shells every 100 km, spokes, a firmer outline
    for z in np.arange(Z0, Z1 + 1, 100.0):
        edge = z in (Z0, Z1)
        ax.plot(*xy(z, aa), color=ink, lw=0.6 if edge else 0.35, alpha=0.55 if edge else 0.22)
    for a in np.linspace(-ALPHA, ALPHA, 13):
        edge = abs(abs(a) - ALPHA) < 1e-9
        ax.plot(*xy(np.array([Z0, Z1]), a), color=ink, lw=0.6 if edge else 0.35, alpha=0.55 if edge else 0.16)

    # the gas: samples that thin with altitude
    z = Z0 + rng.exponential(GAS_EFOLD_KM, 3 * N_GAS)
    z = z[z < Z1 - 3][:N_GAS]
    a = rng.uniform(-0.992, 0.992, z.size) * ALPHA
    ax.scatter(*xy(z, a), s=1.5, color=ink, alpha=0.92, lw=0, rasterized=True, zorder=2)

    # wave fronts: arcs of dots that widen, thicken and thin out as they climb
    for z_c, n, thick, half in FRONTS:
        u = np.clip(rng.normal(0, 0.46, n), -1, 1)
        af = u * half * ALPHA
        zf = z_c - FRONT_DROOP_KM * u ** 2 + rng.normal(0, thick / 2.2, n)
        keep = (zf > Z0 + 4) & (zf < Z1 - 4)
        ax.scatter(*xy(zf[keep], af[keep]), s=2.2, color=ds.ROLE["forcing"], alpha=0.95, lw=0,
                   rasterized=True, zorder=3)

    # exobase
    ax.plot(*xy(ds.EXOBASE_KM, aa), color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.1, zorder=4)

    # forcing ticks under the lower boundary
    for ang in np.linspace(-PATCH, PATCH, 17):
        x0, y0 = CX + (RHO0 - 0.20) * np.sin(ang), CY + (RHO0 - 0.20) * np.cos(ang)
        x1, y1 = CX + (RHO0 - 0.07) * np.sin(ang), CY + (RHO0 - 0.07) * np.cos(ang)
        ax.plot([x0, x1], [y0, y1], color=ds.ROLE["forcing"], lw=0.9, solid_capstyle="butt")

    # clinical labels
    mono = dict(fontfamily=MONO, fontsize=6.5)
    for zt in (100, 400, 700):
        x, y = xy(zt, -ALPHA)
        ax.text(x - 0.10, y, f"{zt}", ha="right", va="center", color=ink, alpha=0.6, **mono)
    x, y = xy(ds.EXOBASE_KM, -ALPHA)
    ax.text(x - 0.10, y, f"{ds.EXOBASE_KM:.0f}", ha="right", va="center", color=ds.ROLE["exobase"], **mono)
    x, y = xy(ds.EXOBASE_KM, ALPHA)
    ax.text(x + 0.10, y, "E X O B A S E", ha="left", va="center", color=ds.ROLE["exobase"], **mono)
    x, y = xy(Z0, -ALPHA)
    ax.text(x - 0.10, y - 0.15, "KM", ha="right", va="center", color=ink, alpha=0.45,
            fontfamily=MONO, fontsize=5.5)
    ax.text(CX, CY + (RHO0 - 0.20) * np.cos(PATCH) - 0.10, "F O R C E D   F L U X   ·   L O W E R   B O U N D A R Y",
            ha="center", va="top", color=ds.ROLE["forcing"], **mono)

    save(fig, "fig01_title_wedge", f"procedural, seed {SEED}")
    fig.savefig(OUT / "fig01_title_wedge_deck_on_midnight.png", dpi=300, facecolor=ds.MIDNIGHT,
                transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
