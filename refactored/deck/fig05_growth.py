"""Slide 5 · FIG 5a amplitude growth (1D spherical vs 1D Cartesian) and FIG 5b slab-vs-shell.

FIG 5a: G(z) = A(z)/A(150 km), the per-cycle peak density perturbation relative to the base cell.
  1D spherical: last forcing cycle of run 1D_5p (the Icarus 2026 run), from the radial_1d cache.
  1D Cartesian: Tian et al. (2023), digitised table dsmclib.constants.TIAN_GROW.
  These are the curves behind the slide's x4.8 vs x5.8 near 300 km.

  A single cycle's peak carries cell-to-cell noise (cycle 5 steps back from 4.50 at 285 km to 4.46
  at 294 km; the five cycles scatter by ~15% there). The spherical line is therefore a 3-cell
  (+/- 9 km) geometric running mean, end cells raw, and the raw cycle-5 cell values are drawn as dots.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from _deck import ds, save
from dsmclib import constants, growth, registry

Z_MARK = 300.0  # km, where the slide quotes the growth factors
TIAN_BIN_KM = 8.0


def load_curves():
    data = growth.cached_growth("1D_5p")
    res = data["growth_results"][0]
    z_sph = np.asarray(res["radial_distances"][res["cell_min_idx"]: res["cell_max_idx"] + 1])
    g_sph = np.asarray(res["growth_peaks"])[:, -1]            # last (5th) forcing cycle

    # The Tian table is a dense hand digitisation (102 points, repeated altitudes, scatter of a few %
    # in G). Median it in TIAN_BIN_KM altitude bins so the dashed line is single-valued.
    z_raw = np.asarray(constants.TIAN_ALTITUDES, float)
    g_raw = np.asarray(constants.TIAN_GROW, float)
    edges = np.arange(150.0, z_raw.max() + TIAN_BIN_KM, TIAN_BIN_KM)
    which = np.digitize(z_raw, edges)
    bins = np.unique(which)
    z_t = np.array([z_raw[which == b].mean() for b in bins])
    g_t = np.array([np.median(g_raw[which == b]) for b in bins])
    path = growth.cache_path(registry.run("1D_5p").campaign, "1D_5p")
    return (z_sph, g_sph), (z_t, g_t), path


def running_geomean(g, half=1):
    """Centred (2*half+1)-cell geometric mean; the end cells are left raw."""
    lg = np.log(g)
    out = lg.copy()
    for i in range(half, g.size - half):
        out[i] = lg[i - half: i + half + 1].mean()
    return np.exp(out)


def fig5a():
    (z_s, g_raw), (z_t, g_t), path = load_curves()
    g_s = running_geomean(g_raw)
    gs300, gt300 = np.interp(Z_MARK, z_s, g_s), np.interp(Z_MARK, z_t, g_t)
    print(f"G at {Z_MARK:.0f} km: spherical {gs300:.2f} (raw cell interp "
          f"{np.interp(Z_MARK, z_s, g_raw):.2f}), Cartesian {gt300:.2f}, "
          f"deficit {100 * (1 - gs300 / gt300):.0f}%")

    fig, ax = ds.figure("evidence")
    ax.plot(g_t, z_t, color=ds.ROLE["prior_work"], ls=ds.LINESTYLE["prior_work"], lw=1.5)
    ax.plot(g_s, z_s, color=ds.ROLE["this_work"], lw=2.4)
    ax.plot(g_raw, z_s, "o", ms=4.2, mew=0, color=ds.ROLE["this_work"], zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.94, 7.0)
    ax.set_ylim(150, 325)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5, 6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax.set_yticks([150, 200, 250, 300])
    ds.exobase(ax)

    ds.label(ax, "1D spherical\nthis work", (1.72, 262), color=ds.ROLE["this_work"],
             ha="right", va="center", linespacing=1.0)
    ds.label(ax, "1D Cartesian\nTian et al. 2023", (3.05, 196), color=ds.ROLE["prior_work"],
             ha="left", va="center", linespacing=1.0)
    ds.takeaway(ax, "~20% less growth at 300 km", xy=(gs300 * 0.985, Z_MARK),
                xytext=(1.03, Z_MARK), ha="left", va="center")
    ax.plot([gs300, gt300], [Z_MARK, Z_MARK], color=ds.ROLE["uncertainty"], lw=0.8, zorder=1)

    ax.set_xlabel(r"Amplitude growth, $G = A(z)/A_{150}$")
    ax.set_ylabel("Altitude (km)")
    save(fig, "fig05a_amplitude_growth", path, "dsmclib.constants.TIAN_GROW (Tian et al. 2023)")
    plt.close(fig)


def fig5b():
    """Pictogram: the same wave fills a constant area in a slab, a growing area in a shell."""
    fig, ax = ds.figure("inset")
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    wall = dict(color=ds.ROLE["this_work"], lw=2.0, solid_capstyle="round")
    levels = np.linspace(1.9, 6.6, 5)

    # slab: parallel walls, fronts of equal width
    for x in (1.25, 3.45):
        ax.plot([x, x], [1.5, 7.0], **wall)
    for y in levels:
        ax.plot([1.6, 3.1], [y, y], color=ds.ROLE["forcing"], lw=2.0, solid_capstyle="round")

    # shell: diverging walls about a centre below the frame, fronts widen as r grows
    cx, cy, half = 7.35, -3.2, np.deg2rad(13.5)
    r0, r1 = 1.5 - cy, 7.0 - cy
    for sgn in (-1, 1):
        ax.plot([cx + sgn * r0 * np.sin(half), cx + sgn * r1 * np.sin(half)],
                [cy + r0 * np.cos(half), cy + r1 * np.cos(half)], **wall)
    th = np.linspace(-half * 0.8, half * 0.8, 40)
    for y in levels:
        r = y - cy
        ax.plot(cx + r * np.sin(th), cy + r * np.cos(th), color=ds.ROLE["forcing"],
                lw=2.0 * (levels[0] - cy) / r, solid_capstyle="round")   # thins as area grows

    ds.label(ax, "slab", (2.35, 0.25), ha="center", va="bottom")
    ds.label(ax, "shell", (cx, 0.25), ha="center", va="bottom")
    save(fig, "fig05b_slab_vs_shell", "schematic (no data)")
    plt.close(fig)


if __name__ == "__main__":
    fig5a()
    fig5b()
