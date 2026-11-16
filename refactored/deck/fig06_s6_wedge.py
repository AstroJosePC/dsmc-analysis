"""Slide 6 · FIG 6a: S6 density perturbation in true curved-wedge geometry (1:1 scale).

delta n / n = (n - n_bg) / n_bg with the azimean_preforcing background (mean over the last 17
pre-forcing outputs and all azimuth columns; dsmclib.background). One output is drawn: an
eighth of a period after the cycle-5 trough at the exobase (found from the mean over the middle
of the forced patch), when a compression front and a rarefaction front are both in view.
Radius x azimuth is mapped to x, y with equal km per inch in both directions.

Shown to 450 km: above that a single output is particle noise (sigma ~ 0.14 at 500 km,
~ 0.9 at 700 km for this run).
"""
import matplotlib.pyplot as plt
import numpy as np

from _deck import R_MARS_KM, ds, save, wedge_xy
from dsmclib import background as bg
from dsmclib import growth, macro, registry
from dsmclib.run_data import DSMCRun

ALIAS = "S6"
Z_TOP_KM = 450.0          # top of the drawn field
HALF_ANGLE_DEG = 22.0     # view half-width about the patch centre (patch half-width is 15.5 deg)
VMAX = 0.4                # symmetric colour limit
PHASE_AFTER_TROUGH = 1 / 8      # of a forcing period
SCALE_BAR_KM = 450.0      # = lambda_h
S_ARROW_KM = (30.0, 300.0)      # arc distance past the patch edge, at Z_S_KM
Z_S_KM = 165.0
DS_COLUMNS_KM = (110.0, 330.0)  # the column pair for the Delta-s ticks, past the other edge
Z_DS_KM = (128.0, 202.0)


def radius(alt_km):
    return R_MARS_KM + alt_km


def load_snapshot():
    run = DSMCRun(registry.run(ALIAS).path, alias=ALIAS)
    p = run.load_parameters()
    density = macro.load_field(run.data_dir, "density", run.macro, mmap_mode="r")

    end = bg.forcing_start_index(p)                    # azimean_preforcing recipe, density only
    n_bg = np.asarray(density[end - bg.PREFORCING_SNAPSHOTS:end]).mean(axis=(0, 2))

    cols = growth.forced_columns(run)
    zi = run.altitude_index_nearest(ds.EXOBASE_KM)
    i0 = int(np.searchsorted(p.time, p.wave_start_time + 4 * p.wave_period))   # cycle 5
    i1 = int(np.searchsorted(p.time, p.wave_end_time))
    mid = slice(cols.min() + len(cols) // 3, cols.max() - len(cols) // 3 + 1)
    exobase_series = np.asarray(density[i0:i1, zi, mid]).mean(axis=1)
    dt = float(p.time[1] - p.time[0])
    idx = i0 + int(exobase_series.argmin()) + int(round(PHASE_AFTER_TROUGH * p.wave_period / dt))

    dn = bg.relative_perturbation(np.asarray(density[idx])[None], n_bg)[0]
    alt_edges = p.radial_edges / 1e3 - p.mars_radius
    phi_edges = np.deg2rad(p.azi_edges)
    return dn, alt_edges, phi_edges, cols, idx, float(p.time[idx]), run.data_dir


def arc(ax, alt, a0, a1, **kw):
    a = np.linspace(a0, a1, 200)
    return ax.plot(*wedge_xy(alt, a), **kw)


def main():
    dn, alt_e, phi_e, cols, idx, t_snap, data_dir = load_snapshot()
    r_base = radius(alt_e[0])
    phi_l, phi_r = phi_e[cols.min()], phi_e[cols.max() + 1]
    phi0 = 0.5 * (phi_l + phi_r)
    half_patch = 0.5 * (phi_r - phi_l)
    half = np.deg2rad(HALF_ANGLE_DEG)
    print(f"output {idx}, t = {t_snap:.1f} s; forced footprint {2 * half_patch * r_base:,.0f} km "
          f"at {alt_e[0]:.0f} km, {2 * half_patch * radius(ds.EXOBASE_KM):,.0f} km at the exobase")

    jj = np.where((phi_e[:-1] >= phi0 - half) & (phi_e[1:] <= phi0 + half))[0]
    ii = np.where(alt_e[1:] <= Z_TOP_KM)[0]
    a_edges = phi_e[jj[0]: jj[-1] + 2] - phi0
    z_edges = alt_e[: ii[-1] + 2]
    z_top, a_max = z_edges[-1], a_edges[-1]
    A, Z = np.meshgrid(a_edges, z_edges)
    X, Y = wedge_xy(Z, A)

    fig, ax = ds.figure("evidence")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_aspect("equal")
    w, h = ds.SIZES["evidence"]
    x_cut = radius(z_top) * np.sin(a_max)                      # km
    ax.set_xlim(-x_cut - 290, x_cut + 90)                      # left margin holds the altitude labels
    span = (2 * x_cut + 380) * h / w
    content_top, content_bottom = z_top, -545.0                # wedge top to colour-bar label
    y_top = content_top + 0.5 * (span - (content_top - content_bottom))
    ax.set_ylim(y_top - span, y_top)

    mesh = ax.pcolormesh(X, Y, dn[np.ix_(ii[: len(z_edges) - 1], jj)], cmap="rarefied_div",
                         vmin=-VMAX, vmax=VMAX, rasterized=True, shading="flat")

    arc(ax, z_edges[0], -a_max, a_max, color=ds.GRANITE, lw=0.8)        # lower boundary

    # altitude scale on the left cut
    for z in (z_edges[0], ds.EXOBASE_KM, z_top):
        x, y = wedge_xy(z, -a_max)
        ax.text(x - 30, y, f"{round(z, -1):.0f} km", ha="right", va="center", fontsize=12)

    # exobase
    arc(ax, ds.EXOBASE_KM, -a_max, a_max, color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5)
    a_lab = a_max - 150.0 / radius(ds.EXOBASE_KM)
    ax.text(*wedge_xy(ds.EXOBASE_KM + 16, a_lab), "EXOBASE", ha="center", va="bottom", fontsize=11,
            fontweight="semibold", rotation=-np.rad2deg(a_lab), rotation_mode="anchor")

    # forced footprint: bracket under the lower boundary
    z_br = z_edges[0] - 45
    arc(ax, z_br, -half_patch, half_patch, color=ds.ROLE["forcing"], lw=2.4, solid_capstyle="butt")
    for a in (-half_patch, half_patch):
        ax.plot(*wedge_xy([z_br - 18, z_edges[0] - 12], [a, a]), color=ds.ROLE["forcing"], lw=2.4)
    ds.label(ax, "forced patch ~1,900 km", wedge_xy(z_br - 55, 0.0), ha="center", va="top")

    # s: arc distance past the patch edge (card 1)
    a_s = half_patch + np.array(S_ARROW_KM) / radius(Z_S_KM)
    aa = np.linspace(*a_s, 60)
    xs, ys = wedge_xy(Z_S_KM, aa)
    ax.plot(xs[:-3], ys[:-3], color=ds.MIDNIGHT, lw=1.2)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-8], ys[-8]),
                arrowprops=dict(arrowstyle="-|>", color=ds.MIDNIGHT, lw=1.2, shrinkA=0, shrinkB=0))
    ax.plot(*wedge_xy([Z_S_KM - 22, Z_S_KM + 22], [a_s[0], a_s[0]]), color=ds.MIDNIGHT, lw=1.2)
    ax.text(*wedge_xy(Z_S_KM, a_s[1] + 42.0 / radius(Z_S_KM)), r"$s$", color=ds.MIDNIGHT,
            ha="center", va="center", fontsize=17)

    # Delta s: a column pair (card 2)
    a_d = -half_patch - np.array(DS_COLUMNS_KM) / radius(np.mean(Z_DS_KM))
    for a in a_d:
        ax.plot(*wedge_xy(Z_DS_KM, [a, a]), color=ds.MIDNIGHT, lw=1.6, solid_capstyle="round")
    ax.text(*wedge_xy(np.mean(Z_DS_KM), a_d.mean()), r"$\Delta s$", color=ds.MIDNIGHT,
            ha="center", va="center", fontsize=15, rotation=-np.rad2deg(a_d.mean()),
            rotation_mode="anchor")

    # colour bar under the arch, scale bar at lower left
    cax = ax.inset_axes([-650, -300, 1300, 40], transform=ax.transData)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.set_label(r"Density perturbation, $\delta n/n$")
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor(ds.GRANITE)

    xb, yb = ax.get_xlim()[0] + 110, -330.0
    ax.plot([xb, xb + SCALE_BAR_KM], [yb, yb], color=ds.GRANITE, lw=1.2)
    for x in (xb, xb + SCALE_BAR_KM):
        ax.plot([x, x], [yb - 20, yb + 20], color=ds.GRANITE, lw=1.2)
    ds.label(ax, f"{SCALE_BAR_KM:.0f} km", (xb + SCALE_BAR_KM / 2, yb - 45), ha="center", va="top")

    save(fig, "fig06a_s6_wedge", data_dir / "macro_Number Density.npy",
         f"output {idx} (t = {t_snap:.1f} s), background azimean_preforcing")
    plt.close(fig)


if __name__ == "__main__":
    main()
