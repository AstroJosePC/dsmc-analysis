"""Slide 6 alternatives (6A, 6B): an enlarged S6 wedge and full-size quantitative panels.

Wedge (fig06a_*), redrawn so the 2D structure carries weight on the slide:
  halfwedge   6A  the coherent part of the wave: Re[Z_sym exp(-i theta)] from the lock-in cache
                  (delta n/n at the forcing frequency over cycles 4-5, folded about the patch
                  centre, the field L_att is fitted to), theta matched to the raw output below.
                  Patch centre to 800 km past the edge, 100-500 km, altitude exaggerated x2.5.
                  Without the particle noise the sideways leakage is visible up to 500 km.
  wedge_band  6B  the raw output of fig06a_s6_wedge.py (one cycle-5 output an eighth of a period
                  after the exobase trough; azimean_preforcing background), both patch edges and
                  700 km beyond each, 100-450 km, altitude exaggerated x1.5
Curvature stays true: shells are concentric arcs about the planet centre, and only the radial
distance above the 100 km boundary is stretched (r = R_base + k (z - z_base)). Horizontal
distances are true at the lower boundary, where the scale bar sits. The view is rotated so the
drawn sector is symmetric about the vertical.

Panels (cached S6 analysis, no recomputation):
  fig06b_latt_fit_*     |Z| at the forcing frequency vs arc distance s past the patch edge, exobase
                        cell (231.4 km), with the exponential fit and the 3-sigma noise floor
                        (lockin_attenuation_S6_azimean_preforcing.npz, dsmclib.attenuation fit).
  fig06c_lambdah_fit_*  crest-arrival delay vs column separation at 231 km with the ODR line
                        (phase_lag_S6_mad_column0.npz). The slope triangle spans one period.
  fig06d_latt_profile_row  L_att(z) with the bootstrap + cycle uncertainty
                        (preforcing_transport_S6_azimean_preforcing.npz), fits with >= 6 points
                        and R^2 >= 0.9.

    conda run -n dsmc python deck/fig06_alternatives.py [wedge] [panels]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from _deck import R_MARS_KM, SVG_ONLY, ds, r_label, save
from dsmclib import attenuation as att
from dsmclib.paths import output_dir
from slots import size_in

ANALYSIS = output_dir("s1_s6_azimuthal", "analysis")
LATT_CACHE = ANALYSIS / "lockin_attenuation_S6_azimean_preforcing.npz"
LAG_CACHE = ANALYSIS / "phase_lag_S6_mad_column0.npz"
TRANSPORT_CACHE = ANALYSIS / "preforcing_transport_S6_azimean_preforcing.npz"
MAX_FIT_KM = 400.0          # attenuation_table default (--max-dist)
VMAX = 0.4
Z_TOP_KM = 450.0            # above this one output is particle noise (fig06_s6_wedge.py)
SCALE_BAR_KM = 450.0


def figure(stem):
    fig = plt.figure(figsize=size_in(stem, ds.SIZES))
    return fig


# ================================================================= wedge
class Wedge:
    """Exaggerated-radius sector about the patch centre; a = azimuth from the centre (rad)."""

    def __init__(self, ax, r_base, z_base, k, rot):
        self.ax, self.rb, self.zb, self.k, self.rot = ax, r_base, z_base, k, rot

    def xy(self, z, a):
        r = self.rb + self.k * (np.asarray(z, float) - self.zb)
        a = np.asarray(a, float) - self.rot
        return r * np.sin(a), r * np.cos(a) - self.rb

    def arc(self, z, a0, a1, n=240, **kw):
        return self.ax.plot(*self.xy(z, np.linspace(a0, a1, n)), **kw)

    def radial(self, a, z0, z1, **kw):
        return self.ax.plot(*self.xy([z0, z1], [a, a]), **kw)


def snapshot_field(a_lo_km, a_hi_km):
    """One raw output (fig06_s6_wedge.load_snapshot), columns within the view."""
    from fig06_s6_wedge import load_snapshot

    dn, alt_e, phi_e, cols, idx, t_snap, data_dir = load_snapshot()
    r_base = R_MARS_KM + alt_e[0]
    phi0 = 0.5 * (phi_e[cols.min()] + phi_e[cols.max() + 1])
    half_patch = 0.5 * (phi_e[cols.max() + 1] - phi_e[cols.min()])
    centres = 0.5 * (phi_e[:-1] + phi_e[1:]) - phi0
    jj = np.where((centres >= a_lo_km / r_base) & (centres <= a_hi_km / r_base))[0]
    ii = np.where(alt_e[1:] <= Z_TOP_KM)[0]
    field = dn[np.ix_(ii, jj)]
    return dict(field=field, a_edges=phi_e[jj[0]: jj[-1] + 2] - phi0, z_edges=alt_e[: ii[-1] + 2],
                r_base=r_base, half_patch=half_patch,
                cbar=r"Density perturbation, $\delta n/n$",
                source=(data_dir / "macro_Number Density.npy",
                        f"output {idx} (t = {t_snap:.1f} s), background azimean_preforcing"),
                note=f"output {idx}")


def lockin_field(z_top_km):
    """Coherent delta n/n at the forcing frequency, Re[Z_sym exp(-i theta)], folded about the patch
    centre (lockin_attenuation_S6_azimean_preforcing.npz: cycles 4-5). theta is the least-squares
    match to the raw output drawn by fig06_s6_wedge.py, so it is the same moment, without the
    incoherent particle noise."""
    from fig06_s6_wedge import load_snapshot

    red = np.load(LATT_CACHE)
    z, kk, Z = red["radial_cells"], red["k"], red["Z_sym"]
    dphi, n_wedge = float(red["dphi"]), int(red["n_wedge"])
    half_patch = 0.5 * n_wedge * dphi
    a_centres = half_patch + (kk + 0.5) * dphi           # column k = 0 is the first outside the patch
    dn, alt_e, phi_e, cols, idx, t_snap, data_dir = load_snapshot()
    assert np.allclose(alt_e[:-1] + np.diff(alt_e) / 2, z, atol=1.0), "radial grids differ"
    j_c = int(red["j_right"]) + 1 + kk                  # right-side columns matching each k
    ii = np.where(alt_e[1:] <= z_top_km)[0]
    raw, Zv = dn[np.ix_(ii, j_c)], Z[ii]
    good = np.abs(Zv) > 0
    c, s = np.linalg.lstsq(np.column_stack([Zv.real[good], Zv.imag[good]]), raw[good], rcond=None)[0]
    theta = np.arctan2(s, c)
    field = (Zv * np.exp(-1j * theta)).real
    a_edges = np.r_[a_centres - dphi / 2, a_centres[-1] + dphi / 2]
    print(f"lock-in phase matched to output {idx}: theta = {np.rad2deg(theta):.0f} deg, "
          f"right columns {j_c[0]}..{j_c[-1]}")
    return dict(field=field, a_edges=a_edges, z_edges=alt_e[: ii[-1] + 2],
                r_base=R_MARS_KM + alt_e[0], half_patch=half_patch,
                cbar=r"$\delta n/n$ at the forcing frequency (cycles 4–5)",
                source=(LATT_CACHE, data_dir / "macro_Number Density.npy",
                        f"Re[Z_sym exp(-i theta)], theta = {np.rad2deg(theta):.1f} deg matched to output {idx}"),
                note=f"lock-in, theta {np.rad2deg(theta):.0f} deg")


def wedge_figure(stem, f, k, *, margin_in, colorbar_at, scale_at, labels, centre=False):
    """Draw one wedge variant from a field dict (snapshot_field / lockin_field).
    margin_in: (left, right, bottom, top) free space in inches; centre: centre the sector
    horizontally instead of honouring left/right."""
    a_edges, z_edges, r_base, half_patch = f["a_edges"], f["z_edges"], f["r_base"], f["half_patch"]
    a_lo, a_hi, z_top = a_edges[0], a_edges[-1], z_edges[-1]

    fig = figure(stem)
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_aspect("equal")
    wd = Wedge(ax, r_base, z_edges[0], k, 0.5 * (a_lo + a_hi))

    A, Z = np.meshgrid(a_edges, z_edges)
    X, Y = wd.xy(Z, A)
    mesh = ax.pcolormesh(X, Y, f["field"], cmap="rarefied_div", vmin=-VMAX, vmax=VMAX,
                         rasterized=True, shading="flat")

    # frame the view: equal km per inch
    w_in, h_in = size_in(stem, ds.SIZES)
    xs, ys = X.ravel(), Y.ravel()
    left, right, bottom, top = margin_in
    kpi = max((xs.max() - xs.min()) / (w_in - left - right), (ys.max() - ys.min()) / (h_in - bottom - top))
    x_min = 0.5 * (xs.min() + xs.max()) - 0.5 * w_in * kpi if centre else xs.min() - left * kpi
    ax.set_xlim(x_min, x_min + w_in * kpi)
    ax.set_ylim(ys.max() + top * kpi - h_in * kpi, ys.max() + top * kpi)
    pt = kpi / 72.0                                         # km per point

    wd.arc(z_edges[0], a_lo, a_hi, color=ds.GRANITE, lw=0.8)        # lower boundary
    for a in (a_lo, a_hi):
        wd.radial(a, z_edges[0], z_top, color=ds.GRANITE, lw=0.5)
    for a in (-half_patch, half_patch):                             # patch edges
        if a_lo < a < a_hi:
            wd.radial(a, z_edges[0], z_top, color=ds.ROLE["forcing"], lw=1.1, ls=(0, (1.2, 2.2)))
    wd.arc(ds.EXOBASE_KM, a_lo, a_hi, color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5)

    # altitude labels on the left cut; the exobase is named there, clear of the field
    alts = labels.get("alts", (z_edges[0], ds.EXOBASE_KM, 300.0, z_top))
    for z in alts:
        x, y = wd.xy(z, a_lo)
        t = r_label(round(z, -1), bound=z in (min(alts), max(alts)))
        ax.text(x - 8 * pt, y, t, ha="right", va="center", fontsize=13)
        if z == ds.EXOBASE_KM:
            ax.text(x - 8 * pt, y - 13 * pt, "EXOBASE", ha="right", va="center", fontsize=11,
                    fontweight="semibold")
    xk, yk = labels["exag_at"]
    ax.text(ax.get_xlim()[0] + xk * kpi, ax.get_ylim()[0] + yk * kpi, f"altitude ×{k:g}",
            ha="left", va="center", fontsize=12, style="italic")

    # forced patch bracket under the lower boundary, label under its lowest point
    z_br = z_edges[0] - 12 * pt / k
    b0, b1 = max(-half_patch, a_lo), min(half_patch, a_hi)
    bx, by = wd.xy(z_br, np.linspace(b0, b1, 200))
    ax.plot(bx, by, color=ds.ROLE["forcing"], lw=2.4, solid_capstyle="butt")
    for a in (-half_patch, half_patch):
        if a_lo <= a <= a_hi:
            wd.radial(a, z_br - 6 * pt / k, z_edges[0] - 3 * pt / k, color=ds.ROLE["forcing"], lw=2.4)
    ds.label(ax, labels["patch"], (0.5 * (bx.min() + bx.max()), by.min() - 9 * pt), ha="center", va="top")

    # s: arc distance past the right patch edge, just above the exobase
    z_s = ds.EXOBASE_KM + 18 * pt / k
    r_s = r_base + k * (z_s - z_edges[0])
    a_s = half_patch + np.array([4.0, labels.get("s_len_km", 300.0)]) / r_s
    xs_, ys_ = wd.xy(z_s, np.linspace(*a_s, 80))
    ax.plot(xs_[:-4], ys_[:-4], color=ds.MIDNIGHT, lw=1.3)
    ax.annotate("", xy=(xs_[-1], ys_[-1]), xytext=(xs_[-10], ys_[-10]),
                arrowprops=dict(arrowstyle="-|>", color=ds.MIDNIGHT, lw=1.3, shrinkA=0, shrinkB=0))
    x, y = wd.xy(z_s + 14 * pt / k, a_s[1])
    ax.text(x, y, r"$s$", color=ds.MIDNIGHT, ha="center", va="bottom", fontsize=18)

    # Delta s: a column pair below the exobase, past the left edge when it is in view
    side = -1 if a_lo < -half_patch - 330 / r_base else 1
    z_ds = (ds.EXOBASE_KM - 34 * pt / k, ds.EXOBASE_KM - 8 * pt / k)
    a_d = side * (half_patch + np.array([110.0, 330.0]) / (r_base + k * (np.mean(z_ds) - z_edges[0])))
    for a in a_d:
        wd.radial(a, *z_ds, color=ds.MIDNIGHT, lw=1.6, solid_capstyle="round")
    x, y = wd.xy(np.mean(z_ds), a_d.mean())
    ax.text(x, y, r"$\Delta s$", color=ds.MIDNIGHT, ha="center", va="center", fontsize=16,
            rotation=-np.rad2deg(a_d.mean() - wd.rot), rotation_mode="anchor")

    # colour bar and scale bar (inches from the lower-left corner)
    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    cx, cy, cw = colorbar_at
    cax = ax.inset_axes([x0 + cx * kpi, y0 + cy * kpi, cw * kpi, 0.12 * kpi], transform=ax.transData)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.ax.set_xticklabels(["−0.4", "−0.2", "0", "0.2", "0.4"])
    cb.ax.tick_params(labelsize=13)
    cax.text(0.5, 1.35, f["cbar"], transform=cax.transAxes, ha="center", va="bottom", fontsize=14)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor(ds.GRANITE)

    sx, sy = x0 + scale_at[0] * kpi, y0 + scale_at[1] * kpi
    ax.plot([sx, sx + SCALE_BAR_KM], [sy, sy], color=ds.GRANITE, lw=1.2)
    for x in (sx, sx + SCALE_BAR_KM):
        ax.plot([x, x], [sy - 5 * pt, sy + 5 * pt], color=ds.GRANITE, lw=1.2)
    ds.label(ax, f"{SCALE_BAR_KM:.0f} km at 100 km", (sx + SCALE_BAR_KM / 2, sy - 9 * pt), ha="center",
             va="top")

    print(f"{stem}: {f['note']}, k = {k}, {kpi:.0f} km/in, view {np.rad2deg(a_lo):.1f}..{np.rad2deg(a_hi):.1f} deg")
    save(fig, stem, *f["source"], f"altitude x{k:g}")
    plt.close(fig)


def wedges():
    wedge_figure("fig06a_s6_halfwedge", lockin_field(500.0), k=2.5,
                 margin_in=(1.05, 0.08, 1.05, 0.05), colorbar_at=(3.95, 0.42, 2.45),
                 scale_at=(0.35, 0.50),
                 labels={"patch": "forced patch, centre to edge", "s_len_km": 330.0,
                         "alts": (100.0, ds.EXOBASE_KM, 300.0, 400.0, 500.0), "exag_at": (0.12, 0.98)})
    wedge_figure("fig06a_s6_wedge_band", snapshot_field(-(940.0 + 700.0), 940.0 + 700.0), k=1.5,
                 margin_in=(0.0, 0.0, 0.40, 0.04), colorbar_at=(9.62, 0.30, 2.40),
                 scale_at=(0.45, 0.40), centre=True,
                 labels={"patch": "forced patch ~1,900 km", "s_len_km": 330.0,
                         "alts": (100.0, ds.EXOBASE_KM, 440.0), "exag_at": (0.40, 0.95)})


# ================================================================= panels
def panel(stem):
    fig = figure(stem)
    ax = fig.add_subplot()
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72)
    return fig, ax


def ylabel_short(ax, text):
    """Short panels: a compact (at most two-line) y label along the axis, so the plot keeps its height."""
    ax.set_ylabel(text, linespacing=1.05)


def latt_fit(stem):
    red = np.load(LATT_CACHE)
    zi = int(np.argmin(np.abs(red["radial_cells"] - ds.EXOBASE_KM)))
    s, amp, sigma = red["s_km"][zi], np.abs(red["Z_sym"][zi]), float(red["sigma_c"][zi])
    i0 = int(np.where(red["k"] == att.K0_FIT_START)[0][0])
    _, i_end = att.fit_range(amp, s, sigma, i0, att.DEFAULT_SNR_CUT, MAX_FIT_KM)
    fit = att.fit_exp_at(amp, s, sigma, i0, i_end)
    L, a0 = fit["L"], fit["a0"]
    print(f"{stem}: L_att = {L:.2f} km at {red['radial_cells'][zi]:.1f} km, n = {fit['n']}")

    fig, ax = panel(stem)
    x_lo, x_hi = -260.0, 620.0
    show = (s >= x_lo) & (s <= x_hi)
    idx = np.arange(s.size)
    in_fit = show & (idx >= i0) & (idx <= i_end)
    other = show & ~in_fit
    ax.axvspan(x_lo, 0.0, color=ds.ROLE["forcing"], alpha=0.12, lw=0, zorder=0)
    ax.axhline(3 * sigma, color=ds.GRANITE, lw=1.0, ls=ds.LINESTYLE["prior_work"], zorder=1)
    ax.text(x_hi, 3 * sigma + 0.012, r"$3\sigma$", ha="right", va="bottom", fontsize=13)
    xx = np.linspace(s[i0], s[i_end], 200)
    ax.plot(xx, a0 * np.exp(-xx / L), color=ds.ROLE["this_work"], lw=2.4, zorder=3)
    ax.plot(s[in_fit], amp[in_fit], "o", ms=5, mew=0, color=ds.ROLE["this_work"], zorder=4)
    ax.plot(s[other], amp[other], "o", ms=5, mew=0.9, mfc="none", color=ds.GRANITE, zorder=2)
    ds.label(ax, "forced", (x_lo + 14, 0.018), ha="left", va="bottom")
    y_L = a0 * np.exp(-1.0)
    ds.takeaway(ax, r"$1/e$ at $s = L_\mathrm{att}$", xy=(L, y_L), xytext=(x_hi, 0.215),
                ha="right", va="center")
    ax.plot([L], [y_L], "o", ms=7, mfc="none", mec=ds.CRIMSON, mew=1.4, zorder=5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 0.31)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"Distance past patch edge, $s$ (km)" if stem.endswith("col") else
                  r"Distance past patch, $s$ (km)")
    ylabel_short(ax, "Amplitude\n" + r"$|\delta n/n|$")
    save(fig, stem, LATT_CACHE, f"exobase cell {red['radial_cells'][zi]:.1f} km, L_att = {L:.2f} km")
    plt.close(fig)


def lambdah_fit(stem):
    d = np.load(LAG_CACHE)
    used = d["used"]
    x, y = d["distance_km"][used], d["lag_periods"][used]
    slope, b = float(d["slope_per_km"]), float(d["intercept_periods"])
    lam = float(d["lambda_h_km"])
    x0 = -b / slope                                    # where the line crosses zero delay
    print(f"{stem}: lambda_h = {lam:.2f} +/- {float(d['lambda_h_err_km']):.2f} km, n = {int(d['n'])}")

    fig, ax = panel(stem)
    xd = np.array([x0, x.max()])
    ax.plot(xd, slope * xd + b, color=ds.ROLE["this_work"], lw=2.4, zorder=3)
    xe = np.array([x.max(), x0 + lam])
    ax.plot(xe, slope * xe + b, color=ds.ROLE["this_work"], lw=1.2, ls=(0, (1, 2.2)), zorder=3)
    ax.plot(x, y, "o", ms=5, mew=0, color=ds.ROLE["this_work"], alpha=0.85, zorder=4)
    # slope triangle: one wavelength of separation = one period of delay
    tri = dict(color=ds.GRANITE, lw=1.0, zorder=2)
    ax.plot([x0 + lam, x0 + lam], [0, 1], **tri)
    ax.annotate("", xy=(x0 + lam, 0), xytext=(x0, 0),
                arrowprops=dict(arrowstyle="<|-|>", color=ds.GRANITE, lw=1.0, mutation_scale=9,
                                shrinkA=0, shrinkB=0), zorder=2)
    lab = dict(fontsize=13, color=ds.GRANITE, fontweight="semibold", ha="right")
    ax.text(x0 + lam - 10, 0.06, rf"$\lambda_h$ = {lam:.0f} km", va="bottom", **lab)
    ax.plot([-15, x0 + lam], [1, 1], color=ds.GRANITE, lw=0.8, ls=(0, (1, 2.5)), zorder=1)
    ax.set_xlim(-15, 520)
    ax.set_ylim(-0.06, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.spines["bottom"].set_bounds(0, 500)
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xlabel(r"Column separation, $\Delta s$ (km)")
    ylabel_short(ax, r"Delay $\Delta\tau$" + "\n(periods)")
    save(fig, stem, LAG_CACHE, f"ODR lambda_h = {lam:.2f} km at {float(d['altitude_km']):.1f} km")
    plt.close(fig)


def latt_profile(stem):
    t = np.load(TRANSPORT_CACHE)
    z, L, e = t["altitude_km"], t["L_att_km"], t["sigma_L_total_km"]
    ok = np.isfinite(L) & (t["L_att_n_points"] >= 6) & (t["L_att_r_squared"] >= 0.9) & (z <= 450)
    fig, ax = panel(stem)
    ax.errorbar(L[ok], z[ok], xerr=e[ok], fmt="none", ecolor=ds.ROLE["uncertainty"], elinewidth=0.8, zorder=2)
    ax.plot(L[ok], z[ok], "-o", color=ds.ROLE["this_work"], lw=1.6, ms=3.6, mew=0, zorder=3)
    ds.exobase(ax, label=False)
    ax.text(4, ds.EXOBASE_KM + 10, "EXOBASE", ha="left", va="bottom",       # left: clear of the data
            fontsize=plt.rcParams["legend.fontsize"] - 2, color=ds.GRANITE, fontweight="semibold")
    for zq in (148.3, 231.4, 315.4):
        i = int(np.argmin(np.abs(z - zq)))
        ax.plot([L[i]], [z[i]], "o", ms=7, color=ds.ROLE["this_work"], zorder=4)
        ax.text(L[i] - 9, z[i] + 6, f"{L[i]:.0f}", ha="right", va="bottom", fontsize=13,
                color=ds.MIDNIGHT, fontweight="semibold")
    ax.set_xlim(0, 200)
    ax.set_ylim(100, 450)
    ax.set_yticks([100, 200, 300, 400])
    ax.set_xlabel(r"Attenuation length, $L_\mathrm{att}$ (km)")
    ylabel_short(ax, "Altitude\n(km)")
    save(fig, stem, TRANSPORT_CACHE, "L_att_km, sigma_L_total_km; n >= 6, R2 >= 0.9, z <= 450 km")
    plt.close(fig)


def write_facts():
    """Slide 6 hero numbers for slides/build_deck.py (deck/data/slide06_facts.json)."""
    if SVG_ONLY:                     # SVG render for the pptx export: leave deck/data alone
        return
    import json
    from datetime import date
    from pathlib import Path
    from _deck import commit
    t = np.load(TRANSPORT_CACHE)
    z = t["altitude_km"]
    i = int(np.argmin(np.abs(z - ds.EXOBASE_KM)))
    lag = np.load(LAG_CACHE)
    out = {"created": date.today().isoformat(), "run": "S6",
           "sources": {"transport_cache": str(TRANSPORT_CACHE), "phase_lag_cache": str(LAG_CACHE),
                       "dsmc_analyzer_commit": commit()},
           "L_att_exobase_km": float(t["L_att_km"][i]), "sigma_L_exobase_km": float(t["sigma_L_total_km"][i]),
           "exobase_cell_km": float(z[i]),
           "lambda_h_odr_km": float(lag["lambda_h_km"]), "lambda_h_odr_err_km": float(lag["lambda_h_err_km"])}
    path = Path(__file__).resolve().parent / "data" / "slide06_facts.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{path.name}: L_att = {out['L_att_exobase_km']:.1f} +/- {out['sigma_L_exobase_km']:.1f} km")


def panels():
    write_facts()
    for suffix in ("col", "row"):
        latt_fit(f"fig06b_latt_fit_{suffix}")
        lambdah_fit(f"fig06c_lambdah_fit_{suffix}")
    latt_profile("fig06d_latt_profile_row")


if __name__ == "__main__":
    which = sys.argv[1:] or ["wedge", "panels"]
    if "panels" in which:
        panels()
    if "wedge" in which:
        wedges()
