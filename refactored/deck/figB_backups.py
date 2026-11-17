"""Backup slides B1-B6 (Q&A): FIG B1-B6, drawn from the caches written by deck/cache_backups.py.

    cd refactored && conda run -n dsmc python deck/figB_backups.py [B1 B2 ...]

B1 run table (strip slot); B2-B6 one evidence-field figure each. Sizes are existing deckstyle slots, so
no style token changes. Curved-geometry profiles (B2, B3: the wheel sector) label altitude with the
radial variable r; flat 1-D and fluid profiles (B4, B5) read "Altitude (km)", as on slides 5 and 8.
"""
import json
import sys

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from _deck import REFACTORED, ds, save

DATA = REFACTORED / "deck" / "data"
F = json.loads((DATA / "backup_facts.json").read_text())
FACTS = "deck/data/backup_facts.json"
CAPS = dict(fontsize=12, fontweight="semibold", color=ds.GRANITE)
R_ALT = r"Altitude $r$ (km)"


def panel_label(ax, text, x=0.0, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=13, fontweight="semibold",
            color=ds.GRANITE)


def thousands(v, _=None):
    return f"{v:,.0f}"


# ================================================================= B1 run table
def figB1():
    b = F["B1"]
    fig, ax = ds.figure("strip")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    w, h = ds.SIZES["strip"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    cols = [("Run", ""), ("Azimuth", "cells"), ("Cell width", "at 231 km (km)"), ("Cells per", r"$\lambda_h$"),
            ("Particles", "in total"), ("Particles per", "cell at 231 km"), ("Peak growth", r"$G_{\max}$"),
            (r"$L_{\mathrm{att}}$ at", "231 km (km)")]
    xc = [0.62, 2.02, 3.55, 5.02, 6.55, 8.15, 9.70, 11.20]
    y_h1, y_h2, y_rule = 2.98, 2.74, 2.60
    for (a, b2), x in zip(cols, xc):
        ha = "left" if x == xc[0] else "center"
        x0 = 0.18 if x == xc[0] else x
        ax.text(x0, y_h1, a, ha=ha, va="baseline", fontsize=14, fontweight="semibold")
        if b2:
            ax.text(x0, y_h2, b2, ha=ha, va="baseline", fontsize=14, fontweight="semibold")
    ax.plot([0.12, w - 0.12], [y_rule] * 2, color=ds.GRANITE, lw=0.8)
    pitch, y0 = 0.31, 2.36
    for i, r in enumerate(b["rows"]):
        y = y0 - i * pitch - (0.08 if i >= 3 else 0)
        last = r["run"] == "S6"
        kw = dict(fontsize=15, va="baseline", ha="center", color=ds.MIDNIGHT if last else ds.GRANITE,
                  fontweight="semibold" if last else "normal")
        ax.add_patch(mpl.patches.Circle((0.27, y + 0.055), 0.055, fc=ds.RUN_RAMP[r["run"]], lw=0))
        ax.text(0.42, y, r["run"], **(kw | dict(ha="left", fontweight="semibold")))
        mant, exp = f"{r['particles']:.2e}".split("e")
        dag = "†" if r["ensemble"] else ""
        L = "—" if r["L_att_km"] is None else f"{r['L_att_km']:.0f} ± {r['sigma_L_km']:.0f}"
        vals = [f"{r['n_azi']:,}", f"{r['cell_km']:.1f}", f"{r['cells_per_lambda']:.1f}",
                rf"${float(mant):.1f}\times10^{{{int(exp)}}}${dag}", f"{r['ppc_230']:,}{dag}",
                f"{r['g_max']:.2f}", L + ("‡" if r["L_att_km"] is None else "")]
        for x, v in zip(xc[1:], vals):
            ax.text(x, y, v, **kw)
        if i == 2:
            ax.plot([0.12, w - 0.12], [y - 0.13] * 2, color=ds.GRANITE, lw=0.5, alpha=0.45)
    ax.plot([0.12, w - 0.12], [y0 - 5 * pitch - 0.2] * 2, color=ds.GRANITE, lw=0.8)
    notes = ("† S1–S3 are ten-member ensemble means; particle counts are per member.   "
             "‡ No accepted fit: at most 3 points outside the forced wedge.",
             rf"$G_{{\max}}$: peak of $A(z)/A_{{150}}$ up to 315 km.   Cells per $\lambda_h$ use "
             rf"$\lambda_h$ = {b['lambda_h_km']:.0f} km at 231 km (S6).")
    for k, t in enumerate(notes):
        ax.text(0.18, 0.40 - 0.24 * k, t, ha="left", va="baseline", fontsize=13, fontweight=300)
    save(fig, "figB1_run_table", FACTS + " B1", *b["sources"])


# ================================================================= B2 wheel-sector boundary fix
def figB2():
    d = np.load(DATA / "figB2_wheel_profiles.npz")
    z = d["z_km"]
    fig, axs = ds.figure("evidence", ncols=2, sharey=True)
    pre_kw = dict(color=ds.ROLE["prior_work"], ls=ds.LINESTYLE["prior_work"], lw=1.5)
    fix_kw = dict(color=ds.ROLE["this_work"], lw=2.4)
    for ax, q, xl in ((axs[0], "n", "Density, 2-D / 1-D − 1 (%)"), (axs[1], "T", "Temperature, 2-D / 1-D − 1 (%)")):
        ax.axvline(0, color=ds.GRANITE, lw=0.6, zorder=1)
        for tag, kw in (("pre", pre_kw), ("fix", fix_kw)):
            m, se = d[f"{q}_{tag}"], d[f"se_{q}_{tag}"]
            ax.fill_betweenx(z, m - se, m + se, color=kw["color"], alpha=0.15, lw=0)   # ±1 SE band
            ax.plot(m, z, **kw)
        ds.exobase(ax)
        ax.set_xlabel(xl)
    k = slice(None, None, 4)
    axs[0].plot(d["model"][k], z[k], ls="none", marker="o", mfc="none", mec=ds.ROLE["theory"], mew=1.2, ms=6)
    axs[0].set_xlim(-7.5, 1.5)
    axs[1].set_xlim(-0.6, 1.0)
    axs[0].set_ylim(140, 400)
    axs[0].set_ylabel(R_ALT)
    panel_label(axs[0], "A · DENSITY")
    panel_label(axs[1], "B · TEMPERATURE")
    ds.label(axs[0], "Plane wall", (-4.6, 384), color=ds.GRANITE)
    ds.label(axs[0], "Cone wall", (0.25, 300), color=ds.MIDNIGHT)
    axs[0].text(-7.3, 160, r"$(R_0/R)^{\cos^2\beta} - 1$", color=ds.ROLE["theory"], fontsize=14)
    ds.takeaway(axs[1], "+0.16–0.35%", xy=(0.40, 352), xytext=(0.30, 285))
    save(fig, "figB2_wheel_fix", FACTS + " B2", *F["B2"]["sources"][1:3])


# ================================================================= B3 Stage 1B floor -> Stage 2 rule
def figB3():
    d = np.load(DATA / "figB3_floor.npz")
    b = F["B3"]
    fig, ax = ds.figure("evidence")
    names = {"n": "n′/n", "T": "T′/T", "u": "u", "w": "w"}
    colors = {"n": ds.MIDNIGHT, "T": ds.CELTIC, "u": ds.VIOLET, "w": ds.GRANITE}
    ax.axvspan(10, 60, color=ds.GRANITE, alpha=0.07, lw=0)
    for f in "nTuw":
        ax.plot(100 * d[f"{f}_term"], d[f"{f}_z_km"], color=colors[f], lw=2.0,
                ls=(0, (3, 2.5)) if f == "w" else "-", label=names[f])
    ax.axvline(100 * b["delta_phys"], color=ds.GRANITE, lw=1.2, ls=(0, (1, 2.5)))
    ax.axvline(100 * b["score_cut"], color=ds.GRANITE, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(1, 50)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(140, 295)
    ds.exobase(ax, label=False)
    ax.text(48, 232, "EXOBASE", **CAPS, ha="right", va="bottom")
    ax.text(4.8, 288, r"$\delta_{\mathrm{phys}}$ = 5%", ha="right", va="top", fontsize=14)
    ax.text(10.8, 145, "NOT SCORED ABOVE\nTHE 10% CROSSING", ha="left", va="bottom", **CAPS)
    ax.set_xlabel(r"Detectable difference, $3\,\delta_{\mathrm{floor}}/\sqrt{5}$")
    ax.set_ylabel(R_ALT)
    ax.legend(loc="upper left", handlelength=1.8)
    save(fig, "figB3_noise_floor", FACTS + " B3", *b["sources"][:2])


# ================================================================= B4 self-induced westward wind
def figB4():
    d = np.load(DATA / "figB4_meanflow.npz")
    b = F["B4"]
    fig, ax = ds.figure("evidence")
    fig.set_layout_engine("none")
    ax.set_position([0.105, 0.19, 0.70, 0.77])
    cax = fig.add_axes([0.835, 0.19, 0.018, 0.77])
    vmax = 6.5
    m = ax.pcolormesh(d["t_h"], d["z_km"], d["u"].T, cmap=ds.DIVERGING, vmin=-vmax, vmax=vmax,
                      shading="nearest", rasterized=True)
    cs = ax.contour(d["t_h"], d["z_km"], d["u"].T, levels=[-4, -2, -1], colors=[ds.CULTURED, ds.GRANITE, ds.GRANITE],
                    linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:g}".replace("-", "−"), fontsize=12, inline_spacing=3)
    ax.axhspan(180, 200, color=ds.CULTURED, alpha=0.6, lw=0, zorder=2)
    ax.text(0.25, 190, "ABSORBING LAYER", va="center", **CAPS, zorder=3)
    ds.input_altitudes(ax)
    ax.text(11.85, 141.5, "DSMC INPUTS", ha="right", va="bottom", **CAPS)
    ds.takeaway(ax, f"{b['u_peak_m_s']:.1f} m/s at {b['u_peak_z_km']:.0f} km".replace("-", "−"),
                xy=(11.9, b["u_peak_z_km"]), xytext=(0.3, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 200)
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks(range(0, 201, 40))
    ax.set_xlabel("Time since wave onset (h)")
    ax.set_ylabel("Altitude (km)")
    cb = fig.colorbar(m, cax=cax, ticks=[-6, -4, -2, 0, 2, 4, 6])
    cb.set_label("Mean wind along the\nwave's path (m/s)")
    save(fig, "figB4_westward_wind", FACTS + " B4", b["sources"][0])


# ================================================================= B5 radial grid across the H2 exobase
def figB5():
    d = np.load(DATA / "figB5_grid_ratio.npz")
    b = F["B5"]
    fig, ax = ds.figure("evidence")
    ubc_col = {1: ds.MIDNIGHT, 2: ds.CELTIC, 3: ds.VIOLET}
    ax.axvspan(0.95, 1.05, color=ds.GRANITE, alpha=0.08, lw=0)
    ax.axhspan(b["exo_all_min_km"], b["exo_all_max_km"], color=ds.ROLE["exobase"], alpha=0.25, lw=0)
    ax.axvline(1.0, color=ds.GRANITE, lw=0.6)
    for atm, lw, al in (("H2O", 1.2, 0.55), ("H2", 2.0, 1.0)):
        for u in (1, 2, 3):
            ax.plot(d[f"{atm}_u{u}_ratio"], d[f"{atm}_u{u}_alt_km"], color=ubc_col[u], lw=lw, alpha=al)
    ax.set_yscale("log")
    ax.set_ylim(100, 20000)
    ax.yaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000, 5000, 10000, 20000]))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.92, 1.18)
    ax.set_xticks([0.95, 1.00, 1.05, 1.10, 1.15])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.text(0.955, 125, "±5%", ha="left", va="center", **CAPS)
    ax.text(1.175, b["exo_all_max_km"] * 1.04, r"H$_2$ EXOBASE", ha="right", va="bottom", **CAPS)
    lo, hi = b["res_band_offset_pct_min"], b["res_band_offset_pct_max"]
    ds.takeaway(ax, f"+{lo:.0f}–{hi:.0f}%\nat {b['res_band_lo_km']:,.0f}–{b['res_band_hi_km']:,.0f} km",
                xy=(1.125, 1350), xytext=(1.115, 5200))
    h = [Line2D([], [], color=ubc_col[u], lw=2.0, label=b["ubcs"][str(u)].capitalize()) for u in (1, 2, 3)]
    # thick lines pure H2, thin coupled O + H2: explained in the slide text, not the legend
    ax.legend(handles=h, loc="lower right", fontsize=13, handlelength=1.6, labelspacing=0.3)
    ax.set_xlabel("Density, 1,000 km cells / 250 km cells")
    ax.set_ylabel("Altitude (km)")
    save(fig, "figB5_radial_grid", FACTS + " B5", *b["sources"][:2])


# ================================================================= B6 phase-lag retrieval of lambda_h
def figB6():
    d = np.load(DATA / "figB6_phase_lag.npz")
    b = F["B6"]
    fig, (a1, a2) = ds.figure("evidence", ncols=2)
    use = d["used"]
    for left, color, name, xy in ((True, ds.MIDNIGHT, "LEFT EDGE", (165, 0.13)),
                                  (False, ds.CELTIC, "RIGHT EDGE", (165, 0.06))):
        s = use & (d["is_left"] == left)
        a1.errorbar(d["distance_km"][s], d["lag_periods"][s], yerr=d["lag_err_periods"][s], ls="none", marker="o",
                    ms=5, color=color, ecolor=ds.GRANITE, elinewidth=0.8)
        ds.label(a1, name, xy, color=color)
    x = np.linspace(0, 330, 50)
    a1.plot(x, d["intercept_periods"] + d["slope_per_km"] * x, color=ds.MIDNIGHT, lw=2.0, zorder=1)
    a1.axhline(b["lag_ceiling_periods"], color=ds.GRANITE, lw=1.2, ls=(0, (3, 2.5)))
    a1.text(5, b["lag_ceiling_periods"] + 0.02, "ALIAS LIMIT 0.7 P", **CAPS)
    a1.text(8, 0.55, r"$\lambda_h = 1/\mathrm{slope}$", fontsize=14, color=ds.MIDNIGHT)
    a1.set_xlim(0, 330)
    a1.set_ylim(-0.08, 0.85)
    a1.set_xlabel("Distance past wedge edge (km)")
    a1.set_ylabel("Phase lag (wave periods)")
    panel_label(a1, "A · ONE FIT, 28 COLUMNS")

    B, lam, err, ml = d["baselines_km"], d["coupled_lam"], d["coupled_err"], d["coupled_maxlag"]
    tot = b["sigma_total_km"]
    a2.axhspan(b["lambda_h_km"] - tot, b["lambda_h_km"] + tot, color=ds.MIDNIGHT, alpha=0.15, lw=0)
    a2.axhline(b["lambda_h_km"], color=ds.MIDNIGHT, lw=1.0)
    first_bad = B[ml > b["lag_ceiling_periods"]].min()
    a2.axvspan(0.5 * (first_bad + B[B < first_bad].max()), 850, color=ds.GRANITE, alpha=0.08, lw=0)
    ok = ml <= b["lag_ceiling_periods"]
    a2.errorbar(B, lam, yerr=err, ls="none", ecolor=ds.GRANITE, elinewidth=0.8)
    a2.plot(B[ok], lam[ok], "o", color=ds.MIDNIGHT, ms=6)
    a2.plot(B[~ok], lam[~ok], "o", mfc="none", mec=ds.MIDNIGHT, mew=1.2, ms=6)
    a2.text(830, 575, "LAG > 0.7 P", ha="right", va="top", **CAPS)
    a2.set_xlim(50, 850)
    a2.set_ylim(400, 580)
    a2.set_xticks([200, 400, 600, 800])
    a2.set_xlabel("Fit baseline (km)")
    a2.set_ylabel(r"$\lambda_h$ (km)")
    panel_label(a2, "B · BASELINE SWEEP")
    save(fig, "figB6_phase_lag", FACTS + " B6", *b["sources"][:2])


ALL = {"B1": figB1, "B2": figB2, "B3": figB3, "B4": figB4, "B5": figB5, "B6": figB6}

if __name__ == "__main__":
    for key in sys.argv[1:] or ALL:
        ALL[key]()
