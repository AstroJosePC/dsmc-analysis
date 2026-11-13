"""drift_mb diagnostic figures from the analysis files (run compute_analysis.py first).

    MPLBACKEND=Agg python -B campaigns/drift_mb/make_figures.py

Reads outputs/drift_mb/analysis/*_analysis.nc and outputs/drift_mb/tables/*.csv; writes
outputs/drift_mb/figures/<name>.{png,pdf}. Graphite style (dsmclib.graphite): species ink O blue / H2 orange,
drift amplitude s as a light-to-dark ramp of the species ink (s = 0 grey), period by marker / line style,
perturbation maps on a teal-grey-red diverging scale centred on 0.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from dsmclib import drift, graphite as G, tiers
from dsmclib.paths import output_dir

CAMPAIGN = "drift_mb"
ATMS = ("O", "OH2")
ATM_LABEL = {"O": "pure O", "OH2": "O + H$_2$"}
S_LEVELS = (0.05, 0.10, 0.20)
PERIODS = (165, 330, 660)
MARK = {165: "o", 330: "s", 660: "D"}
LS = {165: ":", 330: "-", 660: "--"}
DISPLAY_MIN_PART = 50     # profiles are drawn only where the baseline holds this many computational particles
DIV = LinearSegmentedColormap.from_list("tgr", ["#0f5c58", "#3f9c95", "#e6e5e1", "#d0617a", "#8c1d3a"])
FIG = output_dir(CAMPAIGN, "figures")
TAB = output_dir(CAMPAIGN, "tables")


def alias(atm, s, P):
    return f"{atm}_s{int(round(s * 100)):03d}_P{P}"


def shade(color, s):
    """Sequential ramp within the species hue: s = 0.05 / 0.10 / 0.20 -> 45 / 70 / 100 % ink; s = 0 grey."""
    if s == 0:
        return G.GREY
    f = {0.05: 0.45, 0.10: 0.70, 0.20: 1.0}[round(s, 2)]
    c = np.array(to_rgb(color))
    return tuple(1 - f * (1 - c))


_cache: dict = {}


def load(a: str) -> xr.Dataset:
    if a not in _cache:
        with xr.open_dataset(tiers.analysis_path(CAMPAIGN, a)) as ds:
            _cache[a] = ds.load()
    return _cache[a]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def shown(ds, sp):
    """Cells drawn in profile plots: resolved and >= DISPLAY_MIN_PART baseline computational particles."""
    return (ds["resolved"].sel(species=sp).values.astype(bool)
            & (ds["baseline_num_part"].sel(species=sp).values >= DISPLAY_MIN_PART))


def species_of(atm):
    return ["O"] if atm == "O" else ["O", "H2"]


def panel_tag(ax, txt):
    ax.set_title(txt, loc="left", fontsize=8.5)


# ------------------------------------------------------------------ boundary VDFs
def fig_vdf(atm, s=0.20, P=330):
    a = alias(atm, s, P)
    ds = load(a)
    sps = species_of(atm)
    y = ds["y"].values
    dy = ds["y_hi"].values - ds["y_lo"].values
    fig, axes = plt.subplots(3, len(sps), figsize=(G.WIDTH * (0.55 if len(sps) == 1 else 1.0), 6.2),
                             squeeze=False, gridspec_kw={"height_ratios": [1.1, 1.0, 1.2]})
    groups = list(ds["vdf_group"].values)
    show = {"p03": "phase 3 (w' $\\approx$ +max)", "p11": "phase 11 (w' $\\approx$ −max)", "ringdown": "ring-down (w' = 0)"}
    for j, sp in enumerate(sps):
        col = G.species_color(sp)
        cnt = ds["vdf_count"].sel(species=sp)
        prob = ds["vdf_model_prob"].sel(species=sp)
        f0 = drift.mixture_bin_prob(np.append(ds["y_lo"].values, ds["y_hi"].values[-1]), [0.0])
        ax0, ax1, ax2 = axes[:, j]
        for g, lab in show.items():
            c = cnt.sel(vdf_group=g).values
            N = c.sum()
            pdf = c / N / dy
            th = prob.sel(vdf_group=g).values
            ink = {"p03": col, "p11": shade(col, 0.05), "ringdown": G.GREY}[g]
            ax0.step(y, pdf, where="mid", color=ink, lw=1.0, label=lab)
            ax0.plot(y, th / dy, color=G.CHARCOAL, lw=0.6, ls="--")
            # ratio to the undrifted flux Maxwellian: 8-bin blocks for noise
            k = 8
            cb = c[: len(c) // k * k].reshape(-1, k).sum(1)
            f0b = f0[: len(c) // k * k].reshape(-1, k).sum(1)
            thb = th[: len(c) // k * k].reshape(-1, k).sum(1)
            yb = y[: len(c) // k * k].reshape(-1, k).mean(1)
            ok = cb > 50
            r = cb / N / f0b
            ax1.errorbar(yb[ok], r[ok], yerr=np.sqrt(cb[ok]) / N / f0b[ok], fmt="o", ms=2.5, color=ink, elinewidth=0.6)
            ax1.plot(yb, thb / f0b, color=ink, lw=0.9)
        ax0.set_xlim(0, 3.2)
        ax0.set_ylabel("f(y)")
        panel_tag(ax0, f"{G.species_label(sp)} injected, {ATM_LABEL[atm]}, s$_O$ = {s}, P{P}")
        if j == 0:
            ax0.legend(loc="upper right")
        ax1.set_xlim(0, 3.2)
        ax1.set_ylim(0.6, 1.6)
        ax1.axhline(1, color=G.LIGHT, lw=0.6)
        ax1.set_ylabel("f(y) / f(y | 0)")
        ax1.set_xlabel("y = v$_n$ / v$_{th}$")
        panel_tag(ax1, "ratio to f(y | s = 0); lines: mixture theory")
        exp_ = cnt.sum("v_bin") * prob
        z = ((cnt - exp_) / np.sqrt(exp_.where(exp_ > 0))).values
        sel = y < 3.2
        im = ax2.pcolormesh(y[sel], np.arange(len(groups)), z[:, sel], cmap=DIV, vmin=-4, vmax=4, shading="nearest")
        ax2.set_yticks(np.arange(len(groups)), [g.replace("p", "") if g != "ringdown" else "ring" for g in groups], fontsize=5.5)
        ax2.set_xlabel("y = v$_n$ / v$_{th}$")
        ax2.set_ylabel("phase bin")
        ax2.grid(False)
        panel_tag(ax2, "(observed − expected) / $\\sqrt{\\mathrm{expected}}$")
        fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    fig.tight_layout()
    save(fig, f"vdf_phasefold_{atm}_s{int(s * 100):03d}_P{P}")


def fig_moments(P=330):
    cases = [("O", "O"), ("OH2", "O"), ("OH2", "H2")]
    rows = [("mean", "mean_unb", "$\\langle y\\rangle$"), ("var", "var_hist", "var(y)"), ("skew", "skew_hist", "skewness"),
            ("exkurt", "exkurt_hist", "excess kurtosis")]
    fig, axes = plt.subplots(len(rows), len(cases), figsize=(G.WIDTH, 6.4), sharex=True, squeeze=False)
    for c, (atm, sp) in enumerate(cases):
        col = G.species_color(sp)
        for s in (0.0,) + S_LEVELS:
            ds = load(alias(atm, s, P))
            ph = np.degrees(ds["phase_angle"].values[:16])
            for r, (key, obs, lab) in enumerate(rows):
                ax = axes[r, c]
                o = ds[f"vdf_{obs}"].sel(species=sp).values[:16]
                th = ds[f"vdf_{key}_th"].sel(species=sp).values[:16]
                se = ds[f"vdf_{key}_se"].sel(species=sp).values[:16]
                ink = shade(col, s)
                ax.errorbar(ph, o, yerr=se, fmt="o", ms=2.6, color=ink, elinewidth=0.6, label=f"s$_O$ = {s}" if r == 0 else None)
                ax.plot(ph, th, color=ink, lw=0.9)
                if c == 0:
                    ax.set_ylabel(lab)
        axes[0, c].set_title(f"{G.species_label(sp)} in {ATM_LABEL[atm]}", loc="left", fontsize=8.5)
        axes[-1, c].set_xlabel("forcing phase [deg]")
        axes[-1, c].set_xticks([0, 90, 180, 270, 360])
    hs, ls_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls_, loc="upper right", ncol=4, fontsize=6.5, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Injected normal-velocity moments vs phase, P{P}\n(points: DSMC $\\pm$ SE; lines: drifting-Maxwellian mixture)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, f"vdf_moments_vs_phase_P{P}")


def fig_budget():
    b = pd.read_csv(TAB / "flux_budget.csv")
    b = b[(b.window == "fit") & b.Ndot_inj_ratio_control.notna()]
    meta = pd.read_csv(TAB / "run_summary.csv").set_index("run")
    b = b.join(meta[["s_O", "s_H2", "period_s", "atmosphere"]], on="run")
    ss = np.linspace(0, 0.22, 100)
    phi = (np.arange(2048) + 0.5) * 2 * np.pi / 2048
    m0 = drift.raw_moments(0.0, 2)

    def cyc(fun, s):
        return np.array([fun(si * np.sin(phi)).mean() for si in s])

    th = {"Ndot": lambda x: drift.m_flux(x),
          "Pdot": lambda x: drift.m_flux(x) * drift.raw_moments(x, 1)[1] / m0[1],
          "Edot": lambda x: drift.m_flux(x) * (drift.raw_moments(x, 2)[2] + 1) / (m0[2] + 1)}
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.4))
    for ax, (k, lab) in zip(axes, (("Ndot", "number (mass) flux"), ("Pdot", "normal momentum flux"), ("Edot", "kinetic energy flux"))):
        for sp in ("O", "H2"):
            col = G.species_color(sp)
            d = b[b.species == sp]
            sv = d.s_O if sp == "O" else d.s_H2
            for P in PERIODS:
                q = d[np.round(d.period_s / 5) * 5 == {165: 165, 330: 330, 660: 660}[P]] if False else d[(d.period_s > P * 0.9) & (d.period_s < P * 1.1)]
                sq = q.s_O if sp == "O" else q.s_H2
                ax.plot(sq, 100 * (q[f"{k}_inj_ratio_control"] - 1), MARK[P], color=col, ms=4, mfc="none" if sp == "H2" else col, ls="none")
            ax.plot(ss, 100 * (cyc(th[k], ss) - 1), color=G.CHARCOAL, lw=0.8)
        ax.set_xlabel("species drift ratio s = w'/v$_{th}$")
        panel_tag(ax, lab)
    axes[0].set_ylabel("cycle-mean injection vs s = 0 control [%]")
    hs = [plt.Line2D([], [], marker=MARK[P], color=G.GRAPHITE, ls="none", label=f"P{P}") for P in PERIODS]
    hs += [plt.Line2D([], [], marker="o", color=G.species_color("O"), ls="none", label="O"),
           plt.Line2D([], [], marker="o", mfc="none", color=G.species_color("H2"), ls="none", label="H$_2$"),
           plt.Line2D([], [], color=G.CHARCOAL, lw=0.8, label="theory")]
    axes[2].legend(handles=hs, loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "injection_budget_vs_s")


# ------------------------------------------------------------------ escape
def fig_escape_series():
    ctrl = load("OH2_s000_P330")
    Rc = float(ctrl["escape_rate"].sel(species="H2").mean())
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.5), sharey=True)
    col = G.species_color("H2")
    for ax, P in zip(axes, PERIODS):
        for s in (0.0,) + S_LEVELS:
            if s == 0 and P != 330:
                continue
            ds = load(alias("OH2", s, P))
            Pd = ds.attrs["period"]
            t = (np.arange(ds.sizes["period"]) + 0.5) * Pd / 1e3
            r = ds["escape_rate_period"].sel(species="H2").values / Rc
            e = ds["escape_rate_period_se"].sel(species="H2").values / Rc
            ax.errorbar(t, r, yerr=e, color=shade(col, s), lw=1.0, elinewidth=0.5, label=f"s$_O$ = {s}")
        ax.axvspan(20 * Pd / 1e3, 25 * Pd / 1e3, color=G.RULE, zorder=0)
        ax.axhline(1, color=G.LIGHT, lw=0.6)
        ax.set_xlabel("time since forcing start [ks]")
        panel_tag(ax, f"P{P} (shaded: ring-down)")
    axes[0].set_ylabel("H$_2$ escape rate / control mean")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "escape_H2_per_period")


def fig_escape_summary():
    e = pd.read_csv(TAB / "escape_summary.csv")
    meta = pd.read_csv(TAB / "run_summary.csv").set_index("run")
    e = e[e.species == "H2"].join(meta[["s_O", "period_s", "is_control"]], on="run")
    e = e[e.is_control == 0]
    col = G.species_color("H2")
    fig, axes = plt.subplots(2, 2, figsize=(G.WIDTH, 4.6))
    for ax, k, lab in ((axes[0, 0], "enhancement", "fit window (periods 10–19)"),
                       (axes[0, 1], "enhancement_common", "common window 10–20 x P165")):
        for P in PERIODS:
            q = e[(e.period_s > P * 0.9) & (e.period_s < P * 1.1)]
            ax.errorbar(q.s_O, 100 * q[k], yerr=100 * q[f"{k}_se"], fmt=MARK[P] + LS[P], color=col, ms=4, lw=0.9,
                        elinewidth=0.6, label=f"P{P}")
        ax.axhline(0, color=G.LIGHT, lw=0.6)
        ax.set_xlabel("s$_O$")
        panel_tag(ax, lab)
    axes[0, 0].set_ylabel("H$_2$ escape enhancement vs s = 0 [%]")
    axes[0, 0].legend(fontsize=6)
    ax = axes[1, 0]
    for P in PERIODS:
        ds = load(alias("OH2", 0.2, P))
        n = ds["escape_npart_phase"].sel(species="H2").values
        rel = n / n.mean() - 1
        ax.errorbar(np.degrees((np.arange(16) + 0.5) * 2 * np.pi / 16), 100 * rel, yerr=100 * np.sqrt(n) / n.mean(),
                    fmt=MARK[P] + LS[P], color=col, ms=3, lw=0.8, elinewidth=0.5, label=f"P{P}")
    ax.axhline(0, color=G.LIGHT, lw=0.6)
    ax.set_xlabel("forcing phase [deg]")
    ax.set_ylabel("phase-folded escape − mean [%]")
    ax.set_xticks([0, 90, 180, 270, 360])
    panel_tag(ax, "s$_O$ = 0.2, fit window")
    ax = axes[1, 1]
    for P in PERIODS:
        q = e[(e.period_s > P * 0.9) & (e.period_s < P * 1.1)]
        ax.plot(q.s_O, q.mean_speed_fit, MARK[P] + LS[P], color=col, ms=4, lw=0.9, label=f"P{P}")
    vc = float(e.mean_speed_control.iloc[0])
    ax.axhline(vc, color=G.GREY, lw=0.8, ls="-", label="s = 0 control")
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, vc - 15), hi)
    ax.set_xlabel("s$_O$")
    ax.set_ylabel("mean H$_2$ escape speed at r$_{top}$ [m/s]")
    ax.legend(fontsize=6)
    panel_tag(ax, "escape-speed mean, fit window")
    fig.tight_layout()
    save(fig, "escape_H2_summary")


def fig_escape_vdf():
    col = G.species_color("H2")
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.6))
    k = 4                                   # 100 m/s bins for the pdf
    kr = 10                                 # 250 m/s bins for ratios
    ds0 = load("OH2_s000_P330")
    v = ds0["speed"].values[:-1]
    dv = 25.0

    def pdf(h, kk):
        h = h[:-1][: len(v) // kk * kk].reshape(-1, kk).sum(1)
        return v[: len(v) // kk * kk].reshape(-1, kk).mean(1), h

    ax = axes[0]
    vb, h = pdf(ds0["escape_vdf_fit"].sel(species="H2").values, k)
    ax.step(vb, h / h.sum() / (k * dv) * 1e3, where="mid", color=G.GREY, lw=1.0, label="s = 0")
    vb, h = pdf(ds0["escape_vdf_spin"].sel(species_base="H2").values, k)
    ax.step(vb, h / h.sum() / (k * dv) * 1e3, where="mid", color=G.CHARCOAL, lw=0.7, ls=":", label="spin-up")
    jp = ds0["escape_vdf_jeans_prob"].sel(species="H2").values
    vbj, hj = pdf(jp, k)
    ax.plot(vbj, hj / (k * dv) * 1e3, color=G.CHARCOAL, lw=0.7, ls="--", label="Jeans, mapped")
    for s in S_LEVELS:
        ds = load(alias("OH2", s, 330))
        vb, h = pdf(ds["escape_vdf_fit"].sel(species="H2").values, k)
        ax.step(vb, h / h.sum() / (k * dv) * 1e3, where="mid", color=shade(col, s), lw=1.0, label=f"s$_O$ = {s}")
    ax.set_xlim(0, 6000)
    ax.set_xlabel("speed at r$_{top}$ [m/s]")
    ax.set_ylabel("pdf [per km/s]")
    ax.legend(fontsize=5.2, loc="upper right", bbox_to_anchor=(1.02, 1.0), handlelength=1.4)
    panel_tag(ax, "H$_2$ escape-speed VDF, P330")
    for ax, cases, title in ((axes[1], [(s, 330) for s in S_LEVELS], "ratio to control, P330"),
                             (axes[2], [(0.2, P) for P in PERIODS], "ratio to control, s$_O$ = 0.2")):
        vb, hc = pdf(load(alias("OH2", 0.2, 330))["escape_vdf_control"].sel(species_base="H2").values, kr)
        for s, P in cases:
            ds = load(alias("OH2", s, P))
            vb, h = pdf(ds["escape_vdf_fit"].sel(species="H2").values, kr)
            vb2, hc = pdf(ds["escape_vdf_control"].sel(species_base="H2").values, kr)
            ok = (h > 20) & (hc > 20)
            r = (h / h.sum()) / (hc / hc.sum())
            err = r * np.sqrt(1 / np.maximum(h, 1) + 1 / np.maximum(hc, 1))
            ink = shade(col, s)
            ax.errorbar(vb[ok], r[ok], yerr=err[ok], fmt=MARK[P] + "-", color=ink, ms=2.5, lw=0.8, elinewidth=0.5,
                        label=f"s$_O$ = {s}, P{P}")
        ax.axhline(1, color=G.LIGHT, lw=0.6)
        ax.set_xlim(0, 6000)
        ax.set_xlabel("speed at r$_{top}$ [m/s]")
        ax.legend(fontsize=5.5)
        panel_tag(ax, title)
    axes[1].set_ylabel("normalised VDF ratio")
    fig.tight_layout()
    save(fig, "escape_H2_speed_vdf")


# ------------------------------------------------------------------ wave fields
def _fields_map(a, mode):
    ds = load(a)
    sps = list(ds["species"].values)
    alt = ds["altitude"].values / 1e3
    res = ds["resolved"].values.astype(bool)
    zmax = 700.0
    fig, axes = plt.subplots(len(sps), 3, figsize=(G.WIDTH, 2.2 * len(sps) + 0.3), squeeze=False)
    names = (("ur", "u$_r$ [m/s]"), ("dn", "$\\Delta$n / n$_0$"), ("dT", "$\\Delta$T [K]"))
    P = ds.attrs["period"]
    for i, sp in enumerate(sps):
        keep = res[i] & (alt <= zmax)
        for j, (nm, lab) in enumerate(names):
            ax = axes[i, j]
            if mode == "time":
                Z = ds[f"{nm}_series"].sel(species=sp).values[:, keep]
                x = ds["elapsed"].values / P
                xl = "time since forcing start [periods]"
            else:
                Z = np.vstack([ds[f"{nm}_phasefold"].sel(species=sp).values] * 2)[:, keep]
                x = (np.arange(32) + 0.5) * 360 / 16
                xl = "forcing phase [deg] (two cycles)"
            lim = np.nanpercentile(np.abs(Z), 99)
            im = ax.pcolormesh(x, alt[keep], Z.T, cmap=DIV, vmin=-lim, vmax=lim, shading="nearest")
            ax.grid(False)
            ax.axhline(ds.attrs["z_exo_O_km"], color=G.CHARCOAL, lw=0.6, ls="--")
            if mode == "time":
                ax.axvline(ds.attrs["n_force_periods"], color=G.CHARCOAL, lw=0.6)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            panel_tag(ax, f"{G.species_label(sp)} {lab}")
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel("altitude [km]")
    fig.suptitle(f"{a}: perturbation vs {'time' if mode == 'time' else 'phase'} (dashed: O exobase)",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, f"{'hovmoller' if mode == 'time' else 'altitude_phase'}_{a}")


def fig_wave_profiles(atm):
    sp = "O"
    fig, axes = plt.subplots(3, 3, figsize=(G.WIDTH, 6.4), sharey=True)
    col = G.species_color(sp)
    for c, P in enumerate(PERIODS):
        for s in S_LEVELS:
            ds = load(alias(atm, s, P))
            alt = ds["altitude"].values / 1e3
            res = shown(ds, sp)
            ink = shade(col, s)
            for r, (nm, lab) in enumerate((("ur_amp", "u$_r$ amplitude [m/s]"), ("ur_phase", "u$_r$ phase [deg]"),
                                           ("dT_amp", "$\\Delta$T amplitude [K]"))):
                ax = axes[r, c]
                v = ds[nm].sel(species=sp).values
                if nm == "ur_phase":
                    ok = res & np.isfinite(v)
                    v = np.where(ok, np.degrees(np.unwrap(np.where(ok, v, 0))), np.nan)
                ax.plot(np.where(res, v, np.nan), alt, color=ink, lw=1.1, label=f"s$_O$ = {s}")
                if nm != "ur_phase":
                    se = ds[nm + "_se"].sel(species=sp).values
                    sig = res & (v - 2 * se > 0); ax.fill_betweenx(alt, np.where(sig, v - 2 * se, np.nan), np.where(sig, v + 2 * se, np.nan),
                                     color=ink, alpha=G.BAND_ALPHA, lw=0)
                    if s == 0.2:
                        ax.plot(np.where(res, ds[nm.replace("_amp", "_null_amp")].sel(species=sp).values, np.nan), alt,
                                color=G.GREY, lw=0.8, ls=":", label="s = 0 noise floor")
            if s == 0.2:
                H = ds.attrs["H_scale_m"]
                a0 = ds["ur_amp"].sel(species=sp).values[0]
                axes[0, c].plot(a0 * np.exp((alt - alt[0]) * 1e3 / (2 * H)), alt, color=G.CHARCOAL, lw=0.6, ls="--",
                                label="e$^{z/2H}$ (no dissipation)")
        for r in range(3):
            axes[r, c].axhline(load(alias(atm, 0.2, P)).attrs["z_exo_O_km"], color=G.LIGHT, lw=0.8)
        axes[0, c].set_xscale("log")
        axes[0, c].set_xlim(0.5, 300)
        axes[2, c].set_xscale("log")
        axes[2, c].set_xlim(0.2, 200)
        axes[0, c].set_title(f"P{P}", loc="left", fontsize=8.5)
        for r, lab in enumerate(("u$_r$ amplitude [m/s]", "u$_r$ phase rel. w' [deg]", "$\\Delta$T amplitude [K]")):
            axes[r, c].set_xlabel(lab)
    for r in range(3):
        axes[r, 0].set_ylabel("altitude [km]")
    axes[0, 0].set_ylim(100, 500)
    axes[0, 0].legend(fontsize=5.5, loc="upper left")
    fig.suptitle(f"O response in {ATM_LABEL[atm]}, fit window ($\\pm$2 SE bands; grey line: O exobase)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, f"wave_profiles_{atm}")


def fig_propagation():
    w = pd.read_csv(TAB / "wave_response.csv")
    meta = pd.read_csv(TAB / "run_summary.csv").set_index("run")
    w = w.join(meta[["s_O", "period_s", "atmosphere", "is_control"]], on="run")
    w = w[w.is_control == 0]
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.5))
    ax = axes[0]
    for (atm, sp), g in w.groupby(["atmosphere", "species"]):
        mfc = G.species_color(sp) if atm == "O" else "none"
        ax.errorbar(g.omega * 1e3, g.c_phase, yerr=g.c_phase_se, fmt="o", ms=3.5, color=G.species_color(sp), mfc=mfc,
                    elinewidth=0.5, label=f"{G.species_label(sp)} in {ATM_LABEL[atm]}")
    r0 = w.iloc[0]
    om = np.linspace(r0.omega_acoustic * 1.05, 0.045, 200)
    ax.plot(om * 1e3, r0.c_s / np.sqrt(1 - (r0.omega_acoustic / om) ** 2), color=G.CHARCOAL, lw=0.8,
            label=r"$c_s\,/\,\sqrt{1-\omega_a^2/\omega^2}$")
    ax.axhline(r0.c_s, color=G.GREY, lw=0.6, ls=":")
    ax.set_xlabel(r"$\omega$ [10$^{-3}$ rad/s]")
    ax.set_ylabel("u$_r$ phase speed below exobase [m/s]")
    ax.set_ylim(400, 900)
    ax.legend(fontsize=5.5)
    panel_tag(ax, "vertical phase speed")
    ax = axes[1]
    for (atm, sp), g in w.groupby(["atmosphere", "species"]):
        if sp != "O":
            continue
        for P in PERIODS:
            q = g[(g.period_s > P * 0.9) & (g.period_s < P * 1.1)]
            ax.plot(q.s_O, q.damping_rate * 1e6, MARK[P] + LS[P], color=G.species_color("O"), ms=3.5, lw=0.8,
                    mfc=G.species_color("O") if atm == "O" else "none")
    ax.set_xlabel("s$_O$")
    ax.set_ylabel("1/2H − d ln A / dz [10$^{-6}$ m$^{-1}$]")
    panel_tag(ax, "amplitude damping rate (O)")
    ax = axes[2]
    for (atm, sp), g in w.groupby(["atmosphere", "species"]):
        if sp != "O":
            continue
        for P in PERIODS:
            q = g[(g.period_s > P * 0.9) & (g.period_s < P * 1.1)].sort_values("s_O")
            ax.errorbar(q.s_O, q.ur_amp_150, yerr=q.ur_amp_se_150, fmt=MARK[P] + LS[P], color=G.species_color("O"), ms=3.5,
                        lw=0.8, mfc=G.species_color("O") if atm == "O" else "none", elinewidth=0.5,
                        label=f"P{P}" if atm == "O" else None)
    ax.set_xlabel("s$_O$")
    ax.set_ylabel("u$_r$ amplitude at 150 km [m/s]")
    ax.legend(fontsize=6)
    panel_tag(ax, "linearity (filled: pure O; open: O+H$_2$)")
    fig.tight_layout()
    save(fig, "wave_propagation_summary")


def fig_heating():
    fig, axes = plt.subplots(2, 3, figsize=(G.WIDTH, 5.4), sharey="row")
    for r, atm in enumerate(ATMS):
        for c, P in enumerate(PERIODS):
            ax = axes[r, c]
            for sp in species_of(atm):
                col = G.species_color(sp)
                for s in S_LEVELS:
                    ds = load(alias(atm, s, P))
                    alt = ds["altitude"].values / 1e3
                    keep = shown(ds, sp)
                    v = np.where(keep, ds["dT_cyclemean"].sel(species=sp).values, np.nan)
                    se = np.where(keep, ds["dT_cyclemean_se"].sel(species=sp).values, np.nan)
                    ink = shade(col, s)
                    ax.plot(v, alt, color=ink, lw=1.0, ls="-" if sp == "O" else "--",
                            label=f"{G.species_label(sp)} s$_O$ = {s}")
                    ax.fill_betweenx(alt, v - 2 * se, v + 2 * se, color=ink, alpha=0.12, lw=0)
            ax.axvline(0, color=G.LIGHT, lw=0.6)
            ax.axhline(load(alias(atm, 0.2, P)).attrs["z_exo_O_km"], color=G.GREY, lw=0.6, ls=":")
            ax.set_ylim(100, 1500 if atm == "OH2" else 500)
            panel_tag(ax, f"{ATM_LABEL[atm]}, P{P}")
            ax.set_xlabel("cycle-mean $\\Delta$T vs control [K]")
        axes[r, 0].set_ylabel("altitude [km]")
    hs, ls_ = axes[1, 2].get_legend_handles_labels()
    fig.legend(hs, ls_, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Mean temperature change over the fit window ($\\pm$2 SE; dotted: O exobase; cells with "
                 f">= {DISPLAY_MIN_PART} computational particles)", x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "heating_profiles")
    h = pd.read_csv(TAB / "heating.csv")
    meta = pd.read_csv(TAB / "run_summary.csv").set_index("run")
    h = h.join(meta[["s_O", "period_s", "atmosphere", "is_control"]], on="run")
    fig, axes = plt.subplots(1, 3, figsize=(G.WIDTH, 2.4), sharey=True)
    for ax, (atm, sp) in zip(axes, (("O", "O"), ("OH2", "O"), ("OH2", "H2"))):
        g = h[(h.atmosphere == atm) & (h.species == sp)]
        for P in PERIODS:
            q = g[(g.period_s > P * 0.9) & (g.period_s < P * 1.1)].sort_values("s_O")
            ax.errorbar(q.s_O, q.dT_exo, yerr=q.dT_exo_se, fmt=MARK[P] + LS[P], color=G.species_color(sp), ms=3.5,
                        lw=0.8, elinewidth=0.5, label=f"P{P}, fit window")
            ax.errorbar(q.s_O + 0.004, q.dT_exo_common, yerr=q.dT_exo_common_se, fmt=MARK[P], color=G.GREY, ms=3,
                        mfc="none", elinewidth=0.5, label=f"P{P}, common window")
        ax.axhline(0, color=G.LIGHT, lw=0.6)
        ax.set_xlabel("s$_O$")
        panel_tag(ax, f"{G.species_label(sp)} in {ATM_LABEL[atm]}")
    axes[0].set_ylabel("$\\Delta$T at the O exobase [K]")
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    save(fig, "heating_exobase_summary")


def fig_ringdown(s=0.2):
    fig, axes = plt.subplots(3, 3, figsize=(G.WIDTH, 5.6), sharex="col")
    for c, P in enumerate(PERIODS):
        for atm in ATMS:
            ds = load(alias(atm, s, P))
            alt = ds["altitude"].values / 1e3
            per = ds["period"].values
            j150 = int(np.argmin(np.abs(alt - 150)))
            res = ds["resolved"].values.astype(bool)
            above = alt >= ds.attrs["z_exo_O_km"]
            for sp in species_of(atm):
                i = list(ds["species"].values).index(sp)
                mfc = G.species_color(sp) if atm == "O" else "none"
                lab = f"{G.species_label(sp)}, {ATM_LABEL[atm]}"
                if sp == "O":
                    axes[0, c].plot(per, ds["ur_amp_period"].values[:, i, j150], "o-", color=G.species_color(sp), mfc=mfc, ms=2.5, lw=0.8, label=lab)
                    axes[0, c].plot(per, np.full(len(per), ds["ur_amp_period_floor"].values[i, j150]), color=G.GREY, lw=0.6, ls=":")
                ex = above & res[i]
                y = np.nanmean(ds["dT_mean_period"].values[:, i][:, ex], axis=1)
                axes[1, c].plot(per, y, "o-", color=G.species_color(sp), mfc=mfc, ms=2.5, lw=0.8, label=lab)
                if sp == "H2":
                    r = ds["escape_rate_period"].sel(species="H2").values
                    e = ds["escape_rate_period_se"].sel(species="H2").values
                    rc = float(load("OH2_s000_P330")["escape_rate"].sel(species="H2").mean())
                    axes[2, c].errorbar(per, r / rc - 1, yerr=e / rc, fmt="o-", color=G.species_color("H2"), ms=2.5,
                                        lw=0.8, elinewidth=0.5)
        for r_ in range(3):
            axes[r_, c].axvspan(19.5, 24.5, color=G.RULE, zorder=0)
            axes[r_, c].axhline(0, color=G.LIGHT, lw=0.5)
        axes[0, c].set_title(f"s$_O$ = {s}, P{P}", loc="left", fontsize=8.5)
        axes[2, c].set_xlabel("period index")
    axes[0, 0].set_ylabel("u$_r$ amp. at 150 km [m/s]")
    axes[1, 0].set_ylabel("exospheric cycle-mean $\\Delta$T [K]")
    axes[2, 0].set_ylabel("H$_2$ escape excess")
    axes[0, 0].legend(fontsize=5.5)
    fig.suptitle(f"Ring-down, s$_O$ = {s} (shaded: periods 20-24, w' = 0)", x=0.01, ha="left", fontsize=8.5)
    axes[1, 0].legend(fontsize=5.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "ringdown_s020")


def main():
    G.setup()
    for atm in ATMS:
        fig_vdf(atm)
        fig_wave_profiles(atm)
        _fields_map(alias(atm, 0.2, 330), "time")
        _fields_map(alias(atm, 0.2, 330), "phase")
    fig_moments(330)
    fig_budget()
    fig_escape_series()
    fig_escape_summary()
    fig_escape_vdf()
    fig_propagation()
    fig_heating()
    fig_ringdown()


if __name__ == "__main__":
    main()
