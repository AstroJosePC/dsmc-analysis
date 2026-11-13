"""Figures for the ubc_hold_carberry report (Graphite Telemetry style; H2 orange is the only species ink).

    MPLBACKEND=Agg /Users/josepc/miniforge/envs/dsmc/bin/python -B docs/reports/ubc_hold_carberry/scripts/make_figures.py [name ...]

All data come from analysis.py, which loads the analysis-tier files of campaigns h2_hold (verification and
campaign runs, collisionless dsmclib.exosphere references) and h2_ubc_rtop (sweep). Panels whose inputs are not available yet carry a 'PENDING'
annotation instead of failing. Writes figures/<name>.{pdf,png}.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))
import analysis as an  # noqa: E402
import style as S  # noqa: E402
from analysis import PAPER  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
S.setup()
A = an.get()
SP = A.source_params()
AN = A.analytic()
R0 = SP["r0"]
W = SP["W"]
ZS = [2500, 5000, 10000, 20000, 40000]
PAPER_LABEL = "Carberry Mogan et al. 2021 (digitised)"


def tau90(z_actual_km):
    return float(AN.tau90[A.ztop_index(AN, z_actual_km)])


def ztop_of(r):
    if r.group == "verification":
        return A._ver_top_km()
    return r.z_top_actual_km


def plain_log(ax, axis="x", ticks=None, fmt="{:g}"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    a.set_minor_formatter(NullFormatter())


def sci_fmt(v, _=None):
    if v <= 0:
        return ""
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10**e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else ""


def binned_rate(t, counts, dt, weight, per_decade=6, t_min=None):
    edges = an.log_bins(t, per_decade, t_min)
    tc, r, e = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (t > lo) & (t <= hi)
        if not s.any():
            continue
        c, d = counts[s].sum(), dt[s].sum()
        tc.append(np.sqrt(lo * hi))
        r.append(c * weight / d)
        e.append(np.sqrt(max(c, 1)) * weight / d)
    return np.array(tc), np.array(r), np.array(e)


# ----------------------------------------------------------------------------------------------
def fig_escape_rtop():
    """Fig. 1 analogue: Phi / phi_J0 vs r_top / r0 (sweep UBC 1-3, campaign UBC 5) and collisionless references."""
    fig = plt.figure(figsize=(S.WIDTH, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=0.085, right=0.985, top=0.80, bottom=0.15, wspace=0.27)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    pj = SP["phiJ0"]
    x = AN.x.values
    for u, name in ((1, "reflect"), (2, "ballistic"), (3, "free_escape")):
        ax.plot(x, AN[f"esc_{name}"].values / pj, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS if u != 3 else (0, (1, 1.4)),
                zorder=1)
    xf = 2.3
    ax.text(xf * 0.94, float(np.interp(xf, x, AN.esc_free_escape.values)) / pj * 0.82, "collisionless\nfree escape",
            ha="right", va="top", fontsize=6.3, color=S.GREY, linespacing=1.15)
    ax.text(1.3, 0.95, "collisionless reflect, ballistic", ha="left", va="top", fontsize=6.3, color=S.GREY)
    ratio_pts = {}
    for u in (1, 2, 3):
        for dr in (250, 1000):
            xs, ys, es = [], [], []
            for z in ZS:
                r = A.get("sweep", u, z, dr)
                if r is None or not r.has_run:
                    continue
                d = A.reduce(r)
                xs.append(d.attrs["r_top"] / R0)
                ys.append(d.attrs["esc_rate"] / pj)
                es.append(ys[-1] / np.sqrt(max(d.attrs["esc_npart"], 1)))
                ratio_pts.setdefault((u, dr), {})[z] = (xs[-1], d.attrs["esc_rate"], d.attrs["esc_npart"])
            if not xs:
                continue
            if dr == 250:
                ax.plot(xs, ys, **S.line_kw(u, lw=1.1), zorder=3)
                ax.errorbar(xs, ys, yerr=es, **S.marker_kw(u, ms=4.6), elinewidth=0.7, zorder=4)
            else:
                ax.errorbar(np.array(xs) * 1.045, ys, yerr=es, **S.marker_kw(u, ms=3.0, mfc=S.PAPER), elinewidth=0.6,
                            alpha=0.85, zorder=4)
    c5 = [A.get("campaign", 5, z, 250) for z in (2500, 10000, 40000)]
    c5 = [A.reduce(r) for r in c5 if r is not None and r.has_run]
    if c5:
        xs = [d.attrs["r_top"] / R0 for d in c5]
        ys = [d.attrs["esc_rate"] / pj for d in c5]
        es = [y / np.sqrt(max(d.attrs["esc_npart"], 1)) for y, d in zip(ys, c5)]
        ax.errorbar(np.array(xs) * 0.955, ys, yerr=es, **S.marker_kw(5, ms=4.4), elinewidth=0.7, zorder=5)
    else:
        S.pending(ax, "explicit RK4 (campaign)", x=0.80, y=0.60, size=6.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.2, 16.5); ax.set_ylim(0.75, 200)
    plain_log(ax, "x", [1.5, 2, 3, 5, 7, 10, 15])
    plain_log(ax, "y", [1, 2, 5, 10, 20, 50, 100, 200])
    hill = AN.attrs["hill"] / R0
    ax.annotate(f"$r_\\mathrm{{Hill}}$ = {hill:.0f} $r_0$", xy=(16.4, 150), xytext=(10.5, 150), fontsize=6.4,
                color=S.GRAPHITE, va="center", ha="right",
                arrowprops=dict(arrowstyle="-|>", color=S.GRAPHITE, lw=0.7, mutation_scale=6))
    ax.set_xlabel("upper boundary $r_\\mathrm{top}/r_0$")
    ax.set_ylabel("$\\Phi\\,/\\,\\phi_{\\mathrm{J},0}$")
    ax.set_title("a  escape rate vs upper boundary", fontsize=8.5)
    S.tick_mono(ax)

    # b: Kepler / reflect at equal r_top, dr
    for dr, ms, mfc in ((250, 4.6, None), (1000, 3.0, S.PAPER)):
        if (2, dr) not in ratio_pts or (1, dr) not in ratio_pts:
            continue
        xs, ys, es = [], [], []
        for z, (xx, p2, k2) in ratio_pts[(2, dr)].items():
            if z in ratio_pts[(1, dr)]:
                _, p1, k1 = ratio_pts[(1, dr)][z]
                xs.append(xx * (1.0 if dr == 250 else 1.045)); ys.append(p2 / p1)
                es.append(p2 / p1 * np.sqrt(1 / k1 + 1 / k2))
        kw = S.marker_kw(2, ms=ms)
        if mfc:
            kw["mfc"] = mfc
        bx.errorbar(xs, ys, yerr=es, **kw, elinewidth=0.7, zorder=4)
    v = A.verification()["analytic"]
    if "E7a_kepler" in v and "E9_reflect" in v:
        s7, s9 = A.escape_series(A.v("E7a_kepler")), A.escape_series(A.v("E9_reflect"))
        rr, ee, _, _ = an.window_ratio(s7, s9, an.T0_VER)
        xv = (SP["R"] + v["E7a_kepler"]["r_top_km"] * 1e3) / R0
        bx.errorbar([xv], [rr], yerr=[ee], color=S.H2, marker="s", ms=4.6, mfc=S.PAPER, mec=S.H2, mew=0.9, ls="none",
                    elinewidth=0.7, zorder=4)
        bx.text(xv * 1.08, rr, "E7a / E9, no collisions\n($z_\\mathrm{top}$ 2 627 km)", fontsize=6.0,
                color=S.GRAPHITE, va="center", ha="left", linespacing=1.15)
    bx.plot(x, AN.esc_ballistic.values / AN.esc_reflect.values, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=1)
    bx.text(13.5, float(AN.esc_ballistic.values[-1] / AN.esc_reflect.values[-1]) + 0.006,
            "collisionless\n(apoapsis $>r_\\mathrm{Hill}$ escapes)", fontsize=6.0, color=S.GREY, ha="right", va="bottom",
            linespacing=1.15)
    bx.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    bx.set_xscale("log"); bx.set_xlim(1.2, 16.5); bx.set_ylim(0.86, 1.18)
    plain_log(bx, "x", [1.5, 2, 3, 5, 7, 10, 15])
    bx.set_xlabel("upper boundary $r_\\mathrm{top}/r_0$")
    bx.set_ylabel("$\\Phi_\\mathrm{Kepler}\\,/\\,\\Phi_\\mathrm{reflect}$")
    bx.set_title("b  Kepler hold relative to reflect", fontsize=8.5)
    S.tick_mono(bx)
    handles = [Line2D([], [], **S.line_kw(u, lw=1.1), marker=S.UBC[u]["marker"], ms=4.2, mfc=S.UBC[u]["mfc"],
                      mec=S.UBC[u]["ink"], label=S.UBC[u]["label"]) for u in (1, 2, 3)]
    handles += [Line2D([], [], color=S.GRAPHITE, marker="o", ms=3.0, mfc=S.PAPER, ls="none",
                       label="small hollow: $\\Delta r$ = 1000 km"),
                Line2D([], [], **S.marker_kw(5, ms=4.2), label="explicit RK4 (campaign, 2e6 s)"),
                Line2D([], [], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, label="collisionless (dsmclib.exosphere)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.995), ncol=3, fontsize=6.6,
               columnspacing=1.3, handletextpad=0.5)
    return S.save(fig, "fig1_escape_rtop")


# ----------------------------------------------------------------------------------------------
def fig_profiles():
    """Fig. 2 analogue: n/n0, T/T0 vs r/r0 per UBC at the largest common z_top, and ratios to reflect."""
    zc = None
    for z in sorted(ZS, reverse=True):
        if all(A.get("sweep", u, z, 250) is not None and A.get("sweep", u, z, 250).has_run for u in (1, 2, 3)):
            zc = z
            break
    fig, axes = plt.subplots(2, 2, figsize=(S.WIDTH, 5.3), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.94, top=0.865, bottom=0.085, wspace=0.10, hspace=0.42)
    if zc is None:
        for ax in axes.ravel():
            S.pending(ax, "no z_top with UBC 1-3 finished")
        return S.save(fig, "fig2_profiles")
    D = {u: A.reduce(A.get("sweep", u, zc, 250)) for u in (1, 2, 3)}
    c5, c1 = A.get("campaign", 5, zc, 250), A.get("campaign", 1, zc, 250)
    if c5 is not None and c5.has_run and c1 is not None and c1.has_run:
        D[5] = A.reduce(c5)
    ref = D[1]
    ref5 = A.reduce(c1) if 5 in D else None   # explicit is compared with the campaign reflect run (same 2e6 s)
    for u, d in D.items():
        ok = d.p_num_part.values >= 10
        y = d.radius.values[ok] / R0
        axes[0, 0].plot(d.p_num_dens.values[ok] / SP["n0"], y, **S.line_kw(u), zorder=3 if u != 1 else 4)
        axes[0, 1].plot(d.p_temp_trans_corr.values[ok] / SP["T0"], y, **S.line_kw(u), zorder=3 if u != 1 else 4)
        if u == 1:
            continue
        for k, f in enumerate(("num_dens", "temp_trans_corr")):
            rr, ee, zz, okc = an.profile_compare(d, ref5 if u == 5 else ref, f)
            if u == 5:   # campaign Kepler / campaign reflect, same 2e6 s: the like-for-like partner of explicit
                d2c = A.reduce(A.get("campaign", 2, zc, 250))
                r2c, _, _, ok2c = an.profile_compare(d2c, ref5, f)
                axes[1, k].plot(r2c[ok2c], (d2c.radius.values / R0)[ok2c], **S.line_kw(2, lw=0.7), alpha=0.8, zorder=3)
            yy = d.radius.values / R0
            axes[1, k].fill_betweenx(yy[okc], (rr - ee)[okc], (rr + ee)[okc], color=S.UBC[u]["ink"], alpha=0.14, lw=0)
            axes[1, k].plot(rr[okc], yy[okc], **S.line_kw(u), zorder=3)
    # analytic collisionless reference (reflect) for orientation
    rr_ = np.linspace(R0 * 1.0005, ref.attrs["r_top"] * 0.999, 70)
    pa = an.exosphere.profile(rr_, SP["n0"], SP["T0"], SP["m"], R0, ref.attrs["r_top"], ubc="reflect",
                              planet_mass=SP["M"], n_v=400)
    axes[0, 0].plot(pa.num_dens.values / SP["n0"], rr_ / R0, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=2)
    axes[0, 1].plot(pa.temp_trans.values / SP["T0"], rr_ / R0, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=2)
    ex = an.exobase_rows(ref, SP, ref.attrs["esc_rate"])
    for ax in axes.ravel():
        for key, ls, lab in (("Kn", (0, (3, 2)), "Kn = 1"), ("KnZero", (0, (5, 1.5, 1, 1.5)), "Kn$_0$ = 1")):
            if ex[key] is not None:
                ax.axhline(ex[key]["rx_r0"], color=S.GRAPHITE, lw=0.7, ls=ls, zorder=1)
    for ax in axes[:, 1]:
        for key, lab in (("Kn", "Kn = 1"), ("KnZero", "Kn$_0$ = 1")):
            if ex[key] is not None:
                ax.text(1.01, ex[key]["rx_r0"], lab, transform=ax.get_yaxis_transform(), fontsize=6.0,
                        color=S.GRAPHITE, ha="left", va="center", clip_on=False)
    nmin = min(float(np.nanmin(np.where(d.p_num_part.values >= 10, d.p_num_dens.values, np.nan))) for d in D.values())
    tmin = min(float(np.nanmin(np.where(d.p_num_part.values >= 10, d.p_temp_trans_corr.values, np.nan))) for d in D.values())
    axes[0, 0].set_xscale("log"); axes[0, 0].set_xlim(10 ** np.floor(np.log10(nmin / SP["n0"])), 1.5)
    axes[0, 1].set_xlim(np.floor(10 * tmin / SP["T0"]) / 10, 1.08)
    axes[1, 0].set_xlim(0.3, 1.25); axes[1, 1].set_xlim(0.6, 1.15)
    for ax in axes[1]:
        ax.axvline(1, color=S.LIGHT, lw=0.6, zorder=0)
    top = ref.attrs["r_top"] / R0
    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.set_ylim(1, top * 1.01)
        plain_log(ax, "y", [1, 1.5, 2, 3, 5, 7, 10])
        S.tick_mono(ax)
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    axes[0, 0].set_xlabel("number density $n/n_0$")
    axes[0, 1].set_xlabel("temperature $T/T_0$ (radial bulk flow removed)")
    axes[1, 0].set_xlabel("$n\\,/\\,n_\\mathrm{reflect}$")
    axes[1, 1].set_xlabel("$T\\,/\\,T_\\mathrm{reflect}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("radial distance $r/r_0$")
    for ax, t in zip(axes.ravel(), ("a  density", "b  temperature", "c  density relative to reflect",
                                    "d  temperature relative to reflect")):
        ax.set_title(t, fontsize=8.5)
    if 5 not in D:
        S.pending(axes[1, 1], "explicit RK4 (campaign, $z_\\mathrm{top}$ 40 000 km):\nrun file built at the end of the 2e6 s run",
                  x=0.31, y=0.22, size=5.6)
    handles = [Line2D([], [], **S.line_kw(u), label=S.UBC[u]["label"]) for u in (1, 2, 3, 5)]
    handles += [Line2D([], [], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, label="collisionless reflect")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 0.995), ncol=3, fontsize=6.6,
               columnspacing=1.3, handletextpad=0.5)
    fig.text(0.985, 0.995, f"H$_2$, $z_\\mathrm{{top}}$ = {zc:,} km, $\\Delta r$ = 250 km, sweep (3e5 s)".replace(",", " "),
             ha="right", va="top", fontsize=6.6, color=S.GRAPHITE)
    if 5 in D:
        fig.text(0.985, 0.955, "explicit RK4: campaign (2e6 s); in c, d against the campaign reflect run,\n"
                 "with the campaign Kepler hold (thin solid) as its same-length partner",
                 ha="right", va="top", fontsize=6.0, color=S.GRAPHITE, linespacing=1.2)
    return S.save(fig, "fig2_profiles")


# ----------------------------------------------------------------------------------------------
def fig_hold_time():
    """Fig. 3 analogue: hold-in / return rates, their ratio, held count and Phi(ballistic)/Phi(reflect) vs time;
    E8 bug-2 and RK4 return-delay panels."""
    fig, axes = plt.subplots(3, 2, figsize=(S.WIDTH, 7.6))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.925, bottom=0.06, wspace=0.27, hspace=0.50)
    (ax_r, ax_q), (ax_h, ax_p), (ax_8, ax_d) = axes
    ver = {"E7a_kepler": 2, "E7b_explicit": 5}
    camp = {u: A.get("campaign", u, 40000, 250) for u in (1, 2, 5)}
    z_ver = A._ver_top_km()
    t90v = tau90(z_ver)

    # (a) hold-in and return rates
    for name, u in ver.items():
        h = A.hold(A.v(name))
        if h is None:
            continue
        t = h.time.values
        for c, alpha, mk in (("n_hold_in", 1.0, S.UBC[u]["marker"]), ("n_hold_ret", 0.5, S.UBC[u]["marker"])):
            tc, rr, ee = binned_rate(t, h[c].values.astype(float), h.avg_time_interval.values, float(h.part_weight))
            tc, rr = tc[rr > 0], rr[rr > 0]
            ax_r.plot(tc, rr, **S.line_kw(u, lw=1.2), alpha=alpha, marker=mk, ms=3.0,
                      mfc=S.UBC[u]["mfc"] if c == "n_hold_in" else S.PAPER, mec=S.UBC[u]["ink"], mew=0.8)
    for u in (2, 5):
        r = camp[u]
        h = A.hold(r) if r is not None and r.has_hold else None
        if h is None or h.n_hold_in.sum() == 0:
            continue
        t = h.time.values
        for c, alpha in (("n_hold_in", 1.0), ("n_hold_ret", 0.5)):
            tc, rr, ee = binned_rate(t, h[c].values.astype(float), h.avg_time_interval.values, float(h.part_weight))
            ok = rr > 0
            ax_r.plot(tc[ok], rr[ok], **S.line_kw(u, lw=0.9), alpha=alpha)
    i_v = A.ztop_index(AN, z_ver)
    i_c = A.ztop_index(AN, 40127.3)
    for i, lab, yo in ((i_v, "$R_\\mathrm{in}$ 2 627 km", 1.18), (i_c, "$R_\\mathrm{in}$ 40 127 km", 0.82)):
        _, _, rin = an.analytic_in_ret(AN, i)
        ax_r.axhline(rin, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=1)
        ax_r.text(2.3e6, rin * yo, f"collisionless {lab}", fontsize=6.0, color=S.GREY, va="bottom" if yo > 1 else "top", ha="right")
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlim(1e3, 2.5e6); ax_r.set_ylim(1e24, 2e29)
    ax_r.yaxis.set_major_formatter(FuncFormatter(sci_fmt)); ax_r.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_r.set_xlabel("time $t$ [s]"); ax_r.set_ylabel("rate [s$^{-1}$]")
    ax_r.set_title("a  rate into hold (solid) and return rate (faint)", fontsize=8.5)
    ax_r.text(0.03, 0.04, "markers: verification,\n$z_\\mathrm{top}$ 2 627 km, no collisions\n"
              "lines: campaign,\n$z_\\mathrm{top}$ 40 127 km, 2e6 s", transform=ax_r.transAxes, ha="left",
              va="bottom", fontsize=5.9, color=S.GRAPHITE, linespacing=1.25)

    # (b) in / return ratio vs t / tau
    for name, u in ver.items():
        h = A.hold(A.v(name))
        if h is None:
            continue
        t = h.time.values
        edges = an.log_bins(t, 6)
        tc, rr, ee = an.binned_ratio(t, h.n_hold_in.values.astype(float), t, h.n_hold_ret.values.astype(float), edges)
        ok = np.isfinite(rr) & (rr < 50)
        ax_q.errorbar(tc[ok] / t90v, rr[ok], yerr=ee[ok], **S.marker_kw(u, ms=3.2), elinewidth=0.6, zorder=4)
        ax_q.plot(tc[ok] / t90v, rr[ok], **S.line_kw(u, lw=0.8), zorder=3)
    live = False
    for u in (2, 5):
        r = camp[u]
        h = A.hold(r) if r is not None and r.has_hold else None
        if h is None or h.n_hold_ret.sum() == 0:
            continue
        t = h.time.values
        tc, rr, ee = an.binned_ratio(t, h.n_hold_in.values.astype(float), t, h.n_hold_ret.values.astype(float),
                                     an.log_bins(t, 6))
        ok = np.isfinite(rr) & (rr < 50)
        ax_q.plot(tc[ok] / tau90(40127.3), rr[ok], **S.line_kw(u, lw=1.0), zorder=3)
        live = True
    for i, tl in ((i_v, None), (i_c, None)):
        tt, q, _ = an.analytic_in_ret(AN, i)
        ok = np.isfinite(q)
        ax_q.plot(tt[ok] / float(AN.tau90[i]), q[ok], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=1)
    ax_q.plot(PAPER["fig3_t"], PAPER["fig3_in_ret"], **S.PAPER_MARK, zorder=5)
    ax_q.text(PAPER["fig3_t"][0] * 0.9, PAPER["fig3_in_ret"][0], "Carberry Mogan\n2021 ($t/t_\\mathrm{orb}$)",
              fontsize=6.0, color=S.GRAPHITE, ha="right", va="center", linespacing=1.15)
    ax_q.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_q.set_xscale("log"); ax_q.set_yscale("log"); ax_q.set_xlim(2e-4, 8); ax_q.set_ylim(0.9, 30)
    plain_log(ax_q, "y", [1, 2, 5, 10, 20])
    ax_q.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.001 else sci_fmt(v)))
    ax_q.set_xlabel("$t/\\tau_{90}$  (paper: $t/t_\\mathrm{orb}$)")
    ax_q.set_ylabel("rate into hold / return rate")
    ax_q.set_title("b  ballistic rate / return rate", fontsize=8.5)
    if not live:
        ax_q.text(0.03, 0.06, "campaign: PENDING (no returns yet)", transform=ax_q.transAxes, fontsize=6.0,
                  color=S.GREY)

    # (c) held molecules vs t
    for name, u in list(ver.items()) + [("E8_main_kepler", 2)]:
        h = A.hold(A.v(name))
        if h is None:
            continue
        kw = S.line_kw(u, lw=1.1) if name != "E8_main_kepler" else dict(color=S.CHARCOAL, lw=1.0, ls=(0, (1, 1.2)))
        ax_h.plot(h.time.values, h.held_end.values * float(h.part_weight), **kw, zorder=3)
    ax_h.text(1.6e3, 6e32, "E8 main (bug 2)", fontsize=6.0, color=S.CHARCOAL, ha="left")
    for z in (10000, 20000, 40000):
        r = A.get("sweep", 2, z, 250)
        if r is None or not r.has_run:
            continue
        d = A.reduce(r)
        ok = d.held_t.values > 0
        ax_h.plot(d.time.values[ok], d.held_t.values[ok], **S.line_kw(2, lw=0.9), alpha=0.75, zorder=2)
        ax_h.text(d.time.values[-1] * 1.1, d.held_t.values[-1] * {10000: 1.3, 20000: 1.0, 40000: 0.5}.get(z, 0.8), f"{z // 1000}k",
                  color=S.H2, va="center", fontproperties=S.mono(5.9), zorder=6,
                  bbox=dict(fc="white", ec="none", pad=0.6))
    for u in (2, 5):
        r = camp[u]
        h = A.hold(r) if r is not None and r.has_hold else None
        if h is None:
            continue
        ok = h.held_end.values > 0
        ax_h.plot(h.time.values[ok], h.held_end.values[ok] * float(h.part_weight), **S.line_kw(u, lw=1.3), zorder=3)
    for i in sorted({i_v, i_c} | {A.ztop_index(AN, z) for z in (10127.3, 20127.3)}):
        ok = AN.held_t.values[i] > 0
        ax_h.plot(AN.t.values[ok], AN.held_t.values[i][ok], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=1)
    ax_h.text(3e6, float(AN.held_steady[i_v]) * 1.15, "collisionless held($t$)\n2.6k, 10k, 20k, 40k km",
              fontsize=6.0, color=S.GREY, ha="right", va="bottom", linespacing=1.15)
    ax_h.set_xscale("log"); ax_h.set_yscale("log")
    ax_h.set_xlim(1e3, 1e7); ax_h.set_ylim(1e30, 2e33)
    ax_h.xaxis.set_major_formatter(FuncFormatter(sci_fmt)); ax_h.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_h.set_xlabel("time $t$ [s]"); ax_h.set_ylabel("molecules held above $r_\\mathrm{top}$")
    ax_h.set_title("c  held reservoir", fontsize=8.5)

    # (d) Phi(ballistic) / Phi(reflect) vs t / tau
    for z in (10000, 20000, 40000):
        for dr in (250,):
            a2, a1 = A.get("sweep", 2, z, dr), A.get("sweep", 1, z, dr)
            if a2 is None or a1 is None or not (a2.has_run and a1.has_run):
                continue
            s2, s1 = A.escape_series(a2), A.escape_series(a1)
            t = s2.time.values
            edges = an.log_bins(t, 4, t_min=4e3)
            tc, rr, ee = an.binned_ratio(t, s2.npart.values, s1.time.values, s1.npart.values, edges)
            x = tc / tau90(a2.z_top_actual_km)
            ax_p.errorbar(x, rr, yerr=ee, **S.marker_kw(2, ms=3.0), elinewidth=0.6, alpha=0.9, zorder=3)
            ax_p.plot(x, rr, **S.line_kw(2, lw=0.7), alpha=0.6, zorder=2)
            ax_p.text(x[-1] * 1.25, rr[-1], f"{z // 1000}k", fontsize=5.9, color=S.H2, va="center",
                      fontproperties=S.mono(5.9))
    s9 = A.escape_series(A.v("E9_reflect"))
    for name, u in ver.items():
        s = A.escape_series(A.v(name))
        if s is None or s9 is None:
            continue
        t = s.time.values
        tc, rr, ee = an.binned_ratio(t, s.npart.values, s9.time.values, s9.npart.values, an.log_bins(t, 3, t_min=4e3))
        ax_p.errorbar(tc / t90v, rr, yerr=ee, **S.marker_kw(u, ms=3.0, mfc=S.PAPER), elinewidth=0.6, zorder=3)
    if camp[1] is not None and camp[1].has_hold:
        s1 = A.escape_series(camp[1])
        for u in (2, 5):
            if camp[u] is None or not camp[u].has_hold:
                continue
            s = A.escape_series(camp[u])
            t = s.time.values
            n = min(len(t), len(s1.time))
            tc, rr, ee = an.binned_ratio(t[:n], s.npart.values[:n], s1.time.values[:n], s1.npart.values[:n],
                                         an.log_bins(t[:n], 2))
            ok = np.isfinite(rr)
            ax_p.errorbar(tc[ok] / tau90(40127.3), rr[ok], yerr=ee[ok], **S.marker_kw(u, ms=3.4), elinewidth=0.6,
                          zorder=4)
    for i in (i_v, i_c):
        ax_p.hlines(float(AN.escz_ballistic[i] / AN.escz_reflect[i]), 2e-4, 8, color=S.GREY, lw=0.8,
                    ls=S.ANALYTIC_LS, zorder=1)
    ax_p.text(7, float(AN.escz_ballistic[i_c] / AN.escz_reflect[i_c]) + 0.012, "collisionless", fontsize=6.0,
              color=S.GREY, ha="right")
    ax_p.plot(PAPER["fig3_t"], PAPER["fig3_phi_ratio"], **S.PAPER_MARK, zorder=5)
    ax_p.text(PAPER["fig3_t"][0] * 0.85, PAPER["fig3_phi_ratio"][0], "Carberry Mogan\n2021 ($t/t_\\mathrm{orb}$)",
              fontsize=6.0, color=S.GRAPHITE, ha="right", va="center", linespacing=1.15)
    ax_p.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_p.set_xscale("log"); ax_p.set_xlim(2e-4, 8); ax_p.set_ylim(0.6, 1.4)
    ax_p.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.001 else sci_fmt(v)))
    ax_p.set_xlabel("$t/\\tau_{90}$  (paper: $t/t_\\mathrm{orb}$)")
    ax_p.set_ylabel("$\\Phi_\\mathrm{ballistic}\\,/\\,\\Phi_\\mathrm{reflect}$")
    ax_p.set_title("d  ballistic escape relative to reflect", fontsize=8.5)

    # (e) E8 bug-2
    h8, h7 = A.hold(A.v("E8_main_kepler")), A.hold(A.v("E7a_kepler"))
    if h8 is not None and h7 is not None:
        ax_8.plot(h8.time.values / 1e3, np.cumsum(h8.n_hold_in.values), color=S.LIGHT, lw=2.6, zorder=1)
        ax_8.plot(h8.time.values / 1e3, h8.held_end.values, color=S.CHARCOAL, lw=1.0, ls=(0, (1, 1.2)), zorder=3)
        ax_8.plot(h7.time.values / 1e3, h7.held_end.values, **S.line_kw(2, lw=1.4), zorder=2)
        ax_8.text(22, float(h8.held_end.values[-1]) * 0.92, "E8 main: held = cumulative entries\n(no returns, bug 2)",
                  fontsize=6.0, color=S.CHARCOAL, ha="left", va="top", linespacing=1.2)
        ax_8.text(58, float(h7.held_end.values[-1]) * 1.12, "E7a fixed Kepler hold", fontsize=6.0, color=S.H2,
                  ha="right", va="bottom")
        ax_8.set_xlim(0, 61); ax_8.set_ylim(0, float(h8.held_end.values[-1]) * 1.08)
    else:
        S.pending(ax_8, "E8 / E7a hold diagnostics")
    ax_8.set_xlabel("time $t$ [ks]"); ax_8.set_ylabel("held computational particles")
    ax_8.set_title("e  E8: main-branch Kepler hold never returns", fontsize=8.5)

    # (f) RK4 - Kepler return delay
    dt = SP["dt"]
    for name, mk in (("E7b_explicit", "D"), ("E7c_explicit_dbg", "o"), ("E7d_explicit_restart", "^")):
        h = A.hold(A.v(name))
        if h is None:
            continue
        ok = h.n_hold_ret.values > 0
        ax_d.plot(h.time.values[ok] / 1e3, h.ret_dt_mean.values[ok], color=S.H2, marker=mk, ms=2.6, mfc=S.PAPER,
                  mec=S.H2, mew=0.7, ls="none", zorder=3)
        ax_d.plot(h.time.values[ok] / 1e3, h.ret_dt_std.values[ok], color=S.GRAPHITE, marker=mk, ms=2.4,
                  mfc=S.PAPER, mec=S.GRAPHITE, mew=0.6, ls="none", zorder=3)
    ax_d.axhline(dt / 2, color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS)
    ax_d.axhline(dt / np.sqrt(12), color=S.GREY, lw=0.8, ls=(0, (1, 1.4)))
    ax_d.text(60, dt / 2 + 0.012, "mean: $\\Delta t/2$", fontsize=6.0, color=S.GREY, ha="right", va="bottom")
    ax_d.text(60, dt / np.sqrt(12) - 0.012, "s.d.: $\\Delta t/\\sqrt{12}$", fontsize=6.0, color=S.GREY, ha="right",
              va="top")
    ax_d.set_xlim(0, 61); ax_d.set_ylim(0.2, 0.6)
    ax_d.set_xlabel("time $t$ [ks]"); ax_d.set_ylabel("RK4 $-$ Kepler return [s]")
    ax_d.set_title("f  explicit return delay per interval", fontsize=8.5)
    ax_d.legend(handles=[Line2D([], [], color=S.H2, marker=m, ms=2.8, mfc=S.PAPER, ls="none", label=l)
                         for m, l in (("D", "E7b"), ("o", "E7c -fcheck"), ("^", "E7d restart"))],
                loc="center right", fontsize=6.0, handletextpad=0.3, borderaxespad=0.3)
    for ax in axes.ravel():
        S.tick_mono(ax)
    handles = [Line2D([], [], **S.line_kw(u), marker=S.UBC[u]["marker"], ms=3.4, mfc=S.UBC[u]["mfc"],
                      mec=S.UBC[u]["ink"], label=S.UBC[u]["label"]) for u in (2, 5)]
    handles += [Line2D([], [], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, label="collisionless (dsmclib.exosphere)"),
                Line2D([], [], **S.PAPER_MARK, label=PAPER_LABEL)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.095, 0.998), ncol=2, fontsize=6.6,
               columnspacing=1.5, handletextpad=0.5)
    return S.save(fig, "fig3_hold_time")


# ----------------------------------------------------------------------------------------------
def fig_verification():
    """Verification E7a/E7b/E9: n_DSMC / n_analytic and corrected T vs altitude (collisionless, z_top 2 627 km)."""
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, 2.7), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.80, bottom=0.16, wspace=0.16)
    runs = (("E9_reflect", 1, "reflect"), ("E7a_kepler", 2, "ballistic"), ("E7b_explicit", 5, "ballistic"))
    any_ = False
    for name, u, ubc in runs:
        r = A.v(name)
        if not r.has_run:
            continue
        d = A.reduce(r)
        na, Ta = d.analytic_num_dens.values, d.analytic_temp_trans.values  # cell-averaged, campaigns/h2_hold
        ok = d.p_num_part.values > 5
        z = d.altitude.values[ok] / 1e3
        rn = d.p_num_dens.values[ok] / na[ok]
        en = d.p_num_dens_sem.values[ok] / na[ok]
        axes[0].fill_betweenx(z, rn - en, rn + en, color=S.UBC[u]["ink"], alpha=0.12, lw=0)
        axes[0].plot(rn, z, **S.line_kw(u), marker=S.UBC[u]["marker"], ms=3.0, mfc=S.UBC[u]["mfc"], mec=S.UBC[u]["ink"])
        axes[1].plot(d.p_temp_trans_corr.values[ok] / Ta[ok], z, **S.line_kw(u), marker=S.UBC[u]["marker"], ms=3.0,
                     mfc=S.UBC[u]["mfc"], mec=S.UBC[u]["ink"])
        any_ = True
    if not any_:
        for ax in axes:
            S.pending(ax, "verification run files")
    for ax in axes:
        ax.axvline(1, color=S.LIGHT, lw=0.6, zorder=0)
        S.tick_mono(ax)
    axes[0].set_xlim(0.985, 1.015); axes[1].set_xlim(0.97, 1.03)
    axes[0].set_xlabel("$n_\\mathrm{DSMC}\\,/\\,n_\\mathrm{collisionless}$")
    axes[1].set_xlabel("$T_\\mathrm{DSMC}\\,/\\,T_\\mathrm{collisionless}$ (bulk flow removed)")
    axes[0].set_ylabel("altitude [km]")
    axes[0].set_title("a  density", fontsize=8.5); axes[1].set_title("b  temperature", fontsize=8.5)
    handles = [Line2D([], [], **S.line_kw(u), marker=S.UBC[u]["marker"], ms=3.0, mfc=S.UBC[u]["mfc"],
                      mec=S.UBC[u]["ink"], label=f"{n.split('_')[0]}  {S.UBC[u]['short']}") for n, u, _ in runs]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.085, 0.995), ncol=3, fontsize=6.6,
               columnspacing=1.5)
    fig.text(0.985, 0.995, f"collisions off, t > {an.T0_VER / 1e3:.0f} ks, cell-volume-averaged analytic", ha="right",
             va="top", fontsize=6.4, color=S.GRAPHITE)
    return S.save(fig, "fig4_verification")


# ----------------------------------------------------------------------------------------------
def fig_production():
    """Campaign (2e6 s, collisions on): explicit RK4 (UBC 5) against the Kepler hold (UBC 2) at every top."""
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 2.75))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.78, bottom=0.17, wspace=0.38)
    ax_e, ax_h, ax_n = axes
    tops = ((2500, "2.6k"), (10000, "10k"), (40000, "40k"))
    greys = {2500: S.LIGHT, 10000: S.GREY, 40000: S.CHARCOAL}
    any_ = False
    for z, lab in tops:
        r2, r5 = A.get("campaign", 2, z, 250), A.get("campaign", 5, z, 250)
        if not (r2.has_run and r5.has_run and r2.has_hold and r5.has_hold):
            continue
        any_ = True
        h2, h5 = A.hold(r2), A.hold(r5)
        d2, d5 = A.reduce(r2), A.reduce(r5)
        # (a) cumulative escapes, explicit / Kepler, with a Poisson band
        c2 = np.cumsum((h2.n_esc_direct + h2.n_esc_orbit).values.astype(float))
        c5 = np.cumsum((h5.n_esc_direct + h5.n_esc_orbit).values.astype(float))
        n = min(len(c2), len(c5))
        t = h2.time.values[:n]
        ok = (c2[:n] > 0) & (c5[:n] > 0)
        q = c5[:n][ok] / c2[:n][ok]
        e = q * np.sqrt(1 / c5[:n][ok] + 1 / c2[:n][ok])
        late = t[ok] >= 5e4   # skip the first 50 ks, where a handful of escapes make the ratio meaningless
        ax_e.fill_between(t[ok][late] / 1e3, (q - e)[late], (q + e)[late], color=S.H2, alpha=0.10, lw=0)
        ax_e.plot(t[ok][late] / 1e3, q[late], color=greys[z], lw=1.0)
        ax_e.text(t[-1] / 1e3 * 1.01, q[-1], lab, fontproperties=S.mono(5.9), color=greys[z], va="center")
        # (b) held molecules: Kepler solid, explicit dashed, collisionless held(t) grey
        for u, h in ((2, h2), (5, h5)):
            okh = h.held_end.values > 0
            ax_h.plot(h.time.values[okh] / 1e3, h.held_end.values[okh] * float(h.part_weight),
                      **S.line_kw(u, lw=1.2))
        i = A.ztop_index(AN, r2.z_top_actual_km)
        tt = AN.t.values
        sel = (tt >= 1e3) & (tt <= 2.2e6)
        ax_h.plot(tt[sel] / 1e3, AN.held_t.values[i][sel], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, zorder=1)
        k0 = int(np.argmax(h2.held_end.values > 0))
        ax_h.text(h2.time.values[k0] / 1e3 * 0.8, float(h2.held_end.values[k0]) * float(h2.part_weight), lab,
                  fontproperties=S.mono(5.9), color=S.H2, va="center", ha="right")
        # (c) density profile, explicit / Kepler, +-1 sigma
        rn, en, zn, okp = an.profile_compare(d5, d2, "num_dens")
        y = d2.radius.values / R0
        zo = {2500: 4, 10000: 3, 40000: 2}[z]
        ax_n.fill_betweenx(y[okp], (rn - en)[okp], (rn + en)[okp], color=S.H2, alpha=0.10, lw=0, zorder=1)
        ax_n.plot(rn[okp], y[okp], color=greys[z], lw=1.0, zorder=zo)
    if not any_:
        for ax in axes:
            S.pending(ax, "no finished campaign pair")
    ax_e.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_e.set_xlim(0, 2.3e3); ax_e.set_ylim(0.95, 1.05)
    ax_e.set_xlabel("simulated time [ks]"); ax_e.set_ylabel("escapes, explicit / Kepler")
    ax_h.set_xscale("log"); ax_h.set_yscale("log"); ax_h.set_xlim(1, 4e3); ax_h.set_ylim(1e28, 1e33)
    ax_h.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_h.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_h.set_xlabel("simulated time [ks]"); ax_h.set_ylabel("molecules held above $r_\\mathrm{top}$")
    ax_n.axvline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_n.set_xlim(0.8, 1.2); ax_n.set_yscale("log")
    plain_log(ax_n, "y", [1, 1.5, 2, 3, 5, 7, 10])
    ax_n.set_xlabel("$n_\\mathrm{explicit}\\,/\\,n_\\mathrm{Kepler}$"); ax_n.set_ylabel("radial distance $r/r_0$")
    for ax, t in zip(axes, ("a  escapes", "b  held reservoir", "c  density")):
        ax.set_title(t, fontsize=8.5)
        S.tick_mono(ax)
    handles = [Line2D([], [], **S.line_kw(u), label=S.UBC[u]["label"]) for u in (2, 5)]
    handles += [Line2D([], [], color=S.GREY, lw=0.8, ls=S.ANALYTIC_LS, label="collisionless held($t$)")]
    handles += [Line2D([], [], color=greys[z], lw=1.0, label=f"ratio, {lab} km") for z, lab in tops]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995), ncol=3, fontsize=6.4,
               columnspacing=1.4, handletextpad=0.5)
    fig.text(0.985, 0.995, "h2_hold campaign, 2e6 s, collisions on, $\\Delta r$ = 250 km", ha="right",
             va="top", fontsize=6.4, color=S.GRAPHITE)
    return S.save(fig, "fig5_production")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["escape_rtop", "profiles", "hold_time", "verification", "production"]
    for name in todo:
        print(globals()[f"fig_{name}"]())
