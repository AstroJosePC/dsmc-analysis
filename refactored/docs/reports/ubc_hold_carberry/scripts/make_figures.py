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
from matplotlib.legend_handler import HandlerTuple
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
KM = lambda z: f"{z:,.0f}".replace(",", " ")   # noqa: E731  thousands separated by a thin gap
TOP_LABEL = {2500: "2.6k", 10000: "10k", 20000: "20k", 40000: "40k"}   # nominal top -> curve label (altitude, km)


def top_tag(ax, x, y, z, **kw):
    """Curve label: top altitude in km, monospaced, in the ballistic ink, on a paper patch."""
    opt = dict(color=S.H2, va="center", fontproperties=S.mono(5.9), zorder=6,
               bbox=dict(fc=S.PAPER, ec="none", pad=0.6))
    opt.update(kw)
    ax.text(x, y, TOP_LABEL[z], **opt)


def fig_escape_rtop():
    """Fig. 1 analogue: Phi / phi_J0 vs r_top / r0 (sweep UBC 1-3, campaign UBC 5) and collisionless references."""
    fig = plt.figure(figsize=(S.WIDTH, 3.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=0.085, right=0.985, top=0.735, bottom=0.135, wspace=0.27)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    pj = SP["phiJ0"]
    x = AN.x.values
    for name in ("reflect", "ballistic", "free_escape"):
        ax.plot(x, AN[f"esc_{name}"].values / pj, **S.ANALYTIC, zorder=1)
    xf = 2.3
    ax.text(xf * 0.94, float(np.interp(xf, x, AN.esc_free_escape.values)) / pj * 0.82, "collisionless\nfree escape",
            ha="right", va="top", fontsize=6.0, color=S.GREY, linespacing=1.15)
    ax.text(1.3, 0.88, "collisionless reflect, ballistic", ha="left", va="top", fontsize=6.0, color=S.GREY)
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
                ax.plot(xs, ys, **S.line_kw(u, "sweep"), zorder=3)
                ax.errorbar(xs, ys, yerr=es, **S.marker_kw(u, "sweep", ms=S.MS[250]), elinewidth=0.7, zorder=4)
            else:
                ax.errorbar(np.array(xs) * 1.045, ys, yerr=es, **S.marker_kw(u, "sweep", ms=S.MS[1000]),
                            elinewidth=0.6, zorder=4)
    c5 = [A.get("campaign", 5, z, 250) for z in (2500, 10000, 40000)]
    c5 = [A.reduce(r) for r in c5 if r is not None and r.has_run]
    if c5:
        xs = [d.attrs["r_top"] / R0 for d in c5]
        ys = [d.attrs["esc_rate"] / pj for d in c5]
        es = [y / np.sqrt(max(d.attrs["esc_npart"], 1)) for y, d in zip(ys, c5)]
        ax.errorbar(np.array(xs) * 0.955, ys, yerr=es, **S.marker_kw(5, "campaign", ms=S.MS[250]), elinewidth=0.7,
                    zorder=5)
    else:
        S.pending(ax, "explicit RK4 (campaign)", x=0.80, y=0.60, size=6.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.2, 16.5); ax.set_ylim(0.62, 200)
    plain_log(ax, "x", [1.5, 2, 3, 5, 7, 10, 15])
    plain_log(ax, "y", [1, 2, 5, 10, 20, 50, 100, 200])
    hill = AN.attrs["hill"] / R0
    ax.annotate(f"$r_\\mathrm{{Hill}}$ = {hill:.0f} $r_0$", xy=(16.4, 150), xytext=(10.5, 150), fontsize=6.0,
                color=S.GRAPHITE, va="center", ha="right",
                arrowprops=dict(arrowstyle="-|>", color=S.GRAPHITE, lw=0.7, mutation_scale=6))
    ax.set_xlabel("upper boundary $r_\\mathrm{top}/r_0$")
    ax.set_ylabel("$\\Phi\\,/\\,\\phi_{\\mathrm{J},0}$")
    S.tick_mono(ax)

    # b: Kepler / reflect at equal r_top, dr
    for dr in (250, 1000):
        if (2, dr) not in ratio_pts or (1, dr) not in ratio_pts:
            continue
        xs, ys, es = [], [], []
        for z, (xx, p2, k2) in ratio_pts[(2, dr)].items():
            if z in ratio_pts[(1, dr)]:
                _, p1, k1 = ratio_pts[(1, dr)][z]
                xs.append(xx * (1.0 if dr == 250 else 1.045)); ys.append(p2 / p1)
                es.append(p2 / p1 * np.sqrt(1 / k1 + 1 / k2))
        bx.errorbar(xs, ys, yerr=es, **S.marker_kw(2, "sweep", ms=S.MS[dr]), elinewidth=0.7, zorder=4)
    v = A.verification()["analytic"]
    if "E7a_kepler" in v and "E9_reflect" in v:
        s7, s9 = A.escape_series(A.v("E7a_kepler")), A.escape_series(A.v("E9_reflect"))
        rr, ee, _, _ = an.window_ratio(s7, s9, an.T0_VER)
        xv = (SP["R"] + v["E7a_kepler"]["r_top_km"] * 1e3) / R0
        bx.errorbar([xv], [rr], yerr=[ee], **S.marker_kw(2, "verification", ms=S.MS[250]), elinewidth=0.7, zorder=4)
    bx.plot(x, AN.esc_ballistic.values / AN.esc_reflect.values, **S.ANALYTIC, zorder=1)
    bx.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    bx.set_xscale("log"); bx.set_xlim(1.2, 16.5); bx.set_ylim(0.86, 1.18)
    plain_log(bx, "x", [1.5, 2, 3, 5, 7, 10, 15])
    bx.set_xlabel("upper boundary $r_\\mathrm{top}/r_0$")
    bx.set_ylabel("$\\Phi_\\mathrm{Kepler}\\,/\\,\\Phi_\\mathrm{reflect}$")
    S.tick_mono(bx)

    # legends: one per panel, each listing only what that panel draws
    ha = [S.handle(u, "sweep", ms=4.0) for u in (1, 2, 3)]
    ha += [S.handle(5, "campaign", line=False, ms=4.0, label="explicit RK4, campaign ($2\\times10^6$ s)"),
           Line2D([], [], color=S.CHARCOAL, marker="o", ms=S.MS[1000], ls="none",
                  label="small markers: $\\Delta r$ = 1000 km"),
           S.analytic_handle()]
    S.panel(ax, "a  escape rate vs upper boundary", ha, ncol=2, rows=3)
    big = Line2D([], [], **S.marker_kw(2, "sweep", ms=4.0))
    small = Line2D([], [], **S.marker_kw(2, "sweep", ms=S.MS[1000]))
    hb = [(big, small),
          Line2D([], [], **S.marker_kw(2, "verification", ms=4.0)),
          S.analytic_handle()]
    S.panel(bx, "b  Kepler hold relative to reflect", hb, rows=3,
            labels=["sweep ($\\Delta r$ = 250 km; small: 1000 km)", "collisionless verification (E7a / E9)",
                    "collisionless, analytic"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    S.header(fig, "H$_2$  ·  collisional sweep, $3\\times10^5$ s  ·  bars: Poisson $1\\sigma$", 0.085)
    return S.save(fig, "fig1_escape_rtop")


# ----------------------------------------------------------------------------------------------
def fig_profiles():
    """Fig. 2 analogue: n/n0, T/T0 vs r/r0 per UBC at the largest common z_top, and ratios to reflect."""
    zc = None
    for z in sorted(ZS, reverse=True):
        if all(A.get("sweep", u, z, 250) is not None and A.get("sweep", u, z, 250).has_run for u in (1, 2, 3)):
            zc = z
            break
    fig, axes = plt.subplots(2, 2, figsize=(S.WIDTH, 5.75), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.94, top=0.875, bottom=0.08, wspace=0.10, hspace=0.56)
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
        ds = "campaign" if u == 5 else "sweep"
        ok = d.p_num_part.values >= 10
        y = d.radius.values[ok] / R0
        zo = {1: 4, 5: 5}.get(u, 3)
        S.plot(axes[0, 0], d.p_num_dens.values[ok] / SP["n0"], y, u, ds, zorder=zo)
        S.plot(axes[0, 1], d.p_temp_trans_corr.values[ok] / SP["T0"], y, u, ds, zorder=zo)
        if u == 1:
            continue
        for k, f in enumerate(("num_dens", "temp_trans_corr")):
            rr, ee, zz, okc = an.profile_compare(d, ref5 if u == 5 else ref, f)
            if u == 5:   # campaign Kepler / campaign reflect, same 2e6 s: the like-for-like partner of explicit
                d2c = A.reduce(A.get("campaign", 2, zc, 250))
                r2c, _, _, ok2c = an.profile_compare(d2c, ref5, f)
                S.plot(axes[1, k], r2c[ok2c], (d2c.radius.values / R0)[ok2c], 2, "campaign", zorder=4)
            yy = d.radius.values / R0
            axes[1, k].fill_betweenx(yy[okc], (rr - ee)[okc], (rr + ee)[okc], color=S.UBC[u]["ink"], alpha=0.14, lw=0)
            S.plot(axes[1, k], rr[okc], yy[okc], u, ds, zorder=5 if u == 5 else 3)
    # analytic collisionless reference (reflect) for orientation
    rr_ = np.linspace(R0 * 1.0005, ref.attrs["r_top"] * 0.999, 70)
    pa = an.exosphere.profile(rr_, SP["n0"], SP["T0"], SP["m"], R0, ref.attrs["r_top"], ubc="reflect",
                              planet_mass=SP["M"], n_v=400)
    axes[0, 0].plot(pa.num_dens.values / SP["n0"], rr_ / R0, **S.ANALYTIC, zorder=2)
    axes[0, 1].plot(pa.temp_trans.values / SP["T0"], rr_ / R0, **S.ANALYTIC, zorder=2)
    ex = an.exobase_rows(ref, SP, ref.attrs["esc_rate"])
    for ax in axes.ravel():
        for key, ls in (("Kn", (0, (3, 2))), ("KnZero", (0, (5, 1.5, 1, 1.5)))):
            if ex[key] is not None:
                ax.axhline(ex[key]["rx_r0"], color=S.GRAPHITE, lw=0.6, ls=ls, zorder=1)
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
    if 5 not in D:
        S.pending(axes[1, 1], "explicit RK4 (campaign, $z_\\mathrm{top}$ 40 000 km):\nrun file built at the end of the 2e6 s run",
                  x=0.31, y=0.22, size=5.6)
    # row legends: a-b absolute profiles, c-d ratios to the reflect run of the same dataset
    h1 = [S.handle(u, "sweep") for u in (1, 2, 3)]
    h1 += [S.handle(5, "campaign")] if 5 in D else []
    h1 += [S.analytic_handle("collisionless reflect, analytic")]
    S.panel(axes[0, 0], "a  density", h1, ncol=5, rows=1)
    S.panel(axes[0, 1], "b  temperature", rows=1)
    h2 = [S.handle(2, "sweep", label="Kepler hold, sweep"), S.handle(3, "sweep", label="free escape, sweep")]
    if 5 in D:
        h2 += [S.handle(2, "campaign", label="Kepler hold, campaign"),
               S.handle(5, "campaign", label="explicit RK4, campaign")]
    h2 += [S.band_handle()]
    S.panel(axes[1, 0], "c  density relative to reflect", h2, ncol=5, rows=1, columnspacing=1.1)
    S.panel(axes[1, 1], "d  temperature relative to reflect", rows=1)
    S.header(fig, f"H$_2$  ·  top at {KM(zc)} km altitude  ·  $\\Delta r$ = 250 km  ·  fine lines: sweep, "
             "$3\\times10^5$ s  ·  heavy lines and dashed explicit RK4: campaign, $2\\times10^6$ s", 0.085)
    return S.save(fig, "fig2_profiles")


# ----------------------------------------------------------------------------------------------
def fig_hold_time():
    """Fig. 3 analogue: hold-in / return rates, their ratio, held count and Phi(ballistic)/Phi(reflect) vs time;
    frozen-hold (E8) and RK4 return-delay panels."""
    fig, axes = plt.subplots(3, 2, figsize=(S.WIDTH, 7.15))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.06, wspace=0.27, hspace=0.62)
    (ax_r, ax_q), (ax_h, ax_p), (ax_8, ax_d) = axes
    ver = {"E7a_kepler": 2, "E7b_explicit": 5}
    camp = {u: A.get("campaign", u, 40000, 250) for u in (1, 2, 5)}
    z_ver = A._ver_top_km()
    t90v = tau90(z_ver)
    FAINT = 0.42
    ver_lab = f"verification, top {z_ver / 1e3:.1f}k"
    BEFORE_FIX = (0, (4, 1.2, 1, 1.2, 1, 1.2))   # the Kepler hold before the fix (E8): Kepler ink, broken stroke

    # (a) hold-in and return rates
    for name, u in ver.items():
        h = A.hold(A.v(name))
        if h is None:
            continue
        t = h.time.values
        for c, alpha in (("n_hold_in", 1.0), ("n_hold_ret", FAINT)):
            tc, rr, ee = binned_rate(t, h[c].values.astype(float), h.avg_time_interval.values, float(h.part_weight))
            tc, rr = tc[rr > 0], rr[rr > 0]
            S.plot(ax_r, tc, rr, u, "verification", ms=3.2, alpha=alpha, zorder=4 if u == 5 else 3)
    for u in (2, 5):
        r = camp[u]
        h = A.hold(r) if r is not None and r.has_hold else None
        if h is None or h.n_hold_in.sum() == 0:
            continue
        t = h.time.values
        for c, alpha in (("n_hold_in", 1.0), ("n_hold_ret", FAINT)):
            tc, rr, ee = binned_rate(t, h[c].values.astype(float), h.avg_time_interval.values, float(h.part_weight))
            ok = rr > 0
            S.plot(ax_r, tc[ok], rr[ok], u, "campaign", alpha=alpha, zorder=4 if u == 5 else 3)
    i_v = A.ztop_index(AN, z_ver)
    i_c = A.ztop_index(AN, 40127.3)
    for i in (i_v, i_c):
        _, _, rin = an.analytic_in_ret(AN, i)
        ax_r.axhline(rin, **S.ANALYTIC, zorder=1)
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlim(1e3, 2.5e6); ax_r.set_ylim(1e24, 2e29)
    ax_r.yaxis.set_major_formatter(FuncFormatter(sci_fmt)); ax_r.xaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_r.set_xlabel("time $t$ [s]"); ax_r.set_ylabel("rate [s$^{-1}$]")
    S.inset_legend(ax_r, [S.handle(2, "verification", ms=3.2, label=ver_lab),
                          S.handle(2, "campaign", label="campaign, top 40k"),
                          S.handle(2, "campaign", alpha=FAINT, label="faint: return rate")], "upper right",
                   bbox_to_anchor=(2.3e6, 2.4e28), bbox_transform=ax_r.transData, borderaxespad=0)

    # (b) in / return ratio vs t / tau
    for name, u in ver.items():
        h = A.hold(A.v(name))
        if h is None:
            continue
        t = h.time.values
        edges = an.log_bins(t, 6)
        tc, rr, ee = an.binned_ratio(t, h.n_hold_in.values.astype(float), t, h.n_hold_ret.values.astype(float), edges)
        ok = np.isfinite(rr) & (rr < 50)
        ax_q.errorbar(tc[ok] / t90v, rr[ok], yerr=ee[ok], **S.marker_kw(u, "verification", ms=3.2), elinewidth=0.6,
                      zorder=5 if u == 5 else 4)
        ax_q.plot(tc[ok] / t90v, rr[ok], **S.line_kw(u, "verification"), zorder=3)
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
        S.plot(ax_q, tc[ok] / tau90(40127.3), rr[ok], u, "campaign", zorder=4 if u == 5 else 3)
        live = True
    for i in (i_v, i_c):
        tt, q, _ = an.analytic_in_ret(AN, i)
        ok = np.isfinite(q)
        ax_q.plot(tt[ok] / float(AN.tau90[i]), q[ok], **S.ANALYTIC, zorder=1)
    ax_q.plot(PAPER["fig3_t"], PAPER["fig3_in_ret"], **S.PAPER_MARK, zorder=5)
    ax_q.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_q.set_xscale("log"); ax_q.set_yscale("log"); ax_q.set_xlim(2e-4, 8); ax_q.set_ylim(0.9, 30)
    plain_log(ax_q, "y", [1, 2, 5, 10, 20])
    ax_q.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.001 else sci_fmt(v)))
    ax_q.set_xlabel("$t/\\tau_{90}$  (paper: $t/t_\\mathrm{orb}$)")
    ax_q.set_ylabel("rate into hold / return rate")
    if not live:
        ax_q.text(0.03, 0.06, "campaign: PENDING (no returns yet)", transform=ax_q.transAxes, fontsize=6.0,
                  color=S.GREY)
    paper_h = Line2D([], [], **S.PAPER_MARK, label="Carberry Mogan et al. 2021,\ndigitised, against $t/t_\\mathrm{orb}$")
    S.inset_legend(ax_q, [S.handle(2, "verification", ms=3.2, label=ver_lab),
                          S.handle(2, "campaign", label="campaign, top 40k"), paper_h], "upper right")

    # (c) held molecules vs t
    for name, u in ver.items():
        h = A.hold(A.v(name))
        if h is None:
            continue
        n = len(h.time)
        every = list(np.unique(np.geomspace(1, n - 1, 7).astype(int)))
        S.plot(ax_h, h.time.values, h.held_end.values * float(h.part_weight), u, "verification", ms=3.0,
               markevery=every, zorder=4 if u == 5 else 3)
    h8 = A.hold(A.v("E8_main_kepler"))
    if h8 is not None:
        ax_h.plot(h8.time.values, h8.held_end.values * float(h8.part_weight), **S.line_kw(2, "verification", ls=BEFORE_FIX),
                  zorder=3)
    for z in (10000, 20000, 40000):
        r = A.get("sweep", 2, z, 250)
        if r is None or not r.has_run:
            continue
        d = A.reduce(r)
        ok = d.held_t.values > 0
        S.plot(ax_h, d.time.values[ok], d.held_t.values[ok], 2, "sweep", zorder=5)
        if z != 40000:
            top_tag(ax_h, d.time.values[-1] * 1.12, d.held_t.values[-1] * {10000: 1.25, 20000: 0.95}[z], z)
    for u in (2, 5):
        r = camp[u]
        h = A.hold(r) if r is not None and r.has_hold else None
        if h is None:
            continue
        ok = h.held_end.values > 0
        S.plot(ax_h, h.time.values[ok], h.held_end.values[ok] * float(h.part_weight), u, "campaign",
               zorder=4 if u == 5 else 3)
        if u == 2:
            top_tag(ax_h, h.time.values[-1] * 1.15, h.held_end.values[-1] * float(h.part_weight) * 0.62, 40000,
                    ha="left")
    for i in sorted({i_v, i_c} | {A.ztop_index(AN, z) for z in (10127.3, 20127.3)}):
        ok = AN.held_t.values[i] > 0
        ax_h.plot(AN.t.values[ok], AN.held_t.values[i][ok], **S.ANALYTIC, zorder=1)
    ax_h.set_xscale("log"); ax_h.set_yscale("log")
    ax_h.set_xlim(1e3, 1e7); ax_h.set_ylim(1e30, 2e33)
    ax_h.xaxis.set_major_formatter(FuncFormatter(sci_fmt)); ax_h.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_h.set_xlabel("time $t$ [s]"); ax_h.set_ylabel("molecules held above $r_\\mathrm{top}$")
    S.inset_legend(ax_h, [S.handle(2, "verification", ms=3.0, label="verification, 2.6k"),
                          Line2D([], [], **S.line_kw(2, "verification", ls=BEFORE_FIX),
                                 label="before the fix (E8)"),
                          S.handle(2, "sweep", label="sweep, 10k to 40k"),
                          S.handle(2, "campaign", label="campaign, 40k")], "lower right", handlelength=1.8)

    # (d) Phi(ballistic) / Phi(reflect) vs t / tau
    for z in (10000, 20000, 40000):
        a2, a1 = A.get("sweep", 2, z, 250), A.get("sweep", 1, z, 250)
        if a2 is None or a1 is None or not (a2.has_run and a1.has_run):
            continue
        s2, s1 = A.escape_series(a2), A.escape_series(a1)
        t = s2.time.values
        edges = an.log_bins(t, 4, t_min=4e3)
        tc, rr, ee = an.binned_ratio(t, s2.npart.values, s1.time.values, s1.npart.values, edges)
        x = tc / tau90(a2.z_top_actual_km)
        ax_p.errorbar(x, rr, yerr=ee, **S.marker_kw(2, "sweep", ms=3.0), elinewidth=0.6, zorder=3)
        ax_p.plot(x, rr, **S.line_kw(2, "sweep", lw=0.6), alpha=0.6, zorder=2)
        top_tag(ax_p, x[-1] * 1.35, rr[-1] + {10000: 0.045, 20000: -0.05, 40000: 0.04}.get(z, 0.0), z, fontsize=5.9)
    s9 = A.escape_series(A.v("E9_reflect"))
    for name, u in ver.items():
        s = A.escape_series(A.v(name))
        if s is None or s9 is None:
            continue
        t = s.time.values
        tc, rr, ee = an.binned_ratio(t, s.npart.values, s9.time.values, s9.npart.values, an.log_bins(t, 3, t_min=4e3))
        ax_p.errorbar(tc / t90v, rr, yerr=ee, **S.marker_kw(u, "verification", ms=3.2), elinewidth=0.6, zorder=3)
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
            ax_p.errorbar(tc[ok] / tau90(40127.3), rr[ok], yerr=ee[ok], **S.marker_kw(u, "campaign", ms=3.6),
                          elinewidth=0.6, zorder=4)
    for i in (i_v, i_c):
        ax_p.hlines(float(AN.escz_ballistic[i] / AN.escz_reflect[i]), 2e-4, 8, **S.ANALYTIC, zorder=1)
    ax_p.plot(PAPER["fig3_t"], PAPER["fig3_phi_ratio"], **S.PAPER_MARK, zorder=5)
    ax_p.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_p.set_xscale("log"); ax_p.set_xlim(2e-4, 8); ax_p.set_ylim(0.5, 1.4)
    ax_p.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v >= 0.001 else sci_fmt(v)))
    ax_p.set_xlabel("$t/\\tau_{90}$  (paper: $t/t_\\mathrm{orb}$)")
    ax_p.set_ylabel("$\\Phi_\\mathrm{ballistic}\\,/\\,\\Phi_\\mathrm{reflect}$")
    S.inset_legend(ax_p, [Line2D([], [], **S.marker_kw(2, "verification", ms=3.2), label=ver_lab),
                          Line2D([], [], **S.marker_kw(2, "sweep", ms=3.0), label="sweep, tops 10k to 40k"),
                          Line2D([], [], **S.marker_kw(2, "campaign", ms=3.6), label="campaign, top 40k"),
                          Line2D([], [], **S.PAPER_MARK, label="Carberry Mogan et al. 2021")],
                    "lower left", bbox_to_anchor=(0.0028, 0.505), bbox_transform=ax_p.transData, borderaxespad=0)

    # (e) the frozen hold (E8)
    h7 = A.hold(A.v("E7a_kepler"))
    if h8 is not None and h7 is not None:
        ax_8.plot(h8.time.values / 1e3, np.cumsum(h8.n_hold_in.values), color=S.LIGHT, lw=3.2, zorder=1,
                  solid_capstyle="butt")
        ax_8.plot(h8.time.values / 1e3, h8.held_end.values, **S.line_kw(2, "verification", ls=BEFORE_FIX), zorder=3)
        S.plot(ax_8, h7.time.values / 1e3, h7.held_end.values, 2, "verification", ms=3.0, markevery=6, zorder=2)
        ax_8.text(22, float(h8.held_end.values[-1]) * 0.92, "before the fix (E8): held count\n= cumulative entries "
                  "(grey), no returns", fontsize=6.0, color=S.GRAPHITE, ha="left", va="top", linespacing=1.2)
        ax_8.text(58, float(h7.held_end.values[-1]) * 1.12, "after the fix (E7a)", fontsize=6.0, color=S.GRAPHITE,
                  ha="right", va="bottom")
        ax_8.set_xlim(0, 61); ax_8.set_ylim(0, float(h8.held_end.values[-1]) * 1.08)
    else:
        S.pending(ax_8, "E8 / E7a hold diagnostics")
    ax_8.set_xlabel("time $t$ [ks]"); ax_8.set_ylabel("held computational particles")

    # (f) RK4 - Kepler return delay: three explicit verification runs, told apart by the filled half
    dt = SP["dt"]
    runs_f = (("E7b_explicit", "bottom", "E7b"), ("E7c_explicit_dbg", "left", "E7c, debug build"),
              ("E7d_explicit_restart", "right", "E7d, restarted"))
    for name, fs, _ in runs_f:
        h = A.hold(A.v(name))
        if h is None:
            continue
        ok = h.n_hold_ret.values > 0
        kw = S.marker_kw(5, "verification", ms=2.9, fillstyle=fs, mew=0.6)
        ax_d.plot(h.time.values[ok] / 1e3, h.ret_dt_mean.values[ok], **kw, zorder=3)
        ax_d.plot(h.time.values[ok] / 1e3, h.ret_dt_std.values[ok], **kw, alpha=FAINT, zorder=3)
    ax_d.axhline(dt / 2, **S.ANALYTIC)
    ax_d.axhline(dt / np.sqrt(12), **S.ANALYTIC)
    ax_d.text(60, dt / 2 + 0.03, "mean, expected $\\Delta t/2$", fontsize=6.0, color=S.GRAPHITE, ha="right",
              va="bottom")
    ax_d.text(60, dt / np.sqrt(12) - 0.02, "standard deviation (faint), expected $\\Delta t/\\sqrt{12}$", fontsize=6.0,
              color=S.GRAPHITE, ha="right", va="top")
    ax_d.set_xlim(0, 61); ax_d.set_ylim(0.2, 0.6)
    ax_d.set_xlabel("time $t$ [ks]"); ax_d.set_ylabel("RK4 $-$ Kepler return time [s]")
    S.inset_legend(ax_d, [Line2D([], [], **S.marker_kw(5, "verification", ms=3.2, fillstyle=fs, mew=0.6), label=l)
                          for _, fs, l in runs_f], "center right", handletextpad=0.2)
    for ax in axes.ravel():
        S.tick_mono(ax)
    key = [S.handle(2), S.handle(5), S.analytic_handle()]
    S.panel(ax_r, "a  rate into the hold and return rate", key, ncol=3, rows=1)
    S.panel(ax_q, "b  rate into the hold / return rate", rows=1)
    S.panel(ax_h, "c  held inventory", key, ncol=3, rows=1)
    S.panel(ax_p, "d  ballistic escape relative to reflect", rows=1)
    S.panel(ax_8, "e  frozen hold: no returns before the fix", [S.handle(2)], rows=1)
    S.panel(ax_d, "f  explicit RK4 return delay per output interval", [S.handle(5, "verification", line=False, ms=3.2),
                                                                      S.analytic_handle()], ncol=2, rows=1)
    S.header(fig, "H$_2$  ·  ballistic hold through time  ·  tops are altitudes in km (2.6k = 2 627 km)", 0.10)
    return S.save(fig, "fig3_hold_time")


# ----------------------------------------------------------------------------------------------
def fig_verification():
    """Verification E7a/E7b/E9: n_DSMC / n_analytic and corrected T vs altitude (collisionless, z_top 2 627 km)."""
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, 3.0), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.965, top=0.775, bottom=0.15, wspace=0.16)
    runs = (("E9_reflect", 1), ("E7a_kepler", 2), ("E7b_explicit", 5))
    any_ = False
    for name, u in runs:
        r = A.v(name)
        if not r.has_run:
            continue
        d = A.reduce(r)
        na, Ta = d.analytic_num_dens.values, d.analytic_temp_trans.values  # cell-averaged, campaigns/h2_hold
        ok = d.p_num_part.values > 5
        z = d.altitude.values[ok] / 1e3
        for ax, f, ref in ((axes[0], "num_dens", na), (axes[1], "temp_trans_corr", Ta)):
            q = d[f"p_{f}"].values[ok] / ref[ok]
            e = d[f"p_{f}_sem"].values[ok] / ref[ok]
            ax.fill_betweenx(z, q - e, q + e, color=S.UBC[u]["ink"], alpha=0.12, lw=0)
            S.plot(ax, q, z, u, "verification", ms=3.4, zorder={1: 3, 2: 4, 5: 5}[u])
        any_ = True
    if not any_:
        for ax in axes:
            S.pending(ax, "verification run files")
    for ax in axes:
        ax.axvline(1, color=S.LIGHT, lw=0.6, zorder=0)
        S.tick_mono(ax)
    axes[0].set_xlim(0.985, 1.015); axes[1].set_xlim(0.97, 1.03)
    axes[0].set_xlabel("$n_\\mathrm{DSMC}\\,/\\,n_\\mathrm{analytic}$")
    axes[1].set_xlabel("$T_\\mathrm{DSMC}\\,/\\,T_\\mathrm{analytic}$ (bulk flow removed)")
    axes[0].set_ylabel("altitude [km]")
    names = {1: "Specular reflect (E9)", 2: "Kepler hold (E7a)", 5: "Explicit RK4 (E7b)"}
    handles = [S.handle(u, "verification", ms=3.4, label=names[u]) for _, u in runs] + [S.band_handle("$\\pm1$ s.e.")]
    S.panel(axes[0], "a  density", handles, ncol=4, rows=1, columnspacing=1.8)
    S.panel(axes[1], "b  temperature", rows=1)
    S.header(fig, f"H$_2$  ·  collisionless verification (collisions off)  ·  top at {KM(A._ver_top_km())} km altitude"
             f"  ·  time mean over $t$ > {an.T0_VER / 1e3:.0f} ks", 0.09, right=0.965)
    return S.save(fig, "fig4_verification")


# ----------------------------------------------------------------------------------------------
def fig_production():
    """Campaign (2e6 s, collisions on): explicit RK4 (UBC 5) against the Kepler hold (UBC 2) at every top."""
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 3.15))
    fig.subplots_adjust(left=0.085, right=0.975, top=0.715, bottom=0.15, wspace=0.40)
    ax_e, ax_h, ax_n = axes
    tops = (2500, 10000, 40000)
    any_ = False
    for z in tops:
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
        ax_e.plot(t[ok][late] / 1e3, q[late], color=S.H2, lw=1.0, zorder=3)
        top_tag(ax_e, t[-1] / 1e3 * 1.02, q[-1] + {2500: 0.004, 10000: -0.001}.get(z, 0.0), z, bbox=None)
        # (b) held molecules: Kepler heavy solid, explicit slim dashed on top, collisionless held(t) grey
        for u, h in ((2, h2), (5, h5)):
            okh = h.held_end.values > 0
            S.plot(ax_h, h.time.values[okh] / 1e3, h.held_end.values[okh] * float(h.part_weight), u, "campaign",
                   zorder=4 if u == 5 else 3)
        i = A.ztop_index(AN, r2.z_top_actual_km)
        tt = AN.t.values
        sel = (tt >= 1e3) & (tt <= 2.2e6)
        ax_h.plot(tt[sel] / 1e3, AN.held_t.values[i][sel], **S.ANALYTIC, zorder=1)
        k0 = int(np.argmax(h2.held_end.values > 0))
        top_tag(ax_h, h2.time.values[k0] / 1e3 * 1.45, float(h2.held_end.values[k0]) * float(h2.part_weight) * 0.5,
                z, ha="left", bbox=None)
        # (c) density profile, explicit / Kepler, +-1 sigma
        rn, en, zn, okp = an.profile_compare(d5, d2, "num_dens")
        y = d2.radius.values / R0
        ax_n.fill_betweenx(y[okp], (rn - en)[okp], (rn + en)[okp], color=S.H2, alpha=0.10, lw=0, zorder=1)
        ax_n.plot(rn[okp], y[okp], color=S.H2, lw=1.0, zorder=3)
        ytop = float(y[okp][-1])
        ax_n.plot([1.085, 1.115], [ytop, ytop], color=S.H2, lw=0.6)
        top_tag(ax_n, 1.125, ytop, z, ha="left", bbox=None)
    if not any_:
        for ax in axes:
            S.pending(ax, "no finished campaign pair")
    ax_e.axhline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_e.set_xlim(0, 2.3e3); ax_e.set_ylim(0.95, 1.05)
    ax_e.set_xlabel("simulated time [ks]"); ax_e.set_ylabel("cumulative escapes, explicit / Kepler")
    ax_h.set_xscale("log"); ax_h.set_yscale("log"); ax_h.set_xlim(1, 4e3); ax_h.set_ylim(1e28, 1e33)
    ax_h.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_h.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax_h.set_xlabel("simulated time [ks]"); ax_h.set_ylabel("molecules held above $r_\\mathrm{top}$")
    ax_n.axvline(1, color=S.LIGHT, lw=0.6, zorder=0)
    ax_n.set_xlim(0.8, 1.2); ax_n.set_yscale("log")
    plain_log(ax_n, "y", [1, 1.5, 2, 3, 5, 7, 10])
    ax_n.set_xlabel("$n_\\mathrm{explicit}\\,/\\,n_\\mathrm{Kepler}$"); ax_n.set_ylabel("radial distance $r/r_0$")
    ratio_key = [Line2D([], [], color=S.H2, lw=1.0, label="explicit / Kepler"), S.band_handle(color=S.H2)]
    keys = (ratio_key, [S.handle(2, "campaign"), S.handle(5, "campaign"), S.analytic_handle()], ratio_key)
    for ax, t, k in zip(axes, ("a  escapes", "b  held inventory", "c  density"), keys):
        S.panel(ax, t, k, rows=3)
        S.tick_mono(ax)
    S.header(fig, "H$_2$  ·  collisional campaign, $2\\times10^6$ s  ·  $\\Delta r$ = 250 km  ·  curve labels: top "
             "altitude in km (2.6k = 2 627 km)", 0.085, right=0.975)
    return S.save(fig, "fig5_production")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["escape_rtop", "profiles", "hold_time", "verification", "production"]
    for name in todo:
        print(globals()[f"fig_{name}"]())
