"""Slide 6 · measure-card sparklines (1.22 x 0.47 in, no axes, no text, one baseline).

  L_att:    lock-in amplitude |Z| vs arc distance s past the patch edge at the exobase cell, with
            the Table S3 exponential fit and the dashed 3-sigma noise floor.
  lambda_h: crest-arrival lag vs column distance with the ODR line (points cached by
            deck/compute_phase_lag_cache.py). The ODR alone gives 450.4 +/- 7.6 km; the card's
            450 +/- 14 km is the rev5 total (statistical +/-11 = ODR error x 1.4 from injection
            tests, combined with +/-9 km for fit-window and single-edge choices).

    conda run -n dsmc python deck/fig06_sparklines.py [latt] [lambdah]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from _deck import ds, save
from dsmclib import attenuation as att
from dsmclib.paths import output_dir

ANALYSIS = output_dir("s1_s6_azimuthal", "analysis")
LATT_CACHE = ANALYSIS / "lockin_attenuation_S6_azimean_preforcing.npz"
LAG_CACHE = ANALYSIS / "phase_lag_S6_mad_column0.npz"
MAX_FIT_KM = 400.0       # attenuation_table default (--max-dist)
SHOW_TO_KM = 520.0       # sparkline extent: a few points past the fit, on the noise floor
LW = 0.75


def spark():
    fig, ax = ds.figure("spark")
    fig.set_layout_engine("none")
    ax.set_position([0.03, 0.08, 0.94, 0.84])
    return fig, ax


def baseline(ax, x0, x1, y=0.0):
    ax.plot([x0, x1], [y, y], color=ds.GRANITE, lw=LW, solid_capstyle="butt", zorder=1)


def latt():
    red = np.load(LATT_CACHE)
    zi = int(np.argmin(np.abs(red["radial_cells"] - ds.EXOBASE_KM)))
    s, amp, sigma = red["s_km"][zi], np.abs(red["Z_sym"][zi]), float(red["sigma_c"][zi])
    i0 = int(np.where(red["k"] == att.K0_FIT_START)[0][0])
    _, i_end = att.fit_range(amp, s, sigma, i0, att.DEFAULT_SNR_CUT, MAX_FIT_KM)
    fit = att.fit_exp_at(amp, s, sigma, i0, i_end)
    print(f"L_att = {fit['L']:.2f} km at {red['radial_cells'][zi]:.1f} km, n = {fit['n']}, "
          f"fit to s = {fit['s_end']:.1f} km, R2 = {fit['r2']:.3f}")

    fig, ax = spark()
    show = np.arange(i0, amp.size)[s[i0:] <= SHOW_TO_KM]
    in_fit = show <= i_end
    x0, x1 = s[show[0]] - 12, SHOW_TO_KM
    baseline(ax, x0, x1)
    ax.plot([x0, x1], [3 * sigma] * 2, color=ds.GRANITE, lw=LW, ls=(0, (3, 2.5)), zorder=1)   # 3 sigma
    xx = np.linspace(s[i0], s[i_end], 100)
    ax.plot(xx, fit["a0"] * np.exp(-xx / fit["L"]), color=ds.ROLE["this_work"], lw=LW, zorder=3)
    ax.plot(s[show[in_fit]], amp[show[in_fit]], "o", ms=2.0, mew=0, color=ds.ROLE["this_work"], zorder=4)
    ax.plot(s[show[~in_fit]], amp[show[~in_fit]], "o", ms=2.0, mew=0.5, mfc="none",
            color=ds.ROLE["uncertainty"], zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.04 * amp[i0], 1.08 * amp[show].max())
    save(fig, "fig06_spark_latt", LATT_CACHE, f"exobase cell, L_att = {fit['L']:.2f} km")
    plt.close(fig)


def lambdah():
    d = np.load(LAG_CACHE)
    used = d["used"]
    x, y = d["distance_km"][used], d["lag_periods"][used]
    slope, intercept = float(d["slope_per_km"]), float(d["intercept_periods"])
    print(f"lambda_h = {float(d['lambda_h_km']):.2f} +/- {float(d['lambda_h_err_km']):.2f} km, n = {int(d['n'])}")

    fig, ax = spark()
    x0, x1 = -10.0, x.max() * 1.05
    baseline(ax, x0, x1)
    xx = np.array([max(0.0, -intercept / slope), x1])
    ax.plot(xx, slope * xx + intercept, color=ds.ROLE["this_work"], lw=LW, zorder=3)
    ax.plot(x, y, "o", ms=2.0, mew=0, color=ds.ROLE["this_work"], zorder=4)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.04 * y.max(), 1.10 * y.max())
    save(fig, "fig06_spark_lambdah", LAG_CACHE, f"ODR lambda_h = {float(d['lambda_h_km']):.2f} km")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["latt", "lambdah"]
    if "latt" in which:
        latt()
    if "lambdah" in which:
        lambdah()
