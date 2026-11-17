"""Slide 8 · FIG 8a coupling baseline (Plate 1 panel A) and FIG 8b upper-boundary baseline (Plate 3 panel A).

Both read only the caches written by deck/cache_slides78.py (deck/data/), restyled to Rarefied Order.

FIG 8a: MAGNUS-P run S01B8 (pure O at 270 K; wave period 60 min, horizontal wavelength 400 km, forced
  at the ground with 0.1 m/s). Vertical-wind amplitude 2|c_w| of the fluid model and of exact linear
  viscous theory (2|c_w_linear|, computed by MAGNUS-P's isoo_linear.solve_cell and stored with the
  profile). The labelled misfits at the three DSMC input altitudes are |c_w - c_w_lin|/|c_w_lin|
  (amplitude and phase) at the 100.25 / 120.25 / 140.25 km cells. The dashed reference is undamped
  growth 0.1 exp(z/2H), H = R_O T/g = 37.7 km. The sponge starts at 180 km (it continues to 280 km,
  above this axis). The exobase is far above 200 km, so no exobase line here.
FIG 8b: DSMC campaign h2_ubc_rtop, H2 escape rate vs actual domain-top altitude, 250 km radial cells,
  mean over 100-300 ks, for three upper-boundary treatments. Pure-H2 columns are lines with filled
  markers; coupled O+H2 columns are open markers. The overcount quoted on the figure is Plate 3's
  definition at 2,500 km on these 250 km cells (free escape / mean reflection-and-hold rate).
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from _deck import ds, save

DATA = Path(__file__).resolve().parent / "data"
R_O = 8.314462 / 0.016                     # J/(kg K), atomic oxygen
H_KM = R_O * 270.0 / 3.72 / 1e3            # isothermal scale height at 270 K, Mars gravity


def pct(x: float) -> str:
    """Round half up to 2 decimals, as Plate 1 prints them (0.2150 -> 0.22)."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fig8a():
    d = np.load(DATA / "fig08a_plate1A.npz")
    z, wm, wl = d["z_km"], d["w_model"], d["w_linear"]
    fig, ax = ds.figure("evidence")

    ax.axhspan(180, 200, color=ds.SLATE, alpha=0.15, lw=0, zorder=0)
    ds.label(ax, "absorbing layer", (0.058, 190), color=ds.GRANITE, ha="left", va="center")
    ds.input_altitudes(ax)
    zi = np.linspace(0, 200, 200)
    ax.plot(0.1 * np.exp(zi / (2 * H_KM)), zi, color=ds.ROLE["prior_work"], ls=ds.LINESTYLE["prior_work"],
            lw=1.5, zorder=2)
    ax.plot(wm, z, color=ds.ROLE["this_work"], lw=2.4, zorder=3)
    every = slice(0, None, 20)                                  # 0.5 km cells -> a circle every 10 km
    ax.plot(wl[every], z[every], "o", ms=7.5, mfc="none", mec=ds.ROLE["theory"], mew=1.4, zorder=4)

    for zk, m in zip(d["level_km"], d["misfit_pct"]):
        ax.text(0.058, zk + 2.2, f"{zk:.0f} km", ha="left", va="bottom", fontsize=14)
        ax.text(0.103, zk + 2.2, f"{pct(m)}%", ha="left", va="bottom", fontsize=14,
                fontweight="semibold", color=ds.MIDNIGHT)
    ds.label(ax, "model vs theory", (0.058, 157), ha="left", va="bottom")

    ds.label(ax, "fluid model", (0.235, 166), color=ds.ROLE["this_work"], ha="left", va="center")
    ds.label(ax, "linear viscous\ntheory", (0.195, 22), color=ds.ROLE["theory"], ha="left", va="center",
             linespacing=1.0)
    ds.label(ax, "no viscosity", (0.67, 130), color=ds.ROLE["prior_work"], ha="left", va="center")
    zp = float(d["peak_z_km"])
    ds.takeaway(ax, f"Viscous damping wins\nabove {zp:.0f} km", xy=(np.interp(zp, z, wm) * 1.06, zp),
                xytext=(0.46, 62), ha="left", va="center", linespacing=1.1)

    ax.set_xscale("log")
    ax.set_xlim(0.05, 2.0)
    ax.set_ylim(0, 200)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([0, 40, 80, 120, 160, 200])
    ax.set_xlabel("Wave vertical-wind amplitude (m/s)")
    ax.set_ylabel("Altitude (km)")
    meta = json.loads(str(d["meta"]))
    save(fig, "fig08a_coupling_baseline", "deck/data/fig08a_plate1A.npz",
         f"MAGNUS-P {meta['run']} {Path(meta['source']).name} @ MAGNUS-P {meta['magnusp_commit']}")
    plt.close(fig)


def fig8b():
    e = pd.read_csv(DATA / "fig08b_escape.csv", comment="#")
    fine = e[e["dr_max_km"] == 250]
    base = e[e["ubc_num"].isin([1, 2])].groupby("atmosphere")["dsmc_escape_rate_s"].mean()
    u3 = fine[(fine["ubc_num"] == 3) & (fine["z_top_nominal_km"] == 2500)].set_index("atmosphere")
    mult = u3["dsmc_escape_rate_s"] / base
    print(f"overcount at 2,500 km, 250 km cells: H2 {mult['H2']:.1f}x, O+H2 {mult['H2O']:.1f}x")

    style = {1: (ds.GRANITE, "o"), 2: (ds.MIDNIGHT, "s"), 3: (ds.SLATE, "^")}
    fig, ax = ds.figure("narrow")
    for ubc, (color, mk) in style.items():
        for atm, filled in (("H2", True), ("H2O", False)):
            r = fine[(fine["ubc_num"] == ubc) & (fine["atmosphere"] == atm)].sort_values("z_top_actual_km")
            x, y = r["z_top_actual_km"].to_numpy(), r["dsmc_escape_rate_s"].to_numpy()
            if filled:
                ax.plot(x, y, "-", color=color, lw=1.8, zorder=3)
                ax.plot(x, y, mk, ms=5.5, mew=0, color=color, zorder=4)
            else:
                ax.plot(x, y, mk, ms=5.5, mfc="none", mec=color, mew=1.1, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2100, 50000)
    ax.set_ylim(2.2e26, 9e28)
    ax.xaxis.set_major_locator(FixedLocator([2500, 10000, 40000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2,500", "10,000", "40,000"])
    ax.yaxis.set_major_locator(FixedLocator([1e27, 1e28]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([r"$10^{27}$", r"$10^{28}$"])
    ax.set_xlabel("Domain top (km)")
    ax.set_ylabel(r"H$_2$ escape (s$^{-1}$)")

    ds.label(ax, "reflect ≈ hold", (2600, 5.4e26), color=ds.MIDNIGHT, ha="left", va="bottom")
    ds.label(ax, "free escape", (2600, 2.2e27), color=ds.SLATE, ha="left", va="center")
    ds.label(ax, "filled: pure H$_2$\nopen: O+H$_2$", (46000, 2.9e28), color=ds.GRANITE, ha="right", va="top",
             linespacing=1.1)
    lo, hi = sorted(mult.round().astype(int))
    y0 = float(u3.loc["H2", "dsmc_escape_rate_s"])
    ds.takeaway(ax, f"{lo}–{hi}× too high", xy=(2900, y0 * 0.93), xytext=(5000, 5.6e28), ha="left",
                va="center")
    save(fig, "fig08b_escape_vs_top", "deck/data/fig08b_escape.csv",
         "outputs/h2_ubc_rtop/tables/escape_summary.csv (campaign h2_ubc_rtop)")
    plt.close(fig)


if __name__ == "__main__":
    fig8a()
    fig8b()
