"""Slide 2 · FIG 2a one NGIMS pass and FIG 2b the amplitude climatology against solar zenith angle.

Data: deck/data/fig02_ngims.npz and slide02_facts.json, written by deck/cache_slide02.py from our
reduction of the public NGIMS L2 data (williamson-replication, which reproduces Williamson et al. 2019).

FIG 2a (evidence slot): delta rho / rho_bar against altitude for the inbound pass of orbit 2521
  (16 Jan 2016, nightside), Williamson et al. (2019) Fig. 1b. Species in their SPECIES colours, the
  40% event threshold dotted, and this pass's own exobase (N_tot = 5e18 m^-2, 151.6 km) as the Gold
  dashed line: on the cold nightside it sits well below the 230 km of the model atmosphere, so the
  deck default would misplace it. The axes fill the left 4.5 in; the right part of the field holds the
  key (shared with FIG 2b) above the space where build_deck.py places FIG 2b.
FIG 2b (inset slot): median over passes of each pass's mean |delta rho / rho_bar|, in 15 deg SZA bins
  (>= 50 passes), all 4,282 inbound passes Feb 2015 - Nov 2017, no event selection; bands are the
  bootstrap 16-84% range of the median. Its x axis sits at the same height as FIG 2a's.
"""
import json

import numpy as np
from matplotlib.lines import Line2D

from _deck import REFACTORED, ds, save

DATA = REFACTORED / "deck" / "data"
NPZ = DATA / "fig02_ngims.npz"
FACTS = json.loads((DATA / "slide02_facts.json").read_text())
SP = [("CO2", "CO$_2$"), ("Ar", "Ar"), ("O", "O")]       # drawing / key order, heaviest first
SRC = ("deck/data/fig02_ngims.npz", "deck/data/slide02_facts.json")

# shared geometry (inches): both x axes at the same height above the field bottom
AX_BOTTOM = 0.78
W_A, H_A = ds.SIZES["evidence"]
W_B, H_B = ds.SIZES["inset"]


def fig2a(d):
    fig, ax = ds.figure("evidence")
    fig.set_layout_engine("none")
    left, width, top = 0.80, 3.62, 0.16
    ax.set_position([left / W_A, AX_BOTTOM / H_A, width / W_A, (H_A - AX_BOTTOM - top) / H_A])

    z = d["alt_km"]
    # Ar only where O or CO2 is also measured (the replication's Ar peak rule): above that the Ar
    # count noise alone crosses 40% in many passes
    z_major = z[np.isfinite(d["A_O"]) | np.isfinite(d["A_CO2"])].max()
    d = {**d, "A_Ar": np.where(z <= z_major, d["A_Ar"], np.nan)}
    ax.axvline(0, color=ds.GRANITE, lw=0.6, zorder=1)
    thr = FACTS["threshold_pct"]
    ax.axvline(thr, color=ds.GRANITE, ls=(0, (1, 2.5)), lw=1.1, zorder=1)
    for key, _ in SP[::-1]:
        ax.plot(d[f"A_{key}"], z, color=ds.SPECIES[key], lw=1.7, solid_joinstyle="round", zorder=3)
    ds.exobase(ax, z=FACTS["orbit_z_exo_km"])

    ax.set_xlim(-62, 100)
    ax.set_ylim(140, 230)
    ax.set_xticks([-40, 0, 40, 80])
    ax.set_yticks([140, 160, 180, 200, 220])
    ax.set_xlabel(r"$\delta\rho/\bar\rho$  (%)")
    ax.set_ylabel("Altitude (km)")
    ds.label(ax, f"{thr}% threshold", (thr + 2, 141), ha="left", va="bottom")
    zp, pk = FACTS["orbit_peak_alt_km"], FACTS["orbit_peak_pct"]["CO2"]
    ds.takeaway(ax, f"{pk}% at {zp:.0f} km", xy=(pk + 1.0, zp + 1.2), xytext=(52, 172),
                ha="left", va="center")

    # key and pass caption in the right part of the field, above FIG 2b
    x0 = (left + width + 0.42) / W_A
    fig.text(x0, 1 - 0.30 / H_A, f"ORBIT {FACTS['orbit']} · 16 JAN 2016", fontsize=12,
             fontweight="semibold", color=ds.GRANITE, va="baseline")
    fig.text(x0, 1 - 0.52 / H_A, f"NIGHTSIDE · LOCAL TIME {FACTS['orbit_lst_h']:.0f} H", fontsize=12,
             fontweight="semibold", color=ds.GRANITE, va="baseline")
    handles = [Line2D([], [], color=ds.SPECIES[k], lw=2.4) for k, _ in SP]
    labels = [f"{lab}   peak {FACTS['orbit_peak_pct'][k]}%" for k, lab in SP]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x0 - 0.008, 1 - 0.66 / H_A),
               fontsize=14, handlelength=1.6, handletextpad=0.6, labelspacing=0.35, borderaxespad=0,
               borderpad=0)
    # caption for FIG 2b, which build_deck.py places in the lower right of this field
    fig.text(x0, (H_B + 0.10) / H_A, "ALL PASSES · 2015–2017", fontsize=12, fontweight="semibold",
             color=ds.GRANITE, va="baseline")
    save(fig, "fig02a_ngims_orbit2521", *SRC)


def fig2b(d):
    fig, ax = ds.figure("inset")
    fig.set_layout_engine("none")
    left, right, top = 0.66, 0.24, 0.12
    ax.set_position([left / W_B, AX_BOTTOM / H_B, (W_B - left - right) / W_B,
                     (H_B - AX_BOTTOM - top) / H_B])
    x = d["sza_deg"]
    ax.axvline(90, color=ds.GRANITE, ls=(0, (1, 2.5)), lw=1.1, zorder=1)
    for key, _ in SP[::-1]:
        c = ds.SPECIES[key]
        ds.band(ax, x, d[f"lo_{key}"], d[f"hi_{key}"], c)
        ax.plot(x, d[f"med_{key}"], color=c, lw=1.8, marker="o", ms=3.6, mew=0, zorder=3)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 23)
    ax.set_xticks([0, 90, 180])
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("SZA (°)", labelpad=2)
    ax.set_ylabel(r"$|\delta\rho/\bar\rho|$ (%)", labelpad=3)
    ds.label(ax, "day", (84, 22.8), ha="right", va="top")
    ds.label(ax, "night", (96, 22.8), ha="left", va="top")
    save(fig, "fig02b_ngims_sza", *SRC)


def main():
    d = dict(np.load(NPZ))
    fig2a(d)
    fig2b(d)


if __name__ == "__main__":
    main()
