"""Review-only plate for the slide 3 prototype: the numbers behind FIG 3m (not a deck figure).

    cd refactored && conda run -n dsmc python deck/fig03m_diagnostics.py

(a) Kn = 1 / (sigma N_tot) against true altitude for orbit 2521 and the deep-dip orbit 3314, with the
    hard-sphere sensitivity (dashed); bands at Kn = 0.1 and 10, periapses marked.
(b) True geometry: altitude against orbit arc angle nu, with the part of each pass inside 0.1 < Kn < 10 dark.
(c) What FIG 3m draws for 2521: schematic radius z_s = 230 + 40 ln Kn against nu.
Output: slides/build/slide03_maven_trajectory_diagnostics.pdf
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from _deck import REFACTORED, ds

DATA = REFACTORED / "deck" / "data"
OUT = REFACTORED / "slides" / "build" / "slide03_maven_trajectory_diagnostics.pdf"
STYLE = {2521: (ds.MIDNIGHT, "orbit 2521 (slide), 16 Jan 2016"), 3314: (ds.TANGELO, "orbit 3314 (deep dip), 12 Jun 2016")}


def main():
    facts = json.loads((DATA / "slide03_maven_facts.json").read_text())
    d = np.load(DATA / "slide03_maven.npz")
    fig = plt.figure(figsize=(13.33, 5.2))
    fig.set_layout_engine("none")                   # axes are placed by hand below
    a, b, c = (fig.add_axes(r) for r in ([0.065, 0.15, 0.23, 0.71], [0.385, 0.15, 0.27, 0.71],
                                         [0.715, 0.15, 0.27, 0.71]))
    for name, (k0, k1) in {"collisional": (1e-4, 0.1), "transition": (0.1, 10), "exosphere": (10, 1e6)}.items():
        a.axvspan(k0, k1, color=ds.REGIME[name], alpha=0.15, lw=0)
    for orb, (col, lab) in STYLE.items():
        f = facts["orbits"][str(orb)]
        z, kn, kn_hs = d[f"o{orb}_col_z"], d[f"o{orb}_col_kn"], d[f"o{orb}_col_kn_hs"]
        a.plot(kn, z, color=col, lw=1.8, label=lab)
        a.plot(kn_hs, z, color=col, lw=1.0, ls=(0, (3, 2)))
        a.plot(f["kn_peri"], f["z_peri_km"], "o", color=col, ms=5)
        nu, alt, kn_t = d[f"o{orb}_nu"], d[f"o{orb}_alt"], d[f"o{orb}_kn"]
        o = np.argsort(nu)
        nu, alt, kn_t = nu[o], alt[o], kn_t[o]
        sel = np.abs(nu) <= 60
        b.plot(nu[sel], alt[sel], color=col, lw=1.0, alpha=0.45)
        tr = sel & (kn_t >= 0.1) & (kn_t <= 10)
        b.plot(np.where(tr, nu, np.nan), alt, color=col, lw=2.4)
        coll = sel & (kn_t < 0.1)
        if coll.any():
            b.plot(np.where(coll, nu, np.nan), alt, color=ds.CELTIC, lw=2.4)
        for kv, ls in ((10, ":"), (1, "--")):
            zz = f["z_kn10_km"] if kv == 10 else f["z_exo_km"]
            b.axhline(zz, color=col, lw=0.6, ls=ls, alpha=0.6)
    a.set_xscale("log")
    a.set_xlim(1e-3, 1e5)
    a.set_ylim(110, 350)
    a.set_xlabel(r"$\mathrm{Kn} = 1/(\sigma N_{\mathrm{tot}})$")
    a.set_ylabel("true altitude (km)")
    a.set_title("(a) Knudsen profile, NGIMS inbound", loc="left", fontsize=12)
    a.text(0.97, 0.03, r"solid: $\sigma = 2\times10^{-19}\,\mathrm{m^2}$ (Kn = 1 at $N = 5\times10^{18}\,\mathrm{m^{-2}}$)" "\n"
           r"dashed: hard-sphere O / Ar / CO$_2$ mix",
           transform=a.transAxes, fontsize=9, color=ds.GRANITE, ha="right", multialignment="left")
    a.legend(loc="upper left", fontsize=9, frameon=False)
    b.set_xlim(-60, 60)
    b.set_ylim(100, 700)
    b.set_xlabel(r"orbit arc from periapsis $\nu$ (deg)")
    b.set_ylabel("true altitude (km)")
    b.set_title("(b) true geometry of the passes", loc="left", fontsize=12)
    b.text(0.02, 0.97, "bold: 0.1 < Kn < 10; blue: Kn < 0.1\ndotted: Kn = 10, dashed: Kn = 1 (exobase)", transform=b.transAxes,
           fontsize=9, color=ds.GRANITE, va="top")

    f = facts["orbits"]["2521"]
    nu, zs, alt = d["o2521_nu"], d["o2521_zs"], d["o2521_alt"]
    o = np.argsort(nu)
    for name, (z0, z1) in {"collisional": (100, 230 - 40 * np.log(10)), "transition": (230 - 40 * np.log(10), 230 + 40 * np.log(10)),
                           "exosphere": (230 + 40 * np.log(10), 700)}.items():
        c.axhspan(z0, z1, color=ds.REGIME[name], alpha=0.15, lw=0)
    c.axhline(230, color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5)
    meas = alt[o] <= f["z_top_O_km"]
    c.plot(0.4 * nu[o], np.where(meas, zs[o], np.nan), color=ds.MIDNIGHT, lw=2.2)
    c.plot(0.4 * nu[o], zs[o], color=ds.MIDNIGHT, lw=1.0, alpha=0.5)
    c.set_xlim(-22, 22)
    c.set_ylim(100, 700)
    c.set_xlabel(r"sector angle $= 0.4\,\nu$ (deg)")
    c.set_ylabel(r"schematic radius $z_s = 230 + 40\ln\mathrm{Kn}$ (km)")
    c.set_title("(c) FIG 3m placement, orbit 2521", loc="left", fontsize=12)
    c.text(0.02, 0.97, "solid: column sampled by NGIMS (r ≤ 228 km)\nlight: isothermal top term",
           transform=c.transAxes, fontsize=9, color=ds.GRANITE, va="top")
    fig.suptitle("Slide 3 prototype · review only · MAVEN NGIMS L2 v08 + KP v26; williamson-replication columns",
                 x=0.06, ha="left", fontsize=11, color=ds.GRANITE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"Creator": f"deckstyle {ds.STYLE_VERSION}",
                               "Subject": "deck/data/slide03_maven.npz; slide03_maven_facts.json"})
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    print(OUT)


if __name__ == "__main__":
    main()
