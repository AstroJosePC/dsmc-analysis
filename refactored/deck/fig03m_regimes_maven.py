"""Slide 3 prototype · FIG 3m: the FIG 3 regime sector with MAVEN's orbit 2521 periapsis pass on it.

    cd refactored && conda run -n dsmc python deck/fig03m_regimes_maven.py

Step 1 (_b1) is FIG 3 exactly as deck/fig03_regimes.py draws it (that script is imported and its figure
taken over, not copied); step 2 adds the pass. Inputs: deck/data/slide03_maven.npz and
slide03_maven_facts.json (deck/cache_slide03_maven.py).

Placement. The sector draws an isothermal column with Kn = 1 at 230 km and H = 40 km, so its radius is a
log-Kn axis, z_s = 230 + 40 ln Kn. Each point of the pass sits at the radius of its measured Kn
(Kn = 1 / (sigma N_tot), N_tot the NGIMS inbound column, sigma = 2e-19 m^2, i.e. Kn = 1 at the pass's own
exobase, 152 km) and at sector angle 0.4 nu (nu = orbit arc from periapsis; the 700 km crossings at
nu = +-55 deg land on the sector corners). So the band the line is in is the band of the measured Kn; true
altitudes are not on the r scale and are quoted in the label. Solid where NGIMS sampled the column
(r <= 228 km, the top of the O profile); lighter where Kn comes from the column's isothermal top term.
"""
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from _deck import REFACTORED, Build, ds

import fig03_regimes as base

DATA = REFACTORED / "deck" / "data"
ORBIT = 2521
NU_SCALE = 0.4                       # sector angle = 0.4 x orbit arc angle
ICON = REFACTORED / "slides" / "maven_spacecraft.png"
ICON_W, ICON_ZS = 0.62, 575.0        # icon width (in) and the schematic radius (km) it sits at
LABEL_X, LABEL_Y = 5.26, 4.03        # label anchor (in): right of the escape arrows, above FREE ESCAPE


def schematic():
    """FIG 3 as deck/fig03_regimes.py draws it: run its main() with save() captured."""
    got = {}
    base.save = lambda fig, name, *inputs: got.update(fig=fig)
    base.main()
    return got["fig"]


def icon(width_px=620):
    img = mpimg.imread(ICON)
    ys, xs = np.where(img[..., 3] > 0.02)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    step = max(1, img.shape[1] // width_px)
    return img[::step, ::step]


def main():
    f = json.loads((DATA / "slide03_maven_facts.json").read_text())["orbits"][str(ORBIT)]
    d = np.load(DATA / "slide03_maven.npz")
    nu, alt, zs = d[f"o{ORBIT}_nu"], d[f"o{ORBIT}_alt"], d[f"o{ORBIT}_zs"]
    col_z, col_kn = d[f"o{ORBIT}_col_z"], d[f"o{ORBIT}_col_kn"]

    # Kn above the tabulated column (504 km): continue ln N_tot linearly in z (isothermal top)
    top = col_z > col_z[-1] - 40.0
    slope = np.polyfit(col_z[top], np.log(col_kn[top]), 1)[0]
    ext = ~np.isfinite(zs) & (alt > col_z[-1])
    zs = zs.copy()
    zs[ext] = ds.EXOBASE_KM + base.H_KM * (np.log(col_kn[-1]) + slope * (alt[ext] - col_z[-1]))

    a = np.deg2rad(NU_SCALE * nu)
    inside = (np.abs(a) <= base.ALPHA) & (zs <= base.Z1) & (zs >= base.Z0)
    order = np.argsort(nu)
    nu, alt, zs, a, inside = nu[order], alt[order], zs[order], a[order], inside[order]
    keep = inside & (np.abs(nu) <= 60)
    nu, alt, zs, a = nu[keep], alt[keep], zs[keep], a[keep]
    measured = alt <= f["z_top_O_km"]

    fig = schematic()
    ax = fig.axes[0]
    b = Build(fig)
    b.at(2)

    # no halo: a light stroke would hide step-1 dots (build_deck.check_builds flags ink that vanishes)
    x, y = base.xy(zs, a)
    for seg in (nu <= 0, nu >= 0):                                # extrapolated top: lighter, thinner
        for part, kw in ((seg & ~measured, dict(lw=1.3, alpha=0.55)), (seg & measured, dict(lw=2.2))):
            idx = np.flatnonzero(part)
            if idx.size < 2:
                continue
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if run.size > 1:
                    lo, hi = max(run[0] - 1, 0), min(run[-1] + 2, nu.size)
                    ax.plot(x[lo:hi], y[lo:hi], color=ds.ROLE["this_work"], solid_capstyle="round",
                            zorder=6, **kw)
    ip = int(np.argmin(alt))
    ax.scatter([x[ip]], [y[ip]], s=30, color=ds.ROLE["this_work"], edgecolor=ds.CULTURED, lw=1.2, zorder=7)

    # direction of flight: one small arrowhead on the inbound leg, inside the exosphere band
    j = int(np.argmin(np.abs(zs - 470.0) + 1e3 * (nu > 0)))
    ax.annotate("", xy=(x[j + 3], y[j + 3]), xytext=(x[j], y[j]),
                arrowprops=dict(arrowstyle="-|>", color=ds.ROLE["this_work"], lw=0, mutation_scale=11,
                                shrinkA=0, shrinkB=0), zorder=7)

    # spacecraft on the outbound leg, high in the exosphere, under its label
    k = int(np.argmin(np.abs(zs - ICON_ZS) + 1e3 * (nu < 0)))
    img = icon()
    h_in = ICON_W * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x[k] - ICON_W / 2, x[k] + ICON_W / 2, y[k] - h_in / 2, y[k] + h_in / 2),
              interpolation="antialiased", zorder=8)
    ax.set_xlim(0, ds.SIZES["evidence"][0])
    ax.set_ylim(0, ds.SIZES["evidence"][1])

    # label, upper right: above FREE ESCAPE, clear of the escape arrows
    ds.label(ax, f"MAVEN · ORBIT {ORBIT}", (LABEL_X, LABEL_Y), color=ds.ROLE["this_work"],
             ha="left", va="center")
    ax.text(LABEL_X, LABEL_Y - 0.23, rf"periapsis {f['z_peri_km']:.0f} km, $\mathrm{{Kn}} \approx {f['kn_peri']:.1f}$",
            ha="left", va="center", fontsize=11)

    b.save("fig03m_regime_maven", "deck/data/slide03_maven.npz",
           f"orbit {ORBIT} drawn at Kn = 1/(sigma N_tot), sigma = 2e-19 m^2, angle = {NU_SCALE} x orbit arc")
    plt.close(fig)


if __name__ == "__main__":
    main()
