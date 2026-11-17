"""Slide 7 · FIG 7a: the wheel sector of the truncation-baseline campaign (schematic, not to scale).

Main view (radius x azimuth): a sector one horizontal wavelength wide (6.49 deg = 400 km at the 140 km
base, 50 azimuthal cells of 8 km) from 140 km to the 504.6 km top cell edge, side edges periodic.
A particle leaving through one seam re-enters through the other, rotated about the polar axis by the
sector width with its local velocity unchanged (MarsAtmos Wrap_Sector_Azimuth); its path is
integrated in global Cartesian coordinates, so curvature is exact and only the grid is truncated.
Side view (polar direction): the gas is confined to a band of half-width beta = 7.2 deg between two
cones. The fix reflects a particle in the cone's tangent plane (flips v_theta); the old code reflected
it off a plane of constant z (flipped v_z), drawn as the dashed ghost.

Geometry numbers come from deck/data/slide07_facts.json (deck/cache_slides78.py). Angles are
exaggerated for legibility: the sector is drawn about 1.7x wider and the band 2x thicker than true.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle

from _deck import ds, save

DATA = Path(__file__).resolve().parent / "data"
FACTS = json.loads((DATA / "slide07_facts.json").read_text())
G = FACTS["geometry"]

# main view, inches: sector about a centre far below the figure
CX, CY = 1.95, -5.57
R_IN, R_OUT = 6.60, 9.05                     # 140 km and the 504.6 km top edge
HALF = np.deg2rad(5.6)                       # drawn half-angle (true: 3.25 deg)
# side-view inset frame and band
FX0, FY0, FX1, FY1 = 4.05, 0.42, 7.18, 3.78
BETA_DRAW = np.deg2rad(2 * G["beta_deg"])    # drawn at 2x


def r_of(z_km):
    return R_IN + (z_km - G["base_km"]) / (G["top_edge_km"] - G["base_km"]) * (R_OUT - R_IN)


def pol(r, a):
    """Azimuth a measured from the sector centre line, positive to the right."""
    return CX + r * np.sin(a), CY + r * np.cos(a)


def main_view(ax, notes=True):
    """notes=False leaves out the two caption lines under the sector (slide 7 alternatives)."""
    n_azi, n_rad = G["n_azi"], G["n_radial"]
    a = np.linspace(-HALF, HALF, 200)
    # cell grid: every azimuthal cell, every 4th radial cell
    for k in range(1, n_azi):
        ak = -HALF + 2 * HALF * k / n_azi
        ax.plot(*zip(pol(R_IN, ak), pol(R_OUT, ak)), color=ds.GRANITE, lw=0.35, alpha=0.22, zorder=1)
    for j in range(4, n_rad, 4):
        ax.plot(*pol(R_IN + (R_OUT - R_IN) * j / n_rad, a), color=ds.GRANITE, lw=0.35, alpha=0.22, zorder=1)
    ax.plot(*pol(R_IN, a), color=ds.GRANITE, lw=1.0, zorder=2)
    ax.plot(*pol(R_OUT, a), color=ds.GRANITE, lw=1.0, zorder=2)
    for s in (-1, 1):                                            # the two periodic seams
        ax.plot(*zip(pol(R_IN, s * HALF), pol(R_OUT, s * HALF)), color=ds.MIDNIGHT, lw=2.2,
                solid_capstyle="butt", zorder=3)
    ax.plot(*pol(r_of(230.0), a), color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5, zorder=2)

    # altitude labels at the right seam
    x, y = pol(R_IN, HALF)
    ax.text(x + 0.12, y, f"{G['base_km']:.0f} km", ha="left", va="center", fontsize=13)
    x, y = pol(R_OUT, HALF)
    ax.text(x + 0.12, y - 0.04, f"{G['top_edge_km']:.0f} km", ha="left", va="center", fontsize=13)
    x, y = pol(r_of(230.0), HALF)
    ax.text(x + 0.12, y, "EXOBASE", ha="left", va="center", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)

    # periodic seams: one wrap-around arrow over the top, right seam -> left seam
    top_r = R_OUT + 0.14
    xr, yr = pol(top_r, HALF * 0.92)
    xl, yl = pol(top_r, -HALF * 0.92)
    ax.add_patch(FancyArrowPatch((xr, yr), (xl, yl), connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color=ds.MIDNIGHT, shrinkA=0, shrinkB=0,
                                 zorder=3))
    ax.text(CX, CY + top_r + 0.22, "PERIODIC SEAMS", ha="center", va="bottom", fontsize=12,
            fontweight="semibold", color=ds.MIDNIGHT)

    # one particle: straight flight (Cartesian) out through the right seam, back in at the left
    p0 = np.array(pol(r_of(300.0), HALF * 0.35))
    d = np.array([np.sin(np.deg2rad(64)), np.cos(np.deg2rad(64))])
    # intersection with the right seam line (through the centre at angle HALF)
    n = np.array([np.cos(HALF), -np.sin(HALF)])                  # normal of the right seam
    t = n @ (np.array([CX, CY]) - p0) / (n @ d)
    hit = p0 + t * d
    rot = np.array([[np.cos(2 * HALF), -np.sin(2 * HALF)], [np.sin(2 * HALF), np.cos(2 * HALF)]])  # to the left
    c = np.array([CX, CY])
    hit2 = c + rot @ (hit - c)                                     # same point on the left seam
    d2 = rot @ d
    p1 = hit2 + 0.62 * d2
    path = dict(color=ds.MIDNIGHT, lw=1.3, zorder=5)
    ax.plot([p0[0], hit[0]], [p0[1], hit[1]], **path)
    ax.plot([hit2[0], p1[0]], [hit2[1], p1[1]], **path)
    ax.add_patch(FancyArrowPatch(tuple(p1 - 0.05 * d2), tuple(p1), arrowstyle="-|>", mutation_scale=10,
                                 lw=1.3, color=ds.MIDNIGHT, shrinkA=0, shrinkB=0, zorder=5))
    ax.plot(*p0, "o", ms=5, mew=0, color=ds.MIDNIGHT, zorder=6)
    for q in (hit, hit2):
        ax.plot(*q, "o", ms=5.5, mfc=ds.CULTURED, mec=ds.MIDNIGHT, mew=1.1, zorder=6)

    if not notes:
        return
    xb, yb = pol(R_IN, 0)
    ax.text(xb, yb - 0.30, rf"1 $\lambda_h$ = {G['sector_width_km']:.0f} km  ·  {G['n_azi']} cells × "
            f"{G['azi_cell_km']:.0f} km", ha="center", va="top", fontsize=13)
    ax.text(xb, yb - 0.62, "curvature is exact; only the grid is truncated", ha="center", va="top",
            fontsize=12, style="italic")


def side_view(ax):
    ax.add_patch(Rectangle((FX0, FY0), FX1 - FX0, FY1 - FY0, fill=False, ec=ds.GRANITE, lw=0.8, alpha=0.6))
    ax.text(FX0 + 0.16, FY1 - 0.16, "SIDE VIEW · POLAR BAND", ha="left", va="top", fontsize=11,
            fontweight="semibold", color=ds.GRANITE)
    apex = np.array([FX0 + 0.30, 1.95])
    L = 2.70
    up = np.array([np.cos(BETA_DRAW), np.sin(BETA_DRAW)])
    dn = np.array([np.cos(BETA_DRAW), -np.sin(BETA_DRAW)])
    ax.add_patch(Polygon([apex, apex + L * up, apex + L * dn], closed=True, fc=ds.CELTIC, alpha=0.10,
                         lw=0, zorder=0))
    for e in (up, dn):
        ax.plot(*zip(apex, apex + L * e), color=ds.GRANITE, lw=1.4, zorder=2)
    ax.plot(*zip(apex, apex + (L + 0.05) * np.array([1, 0])), color=ds.GRANITE, lw=0.6, ls=(0, (1, 2.5)),
            zorder=1)
    ax.add_patch(Arc(apex, 1.5, 1.5, theta1=0, theta2=np.rad2deg(BETA_DRAW), color=ds.GRANITE, lw=0.8))
    ax.text(*(apex + 0.82 * np.array([np.cos(BETA_DRAW / 2), np.sin(BETA_DRAW / 2)])),
            rf"$\beta = {G['beta_deg']:.1f}^\circ$", ha="left", va="center", fontsize=13)

    # a particle hits the upper cone wall at q
    q = apex + 2.05 * up
    v_in = np.array([np.cos(np.deg2rad(62)), np.sin(np.deg2rad(62))])
    p_start = q - 0.78 * v_in
    t_hat = up                                                   # cone tangent in this view
    n_hat = np.array([-up[1], up[0]])                            # wall normal (e_theta direction)
    v_fix = v_in - 2 * (v_in @ n_hat) * n_hat                    # flip v_theta
    v_old = np.array([v_in[0], -v_in[1]])                        # flip v_z (plane z = const)
    seg = 0.80
    ax.plot(*zip(p_start, q), color=ds.MIDNIGHT, lw=1.6, zorder=4)
    ax.plot(*p_start, "o", ms=5, mew=0, color=ds.MIDNIGHT, zorder=5)
    ax.plot(*zip(q, q + (seg - 0.06) * v_old), color=ds.GRANITE, lw=1.3, ls=(0, (3, 2.5)), zorder=3)
    ax.add_patch(FancyArrowPatch(tuple(q + (seg - 0.07) * v_old), tuple(q + seg * v_old), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color=ds.GRANITE, shrinkA=0, shrinkB=0, zorder=3))
    ax.add_patch(FancyArrowPatch(tuple(q), tuple(q + seg * v_fix), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.6, color=ds.MIDNIGHT, shrinkA=0, shrinkB=0, zorder=4))
    ax.plot(*zip(q - 0.30 * t_hat, q + 0.30 * t_hat), color=ds.MIDNIGHT, lw=0.6, alpha=0.5, zorder=3)
    ds.takeaway(ax, "fix: reflect off\nthe cone itself", xy=tuple(q + 0.62 * v_fix + np.array([0.04, 0.0])),
                xytext=(FX0 + 1.30, 3.12), ha="left", va="center", linespacing=1.1)
    ax.text(FX0 + 0.16, FY0 + 0.14, "dashed: old flat-plane bounce", ha="left", va="bottom",
            fontsize=12, color=ds.GRANITE)


def main():
    fig, ax = ds.figure("evidence")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    w, h = ds.SIZES["evidence"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.text(FX1, 0.08, "NOT TO SCALE", ha="right", va="bottom", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)
    main_view(ax)
    side_view(ax)
    save(fig, "fig07a_wheel_sector", "deck/data/slide07_facts.json",
         f"schematic; MarsAtmos {FACTS['sources']['marsatmos_commit']}")
    plt.close(fig)


if __name__ == "__main__":
    main()
