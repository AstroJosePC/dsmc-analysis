"""Slide 4 (adopted 4D): the scale hierarchy and one time step in one figure.

Left to right, three zoom levels joined by zoom lines:
  1. the S6 domain at true scale (75 shells x 1,024 columns, 100-801 km; forced patch in Tangelo),
  2. a patch of its cells (schematic, curvature exaggerated) with the three operations of one time
     step numbered on it: 1 move (straight flights that change cell), 2 collide (a random pair in
     one cell), 3 boundaries (wave flux in at the base, escape at the top),
  3. one computational particle = W O atoms with one velocity.
The slide text under the figure explains steps 1-3; the badges here match it.

Build steps (progressive deck, deck/_deck.Build): 1 the domain, 2 the cells, 3 move, 4 collide,
5 boundaries, 6 the particle and W; fig04d_scales_and_step_b1 ... _b5 hold steps 1-5.

Numbers are read, not typed: W and dt from the S6 Info/Simulation_Parameters.ascii, grid from the
run parameters, particles per cell and the shortest collision time from the S6 pre-forcing transport
cache (see fig04_alternatives.s6_numbers). They are also written to deck/data/slide04_facts.json,
which slides/build_deck.py reads for the step text, so the deck build does not need the archive.

    cd refactored && conda run -n dsmc python deck/fig04_method.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge

from _deck import R_MARS_KM, Build, commit, ds
from fig04_alternatives import TRANSPORT, draw_plate, s6_numbers, sci

STEM = "fig04d_scales_and_step"
FACTS = Path(__file__).resolve().parent / "data" / "slide04_facts.json"

# Plate crop in its own (4A) inch coordinates, and where it goes in this figure
PLATE_X, PLATE_Y = (0.60, 6.50), (0.12, 3.98)
STEP = {"domain": 1, "cells": 2, "move": 3, "collide": 4, "boundaries": 5, "particle": 6}


def write_facts(num):
    keys = ("W", "dt", "tau_100", "z_bot", "z_top", "n_r", "n_phi", "n_part", "ppc_exo", "dr_exo",
            "dphi_exo_km")
    out = {"created": date.today().isoformat(), "run": "S6",
           "sources": {"parameters": str(num["source"]), "transport_cache": str(TRANSPORT),
                       "dsmc_analyzer_commit": commit()},
           **{k: float(num[k]) for k in keys}}
    FACTS.write_text(json.dumps(out, indent=1) + "\n")


def main():
    num = s6_numbers()
    write_facts(num)
    w, h = ds.SIZES["plate"]
    fig = plt.figure(figsize=(w, h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    build = Build(fig)
    cap_y, mid_y = 0.40, 2.30
    cap = dict(ha="center", fontsize=12, fontweight="semibold")
    sub = dict(ha="center", va="top", fontsize=13)
    zoom = dict(color=ds.GRANITE, lw=0.5, alpha=0.6, zorder=0)

    # 1 the domain, true scale
    cx, cy, Rm = 1.45, mid_y, 0.86
    k = Rm / R_MARS_KM
    r0, r1 = (R_MARS_KM + num["z_bot"]) * k, (R_MARS_KM + num["z_top"]) * k
    ax.add_patch(Circle((cx, cy), Rm, fc=ds.GRANITE, alpha=0.12, lw=0))
    ax.add_patch(Wedge((cx, cy), r1, 0, 360, width=r1 - r0, fc=ds.MIDNIGHT, alpha=0.18, lw=0))
    for r in (r0, r1):
        ax.add_patch(Circle((cx, cy), r, fc="none", ec=ds.MIDNIGHT, lw=0.6))
    half = np.deg2rad(88 * 360 / num["n_phi"] / 2)                 # forced patch: 88 columns
    aa = np.linspace(-np.pi / 2 - half, -np.pi / 2 + half, 80)
    ax.plot(cx + (r0 - 0.035) * np.cos(aa), cy + (r0 - 0.035) * np.sin(aa), color=ds.ROLE["forcing"],
            lw=3.0, solid_capstyle="butt")
    ax.text(cx, cy, "MARS", ha="center", va="center", fontsize=12, fontweight="semibold")
    a_s = -np.pi / 2 + 0.6 * half                                  # a column inside the patch
    rm = 0.5 * (r0 + r1)
    sx, sy = cx + rm * np.cos(a_s), cy + rm * np.sin(a_s)
    b1 = 0.30
    ax.add_patch(Rectangle((sx - b1 / 2, sy - b1 / 2), b1, b1, fc="none", ec=ds.MIDNIGHT, lw=1.2, zorder=5))
    ax.text(cx, cap_y + 0.02, "THE S6 DOMAIN, TRUE SCALE", va="bottom", **cap)
    ax.text(cx, cap_y - 0.04, f"{num['n_r']} × {num['n_phi']:,} cells, "
            f"{num['z_bot']:.0f}–{num['z_top']:.0f} km", **sub)

    # 2 the cells, one time step (the 4A plate, scaled into an inset axes)
    build.at(STEP["cells"])
    pw_n, ph_n = PLATE_X[1] - PLATE_X[0], PLATE_Y[1] - PLATE_Y[0]
    py0, py1 = 0.76, h - 0.04
    s = (py1 - py0) / ph_n
    pw = pw_n * s
    px0 = 0.5 * (3.55 + 8.70) - pw / 2
    ins = fig.add_axes([px0 / w, py0 / h, pw / w, (py1 - py0) / h])
    ins.set_axis_off()
    ins.patch.set_visible(False)
    ins.set_xlim(*PLATE_X)
    ins.set_ylim(*PLATE_Y)
    ins.set_aspect("equal")
    plate = draw_plate(ins, num, short_labels=True, badge_r=0.17, mark=lambda layer: build.at(STEP[layer]))
    build.at(STEP["cells"])

    def to_main(xn, yn):
        return px0 + (xn - PLATE_X[0]) * s, py0 + (yn - PLATE_Y[0]) * s

    xy, r_edges, a_edges = plate["xy"], plate["r_edges"], plate["a_edges"]
    for (r, a), corner in (((plate["r_top"], a_edges[0]), (sx + b1 / 2, sy + b1 / 2)),
                           ((r_edges[0] - 0.45, a_edges[0]), (sx + b1 / 2, sy - b1 / 2))):
        tx, ty = to_main(*xy(r, a))
        ax.plot([corner[0], tx], [corner[1], ty], **zoom)
    pc = px0 + pw / 2
    ax.text(pc, cap_y + 0.02, "SPHERICAL CELLS, ONE TIME STEP", va="bottom", **cap)
    ax.text(pc, cap_y - 0.04, f"schematic; exobase cell {num['dphi_exo_km']:.0f} × {num['dr_exo']:.0f} km, "
            f"≈ {round(num['ppc_exo'], -3):,.0f} particles", **sub)

    # 3 one computational particle, picked from a quiet cell at the right of the plate
    build.at(STEP["particle"])
    i, j = 2, len(a_edges) - 2
    rc, ac = 0.5 * (r_edges[i] + r_edges[i + 1]), 0.5 * (a_edges[j] + a_edges[j + 1])
    cxy = np.array(xy(rc, ac))
    cand = [np.array([x, y]) for i2, j2, xs_, ys_ in plate["pts"] if (i2, j2) == (i, j)
            for x, y in zip(np.atleast_1d(xs_), np.atleast_1d(ys_))]
    p = cand[int(np.argmin([np.hypot(*(c - cxy)) for c in cand]))]
    ins.plot(*p, "o", ms=8, mfc="none", mec=ds.MIDNIGHT, mew=1.3, zorder=9)
    px, py = to_main(*p)
    lx, ly, lr = 10.75, mid_y, 0.86
    for sgn in (1, -1):
        ax.plot([px + 0.05, lx - 0.20], [py, ly + sgn * lr * 0.80], **zoom)
    ax.add_patch(Circle((lx, ly), lr, fc=ds.CULTURED, ec=ds.MIDNIGHT, lw=1.2, zorder=4))
    rng = np.random.default_rng(10)
    rr = lr * 0.86 * np.sqrt(rng.uniform(0, 1, 2400))
    th = rng.uniform(0, 2 * np.pi, rr.size)
    ax.scatter(lx + rr * np.cos(th), ly + rr * np.sin(th), s=0.5, lw=0, color=ds.GRANITE, alpha=0.75,
               zorder=5, rasterized=True)
    ax.annotate("", xy=(lx + 0.47, ly + 0.26), xytext=(lx - 0.26, ly - 0.16),
                arrowprops=dict(arrowstyle="-|>", color=ds.MIDNIGHT, lw=2.0, mutation_scale=13,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(lx + 0.50, ly + 0.30, r"$\mathbf{v}$", ha="left", va="bottom", fontsize=17, color=ds.MIDNIGHT,
            zorder=8)
    ax.text(lx, cap_y + 0.02, "ONE COMPUTATIONAL PARTICLE", va="bottom", **cap)
    ax.text(lx, cap_y - 0.04, f"= {sci(num['W'])} O atoms, one velocity", **sub)

    build.save(STEM, num["source"], TRANSPORT, "true-scale ring; schematic cells; W")
    plt.close(fig)
    print(f"{STEM}: plate scale {s:.3f}, plate {pw:.2f} in wide at x = {px0:.2f}; facts -> {FACTS.name}")


if __name__ == "__main__":
    sys.exit(main())
