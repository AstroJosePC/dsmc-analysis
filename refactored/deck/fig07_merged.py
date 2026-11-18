"""Slide 7 · FIG 7 (merged slide: the wheel sector and the fluid model that drives it), as built:

  fig07m_ring      split_left  (6.25 x 3.15 in)  build step 1 only: the full ring at true scale, the
                                                 sector dark
  fig07m_sector    split_left  (6.25 x 3.15 in)  from step 2: the wheel sector in 3D (FIG 7e's drawing)
                                                 in the ring's place, the true-scale ring kept as a
                                                 thumbnail (RING_INSET)
  fig07m_budget    split_band  (6.25 x 1.60 in)  from step 1: particles per run, the full ring as 170
                                                 sector tiles
  fig07m_profile   split_right (5.64 x 3.14 in)  step 3: FIG 8a (MAGNUS-P vs linear viscous theory)
                                                 redrawn for the right column, above the COUPLING CHECK card

Slide grid (points, y down; slides/build_deck.slide7): content 150-492; left column 40-490 (6.25 in),
gutter 0.33 in, right column 514-920 (5.64 in). This slide trades the deck's 7.22 / 4.67 in split for
6.25 / 5.64 in so FIG 8a keeps its labels at a legible size; the three slots are deckstyle SIZES
(STYLE_VERSION 1.3).

Drawing code is reused: ring(), sector_3d(), sector_labels() and tiles() from fig07_alternatives.py
(FIG 7e) and the FIG 8a recipe of fig08_baselines.py, with label positions refit to the smaller
slots. Numbers come from deck/data/slide07_facts.json, slide04_facts.json and fig08a_plate1A.npz
(deck/cache_slides78.py).

Each figure passes audit_text() (measured ink) before it is saved: every text inside the figure, no two texts
overlapping, and no in-axes label crossing a plotted line.

    cd refactored && conda run -n dsmc python deck/fig07_merged.py
"""
import itertools
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, NullLocator

import fig07_alternatives as f7
from _deck import R_MARS_KM, ds, save
from fig08_baselines import H_KM, pct

DATA = Path(__file__).resolve().parent / "data"
F7 = f7.F7
SRC7 = (*f7.SOURCES, f"schematic + true-scale ring; MarsAtmos {F7['sources']['marsatmos_commit']}")

SLOT = {"fig07m_ring": "split_left", "fig07m_sector": "split_left", "fig07m_budget": "split_band",
        "fig07m_profile": "split_right"}                  # deckstyle SIZES key of each figure
RING_INSET = True          # keep the true-scale ring as a thumbnail once the sector replaces it
X_TAB = 0.043             # left edge of the misfit table (m/s, log axis)
DOMAIN_CX = 2.62           # common centre (in) of the step-1 ring and the step-2 sector
R_OUT = (R_MARS_KM + f7.F4["z_top"]) / R_MARS_KM      # outer radius of ring() per unit Mars radius


# ================================================================= audit
def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).astype(np.int16).copy()     # RGBA: figures are transparent


def audit_text(fig, name: str, dpi: float = 200.0, tol: int = 30, pad: int = 1):
    """Measured on rendered ink, as build_deck.ink_rect does for placed figures: each text's ink box is
    the difference between the figure with that one text and with no text (annotation arrows left out).
    Checks: every text inked and clear of the figure edge (not clipped); no two text ink boxes within
    pad px; no text ink box over the ink of a plotted line or marker (Line2D, ticks excluded).
    Tick and axis labels are left to constrained layout and not checked.
    Returns the list of problems (empty: clean)."""
    old_dpi, engine = fig.dpi, fig.get_layout_engine()
    fig.set_dpi(dpi)
    fig.canvas.draw()
    fig.set_layout_engine("none")              # freeze: hiding labels must not re-run constrained layout
    managed = set()                            # tick and axis labels: placed by constrained layout
    for ax in fig.axes:
        managed |= {*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label,
                    ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()}
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.figure is fig
             and t not in managed]
    arrows = [t.arrow_patch for t in texts if getattr(t, "arrow_patch", None) is not None]
    lines = [l for l in fig.findobj(Line2D) if l.get_visible() and l.axes is not None
             and l not in l.axes.get_xticklines() + l.axes.get_yticklines()]
    for a in (*texts, *lines):
        a.set_visible(False)
    base = _pixels(fig)
    line_ink = []                              # one mask per line, so a hit names its line
    for l in lines:
        l.set_visible(True)
        line_ink.append((l, np.abs(_pixels(fig) - base).max(-1) > tol))
        l.set_visible(False)
    for a in arrows:
        a.set_visible(False)
    boxes, problems = [], []
    for t in texts:
        t.set_visible(True)
        ink = np.abs(_pixels(fig) - base).max(-1) > tol
        t.set_visible(False)
        label = repr(t.get_text())
        rows, cols = np.nonzero(ink)
        if not rows.size:
            problems.append(f"no ink (outside the figure?): {label}")
            continue
        b = (cols.min(), rows.min(), cols.max(), rows.max())
        if b[0] == 0 or b[1] == 0 or b[2] == ink.shape[1] - 1 or b[3] == ink.shape[0] - 1:
            problems.append(f"clipped at the figure edge: {label}")
        boxes.append((label, b))
        for l, m in line_ink:
            if m[max(b[1] - pad, 0):b[3] + pad + 1, max(b[0] - pad, 0):b[2] + pad + 1].any():
                where = l.get_gid() or f"{l.get_linestyle()} {l.get_color()} line"
                problems.append(f"text on a line: {label} / {where}")
    for (n1, a), (n2, b) in itertools.combinations(boxes, 2):
        if a[0] <= b[2] + pad and b[0] <= a[2] + pad and a[1] <= b[3] + pad and b[1] <= a[3] + pad:
            problems.append(f"text overlap: {n1} / {n2}")
    for a in (*texts, *lines, *arrows):
        a.set_visible(True)
    fig.set_layout_engine(engine)
    fig.set_dpi(old_dpi)
    return sorted(set(problems))


def finish(fig, name, *inputs):
    problems = audit_text(fig, name)
    if problems:
        dbg = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "tmp" / f"{name}_audit.png"
        dbg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dbg, dpi=150)
        raise SystemExit(f"{name} (preview: {dbg}):\n  " + "\n  ".join(problems))
    save(fig, name, *inputs)
    plt.close(fig)


# ================================================================= the domain slot
def domain_ring():
    w, h = ds.SIZES[SLOT["fig07m_ring"]]
    fig, ax = f7.canvas(w, h)
    rm, cy = 0.98, 1.66
    f7.ring(ax, DOMAIN_CX, cy, rm, angle_deg=90.0)
    x_cap = DOMAIN_CX + R_OUT * rm + 0.36
    ax.text(x_cap, cy + 0.06, "FULL RING, TRUE SCALE", ha="left", va="bottom", fontsize=12,
            fontweight="semibold")
    ax.text(x_cap, cy - 0.02, "dark slice: the sector", ha="left", va="top", fontsize=13)
    finish(fig, "fig07m_ring", *SRC7)


def domain_sector():
    w, h = ds.SIZES[SLOT["fig07m_sector"]]
    fig, ax = f7.canvas(w, h)
    f7.PLACE.update(scale=1.42, ox=DOMAIN_CX + 0.10, oy=-1.62)
    lat_f = f7.sector_3d(ax)["lat_front"]
    f7.sector_labels(ax, lat_f, notes=False)
    ax.text(0.02, h - 0.05, "NOT TO SCALE", ha="left", va="top", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)
    if RING_INSET:                                  # top right, clear of the seam arrow and r labels
        rm = 0.36
        rx, ry = w - 0.04 - R_OUT * rm, h - 0.06 - R_OUT * rm
        f7.ring(ax, rx, ry, rm, angle_deg=90.0)
        ax.texts[-1].remove()                       # no "MARS" in a 0.7 in disc
        ax.text(rx, ry - R_OUT * rm - 0.08, "FULL RING,\nTRUE SCALE", ha="center", va="top",
                fontsize=11, fontweight="semibold", color=ds.GRANITE, linespacing=1.15)
    finish(fig, "fig07m_sector", *SRC7)


# ================================================================= particles per run
def budget():
    """The full ring's particles as 170 sector-sized tiles in 5 rows; the one sector dark."""
    w, h = ds.SIZES[SLOT["fig07m_budget"]]
    fig, ax = f7.canvas(w, h)
    ratio = F7["s6"]["particles"] / F7["stage1b"]["particles"]
    n = 10 * round(ratio / 10)
    rows = 5
    cols = n // rows
    gap = 0.032
    tile = (w - 0.04 - (cols - 1) * gap) / cols
    x0, y0 = 0.02, 0.34
    x1, y1 = f7.tiles(ax, x0, y0, tile, gap, n, cols)
    ax.text(x0, y1 + 0.09, "PARTICLES PER RUN", ha="left", va="bottom", fontsize=12, fontweight="semibold")
    mant, exp = f7.sci_parts(F7["s6"]["particles"])
    ax.text(x1, y1 + 0.09, rf"full ring: {mant}$^{{{exp}}}$ = {n} sectors", ha="right", va="bottom",
            fontsize=13)
    mant, exp = f7.sci_parts(F7["stage1b"]["particles"])
    ds.takeaway(ax, rf"one sector: {mant}$^{{{exp}}}$", xy=(x0 + tile + 0.02, y0 + tile / 2),
                xytext=(x0 + tile + 0.40, 0.15), ha="left", va="center")
    if y1 + 0.09 + 0.25 > h:
        raise SystemExit(f"fig07m_budget: header at {y1 + 0.09:.2f} in does not fit {h} in")
    finish(fig, "fig07m_budget", *SRC7)


# ================================================================= FIG 8a for the right column
def profile():
    """FIG 8a (fig08_baselines.fig8a) at 5.64 x 3.14 in: same data, marks and labels; labels that sat on
    the left of the curve move to free space on its right."""
    d = np.load(DATA / "fig08a_plate1A.npz")
    z, wm, wl = d["z_km"], d["w_model"], d["w_linear"]
    fig, ax = plt.subplots(figsize=ds.SIZES[SLOT["fig07m_profile"]])

    ax.axhspan(180, 200, color=ds.SLATE, alpha=0.15, lw=0, zorder=0)
    ds.label(ax, "absorbing layer", (0.25, 190), color=ds.GRANITE, ha="left", va="center")
    ds.input_altitudes(ax)
    for line in ax.lines:
        line.set_gid("input-altitude line")
    zi = np.linspace(0, 200, 200)
    ax.plot(0.1 * np.exp(zi / (2 * H_KM)), zi, color=ds.ROLE["prior_work"], ls=ds.LINESTYLE["prior_work"],
            lw=1.5, zorder=2, gid="no-viscosity line")
    ax.plot(wm, z, color=ds.ROLE["this_work"], lw=2.2, zorder=3, gid="fluid-model curve")
    every = slice(0, None, 20)                                  # 0.5 km cells -> a circle every 10 km
    ax.plot(wl[every], z[every], "o", ms=6.5, mfc="none", mec=ds.ROLE["theory"], mew=1.3, zorder=4,
            gid="linear-theory circles")

    # misfit table on the input-altitude lines: each row's ink centred in the 16 pt between two lines
    # (baseline 3.6 km above its line), the header above the 140 km row, all left of the curve
    for zk, m in zip(d["level_km"], d["misfit_pct"]):
        ax.text(X_TAB, zk + 3.6, f"{zk:.0f} km", ha="left", va="baseline", fontsize=13)
        ax.text(X_TAB * 1.86, zk + 3.6, f"{pct(m)}%", ha="left", va="baseline", fontsize=13,
                fontweight="semibold", color=ds.MIDNIGHT)
    ds.label(ax, "model vs theory", (X_TAB, 163.5), ha="left", va="baseline")

    ds.label(ax, "fluid model", (0.25, 165), color=ds.ROLE["this_work"], ha="left", va="center")
    ds.label(ax, "linear viscous\ntheory", (0.18, 19), color=ds.ROLE["theory"], ha="left", va="center",
             linespacing=1.0)
    # the undamped reference: name in the 120-140 km band, its law in the 100-120 km band, both right
    # of the dashed line
    ds.label(ax, "no viscosity", (0.70, 124.5), color=ds.ROLE["prior_work"], ha="left", va="baseline")
    ax.text(0.70, 102.3, r"$\propto e^{z/2H}$", color=ds.ROLE["prior_work"],     # 12 pt: the superscript
            fontsize=plt.rcParams["legend.fontsize"] - 1, ha="left", va="baseline")  # stays under 120 km
    zp = float(d["peak_z_km"])
    ds.takeaway(ax, f"Viscous damping wins\nabove {zp:.0f} km", xy=(np.interp(zp, z, wm) * 1.07, zp),
                xytext=(0.42, 66), ha="left", va="center", linespacing=1.1)

    ax.set_xscale("log")
    ax.set_xlim(0.04, 2.0)                   # FIG 8a: 0.05; room for the misfit table
    ax.set_ylim(0, 200)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([0, 40, 80, 120, 160, 200])
    ax.set_xlabel("Wave vertical-wind amplitude (m/s)")
    ax.set_ylabel("Altitude (km)")
    meta = json.loads(str(d["meta"]))
    finish(fig, "fig07m_profile", "deck/data/fig08a_plate1A.npz",
           f"MAGNUS-P {meta['run']} {Path(meta['source']).name} @ MAGNUS-P {meta['magnusp_commit']}")


if __name__ == "__main__":
    domain_ring()
    domain_sector()
    budget()
    profile()
