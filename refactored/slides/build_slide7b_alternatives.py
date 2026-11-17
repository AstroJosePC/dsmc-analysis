"""Review PDF for slide 7, round 2: 7B beside two audience-facing redesigns (7B-1, 7B-2).

    cd refactored && conda run -n dsmc python slides/build_slide7b_alternatives.py

Both redesigns keep 7B's two views (the wheel sector in 3D and the full ring at true scale) and drop the
validation ladder, the cone-wall schematic and the no-wave check card. They spend the freed space on
what the 1/170 particle saving buys:
  7B-1  figure left (fig07d_sector_ring); word column: the cost hero, three numbered payoffs, one
        progress line
  7B-2  full-width strip (fig07e_sector_strip: ring, sector, the full ring as 170 sector tiles) over
        three metrics: cells per wavelength, runs per day, what it opens up

Every number is read from deck/data/slide07_facts.json (deck/cache_slides78.py), including the
finished-run wall times. "Fluid coupling adds no measurable cost" is the measured result of the coupled
smoke run in MarsAtmos-TruncBase reports/1B_noise_floor.md (25.2 s/output at np = 2 vs 9.88 s at
np = 5 uncoupled, both about 50 core-s per output).

Outputs: slides/slide7b_alternatives_review.pdf, slides/build/slide7b_review_preview/slide-N.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402
import build_slide7_alternatives as s7  # noqa: E402
from build_deck import (EVID_X, FOOT_RULE_Y, M, RIGHT, SANS, SERIF, TRACK_LABEL, WORD_W, WORD_X,  # noqa: E402
                        Slide, cite, ds, fp, footer, header, number_badge)
from matplotlib.patches import Circle  # noqa: E402

REVIEW_PDF = bd.REFACTORED / "slides" / "slide7b_alternatives_review.pdf"
PREVIEW = bd.OUT / "slide7b_review_preview"
WORK = bd.OUT / "_work_review7b"

F = s7.F
G = F["geometry"]
RUNS = F["runs_1b"]
KICK = "SCALING UP · IN PROGRESS"
CITE = cite("Full ring: Pérez Chávez et al., to be submitted, JGR Planets · wheel-sector runs: in progress")
ALL_DONE = RUNS["n"] == F["stage1b"]["planned"]


def fits(s: Slide, text, prop, width, where):
    if s.width(text, prop) > width:
        raise ValueError(f"slide {where}: '{text}' is wider than {width:.0f} pt")


def progress(s: Slide, x, y):
    """One status line: dot plus tracked caps."""
    color = ds.GREEN if ALL_DONE else ds.TANGELO
    text = (f"{RUNS['n']}/{F['stage1b']['planned']} SECTOR RUNS DONE · ANALYSIS UNDER WAY" if ALL_DONE
            else f"{RUNS['n']}/{F['stage1b']['planned']} SECTOR RUNS DONE")
    s.ax.add_patch(Circle((x + 5, y - 4.5), 4.0, fc=color, lw=0, zorder=4))
    s.tracked(x + 16, y, text, fp(SANS, 10.5), TRACK_LABEL)


# ================================================================= 7B-1: what the savings buy
def slide7b1():
    s = Slide(7)
    header(s, KICK, s7.HEAD)
    s7.tag(s, "REVIEW · 7B-1 · WHAT THE SAVINGS BUY")
    y0 = 160.0
    _, fh = s.place("fig07d_sector_ring_deck", EVID_X, y0)
    x = WORD_X
    s.tracked(x, y0 + 12, "COST PER RUN", fp(SANS, 11), TRACK_LABEL)
    s7.hero_with_unit(s, x, y0 + 58, f"1/{10 * round(s7.RATIO / 10):.0f}",
                      ["of the particles,", f"cells {s7.FINER:.1f}× finer"], size=52.0, unit_size=18.0)
    line = (f"{s7.sci_text(F['stage1b']['particles'])} in the sector vs {s7.sci_text(F['s6']['particles'])} "
            "in the full ring")
    fits(s, line, fp(SANS, 14, 300), WORD_W, "7B-1")
    s.text(x, y0 + 84, line, fp(SANS, 14, 300))

    s.tracked(x, y0 + 124, "WHAT THE SAVINGS BUY", fp(SANS, 11), TRACK_LABEL)
    items = (("Ensembles, not single runs",
              f"{RUNS['n']} runs done in {RUNS['span_h']:.0f} h on one workstation"),
             ("Waves fed in from below", "fluid coupling adds no measurable cost"),
             ("Room to grow", "multi-species gases and parameter sweeps"))
    y = y0 + 152
    for k, (lead, sub) in enumerate(items, 1):
        number_badge(s, x + 10.5, y - 5.5, k)
        fits(s, lead, fp(SANS, 16, "semibold"), WORD_W - 30, "7B-1")
        fits(s, sub, fp(SANS, 14, 300), WORD_W - 30, "7B-1")
        s.text(x + 30, y, lead, fp(SANS, 16, "semibold"), color=ds.MIDNIGHT)
        s.text(x + 30, y + 19, sub, fp(SANS, 14, 300))
        y += 50
    y_status = y0 + fh - 2
    if y - 50 + 19 + 16 > y_status:
        raise ValueError("slide 7B-1: payoffs run into the progress line")
    s.rule(x, y_status - 22, WORD_W, 0.8, ds.GRANITE, alpha=0.35)
    progress(s, x, y_status)
    footer(s, CITE)
    return s


# ================================================================= 7B-2: see the 170x, then three metrics
def metric(s: Slide, x, y0, label, hero, units, lines, size=40.0):
    s.tracked(x, y0, label, fp(SANS, 11), TRACK_LABEL)
    s7.hero_with_unit(s, x, y0 + 40, hero, units, size=size)
    for k, t in enumerate(lines):
        s.text(x, y0 + 40 + 18.5 * (k + 1) + 4, t, fp(SANS, 14, 300))


def slide7b2():
    s = Slide(7)
    header(s, KICK, s7.HEAD)
    s7.tag(s, "REVIEW · 7B-2 · SEE THE 170×")
    y_fig = 148.0
    _, fh = s.place("fig07e_sector_strip_deck", M, y_fig)
    gap = 28.0
    col_w = (RIGHT - M - 2 * gap) / 3
    y0 = y_fig + fh + 24
    lam = G["sector_width_km"]
    ring_cells = lam / F["s6"]["azi_cell_km"]

    metric(s, M, y0, "SHARPER WAVES", f"{G['n_azi']}", ["cells per", f"{lam:.0f} km wave"],
           [f"{ring_cells:.0f} on the full-ring grid:",
            f"{G['azi_cell_km']:.0f} km cells instead of {F['s6']['azi_cell_km']:.0f} km"])
    metric(s, M + col_w + gap, y0, "FAST ENOUGH FOR ENSEMBLES", f"{RUNS['n']}",
           ["runs in", f"{RUNS['span_h']:.0f} hours"],
           [f"{RUNS['n_wave']} with waves, {RUNS['n_nowave']} without",
            f"{RUNS['wall_h_min']:.0f}–{RUNS['wall_h_max']:.0f} h each on {RUNS['np']} cores of one workstation"])

    x = M + 2 * (col_w + gap)
    s.tracked(x, y0, "WHAT IT OPENS UP", fp(SANS, 11), TRACK_LABEL)
    y = y0 + 28
    for t in ("Waves from the fluid model below", "Multi-species atmospheres", "Parameter sweeps"):
        s.ax.add_patch(Circle((x + 5, y - 5), 3.5, fc=ds.MIDNIGHT, lw=0, zorder=4))
        fits(s, t, fp(SANS, 15), col_w - 18, "7B-2")
        s.text(x + 18, y, t, fp(SANS, 15))
        y += 22
    for k in range(2):
        xk = M + (k + 1) * (col_w + gap) - gap / 2
        s.rule(xk, y0 - 10, 0.6, 96, ds.GRANITE, alpha=0.35)
    if y0 + 40 + 2 * 18.5 + 4 + 8 > FOOT_RULE_Y or y - 22 + 8 > FOOT_RULE_Y:
        raise ValueError("slide 7B-2: metrics run into the footer")
    footer(s, CITE)
    return s


def current():
    s = s7.slide7b()
    return s


def main():
    for note in bd.audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    pages = [current(), slide7b1(), slide7b2()]
    for s in pages:
        for stem, *_ in s.placed:
            bd.audit_figure(stem)
        bd.check_layout(s)
    print("figures: sizes, fonts, provenance OK; layout: no clipping or collisions")
    bd.compose(pages, commit, REVIEW_PDF, f"NSBP/NSHP 2026 review: slide 7B and redesigns 7B-1, 7B-2; "
                                          f"figures/deck @ {commit}", work=WORK)
    n, fonts = bd.verify(len(pages), REVIEW_PDF, PREVIEW)
    print(f"{REVIEW_PDF.relative_to(bd.REFACTORED)}: {n} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")


if __name__ == "__main__":
    main()
