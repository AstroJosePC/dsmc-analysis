"""Review PDF for slide 7: the current slide beside two audience-facing alternatives (7A, 7B).

    cd refactored && conda run -n dsmc python slides/build_slide7_alternatives.py

Uses the chrome, audits, collision check and composition of build_deck.py unchanged. The 'current'
page is slide7_v1(), the slide 7 of the deck before 7B-2 was adopted. Figures come
from deck/fig07_alternatives.py; every number is read from deck/data/slide07_facts.json (rerun
deck/cache_slides78.py first when a source changes). The alternatives name each validation step in
plain words instead of the campaign's stage codes. A small Slate tag at the top right names the
variant; it is for review only.

Outputs: slides/slide7_alternatives_review.pdf, slides/build/slide7_review_preview/slide-N.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402
from build_deck import (EVID_BOTTOM, EVID_X, EVID_Y, FOOT_RULE_Y, M, RIGHT, SANS, TRACK_LABEL,  # noqa: E402
                        WORD_W, WORD_X, Card, Slide, cite, ds, facts, footer, fp, header, hero_sci, ladder_row)
from matplotlib.patches import Circle  # noqa: E402

REVIEW_PDF = bd.REFACTORED / "slides" / "slide7_alternatives_review.pdf"
PREVIEW = bd.OUT / "slide7_review_preview"
WORK = bd.OUT / "_work_review7"

F = bd.facts("slide07_facts")
HEAD = "To afford finer cells, we simulate a one-wavelength wheel sector instead of the full ring."
CITE = cite("Truncation-baseline campaign: no-wave test and cone-reflection fix, Nov 2026"
            " · full-ring count: Table S1")
RATIO = F["s6"]["particles"] / F["stage1b"]["particles"]
FINER = F["s6"]["azi_cell_km"] / F["geometry"]["azi_cell_km"]
NULL = F["null_test"]
ST = F["stage1b"]
MINUS = "−"


def sci_text(v):
    """6.24e6 -> '6.2 × $10^{6}$' (mathtext exponent inside Open Sans text)."""
    mant, exp = bd.sci(v)
    return f"{mant}$^{{{exp}}}$"


def steps():
    """Validation steps in plain words: (colour, text, status)."""
    noise_done = ST["finished"] == ST["planned"]
    return ((ds.GREEN, "Equilibrium null test (no wave)", "DONE"),
            (ds.GREEN, "Cone-tangent wall reflection fix", "DONE"),
            (ds.GREEN if noise_done else ds.TANGELO, "Baseline wave noise floor",
             "RUNS DONE" if noise_done else f"RUNNING {ST['finished']}/{ST['planned']}"),
            (ds.SLATE, "Domain-width & boundary tests", "NEXT"))


def ladder(s: Slide, x, y, right, size=14.0, pitch=20.5):
    """Draws the steps from baseline y; returns the last baseline."""
    for k, (color, text, status) in enumerate(steps()):
        yk = y + k * pitch
        s.ax.add_patch(Circle((x + 5, yk - 5), 4.0, fc=color, lw=0, zorder=4))
        s.text(x + 18, yk, text, fp(SANS, size))
        s.tracked(right, yk, status, fp(SANS, 10.5), TRACK_LABEL, ha="right")
    return yk


def tag(s: Slide, text: str):
    s.tracked(RIGHT, 46, text, fp(SANS, 10, "semibold"), 0.14, color=ds.SLATE, ha="right")


hero_with_unit = bd.hero_with_unit


# ================================================================= 7A: zoom strip over three callouts
def slide7a():
    s = Slide(7)
    header(s, "SCALING UP · IN PROGRESS", HEAD)
    tag(s, "REVIEW · 7A · ZOOM STRIP")
    y_fig = 148.0
    _, fh = s.place("fig07b_zoom_strip_deck", M, y_fig)
    gap = 28.0
    col_w = (RIGHT - M - 2 * gap) / 3
    y0 = y_fig + fh + 24
    small = fp(SANS, 14, 300)
    lead = 18.5

    x = M                                                       # 1 the cost
    s.tracked(x, y0, "COST PER RUN", fp(SANS, 11), TRACK_LABEL)
    hero_with_unit(s, x, y0 + 40, f"1/{10 * round(RATIO / 10):.0f}", ["of the", "particles"])
    s.text(x, y0 + 40 + lead + 4, f"{sci_text(F['stage1b']['particles'])} in the sector vs "
           f"{sci_text(F['s6']['particles'])}", small)
    s.text(x, y0 + 40 + 2 * lead + 4, f"in the full ring, with cells {FINER:.1f}× finer", small)

    x = M + col_w + gap                                         # 2 the no-wave check
    s.tracked(x, y0, "NO-WAVE CHECK AGAINST 1D", fp(SANS, 11), TRACK_LABEL)
    hero_with_unit(s, x, y0 + 40, f"{NULL['chi2_density_12x1d_fix']:.2f}",
                   [r"$\chi^2$/dof, density", f"({NULL['chi2_density_12x1d_pre']:,.0f} before the fix)"],
                   unit_size=14.0)
    s.text(x, y0 + 40 + lead + 4, f"particle budget closes: R = {NULL['budget_R_fix']:.1f} ± "
           f"{NULL['budget_R_fix_se']:.1f}".replace("-", MINUS), small)
    s.text(x, y0 + 40 + 2 * lead + 4, f"escape within {abs(NULL['escape_rel_pct_fix']):.1f}% of 1D "
           f"({abs(NULL['escape_rel_pct_pre']):.0f}% low before)", small)

    x = M + 2 * (col_w + gap)                                   # 3 progress
    s.tracked(x, y0, "VALIDATION STEPS", fp(SANS, 11), TRACK_LABEL)
    end = ladder(s, x, y0 + 26, RIGHT, size=13.0, pitch=20.0)
    if end + 8 > FOOT_RULE_Y:
        raise ValueError("slide 7A: validation steps run into the footer")
    footer(s, CITE)
    return s


# ================================================================= 7B: one multi-view figure, word column
def slide7b():
    s = Slide(7)
    header(s, "SCALING UP · IN PROGRESS", HEAD)
    tag(s, "REVIEW · 7B · ONE SECTOR, TWO VIEWS")
    y0 = 160.0
    _, fh = s.place("fig07c_sector_3d_deck", EVID_X, y0)
    x = WORD_X
    s.tracked(x, y0 + 12, "COST PER RUN", fp(SANS, 11), TRACK_LABEL)
    hero_with_unit(s, x, y0 + 58, f"1/{10 * round(RATIO / 10):.0f}", ["of the particles,",
                                                                      f"cells {FINER:.1f}× finer"],
                   size=52.0, unit_size=18.0)
    line = (f"{sci_text(F['stage1b']['particles'])} in the sector vs {sci_text(F['s6']['particles'])} "
            "in the full ring")
    if s.width(line, fp(SANS, 14, 300)) > WORD_W:
        raise ValueError("slide 7B: cost line wraps")
    s.text(x, y0 + 84, line, fp(SANS, 14, 300))
    s.tracked(x, y0 + 120, "VALIDATION STEPS", fp(SANS, 11), TRACK_LABEL)
    end = ladder(s, x, y0 + 145, RIGHT, pitch=20.5)
    card = Card("NO-WAVE CHECK · SECTOR VS 1D",
                eq=[[(rf"$\chi^2/\mathrm{{dof}} = {NULL['chi2_density_12x1d_fix']:.2f}$", fp(SANS, 20), ds.MIDNIGHT),
                     (f"  density ({NULL['chi2_density_12x1d_pre']:,.0f} before the fix)", fp(SANS, 13.5, 300),
                      ds.GRANITE)]],
                lines=[(f"Particle budget closes: R = {NULL['budget_R_fix']:.1f} ± {NULL['budget_R_fix_se']:.1f}"
                        .replace("-", MINUS), 14.5, ds.GRANITE, 300),
                       (f"Escape within {abs(NULL['escape_rel_pct_fix']):.1f}% of 1D "
                        f"({abs(NULL['escape_rel_pct_pre']):.0f}% low before)", 14.5, ds.GRANITE, 300)])
    _, ch = card.layout()
    y_card = y0 + fh - ch
    if end + 12 > y_card:
        raise ValueError(f"slide 7B: validation steps (last baseline {end:.0f}) run into the card at {y_card:.0f}")
    card.draw(s, WORD_X, y_card)
    footer(s, CITE)
    return s


def slide7_v1():
    """Slide 7 as built before 7B-2 was adopted (FIG 7a, ladder, null-test card)."""
    s = Slide(7)
    f = facts("slide07_facts")
    header(s, "SCALING UP · IN PROGRESS",
           "To afford finer cells, we simulate a one-wavelength wheel sector instead of the full ring.")
    s.place("fig07a_wheel_sector_deck", EVID_X, EVID_Y)

    # particles per run: full ring (S6, Table S1) vs wheel sector (Stage 1B, measured)
    s.tracked(WORD_X, EVID_Y + 10, "PARTICLES PER RUN", fp(SANS, 11), TRACK_LABEL)
    desc = fp(SANS, 13, 300)
    rows = ((f["s6"]["particles"], ds.GRANITE, "full ring (S6)", f"{f['s6']['azi_cell_km']:.0f} km cells"),
            (f["stage1b"]["particles"], ds.MIDNIGHT, "wheel sector (1B)",
             f"{f['geometry']['azi_cell_km']:.0f} km cells"))
    y = EVID_Y + 44
    for value, color, a, b in rows:
        w = hero_sci(s, WORD_X, y, value, color=color)
        s.text(WORD_X + 150, y - 14, a, desc, color=color if color == ds.MIDNIGHT else ds.GRANITE)
        s.text(WORD_X + 150, y + 1, b, desc)
        if w > 142:
            raise ValueError("slide 7: hero number runs into its description")
        y += 36
    ratio = f["s6"]["particles"] / f["stage1b"]["particles"]
    finer = f["s6"]["azi_cell_km"] / f["geometry"]["azi_cell_km"]
    s.text(WORD_X, y - 12, f"≈ 1/{10 * round(ratio / 10):.0f} of the particles, with cells "
           f"{finer:.1f}× finer", fp(SANS, 13.5), color=ds.MIDNIGHT)

    # status ladder
    st = f["stage1b"]
    y = EVID_Y + 126
    for color, text, status in ((ds.GREEN, "0C null test, no wave", "DONE"),
                                (ds.GREEN, "Cone-reflection fix", "DONE"),
                                (ds.GREEN if st["finished"] == st["planned"] else ds.TANGELO, "1B noise floor",
                                 "DONE" if st["finished"] == st["planned"]
                                 else f"RUNNING {st['finished']}/{st['planned']}"),
                                (ds.SLATE, "2 truncation twins", "NEXT")):
        ladder_row(s, y, color, text, status)
        y += 19.5

    n = f["null_test"]
    card = Card("NULL TEST · SECTOR VS 1D, NO WAVE",
                eq=[[(rf"$\chi^2/\mathrm{{dof}} = {n['chi2_density_12x1d_fix']:.2f}$", fp(SANS, 20), ds.MIDNIGHT),
                     (f"  density ({n['chi2_density_12x1d_pre']:,.0f} before fix)", fp(SANS, 13.5, 300),
                      ds.GRANITE)]],
                lines=[(f"Budget closes: R = {n['budget_R_fix']:.1f} ± {n['budget_R_fix_se']:.1f} "
                        "per block".replace("-", "−"), 14.5, ds.GRANITE, 300),
                       (f"Escape {abs(n['escape_rel_pct_fix']):.1f}% below 1D "
                        f"({abs(n['escape_rel_pct_pre']):.0f}% before)", 14.5, ds.GRANITE, 300)])
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Truncation-baseline campaign · Stage 0C report and cone-reflection fix, Nov 2026"
                   " · S6: Table S1"))
    return s


def current():
    s = slide7_v1()
    tag(s, "REVIEW · CURRENT")
    return s


def main():
    for note in bd.audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    pages = [current(), slide7a(), slide7b()]
    for s in pages:
        for stem, *_ in s.placed:
            bd.audit_figure(stem)
        bd.check_layout(s)
    print("figures: sizes, fonts, provenance OK; layout: no clipping or collisions")
    bd.compose(pages, commit, REVIEW_PDF, f"NSBP/NSHP 2026 review: slide 7 and alternatives 7A, 7B; "
                                          f"figures/deck @ {commit}", work=WORK)
    n, fonts = bd.verify(len(pages), REVIEW_PDF, PREVIEW)
    print(f"{REVIEW_PDF.relative_to(bd.REFACTORED)}: {n} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")


if __name__ == "__main__":
    main()
