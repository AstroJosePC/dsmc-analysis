"""PROTOTYPE: slides 7 (wheel sector) and 8 (fluid coupling) merged into one slide, as a standalone review PDF.

    cd refactored && conda run -n dsmc python deck/fig07m_merged_proto.py
                     conda run -n dsmc python slides/build_slide07_merged_preview.py [--docs DIR]

Writes slides/build/slide07_merged_preview.pdf: page 1 the full merged slide, pages 2-4 its build steps
  1  the full ring at true scale + particles per run (170 sector tiles)
  2  the wheel sector replaces the ring in the same slot (the budget stays)
  3  + MAGNUS-P vs linear viscous theory (FIG 8a redrawn) and the COUPLING CHECK card
and PNG previews in slides/build/slide07_merged_preview/. NSBP2026_deck_full.pdf and
NSBP2026_deck_progressive.pdf are not touched: this script uses its own work and preview folders.

Everything is build_deck.py's machinery, imported unchanged: Slide, header, footer, Card, the style and
figure audits, check_layout, compose and verify. Only runtime state differs: the prototype slots of
deck/fig07m_merged_proto.py are registered, and the talk counts 8 slides (07 / 08, eight dots).

One rule differs from build_deck.check_builds, because step 2 swaps a figure: a build page may put ink
where the full slide is blank only inside the slot of a figure that a later step removes (the ring).
Everything else must sit exactly where it is on the full slide, and step 3 must equal the full slide.

--docs: folder holding docs/deckstyle.py and docs/FIGURE_STYLE.md for the style-sync audit (default:
refactored/docs; both files are untracked, so a fresh worktree has neither and the audit then only
reports the canonical module).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402
from build_deck import (FOOT_BASE, M, RIGHT, SANS, Card, Slide, ds, fp,  # noqa: E402
                        figure_size, header)
from fig07m_merged_proto import PROTO_SLOTS  # noqa: E402

bd.SLOTS.update(PROTO_SLOTS)          # audit_figure checks each placed PDF against its slot
bd.N_SLIDES = 8                       # talk with 7 and 8 merged: counter 07 / 08, eight dots

OUT = bd.OUT
PDF = OUT / "slide07_merged_preview.pdf"
PREV = OUT / "slide07_merged_preview"
WORK = OUT / "_work_slide07_merged"
Y0 = 150.0                            # top of the content band (slides 7, 8: 148)
Y1 = 492.0                            # common bottom of the budget band and the card
# this slide's split: left 6.25 in (domain + budget), gutter 0.33 in, right 5.64 in (profile + card)
RX = RIGHT - PROTO_SLOTS["fig08m_profile"][0] * 72
RW = RIGHT - RX

KICKER = "SCALING UP · THE NEXT 2D RUNS"
HEADLINE = ("A one-wavelength wheel sector replaces the full ring, and a fluid model drives its waves "
            "from below.")
# both slides' footers, verbatim: together they are 950 pt at 10.5 pt, and one line has ~740 pt
CITATION = (bd.CITE7[0],
            "MAGNUS-P (Srivastava et al. 2022) · viscous theory after Pitteway & Hines 1963; "
            "Vadas & Fritts 2005 · no viscosity: Hines 1960")
CITE_SIZE, CITE_BASES = 10.0, (519.5, 531.5)     # two lines in the footer band (rule at 508)


def footer2(s: Slide, lines):
    """build_deck.footer with a two-line citation (10 pt, Phase 1 allows 10-11); rule, counter and dots
    exactly as on every other slide. On every build step."""
    resume = s.step
    s.reveal(0)
    s.ax.plot([M, RIGHT], [bd.FOOT_RULE_Y] * 2, color=ds.GRANITE, alpha=0.25, lw=0.5, zorder=3)
    num = f"{s.number:02d} / {bd.N_SLIDES:02d}"
    num_w = s.width(num, fp(SANS, 12))
    first_dot = RIGHT - num_w - 16 - (bd.N_SLIDES - 1) * 10 - 2.5
    with s.para():
        for t, y in zip(lines, CITE_BASES):
            if M + s.width(t, fp(SANS, CITE_SIZE)) > first_dot - 12:
                raise SystemExit(f"citation line runs into the progress dots: {t}")
            s.text(M, y, t, fp(SANS, CITE_SIZE), alpha=bd.MUTED)
    s.text(RIGHT, FOOT_BASE, num, fp(SANS, 12), color=ds.MIDNIGHT, ha="right")
    x_last = RIGHT - num_w - 16
    for k in range(bd.N_SLIDES, 0, -1):
        cx = x_last - (bd.N_SLIDES - k) * 10
        color, alpha = ((ds.MIDNIGHT, 1.0) if k == s.number else
                        (ds.GRANITE, 0.45) if k < s.number else (ds.GRANITE, 0.20))
        s.ax.add_patch(bd.Circle((cx, FOOT_BASE - 4.3), 2.5, fc=color, alpha=alpha, lw=0, zorder=4))
    s.reveal(resume)


def place_steps(s: Slide, stem: str, x: float, y: float, first: int, last: int | None = None):
    """A figure shown from build step first through last (None: to the end). Slide.place covers last=None;
    a swapped-out figure needs the explicit last step."""
    w, h = figure_size(stem)
    if last is None:
        s.placed.append((stem, x, y, w, h))
    s.layers.append((stem, x, y, w, h, first, last))
    return w, h


def merged_slide() -> Slide:
    s = Slide(7)
    f8 = bd.facts("slide08_facts")
    header(s, KICKER, HEADLINE)

    # left: the domain slot (ring, then sector) over the particle budget
    _, hd = place_steps(s, "fig07m_ring_deck", M, Y0, 1, 1)
    place_steps(s, "fig07m_sector_deck", M, Y0, 2)
    _, hb = place_steps(s, "fig07m_budget_deck", M, Y0 + hd, 1)
    if abs(Y0 + hd + hb - Y1) > 0.5:
        raise SystemExit(f"left column ends at {Y0 + hd + hb:.1f} pt, not {Y1}")

    # right: the fluid driver and its check (step 3)
    s.reveal(3)
    if RX - (M + PROTO_SLOTS["fig07m_budget"][0] * 72) < 0.33 * 72 - 0.5:
        raise SystemExit("left and right columns closer than the 0.33 in gutter")
    _, hp = place_steps(s, "fig08m_profile_deck", RX, Y0, 3)
    card = Card("COUPLING CHECK",
                eq=[r"$|\hat w_{\mathrm{fluid}} - \hat w_{\mathrm{lin}}|\,/\,|\hat w_{\mathrm{lin}}| \leq 0.22\%$"],
                lines=[(f"at 100–140 km, so the {f8['n_weak_streams']} weak-forcing", 14.5, ds.GRANITE, 300),
                       ("streams are the first input to DSMC.", 14.5, ds.GRANITE, 300)])
    _, ch = card.layout()
    if Y1 - ch < Y0 + hp + 6:
        raise SystemExit(f"card ({ch:.0f} pt) runs into the profile")
    card.draw(s, RX, Y1 - ch, w=RW)

    footer2(s, CITATION)
    return s


def check_swap_builds(s: Slide, pages, prev: Path, tol: int = 40):
    """Previews of [full, step 1, ..., step n]: ink of a build page outside the full slide's ink only
    inside the slots of figures removed later; the last step equal to the full slide."""
    png = sorted(prev.glob("*.png"))
    full = mpimg.imread(png[0])[..., :3] * 255
    bg = np.array(matplotlib.colors.to_rgb(ds.CULTURED)) * 255
    k_px = full.shape[1] / bd.W
    worst = 0.0
    for i, (_, step) in enumerate(pages[1:], start=1):
        a = mpimg.imread(png[i])[..., :3] * 255
        if step == s.n_steps:
            bad = (np.abs(a - full).max(axis=-1) > tol).mean()
        else:
            ink = np.abs(a - bg).max(axis=-1) > tol
            blank = np.abs(full - bg).max(axis=-1) <= tol
            stray = ink & blank
            for stem, x, y, w, h, first, last in s.layers:
                if last is not None and first <= step <= last:
                    y0, y1 = int(np.floor(y * k_px)), int(np.ceil((y + h) * k_px))
                    x0, x1 = int(np.floor(x * k_px)), int(np.ceil((x + w) * k_px))
                    stray[y0:y1, x0:x1] = False
            bad = stray.sum() / max(ink.sum(), 1)
        worst = max(worst, bad)
        if bad > 1e-3:
            raise SystemExit(f"preview page {i + 1} (step {step}): {100 * bad:.2f}% of inked pixels misplaced")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=bd.DOCS)
    bd.DOCS = ap.parse_args().docs
    for note in bd.audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    s = merged_slide()
    for stem in sorted({l[0] for l in s.layers}):
        bd.audit_figure(stem)
    for k in [*s.pages()]:
        bd.check_layout(s, k)
    pages = [(s, s.n_steps)] + [(s, k) for k in s.pages()]
    print(f"figures: sizes, fonts, provenance OK; layout: no clipping or collisions on the full slide "
          f"and build steps {', '.join(map(str, s.pages()))}")
    bd.compose(pages, commit, PDF, work=WORK,
               subject=f"PROTOTYPE slide 7+8 merged: full slide + build steps 1-{s.n_steps}; "
                       f"figures/deck @ {commit}")
    n, fonts = bd.verify(len(pages), PDF, PREV, name="page")
    print(f"{PDF.relative_to(bd.REFACTORED)}: {n} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")
    worst = check_swap_builds(s, pages, PREV)
    print(f"builds: step 2 swaps ring -> sector in one slot; nothing else moves "
          f"(worst mismatch {100 * worst:.3f}% of inked pixels); step {s.n_steps} = full slide")
    plt.close(s.fig)


if __name__ == "__main__":
    main()
