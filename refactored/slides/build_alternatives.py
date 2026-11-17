"""Review PDF: the current slides 3, 4, 6 beside their alternatives (3A-3C, 4A-4C, 6A-6B).

    cd refactored && conda run -n dsmc python slides/build_alternatives.py

Uses the chrome, audit, collision check and composition of build_deck.py unchanged, so every page
obeys the same rules as the deck (figures at natural size, deck fonts only, no clipping or
overlaps). Figures come from deck/fig03_alternatives.py, fig04_alternatives.py and
fig06_alternatives.py. Numbers on the pages are read from the same S6 caches as the figures,
except the rev5 totals already on the current slide 6 (450 +/- 14 km).

A small Slate tag at the top right of each page names the variant; it is for review only and is
not part of the slide design.

Outputs: slides/alternatives_review.pdf, slides/build/review_preview/slide-NN.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402
from build_deck import (EVID_BOTTOM, EVID_H, EVID_X, EVID_Y, FOOT_RULE_Y, M, MATH, RIGHT, SANS,  # noqa: E402
                        SERIF, TRACK_LABEL, W, WORD_W, WORD_X, Card, Slide, chip, cite, ds, fp, footer,
                        header)

import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from fig03_alternatives import Profile, sig2  # noqa: E402
from fig04_alternatives import TRANSPORT, s6_numbers, sci  # noqa: E402

REVIEW_PDF = bd.REFACTORED / "slides" / "alternatives_review.pdf"
PREVIEW = bd.OUT / "review_preview"
WORK = bd.OUT / "_work_review"

PROF = Profile()
NUM = s6_numbers()
T = np.load(TRANSPORT)


def latt(z_km):
    i = int(np.argmin(np.abs(T["altitude_km"] - z_km)))
    return float(T["L_att_km"][i]), float(T["sigma_L_total_km"][i])


L_EXO, E_EXO = latt(ds.EXOBASE_KM)
L_150, L_320 = latt(150.0)[0], latt(320.0)[0]

KICK = {3: "02 · WHY KINETIC", 4: "03 · METHOD", 6: "05 · 2D RESULT · FIRST 2D DSMC WAVE STUDY"}
HEAD = {3: "Near the exobase, molecules rarely collide, so fluid equations lose their footing "
           "and kinetic models don't.",
        4: "DSMC follows representative molecules, not equations: move, collide, sample, repeat.",
        6: "In 2D, wave energy escapes sideways, lowering heating below the 1D prediction."}
CITE = {3: cite("Bird 1994; Leclercq et al. 2020; Tian et al. 2023"),
        4: cite("Bird 1994; HARRAH: Carberry Mogan et al. 2020–2022"),
        6: cite("Pérez Chávez et al., to be submitted, JGR Planets")}
BODY = fp(SANS, 20)
SMALL = 16


# ================================================================= helpers
def tag(s: Slide, text: str):
    s.tracked(RIGHT, 46, text, fp(SANS, 10, "semibold"), 0.14, color=ds.SLATE)
    # tracked() aligns left; re-place the group flush right
    grp = s.texts[-1][1]
    arts = [t for t, g, _ in s.texts if g == grp]
    right = max(a.get_window_extent(s.renderer).x1 for a in arts)
    for a in arts:
        x, y = a.get_position()
        a.set_x(x + RIGHT - right)


def slide(n: int, variant: str) -> Slide:
    s = Slide(n)
    header(s, KICK[n], HEAD[n])
    tag(s, f"REVIEW · {variant}")
    footer(s, CITE[n])
    return s


def badge(s: Slide, x, y, n):
    s.ax.add_patch(Circle((x, y), 10.5, fc=ds.MIDNIGHT, lw=0, zorder=4))
    s.text(x, y + 4.6, str(n), fp(SANS, 13, "semibold"), color=ds.CULTURED, ha="center")


def statements(s: Slide, x, y, items, width, prop=BODY, leading=26.0, gap=18.0):
    for t in items:
        y = s.paragraph(x, y, t, prop, width, leading) + leading + gap
    return y


def kn_card(eq_tail=True):
    eq = [(r"$\mathrm{Kn} = \lambda_{\mathrm{mfp}}/H \;\to\; 1$", fp(SANS, 20), ds.MIDNIGHT)]
    if eq_tail:
        eq.append(("  at the exobase", fp(SANS, 16, 300), ds.GRANITE))
    return Card("WHERE FLUIDS FAIL", eq=[eq],
                lines=[(r"$\lambda_{\mathrm{mfp}}$: distance between collisions", SMALL, ds.GRANITE, 300),
                       (f"H: scale height ({PROF.at('H', ds.EXOBASE_KM):.0f} km at the exobase)", SMALL,
                        ds.GRANITE, 300)])


def w_card():
    return Card("ONE DOT = MANY ATOMS",
                eq=[rf"$W = n\,V_{{\mathrm{{cell}}}}/N_{{\mathrm{{cp}}}} \approx {NUM['W'] / 1e23:.1f}\times10^{{23}}$"],
                lines=[("O atoms per computational particle (S6).", SMALL, ds.GRANITE, 300),
                       ("Finer cells need more particles: slide 7.", SMALL, ds.GRANITE, 300)])


# ================================================================= slide 3
def s3a():
    s = slide(3, "3A · ZOOM WINDOWS")
    s.place("fig03a_regime_windows_deck", EVID_X, EVID_Y)
    statements(s, WORD_X, EVID_Y + 15, ["Fluid models need many collisions per wave period.",
                                         "DSMC has no closure: heating and dissipation emerge from the collisions."],
               WORD_W)
    c = kn_card()
    c.draw(s, WORD_X, bd.EVID_BOTTOM - c.layout()[1])
    return s


def s3b():
    s = slide(3, "3B · ONE MOLECULE'S CLIMB")
    s.place("fig03b_kn_profile_deck", EVID_X, EVID_Y)
    statements(s, WORD_X, EVID_Y + 15,
               ["Low down, a molecule collides every few kilometres: a random walk, so a fluid.",
                "Above the exobase it flies free and can escape: no fluid closure holds."], WORD_W)
    c = kn_card()
    c.draw(s, WORD_X, bd.EVID_BOTTOM - c.layout()[1])
    return s


def s3c():
    s = slide(3, "3C · COLLISION CLOCK")
    y_fig = 156.0
    _, fh = s.place("fig03c_collision_clock_deck", M, y_fig)
    top = y_fig + fh + 20
    col_w, gap = 240.0, 25.0
    statements(s, M, top + 18, ["Fluid models need many collisions per wave period."], col_w)
    nu_exo = PROF.coll_per_period(ds.EXOBASE_KM)
    z1 = float(np.interp(0.0, -np.log(PROF.c / PROF.lam * PROF.P), PROF.z))   # one collision per period
    statements(s, M + col_w + gap, top + 18,
               [f"At the exobase only ~{nu_exo:.0f}; above ~{round(z1, -1):.0f} km, fewer than one."], col_w)
    c = kn_card(eq_tail=False)
    x_c = M + 2 * (col_w + gap)
    c.draw(s, x_c, FOOT_RULE_Y - 8 - c.layout()[1], w=RIGHT - x_c)
    return s


# ================================================================= slide 4
def s4a():
    s = slide(4, "4A · ONE TIME STEP")
    s.place("fig04a_timestep_plate_deck", EVID_X, EVID_Y)
    items = [("Move", f"fly straight for Δt = {NUM['dt']:.2f} s, then find the new cell"),
             ("Collide", "random pairs within a cell; momentum and energy kept"),
             ("Boundaries", f"wave flux in at {NUM['z_bot']:.0f} km; escape out the top")]
    y = EVID_Y + 14
    prop, lead = fp(SANS, 18), 23.0
    for k, (verb, rest) in enumerate(items, 1):
        badge(s, WORD_X + 10.5, y - 6.5, k)
        lines = s.wrap(f"{verb}: {rest}", prop, WORD_W - 30)
        for i, line in enumerate(lines):
            if i == 0:
                s.runs(WORD_X + 30, y, [(f"{verb}:", fp(SANS, 18, "semibold"), ds.MIDNIGHT),
                                        (line[len(verb) + 1:], prop, ds.GRANITE)])
            else:
                s.text(WORD_X + 30, y + i * lead, line, prop)
        y += len(lines) * lead + 10
    c = w_card()
    c.draw(s, WORD_X, bd.EVID_BOTTOM - c.layout()[1])
    if y - lead > bd.EVID_BOTTOM - c.layout()[1] - 8:
        raise ValueError("slide 4A: numbered steps run into the card")
    return s


def s4b():
    s = slide(4, "4B · POWERS OF TEN")
    y_fig = 156.0
    _, fh = s.place("fig04b_powers_of_ten_deck", M, y_fig)
    top = y_fig + fh + 18
    c = w_card()
    c.draw(s, M, FOOT_RULE_Y - 8 - c.layout()[1], w=330.0)
    x2 = M + 330 + 28
    statements(s, x2, top + 20,
               ["Particles fly straight in x, y, z; cells are true spherical shells, so curvature is exact."],
               250.0, prop=fp(SANS, 18), leading=23.0)
    x3 = x2 + 250 + 28
    s.tracked(x3, top + 12, "DSMC CONSTRAINTS", fp(SANS, 11), bd.TRACK_LABEL)
    y = top + 22
    for t in ("Δt < mean collision time", "cell ≈ mean free path", "≥ 20 particles per cell"):
        chip(s, x3, y, t)
        y += 26.0
    return s


def s4c():
    s = slide(4, "4C · REAL GAS VS DSMC")
    s.place("fig04c_split_step_deck", EVID_X, EVID_Y)
    statements(s, WORD_X, EVID_Y + 15,
               ["Each step: free flight for Δt, then collisions.",
                "Partners are drawn at random within a cell, so cells must be smaller than the mean free path."],
               WORD_W, prop=fp(SANS, 19), leading=25.0, gap=14.0)
    c = Card("WHY THE SPLIT WORKS",
             eq=[rf"$\Delta t = {NUM['dt']:.2f}\,\mathrm{{s}} \;\ll\; \tau_{{\mathrm{{coll}}}}$"],
             lines=[(r"$\tau_{\mathrm{coll}}$: mean time between collisions,", SMALL, ds.GRANITE, 300),
                    (f"shortest in S6: {NUM['tau_100']:.1f} s, at 100 km", SMALL, ds.GRANITE, 300)])
    c.draw(s, WORD_X, bd.EVID_BOTTOM - c.layout()[1])
    chip(s, EVID_X + 3.90 * 72, bd.EVID_BOTTOM + 8, f"one particle = {NUM['W'] / 1e23:.1f} × 10²³ O atoms")
    return s


# ================================================================= slide 6
def s6a():
    s = bd.slide6()
    tag(s, "REVIEW · 6A · TALL WEDGE + PLOT COLUMN")
    return s


def s6b():
    s = slide(6, "6B · WEDGE BAND + THREE PANELS")
    y0 = 156.0
    _, fh = s.place("fig06a_s6_wedge_band_deck", EVID_X, y0)
    pw = ds.SIZES["full"][0] * 72
    bw = bd.figure_size("fig06b_latt_fit_row_deck")[0]
    gap = (pw - 3 * bw) / 2
    y = y0 + fh + 6 + 30
    heroes = [(f"{L_EXO:.0f} ± {E_EXO:.0f}", "km", ["ATTENUATION", "LENGTH"], "fig06b_latt_fit_row_deck"),
              ("450 ± 14", "km", ["HORIZONTAL", "WAVELENGTH"], "fig06c_lambdah_fit_row_deck"),
              (f"×{L_320 / L_150:.0f}", None, ["ATTENUATION LENGTH", "FROM 150 TO 320 KM"],
               "fig06d_latt_profile_row_deck")]
    for k, (hero, unit, labels, stem) in enumerate(heroes):
        x = M + k * (bw + gap)
        bd.hero_row(s, x, y, bw, hero, unit, labels, size=34.0)
        s.place(stem, x, y + 7)
    return s


# ================================================================= current slides and cover
# ================================================================= pre-adoption slides 4 and 6
# Kept here so the review PDF still shows the slides as they were when the alternatives were judged.
def slide4_v1():
    """Slide 4 as it was before 4D was adopted."""
    s = Slide(4)
    header(s, "03 · METHOD",
           "DSMC follows representative molecules, not equations: move, collide, sample, repeat.")
    s.place("fig04_harrah_loop_deck", M, 148)
    card = Card("ONE DOT = MANY ATOMS",
                eq=[r"$W = n\,V_{\mathrm{cell}}/N_{\mathrm{cell}} \approx 10^{24}$"],
                lines=[("Particle weight. Finer cells need more simulated particles,", 16, ds.GRANITE, 300),
                       ("which is the cost on slide 7.", 16, ds.GRANITE, 300)])
    _, ch = card.layout()
    cw = 452.0
    y_card = FOOT_RULE_Y - 9 - ch
    card.draw(s, RIGHT - cw, y_card, w=cw)
    s.tracked(M, y_card + 20, "DSMC CONSTRAINTS", fp(SANS, 11), TRACK_LABEL)
    y = y_card + 30
    for t in ("Δt < mean collision time", "cell ≈ mean free path", "≥ 20 particles per cell"):
        chip(s, M, y, t)
        y += 26.0
    footer(s, cite("Bird 1994; HARRAH: Carberry Mogan et al. 2020–2022"))
    return s


def slide6_v1():
    """Slide 6 as it was before 6A was adopted."""
    s = Slide(6)
    header(s, "05 · 2D RESULT · FIRST 2D DSMC WAVE STUDY",
           "In 2D, wave energy escapes sideways, lowering heating below the 1D prediction.")
    s.place("fig06a_s6_wedge_deck", EVID_X, EVID_Y)
    card_h = (EVID_H - 8) / 2
    small = 14.5
    Card("LATERAL ATTENUATION LENGTH", hero="119 ± 3", unit="km", spark="fig06_spark_latt_deck",
         eq=[r"$A(s) = A_0\,e^{-s/L_{\mathrm{att}}}$"],
         lines=[("A: lock-in amplitude  ·  s: distance past patch", small, ds.GRANITE, 300),
                ("150 / 230 / 320 km:  49, 119, 148 km", small, ds.MIDNIGHT, "normal")]
         ).draw(s, WORD_X, EVID_Y, h=card_h)
    Card("HORIZONTAL WAVELENGTH", hero="450 ± 14", unit="km", spark="fig06_spark_lambdah_deck",
         eq=[r"$\lambda_h = c_h P,\quad c_h = \Delta s/\Delta\tau$"],
         lines=[("Δτ: crest-arrival delay between columns", small, ds.GRANITE, 300),
                ("Slope by ODR fit over many column pairs", small, ds.GRANITE, 300)]
         ).draw(s, WORD_X, EVID_Y + card_h + 8, h=card_h)
    # resolution strip (rev5 §3.6 / Table 4 numbers), with the Crimson tick
    y = EVID_BOTTOM + 25
    s.rule(M, y - 13, 3, 17, ds.CRIMSON)
    strip = fp(SANS, 15)
    s.runs(M + 12, y, [
        ("Grid rule:  ", fp(SANS, 15, "semibold"), ds.GRANITE),
        (r"≈ 8 cells per $\lambda_h$ (57 km) for amplitude growth  ·  ≤ 45 km to fit $L_{\mathrm{att}}$"
         r"  ·  $\lambda_h$ still shifts 8% from 13 to 21 cells", strip, ds.GRANITE)])
    footer(s, cite("Pérez Chávez et al., to be submitted, JGR Planets"))
    return s


def current(n):
    s = {3: bd.slide3, 4: slide4_v1, 6: slide6_v1}[n]()
    tag(s, "REVIEW · CURRENT")
    return s


def cover():
    s = Slide(1)
    s.tracked(M, 46, "NSBP · NSHP 2026 · REVIEW", fp(SANS, 12.5), bd.TRACK_KICKER)
    s.text(M, 96, "Slides 3, 4 and 6: alternatives for review", fp(SERIF, 34), color=ds.MIDNIGHT)
    s.rule(M, 113, 0.9 * 72, 2.5, ds.CRIMSON)
    rows = [("02 · WHY KINETIC", ["Current", "3A · Zoom windows: one molecule in 40 km windows at 120, 230, 400 km",
                                 "3B · One molecule's climb beside the S6 mean free path and scale height",
                                 "3C · Collision clock: collisions per wave period at six altitudes"]),
            ("03 · METHOD", ["Current", "4A · One time step on curved cells: move, collide, boundaries",
                             "4B · Powers of ten: the S6 ring, one exobase cell, one particle",
                             "4C · Real gas vs DSMC: contact collisions vs random partners in a cell"]),
            ("05 · 2D RESULT", ["Current", "6A · Coherent half-wedge (altitude ×2.5) beside the two fits",
                                "6B · Raw-snapshot wedge band (altitude ×1.5) over three panels"])]
    y = 158.0
    for head, items in rows:
        s.tracked(M, y, head, fp(SANS, 11), bd.TRACK_LABEL)
        y += 24
        for it in items:
            s.text(M + 14, y, it, fp(SANS, 16, 300 if it != "Current" else "normal"),
                   color=ds.GRANITE if it != "Current" else ds.MIDNIGHT)
            y += 21
        y += 14
    s.text(M, 500, "Numbers on the new pages are read from the S6 run parameters and analysis caches; "
                   "see deck/fig0{3,4,6}_alternatives.py.", fp(SANS, 12, 300))
    return s


def main():
    for note in bd.audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    pages = [cover(), current(3), s3a(), s3b(), s3c(), current(4), s4a(), s4b(), s4c(),
             current(6), s6a(), s6b()]
    for s in pages:
        for stem, *_ in s.placed:
            bd.audit_figure(stem)
        bd.check_layout(s)
    print("figures: sizes, fonts, provenance OK; layout: no clipping or collisions")
    bd.compose(pages, commit, REVIEW_PDF, f"NSBP/NSHP 2026 review: slides 3, 4, 6 and alternatives; "
                                          f"figures/deck @ {commit}", work=WORK)
    n, fonts = bd.verify(len(pages), REVIEW_PDF, PREVIEW)
    print(f"{REVIEW_PDF.relative_to(bd.REFACTORED)}: {n} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")


if __name__ == "__main__":
    main()
