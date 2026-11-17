"""Assemble the NSBP/NSHP 2026 deck, slides 1, 3, 4, 5, 6, as one 13.333 x 7.5 in PDF.

    cd refactored && conda run -n dsmc python slides/build_deck.py

Slide 2 waits for the reduced NGIMS data; slides 7-9 are the next phase. Page numbers and progress
dots already count the full nine-slide talk, so these pages drop into the final deck unchanged.

Pipeline
  1. Audit. style/deckstyle.py is the one canonical style module: if docs/deckstyle.py or the
     Appendix B block of docs/FIGURE_STYLE.md exist, they must match it byte for byte. Every placed
     figure PDF must be at its deckstyle size, embed only deck fonts, and carry provenance metadata.
  2. Slide chrome. matplotlib draws each slide (field, kicker, headline, Crimson rule, word column,
     measure cards, footer) on a 960 x 540 pt canvas whose data units are points, y down.
     Layout follows Phase1_Design_Philosophy_and_Visual_System.md, Phase2_Slide_Scaffold.md (v2)
     and the proportions of NSBP2026_Design_Preview.pdf (40 pt margins, 7.22 in evidence field,
     4.67 in word column, cards and footer as on preview page 3).
  3. Collision check. Every text box must sit inside the page and must not touch another text box
     or the ink of a placed figure (measured from the figure's transparent PNG preview).
  4. Composition. tectonic (xdvipdfmx) places each figure PDF over its chrome page at natural size:
     vector, no scaling, embedded fonts untouched.
  5. Verification. Page count and size, embedded-font whitelist (no substitutes), PNG previews.

Outputs (refactored/slides/build/): NSBP2026_slides_1-6.pdf, preview/slide-NN.png.
"""
from __future__ import annotations

import itertools
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont, get_font  # noqa: E402
from matplotlib.ft2font import LoadFlags  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

REFACTORED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFACTORED / "style"))
sys.path.insert(0, str(REFACTORED / "deck"))
import deckstyle as ds  # noqa: E402
from slots import SLOTS  # noqa: E402

ds.use("deck", font_dir=REFACTORED / "style" / "fonts")

FIGS = REFACTORED / "figures" / "deck"
DOCS = REFACTORED / "docs"
OUT = REFACTORED / "slides" / "build"
WORK = OUT / "_work"
DECK_PDF = OUT / "NSBP2026_slides_1-6.pdf"

# ---------------------------------------------------------------- page geometry (points, y down)
W, H = 960.0, 540.0
M = 40.0                                   # outer margin (0.55 in)
RIGHT = W - M
EVID_X, EVID_Y = M, 168.0                  # evidence field, deckstyle SIZES["evidence"]
EVID_W, EVID_H = (v * 72 for v in ds.SIZES["evidence"])
WORD_X = RIGHT - ds.SIZES["narrow"][0] * 72   # word column, deckstyle SIZES["narrow"] wide
WORD_W = RIGHT - WORD_X
EVID_BOTTOM = EVID_Y + EVID_H
FOOT_RULE_Y, FOOT_BASE = 508.0, 526.0
N_SLIDES = 9

# ---------------------------------------------------------------- type
SERIF = "Playfair Deck LF"                 # Playfair Display, lining figures (style/fonts)
SANS = "Open Sans"
MATH = "stix"                              # equation-editor look, as in the preview
CARD_FILL = "#ECECE6"                      # Phase 1: "Cultured, one step darker"
MUTED = 0.80                               # citation footer: Granite at 80%
TRACK_KICKER, TRACK_LABEL = 0.15, 0.16     # letter-spacing in em ("+120" and wider for 11 pt)

# Figures drawn at a deckstyle SIZES slot; every other placed figure has a layout slot in deck/slots.py.
SIZE_KEY = {"fig03_regime_schematic": "evidence", "fig03a_regime_windows": "evidence",
            "fig03b_kn_profile": "evidence", "fig04_harrah_loop": "full",
            "fig04a_timestep_plate": "evidence", "fig04c_split_step": "evidence",
            "fig05a_amplitude_growth": "evidence", "fig05b_slab_vs_shell": "inset",
            "fig06a_s6_wedge": "evidence", "fig06_spark_latt": "spark", "fig06_spark_lambdah": "spark"}
ALLOWED_FONTS = re.compile(r"^(OpenSans|PlayfairDeckLF|IBMPlexMono|STIX)")


# ================================================================= text machinery
def fp(family=SANS, size=14.0, weight="normal", style="normal") -> FontProperties:
    return FontProperties(family=family, size=size, weight=weight, style=style)


def advance(ch: str, prop: FontProperties) -> float:
    """Horizontal advance of one character in points (unhinted)."""
    font = get_font(findfont(prop))
    font.set_size(prop.get_size_in_points(), 72)
    return font.load_char(ord(ch), flags=LoadFlags.NO_HINTING).linearHoriAdvance / 65536


@dataclass
class Slide:
    number: int
    dark: bool = False
    fig: plt.Figure = None
    ax: plt.Axes = None
    texts: list = field(default_factory=list)      # (artist, group, name)
    placed: list = field(default_factory=list)     # (stem, x, y_top, w, h)
    panels: list = field(default_factory=list)     # (x, y, w, h, name) card / chip rectangles
    _group: itertools.count = field(default_factory=itertools.count)

    def __post_init__(self):
        self.fig = plt.figure(figsize=(W / 72, H / 72), dpi=72)
        self.fig.set_layout_engine("none")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_axis_off()
        self.field = ds.MIDNIGHT if self.dark else ds.CULTURED
        self.ax.add_patch(Rectangle((0, 0), W, H, fc=self.field, lw=0, zorder=-10))
        self.renderer = self.fig.canvas.get_renderer()

    # ---- measuring
    def width(self, s: str, prop: FontProperties) -> float:
        t = self.ax.text(0, 0, s, fontproperties=prop, math_fontfamily=MATH)
        w = t.get_window_extent(self.renderer).width
        t.remove()
        return w

    # ---- drawing
    def text(self, x, y, s, prop, color=None, ha="left", va="baseline", alpha=1.0, name=None, group=None):
        t = self.ax.text(x, y, s, fontproperties=prop, color=color or ds.GRANITE, ha=ha, va=va,
                         alpha=alpha, math_fontfamily=MATH, zorder=5)
        self.texts.append((t, next(self._group) if group is None else group, name or s))
        return t

    def tracked(self, x, y, s, prop, tracking, color=None, alpha=1.0, ha="left"):
        """Letter-spaced caps (matplotlib has no tracking): one glyph per artist, one group."""
        step = [advance(c, prop) + tracking * prop.get_size_in_points() for c in s]
        total = sum(step) - tracking * prop.get_size_in_points()
        x0 = x - total if ha == "right" else x
        g = next(self._group)
        for c, dx in zip(s, step):
            if c != " ":
                self.text(x0, y, c, prop, color=color, alpha=alpha, group=g, name=s)
            x0 += dx
        return total

    def runs(self, x, y, parts, ha="left"):
        """One line from (text, prop, color) parts, e.g. an italic journal name in a citation."""
        widths = [sum(advance(c, p) for c in s) if "$" not in s else self.width(s, p) for s, p, _ in parts]
        x0 = x - sum(widths) if ha == "right" else x
        g = next(self._group)
        for (s, p, c), w in zip(parts, widths):
            self.text(x0, y, s, p, color=c, group=g, name="".join(q[0] for q in parts))
            x0 += w
        return sum(widths)

    def wrap(self, s, prop, max_w):
        lines, cur = [], ""
        for word in s.split(" "):
            trial = f"{cur} {word}".strip()
            if cur and self.width(trial, prop) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + [cur]

    def paragraph(self, x, y, s, prop, max_w, leading, color=None, alpha=1.0, max_lines=None):
        lines = self.wrap(s, prop, max_w)
        if max_lines and len(lines) > max_lines:
            raise ValueError(f"slide {self.number}: '{s[:40]}...' needs {len(lines)} lines (max {max_lines})")
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, prop, color=color, alpha=alpha)
        return y + (len(lines) - 1) * leading

    def place(self, stem, x, y_top):
        """Reserve a figure at natural size; the PDF is laid over the chrome by tectonic."""
        w, h = figure_size(stem)
        self.placed.append((stem, x, y_top, w, h))
        return w, h

    def rule(self, x, y, w, h, color, alpha=1.0):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=color, lw=0, alpha=alpha, zorder=4))


# ================================================================= recurring components
def header(s: Slide, kicker: str, headline: str):
    s.tracked(M, 46, kicker, fp(SANS, 12.5), TRACK_KICKER)
    lines = s.wrap(headline, fp(SERIF, 32), RIGHT - M)
    if len(lines) > 2:
        raise ValueError(f"slide {s.number}: headline needs {len(lines)} lines (max 2)")
    for i, line in enumerate(lines):
        s.text(M, 88 + 34 * i, line, fp(SERIF, 32), color=ds.MIDNIGHT)
    s.rule(M, 139, 0.9 * 72, 2.5, ds.CRIMSON)


def footer(s: Slide, parts):
    s.ax.plot([M, RIGHT], [FOOT_RULE_Y] * 2, color=ds.GRANITE, alpha=0.25, lw=0.5, zorder=3)
    s.runs(M, FOOT_BASE, [(t, p, ds.GRANITE) for t, p in parts])
    for t in s.texts[-len(parts):]:
        t[0].set_alpha(MUTED)
    num = f"{s.number:02d} / {N_SLIDES:02d}"
    w = s.width(num, fp(SANS, 12))
    s.text(RIGHT, FOOT_BASE, num, fp(SANS, 12), color=ds.MIDNIGHT, ha="right")
    x_last = RIGHT - w - 16
    for k in range(N_SLIDES, 0, -1):
        cx = x_last - (N_SLIDES - k) * 10
        color, alpha = ((ds.MIDNIGHT, 1.0) if k == s.number else
                        (ds.GRANITE, 0.45) if k < s.number else (ds.GRANITE, 0.20))
        s.ax.add_patch(Circle((cx, FOOT_BASE - 4.3), 2.5, fc=color, alpha=alpha, lw=0, zorder=4))


def cite(*parts):
    """Footer citation from plain strings and ('italic', text) tuples."""
    return [(p[1], fp(SANS, 10.5, style="italic")) if isinstance(p, tuple) else (p, fp(SANS, 10.5))
            for p in parts]


@dataclass
class Card:
    label: str
    eq: list                                 # one or more equation lines (mathtext + text)
    lines: list = field(default_factory=list)  # (text, size, color, weight)
    hero: str | None = None
    unit: str | None = None
    spark: str | None = None

    PAD = 16.0

    def layout(self):
        """Baselines relative to the card top; returns (rows, height)."""
        rows, y = [], 20.0
        rows.append(("label", y))
        if self.hero:
            y += 35.0
            rows.append(("hero", y))
            y += 36.0
        else:
            y += 31.0
        for k, _ in enumerate(self.eq):
            rows.append((f"eq{k}", y))
            y += 26.0
        y -= 26.0
        for k, (_, size, _, _) in enumerate(self.lines):
            y += size * 1.25 if k else 22.0
            rows.append((f"line{k}", y))
        return rows, y + 13.0

    def draw(self, s: Slide, x, y, w=WORD_W, h=None):
        rows, need = self.layout()
        h = h or need
        if need > h + 0.5:
            raise ValueError(f"slide {s.number}: card '{self.label}' needs {need:.0f} pt, has {h:.0f}")
        s.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                      fc=CARD_FILL, ec=(*matplotlib.colors.to_rgb(ds.GRANITE), 0.35),
                                      lw=1.0, zorder=1))
        s.rule(x, y + 7, 4, h - 14, ds.MIDNIGHT)
        s.panels.append((x, y, w, h, f"card {self.label}"))
        tx, text_w = x + self.PAD, w - 2 * self.PAD
        base = dict(rows)
        s.tracked(tx, y + base["label"], self.label, fp(SANS, 11), TRACK_LABEL)
        if self.hero:
            hw = s.width(self.hero, fp(SERIF, 40))
            s.text(tx, y + base["hero"], self.hero, fp(SERIF, 40), color=ds.MIDNIGHT)
            if self.unit:
                s.text(tx + hw + 8, y + base["hero"], self.unit, fp(SANS, 19, 300))
        if self.spark:
            sw, sh = figure_size(self.spark)
            s.place(self.spark, x + w - self.PAD - sw, y + 28)
            text_w_hero = w - 2 * self.PAD - sw - 8
            if self.hero and hw + 60 > text_w_hero:
                raise ValueError(f"slide {s.number}: hero '{self.hero}' runs into the sparkline")
        for k, line in enumerate(self.eq):
            parts = line if isinstance(line, list) else [(line, fp(SANS, 20), ds.MIDNIGHT)]
            s.runs(tx, y + base[f"eq{k}"], parts)
        for k, (t, size, color, weight) in enumerate(self.lines):
            if s.width(t, fp(SANS, size, weight)) > text_w:
                raise ValueError(f"slide {s.number}: card line too long: {t}")
            s.text(tx, y + base[f"line{k}"], t, fp(SANS, size, weight), color=color)
        return h


def chip(s: Slide, x, y, text, size=14.0, h=22.0):
    w = s.width(text, fp(SANS, size)) + 24
    s.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                  fc="none", ec=ds.GRANITE, alpha=0.55, lw=1.0, zorder=1))
    s.panels.append((x, y, w, h, f"chip {text}"))
    s.text(x + 12, y + h / 2 + 0.36 * size, text, fp(SANS, size))
    return w


# ================================================================= slides
def slide1():
    s = Slide(1, dark=True)
    x = 56.0
    s.tracked(x, 74, "NSBP · NSHP 2026 JOINT ANNUAL CONFERENCE", fp(SANS, 10), 0.14, color=ds.SUNGLOW)
    s.tracked(x, 89, "EARTH & PLANETARY SCIENCE · THURSDAY 10:00", fp(SANS, 10), 0.14,
              color=ds.CULTURED, alpha=0.6)
    title = ["Molecular Kinetic", "Simulations of", "Acoustic-Gravity Wave",
             "Energy Transport into the", "Martian Exosphere"]
    for i, line in enumerate(title):
        s.text(x, 140 + 31.5 * i, line, fp(SERIF, 30), color=ds.CULTURED)
    s.rule(x, 284, 48, 2.5, ds.CRIMSON)
    s.text(x, 322, "José Ángel Pérez Chávez", fp(SANS, 15), color=ds.CULTURED)
    s.text(x, 340, "O. J. Tucker · R. E. Johnson · S. Carberry Mogan · C. Boxe", fp(SANS, 11, 300),
           color=ds.CULTURED, alpha=0.75)
    s.tracked(x, 364, "HOWARD UNIVERSITY", fp(SANS, 9.5), 0.2, color=ds.CULTURED, alpha=0.75)
    fw, fh = figure_size("fig01_title_wedge_deck")
    s.place("fig01_title_wedge_deck", W - fw - 16, H - fh - 32)
    return s


def slide3():
    s = Slide(3)
    header(s, "02 · WHY KINETIC",
           "Near the exobase, molecules rarely collide, so fluid equations lose their footing "
           "and kinetic models don't.")
    s.place("fig03_regime_schematic_deck", EVID_X, EVID_Y)
    body = fp(SANS, 20)
    y = EVID_Y + 15
    for stmt in ("Fluid models need many collisions per wave period.",
                 "DSMC has no closure: heating and dissipation emerge from the collisions."):
        y = s.paragraph(WORD_X, y, stmt, body, WORD_W, 27) + 42
    card = Card("WHERE FLUIDS FAIL",
                eq=[[(r"$\mathrm{Kn} = \lambda_{\mathrm{mfp}}/H \;\to\; 1$", fp(SANS, 20), ds.MIDNIGHT),
                     ("  at the exobase", fp(SANS, 16, 300), ds.GRANITE)]],
                lines=[(r"$\lambda_{\mathrm{mfp}}$: distance between collisions", 16, ds.GRANITE, 300),
                       ("H: scale height (40 km here)", 16, ds.GRANITE, 300)])
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Bird 1994; Leclercq et al. 2020; Tian et al. 2023"))
    return s


def slide4():
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


def slide5():
    s = Slide(5)
    header(s, "04 · 1D RESULT · PUBLISHED",
           "Spherical geometry cuts wave-driven heating by 40–56% compared with slab models.")
    s.place("fig05a_amplitude_growth_deck", EVID_X, EVID_Y)
    hero, unit = fp(SERIF, 54), fp(SANS, 18, 300)
    y1 = EVID_Y + 38
    w1 = s.width("1.7–2.3×", hero)
    s.text(WORD_X, y1, "1.7–2.3×", hero, color=ds.MIDNIGHT)
    s.text(WORD_X + w1 + 14, y1 - 21, "lower heating", unit)
    s.text(WORD_X + w1 + 14, y1, "gradient", unit)
    # FIG 5b beside the second hero number; its top 0.25 in is empty margin above the walls
    y5b = y1 - 4
    bw, _ = s.place("fig05b_slab_vs_shell_deck", WORD_X - 6, y5b)
    x2 = WORD_X - 6 + bw + 10
    y2 = y5b + 18 + 40
    s.text(x2, y2, "~20%", hero, color=ds.MIDNIGHT)
    for k, t in enumerate(("less amplitude", "growth at 300 km", "×4.8 vs ×5.8")):
        s.text(x2, y2 + 25 + 21 * k, t, unit)
    card = Card("HOW WE MEASURE IT",
                eq=[r"$G = A(z)/A_{150},\quad d\langle T\rangle/dr$"],
                lines=[("Growth relative to 150 km; heating as", 16, ds.GRANITE, 300),
                       ("the cycle-5 mean temperature gradient.", 16, ds.GRANITE, 300)])
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Pérez Chávez et al. 2026, ", ("italic", "Icarus"), " 447, 116900"))
    return s


def slide6():
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


# ================================================================= audit and checks
def figure_size(stem: str):
    """MediaBox of a figure PDF in points."""
    raw = (FIGS / f"{stem}.pdf").read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    return float(m.group(1)), float(m.group(2))


def expected_size(stem: str):
    base = stem.removesuffix("_deck")
    if base in SLOTS:
        return SLOTS[base]
    return ds.SIZES[SIZE_KEY[base]]


def pdf_fonts(path: Path):
    out = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines()[2:]:
        cols = line.split()
        name = cols[0].split("+")[-1]
        emb = cols[-5]
        rows.append((name, emb))
    return rows


def audit_style():
    canon = (REFACTORED / "style" / "deckstyle.py").read_text()
    notes = [f"canonical style module: style/deckstyle.py ({ds.STYLE_VERSION})"]
    copy = DOCS / "deckstyle.py"
    if copy.exists():
        if copy.read_text() != canon:
            raise SystemExit("docs/deckstyle.py differs from style/deckstyle.py: reconcile before building")
        notes.append("docs/deckstyle.py: identical copy")
    spec = DOCS / "FIGURE_STYLE.md"
    if spec.exists():
        blocks = [b for b in re.findall(r"```python\n(.*?)```", spec.read_text(), re.S) if "STYLE_VERSION" in b]
        if not blocks or blocks[-1] != canon:
            raise SystemExit("FIGURE_STYLE.md Appendix B differs from style/deckstyle.py")
        notes.append("FIGURE_STYLE.md Appendix B: identical")
    return notes


def audit_figure(stem: str):
    path = FIGS / f"{stem}.pdf"
    w, h = figure_size(stem)
    ew, eh = expected_size(stem)
    if abs(w - ew * 72) > 0.05 or abs(h - eh * 72) > 0.05:
        raise SystemExit(f"{stem}: {w:.2f} x {h:.2f} pt, expected {ew * 72:.2f} x {eh * 72:.2f}")
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout
    if not re.search(r"^Subject:\s+\S.*@ [0-9a-f]{7}", info, re.M):
        raise SystemExit(f"{stem}: no provenance (Subject '... @ commit')")
    if f"deckstyle {ds.STYLE_VERSION}" not in info:
        raise SystemExit(f"{stem}: not made with deckstyle {ds.STYLE_VERSION}")
    for name, emb in pdf_fonts(path):
        if emb != "yes" or not ALLOWED_FONTS.match(name):
            raise SystemExit(f"{stem}: font {name} (embedded={emb}) is not a deck font")


def ink_rect(stem: str, x, y_top):
    """Bounding box (x0, y0, x1, y1, pt, y down) of the non-transparent pixels of a placed figure."""
    img = mpimg.imread(FIGS / f"{stem}.png")
    alpha = img[..., 3] if img.shape[-1] == 4 else 1 - img[..., :3].min(axis=-1)
    rows, cols = np.where(alpha > 0.03)
    k = 72.0 / 300.0
    return (x + cols.min() * k, y_top + rows.min() * k, x + (cols.max() + 1) * k, y_top + (rows.max() + 1) * k)


def check_layout(s: Slide):
    """Text inside the page, no text/text, text/figure-ink or panel/figure-ink collisions."""
    boxes = []
    for t, g, name in s.texts:
        b = t.get_window_extent(s.renderer)                 # display coords, y up, 1 px = 1 pt
        # vertical extent from the glyph ink (the layout box always spans ascender to descender)
        prop = t.get_fontproperties().copy()
        prop.set_math_fontfamily(MATH)
        ink = TextPath((0, 0), t.get_text(), prop=prop).get_extents()
        base = t.get_position()[1]
        boxes.append(((b.x0, base - ink.y1, b.x1, base - ink.y0), g, name))
    problems = []
    margin = 6.0
    for (x0, y0, x1, y1), _, name in boxes:
        if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
            problems.append(f"outside page: {name}")
        if not s.dark and y0 < FOOT_RULE_Y < y1:
            problems.append(f"text crosses the footer rule: {name}")
    for px, py, pw, ph, pname in s.panels:
        if not s.dark and py + ph > FOOT_RULE_Y - 4:
            problems.append(f"{pname} reaches the footer rule")
    groups = {}
    for box, g, name in boxes:
        groups.setdefault(g, [box, name])
        gb = groups[g][0]
        groups[g][0] = (min(gb[0], box[0]), min(gb[1], box[1]), max(gb[2], box[2]), max(gb[3], box[3]))

    def hit(a, b, pad=1.0):
        return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad

    items = list(groups.values())
    for (a, na), (b, nb) in itertools.combinations(items, 2):
        if hit(a, b):
            problems.append(f"text overlap: '{na}' / '{nb}'")
    for stem, x, y, w, h in s.placed:
        ink = ink_rect(stem, x, y)
        if ink[0] < 0 or ink[1] < 0 or ink[2] > W or ink[3] > H:
            problems.append(f"figure ink off page: {stem}")
        for box, name in items:
            if hit(box, ink):
                problems.append(f"text over figure ink: '{name}' / {stem}")
        for px, py, pw, ph, pname in s.panels:
            if "spark" in stem and pname.startswith("card"):
                continue
            if hit((px, py, px + pw, py + ph), ink):
                problems.append(f"{pname} over figure ink: {stem}")
    if problems:
        raise SystemExit(f"slide {s.number} layout:\n  " + "\n  ".join(problems))


# ================================================================= composition
TEX_HEAD = r"""\documentclass{article}
\usepackage[paperwidth=960bp,paperheight=540bp,margin=0pt]{geometry}
\usepackage{graphicx,eso-pic}
\pagestyle{empty}
\setlength{\unitlength}{1bp}
\special{pdf:docinfo<</Title (%s) /Subject (%s) /Creator (slides/build_deck.py, deckstyle %s)>>}
\begin{document}
"""


TITLE = "Molecular Kinetic Simulations of Acoustic-Gravity Wave Energy Transport into the Martian Exosphere"


def compose(slides, commit: str, out_pdf: Path = DECK_PDF, subject: str | None = None, work: Path = WORK):
    """Lay each slide's figure PDFs over its chrome page at natural size; one PDF out."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    body = []
    for k, s in enumerate(slides):
        chrome = work / f"chrome{k:02d}.pdf"
        s.fig.savefig(chrome, facecolor=s.field, transparent=False)
        page = [rf"\AddToShipoutPictureBG*{{\put(0,0){{\includegraphics{{{chrome.name}}}}}}}"]
        for stem, x, y, w, h in s.placed:
            shutil.copy(FIGS / f"{stem}.pdf", work / f"{stem}.pdf")
            page.append(rf"\AddToShipoutPictureFG*{{\put({x:.3f},{H - y - h:.3f}){{\includegraphics{{{stem}.pdf}}}}}}")
        body.append("\n".join(page) + "\n\\null\\newpage\n")
    subject = subject or f"NSBP/NSHP 2026, slides 1, 3-6; figures/deck @ {commit}"
    tex = TEX_HEAD % (TITLE, subject, ds.STYLE_VERSION)
    tex += "".join(body) + r"\end{document}" + "\n"
    (work / "deck.tex").write_text(tex)
    subprocess.run(["tectonic", "-X", "compile", "deck.tex"], cwd=work, check=True, capture_output=True)
    shutil.copy(work / "deck.pdf", out_pdf)


def verify(n_pages: int, pdf: Path = DECK_PDF, prev: Path = OUT / "preview"):
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups()
    assert pages == n_pages, f"{pages} pages, expected {n_pages}"
    assert tuple(map(float, size)) == (W, H), f"page size {size}"
    fonts = sorted({(n, e) for n, e in pdf_fonts(pdf)})
    bad = [f for f in fonts if f[1] != "yes" or not ALLOWED_FONTS.match(f[0])]
    if bad:
        raise SystemExit(f"non-deck or unembedded fonts in {pdf.name}: {bad}")
    if prev.exists():
        shutil.rmtree(prev)
    prev.mkdir(parents=True)
    subprocess.run(["pdftoppm", "-r", "110", "-png", str(pdf), str(prev / "slide")], check=True)
    return pages, sorted({n for n, _ in fonts})


def main():
    for note in audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    slides = [slide1(), slide3(), slide4(), slide5(), slide6()]
    for s in slides:
        for stem, *_ in s.placed:
            audit_figure(stem)
        check_layout(s)
    print("figures: sizes, fonts, provenance OK; layout: no clipping or collisions")
    compose(slides, commit)
    pages, fonts = verify(len(slides))
    rel = DECK_PDF.relative_to(REFACTORED)
    print(f"{rel}: {pages} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")
    for s in slides:
        plt.close(s.fig)


if __name__ == "__main__":
    main()
