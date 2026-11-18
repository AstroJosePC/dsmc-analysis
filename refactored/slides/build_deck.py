"""Assemble the NSBP/NSHP 2026 deck, slides 1-8 plus backups B1-B6, as 13.333 x 7.5 in PDFs: the full
deck (one page per slide) and the progressive deck for live delivery (slides 2-8 built up click by click).

    cd refactored && conda run -n dsmc python slides/build_deck.py

Backup slides B1-B6 (Q&A) follow slide 8: their counter reads B1-B6 and all eight talk dots show
as done.

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

Progressive builds. Each slide is laid out once; Slide.reveal(k) marks what first appears at build
step k (header and footer are step 0, on every page). Page k of the progressive deck is that same
layout with the later steps hidden, so nothing is placed twice and nothing moves between clicks.
A figure that builds up inside itself (FIG 2a, 4, 9) has variants <stem>_b1, _b2, ... written by
its script with deck/_deck.Build: the finished figure with later artists hidden, at the same size.
A figure can also be swapped out (Slide.place(..., until=k)): on slide 7 the full ring gives way to the
wheel sector in the same slot at step k + 1.
Steps 1-3 run on every build page as well; check_builds then compares the previews pixel by pixel:
each inked pixel of a build page must equal the full slide's (except inside the slot of a figure that
is swapped out later), and a slide's last page must equal it.
The title and backup slides stay single pages.

Numbers on slides 2, 4, 7 and B1-B6 are read from deck/data/*.json, written and checked against their
sources by deck/cache_slide02.py, deck/cache_slides78.py and deck/cache_backups.py (rerun it, then the figure scripts, before rebuilding when a source changes,
e.g. when the last Stage 1B replicates finish).

Outputs (refactored/slides/build/): NSBP2026_deck_full.pdf, preview/slide-N.png (page order);
NSBP2026_deck_progressive.pdf, preview_progressive/page-N.png. slides/export_pptx.py writes the same 14
slides (deck_slides()) as an editable PowerPoint file from these Slide objects (Slide.para and Slide.card_eqs exist for it).
"""
from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
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
DECK_PDF = OUT / "NSBP2026_deck_full.pdf"
PROG_PDF = OUT / "NSBP2026_deck_progressive.pdf"
DECK_DATA = REFACTORED / "deck" / "data"

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
CITE2_SIZE, CITE2_BASES = 10.0, (519.5, 531.5)   # two-line citation (slide 7) in the footer band
N_SLIDES = 8                               # talk slides; backups follow with every dot done

# ---------------------------------------------------------------- type
SERIF = "Playfair Deck LF"                 # Playfair Display, lining figures (style/fonts)
SANS = "Open Sans"
MATH = "stix"                              # equation-editor look, as in the preview
CARD_FILL = "#ECECE6"                      # Phase 1: "Cultured, one step darker"
MUTED = 0.80                               # citation footer: Granite at 80%
TRACK_KICKER, TRACK_LABEL = 0.15, 0.16     # letter-spacing in em ("+120" and wider for 11 pt)

# Figures drawn at a deckstyle SIZES key; review-only alternatives have a layout slot in deck/slots.py.
SIZE_KEY = {"fig01_title_wedge": "title", "fig04d_scales_and_step": "plate",
            "fig02a_ngims_orbit2521": "evidence", "fig02b_ngims_sza": "inset",
            "fig06a_s6_halfwedge": "evidence_tall", "fig06b_latt_fit_col": "column",
            "fig06c_lambdah_fit_col": "column", "fig07c_sector_3d": "evidence_tall", "fig07d_sector_ring": "evidence_tall",
            "fig07e_sector_strip": "strip",
            "fig03_regime_schematic": "evidence", "fig03a_regime_windows": "evidence",
            "fig03b_kn_profile": "evidence", "fig04_harrah_loop": "full",
            "fig04a_timestep_plate": "evidence", "fig04c_split_step": "evidence",
            "fig05a_amplitude_growth": "evidence", "fig05b_slab_vs_shell": "inset",
            "fig06a_s6_wedge": "evidence", "fig06_spark_latt": "spark", "fig06_spark_lambdah": "spark",
            "fig07a_wheel_sector": "evidence", "fig08a_coupling_baseline": "evidence",
            "fig07m_ring": "split_left", "fig07m_sector": "split_left", "fig07m_budget": "split_band",
            "fig07m_profile": "split_right",
            "fig08b_escape_vs_top": "narrow", "fig09_maven_roadmap": "strip",
            "figB1_run_table": "strip", "figB1b_growth_convergence": "column", "figB2_wheel_fix": "evidence", "figB3_noise_floor": "evidence",
            "figB4_westward_wind": "evidence", "figB5_radial_grid": "evidence", "figB6_phase_lag": "evidence"}
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
    counter: str | None = None                     # "B1" on backup slides
    fig: plt.Figure = None
    ax: plt.Axes = None
    texts: list = field(default_factory=list)      # (artist, group, name)
    placed: list = field(default_factory=list)     # (stem, x, y_top, w, h) as on the full slide
    panels: list = field(default_factory=list)     # (x, y, w, h, name, step) card / chip rectangles
    layers: list = field(default_factory=list)     # (stem, x, y_top, w, h, first, last) incl. build variants
    swapped: list = field(default_factory=list)    # (x, y_top, w, h, first, last) of figures swapped out
    images: list = field(default_factory=list)     # (x0, y0, x1, y1, name, first, last) raster chrome images
    art_until: dict = field(default_factory=dict)  # chrome artist -> last build step it shows on
    step: int = 0                                  # build step of what is drawn now (0: every page)
    art_step: dict = field(default_factory=dict)   # chrome artist -> build step
    para_of: dict = field(default_factory=dict)    # text artist -> paragraph id (slides/export_pptx.py only)
    card_eqs: dict = field(default_factory=dict)   # text artist -> (card label, equation row source)
    _group: itertools.count = field(default_factory=itertools.count)
    _paras: itertools.count = field(default_factory=itertools.count)
    _para: int | None = None

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
        self._seen = set(self.ax.get_children())

    # ---- build steps (progressive deck)
    def _sweep(self):
        for a in self.ax.get_children():
            if a not in self._seen:
                self._seen.add(a)
                self.art_step[a] = self.step

    def reveal(self, k: int):
        """What is drawn from here on first appears at build step k. Positions never depend on k:
        the slide is laid out once and page k hides the later steps, so nothing moves between pages."""
        self._sweep()
        self.step = k

    @property
    def n_steps(self) -> int:
        self._sweep()
        return max([0, *self.art_step.values(), *(p[5] for p in self.panels), *(l[5] for l in self.layers)])

    def show(self, k: int):
        """Show build step k (chrome artists of later steps hidden); returns the figures on that page."""
        self._sweep()
        for a, s in self.art_step.items():
            a.set_visible(s <= k <= self.art_until.get(a, k))
        return [l[:5] for l in self.layers if l[5] <= k and (l[6] is None or k <= l[6])]

    def pages(self) -> range:
        """Build steps that become pages of the progressive deck (one page if nothing is staged)."""
        n = self.n_steps
        return range(min(1, n), n + 1)

    # ---- measuring
    def width(self, s: str, prop: FontProperties) -> float:
        t = self.ax.text(0, 0, s, fontproperties=prop, math_fontfamily=MATH)
        w = t.get_window_extent(self.renderer).width
        t.remove()
        return w

    # ---- drawing
    @contextmanager
    def para(self):
        """Lines drawn inside belong to one paragraph: the editable PowerPoint export writes them as one
        text box with their leading. Drawing and the collision check do not see it."""
        outer = self._para
        if outer is None:
            self._para = next(self._paras)
        try:
            yield
        finally:
            self._para = outer

    def text(self, x, y, s, prop, color=None, ha="left", va="baseline", alpha=1.0, name=None, group=None):
        t = self.ax.text(x, y, s, fontproperties=prop, color=color or ds.GRANITE, ha=ha, va=va,
                         alpha=alpha, math_fontfamily=MATH, zorder=5)
        self.texts.append((t, next(self._group) if group is None else group, name or s))
        if self._para is not None:
            self.para_of[t] = self._para
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
        with self.para():
            for i, line in enumerate(lines):
                self.text(x, y + i * leading, line, prop, color=color, alpha=alpha)
        return y + (len(lines) - 1) * leading

    def place(self, stem, x, y_top, builds=0, until=None):
        """Reserve a figure at natural size; the PDF is laid over the chrome by tectonic.

        builds = n: the progressive deck shows the figure's build variants <base>_b1 ... _bn (same size,
        deck/_deck.Build) at steps step ... step + n - 1 and the finished figure from step + n on.
        until = k: the figure shows at steps step ... k only and is not on the full slide (another
        figure takes its slot at step k + 1)."""
        w, h = figure_size(stem)
        if until is not None:
            if builds or until < self.step:
                raise ValueError(f"{stem}: until={until} needs builds=0 and a step >= {self.step}")
            self.layers.append((stem, x, y_top, w, h, self.step, until))
            self.swapped.append((x, y_top, w, h, self.step, until))
            return w, h
        self.placed.append((stem, x, y_top, w, h))
        base = stem.removesuffix("_deck")
        for i in range(1, builds + 1):
            v = f"{base}_b{i}_deck"
            if figure_size(v) != (w, h):
                raise SystemExit(f"{v}: not the size of {stem}")
            self.layers.append((v, x, y_top, w, h, self.step + i - 1, self.step + i - 1))
        self.layers.append((stem, x, y_top, w, h, self.step + builds, None))
        return w, h

    def image(self, rgba, x, y_top, w=None, h=None, name="image", until=None):
        """A raster image (RGBA array, trimmed to its ink) drawn in the chrome at width w or height h,
        aspect kept. until = k: shown at steps step ... k only (not on the full slide); check_builds
        then allows its ink where the full slide is blank. check_layout keeps it clear of text, panels
        and the ink of placed figures."""
        ar = rgba.shape[1] / rgba.shape[0]
        w, h = (w, w / ar) if w is not None else (h * ar, h)
        im = self.ax.imshow(rgba, extent=(x, x + w, y_top + h, y_top), interpolation="none", zorder=5)
        if until is not None:
            if until < self.step:
                raise ValueError(f"{name}: until={until} before step {self.step}")
            self._sweep()
            self.art_until[im] = until
            self.swapped.append((x, y_top, w, h, self.step, until))
        self.images.append((x, y_top, x + w, y_top + h, name, self.step, until))
        return w, h

    def panel(self, x, y, w, h, name):
        self.panels.append((x, y, w, h, name, self.step))

    def rule(self, x, y, w, h, color, alpha=1.0):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=color, lw=0, alpha=alpha, zorder=4))


# ================================================================= recurring components
def header(s: Slide, kicker: str, headline: str):
    s.tracked(M, 46, kicker, fp(SANS, 12.5), TRACK_KICKER)
    lines = s.wrap(headline, fp(SERIF, 32), RIGHT - M)
    if len(lines) > 2:
        raise ValueError(f"slide {s.number}: headline needs {len(lines)} lines (max 2)")
    with s.para():
        for i, line in enumerate(lines):
            s.text(M, 88 + 34 * i, line, fp(SERIF, 32), color=ds.MIDNIGHT)
    s.rule(M, 139, 0.9 * 72, 2.5, ds.CRIMSON)


def footer(s: Slide, parts=None, lines=None):
    """Citation, counter and progress dots: on every build step of the slide.

    parts: one citation line (cite()). lines: two plain citation lines at 10 pt instead (Phase 1 allows
    10-11 pt), for a slide whose sources do not fit one line before the progress dots (slide 7)."""
    resume = s.step
    s.reveal(0)
    s.ax.plot([M, RIGHT], [FOOT_RULE_Y] * 2, color=ds.GRANITE, alpha=0.25, lw=0.5, zorder=3)
    num = s.counter or f"{s.number:02d} / {N_SLIDES:02d}"
    w = s.width(num, fp(SANS, 12))
    x_last = RIGHT - w - 16
    if lines is None:
        s.runs(M, FOOT_BASE, [(t, p, ds.GRANITE) for t, p in parts])
        for t in s.texts[-len(parts):]:
            t[0].set_alpha(MUTED)
    else:
        first_dot = x_last - (N_SLIDES - 1) * 10 - 2.5
        with s.para():
            for t, y in zip(lines, CITE2_BASES, strict=True):
                if M + s.width(t, fp(SANS, CITE2_SIZE)) > first_dot - 12:
                    raise ValueError(f"slide {s.number}: citation line runs into the progress dots: {t}")
                s.text(M, y, t, fp(SANS, CITE2_SIZE), alpha=MUTED)
    s.text(RIGHT, FOOT_BASE, num, fp(SANS, 12), color=ds.MIDNIGHT, ha="right")
    for k in range(N_SLIDES, 0, -1):
        cx = x_last - (N_SLIDES - k) * 10
        color, alpha = ((ds.MIDNIGHT, 1.0) if k == s.number else
                        (ds.GRANITE, 0.45) if k < s.number else (ds.GRANITE, 0.20))
        s.ax.add_patch(Circle((cx, FOOT_BASE - 4.3), 2.5, fc=color, alpha=alpha, lw=0, zorder=4))
    s.reveal(resume)


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
        s.panel(x, y, w, h, f"card {self.label}")
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
            n0 = len(s.texts)
            s.runs(tx, y + base[f"eq{k}"], parts)
            src = "".join(p[0] for p in parts)
            if "$" in src:
                for t in s.texts[n0:]:
                    s.card_eqs[t[0]] = (self.label, src)
        for k, (t, size, color, weight) in enumerate(self.lines):
            if s.width(t, fp(SANS, size, weight)) > text_w:
                raise ValueError(f"slide {s.number}: card line too long: {t}")
            s.text(tx, y + base[f"line{k}"], t, fp(SANS, size, weight), color=color)
        return h


def chip(s: Slide, x, y, text, size=14.0, h=22.0):
    w = s.width(text, fp(SANS, size)) + 24
    s.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                  fc="none", ec=ds.GRANITE, alpha=0.55, lw=1.0, zorder=1))
    s.panel(x, y, w, h, f"chip {text}")
    s.text(x + 12, y + h / 2 + 0.36 * size, text, fp(SANS, size))
    return w


# ================================================================= slides
SLIDES_DIR = Path(__file__).resolve().parent


def trimmed_png(name: str) -> np.ndarray:
    """RGBA array of slides/<name>, cropped to its non-transparent pixels."""
    img = mpimg.imread(SLIDES_DIR / name).copy()
    ys, xs = np.where(img[..., 3] > 0.02)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def place_title_logos(s: Slide, x: float, y_center: float, howard_h: float = 112.0, nasa_h: float = 90.0,
                      gap: float = 34.0):
    """Place Howard University (light-on-Midnight reverse) and NASA logos in the bottom-left of slide 1,
    centres level at y_center."""
    howard = trimmed_png("howard_logo.png")
    # Convert Howard Blue (#003963) -> Cultured and white clock face -> Midnight; keep crimson 1867 & rule
    r, g, b = howard[..., 0], howard[..., 1], howard[..., 2]
    non_red = (r - b) < 0.15
    t = np.clip((g - 0.2235) / (1.0 - 0.2235), 0.0, 1.0)
    cultured = np.array(matplotlib.colors.to_rgb(ds.CULTURED))
    midnight = np.array(matplotlib.colors.to_rgb(ds.MIDNIGHT))
    inv_rgb = (1.0 - t)[..., None] * cultured + t[..., None] * midnight
    howard[..., :3] = np.where(non_red[..., None], inv_rgb, howard[..., :3])
    hw, _ = s.image(howard, x, y_center - howard_h / 2, h=howard_h, name="Howard logo")
    s.image(trimmed_png("nasa_logo.png"), x + hw + gap, y_center - nasa_h / 2, h=nasa_h, name="NASA logo")


def slide1():
    s = Slide(1, dark=True)
    x = 56.0
    s.tracked(x, 74, "NSBP · NSHP 2026 JOINT ANNUAL CONFERENCE", fp(SANS, 10), 0.14, color=ds.SUNGLOW)
    s.tracked(x, 89, "EARTH & PLANETARY SCIENCE · THURSDAY 10:00", fp(SANS, 10), 0.14,
              color=ds.CULTURED, alpha=0.6)
    title = ["Molecular Kinetic", "Simulations of", "Acoustic-Gravity Wave",
             "Energy Transport into the", "Martian Exosphere"]
    with s.para():
        for i, line in enumerate(title):
            s.text(x, 140 + 31.5 * i, line, fp(SERIF, 30), color=ds.CULTURED)
    s.rule(x, 284, 48, 2.5, ds.CRIMSON)
    s.text(x, 322, "José Ángel Pérez Chávez", fp(SANS, 15), color=ds.CULTURED)
    s.text(x, 340, "O. J. Tucker · R. E. Johnson · S. Carberry Mogan · C. Blaszczak-Boxe", fp(SANS, 11, 300),
           color=ds.CULTURED, alpha=0.75)
    s.tracked(x, 364, "HOWARD UNIVERSITY", fp(SANS, 9.5), 0.2, color=ds.CULTURED, alpha=0.75)
    place_title_logos(s, x, y_center=448.0)
    fw, fh = figure_size("fig01_title_wedge_deck")
    s.place("fig01_title_wedge_deck", W - fw - 16, H - fh - 32)
    return s


def slide2():
    """What MAVEN sees: one NGIMS pass (FIG 2a) with the all-pass SZA climatology (FIG 2b) inset in the
    lower right of the evidence field, three callouts and the perturbation measure card."""
    s = Slide(2)
    f = facts("slide02_facts")
    header(s, "WHAT MAVEN SEES",
           "MAVEN sees large, species-dependent waves right where Mars' atmosphere stops behaving "
           "like a fluid.")
    s.reveal(1)                                   # build: one pass, its peak, all passes, what they show
    s.place("fig02a_ngims_orbit2521_deck", EVID_X, EVID_Y, builds=2)   # 2: peak callout, 3: FIG 2b caption
    maven = trimmed_png("maven_spacecraft.png")
    w_big = WORD_W - 16                           # steps 1-3: the word column is still empty
    h_big = w_big * maven.shape[0] / maven.shape[1]
    s.image(maven, WORD_X + 8, EVID_Y + (EVID_H - h_big) / 2, w=w_big, name="MAVEN (large)", until=3)
    iw, ih = (v * 72 for v in ds.SIZES["inset"])
    s.reveal(3)
    s.place("fig02b_ngims_sza_deck", EVID_X + EVID_W - iw, EVID_BOTTOM - ih)   # x axes level with 2a

    s.reveal(4)
    # from step 4 a thumbnail in FIG 2a's empty upper right: right of the 40% line, left of the legend,
    # above the 87% callout (check_layout: no figure ink under it or within 4 pt)
    s.image(maven, EVID_X + 233, EVID_Y + 40, w=100, name="MAVEN (small)")
    y = EVID_Y + 31
    for hero, labels in ((f"> {f['threshold_pct']}%", ["PEAK DENSITY", "PERTURBATIONS"]),
                         ("CO$_2$, Ar > O", ["HEAVIER SPECIES,", "LARGER AMPLITUDES"]),
                         (r"$\uparrow$ with SZA", ["LARGER AT HIGH", "SOLAR ZENITH ANGLE"])):
        hero_row(s, WORD_X, y, WORD_W, hero, None, labels)
        y += 54
    y_src = y - 54 + 30
    s.text(WORD_X, y_src, f"NGIMS · {f['williamson_orbits']:,} orbits · 2015–2017", fp(SANS, 14, 300))
    card = Card('WHAT "PERTURBATION" MEANS',
                eq=[r"$\delta\rho/\bar\rho = (\rho - \bar\rho_{\mathrm{fit}})\,/\,\bar\rho_{\mathrm{fit}}$"],
                lines=[(r"$\bar\rho_{\mathrm{fit}}$ is a smooth fit to each pass, so", 15, ds.GRANITE, 300),
                       ("the wave depends on that choice.", 15, ds.GRANITE, 300)])
    card_bottom(s, card, y_src)
    footer(s, cite("England et al. 2017; Williamson et al. 2019; Yiğit et al. 2021"))
    return s


def slide3():
    s = Slide(3)
    header(s, "WHY KINETIC",
           "Near the exobase, gas becomes rarefied and mean free paths grow, "
           "so non-equilibrium solutions require kinetic models.")
    s.reveal(1)                                   # build: regimes and Kn, then the two statements
    s.place("fig03_regime_schematic_deck", EVID_X, EVID_Y)
    body = fp(SANS, 20)
    y = EVID_Y + 8
    for k, stmt in enumerate((
        r"Navier-Stokes errors become significant when local $\mathrm{Kn}>0.1$, "
        r"and continuum models must be replaced by molecular at $\mathrm{Kn}\sim0.2$.",
        "Heating and dissipation effects self-consistently emerge from collisions.",
    )):
        s.reveal(2 + k)
        y = s.paragraph(WORD_X, y, stmt, body, WORD_W, 26) + 37
    s.reveal(1)
    card = Card("WHERE FLUIDS FAIL",
                eq=[[(r"$\mathrm{Kn} = \lambda_{\mathrm{mfp}}/H \;\to\; 1$", fp(SANS, 20), ds.MIDNIGHT),
                     ("  at the exobase", fp(SANS, 16, 300), ds.GRANITE)]],
                lines=[(r"$\lambda_{\mathrm{mfp}}$: distance between collisions", 16, ds.GRANITE, 300),
                       ("H: scale height (40 km here)", 16, ds.GRANITE, 300)])
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Bird 1994; Leclercq et al. 2020; Tian et al. 2023"))
    return s


def number_badge(s: Slide, x, y, n):
    """Midnight disc with a white numeral, matching the numbered badges inside FIG 4."""
    s.ax.add_patch(Circle((x, y), 10.5, fc=ds.MIDNIGHT, lw=0, zorder=4))
    s.text(x, y + 4.6, str(n), fp(SANS, 13, "semibold"), color=ds.CULTURED, ha="center")


def slide4():
    """Adopted 4D: scale hierarchy and one time step (FIG 4), the three steps spelled out below."""
    s = Slide(4)
    header(s, "METHOD",
           "DSMC follows representative molecules, not equations: move, collide, sample, repeat.")
    y_fig = 148.0
    # build (deck/fig04_method.py): 1 domain, 2 cells, 3-5 move / collide / boundaries with their
    # text below, 6 one particle and W
    s.reveal(1)
    _, fh = s.place("fig04d_scales_and_step_deck", M, y_fig, builds=5)
    f = facts("slide04_facts")
    ratio = int(f["tau_100"] / f["dt"])          # shortest collision time in S6 (100 km) over the step
    nb = " "                                # keep a value with its symbol and unit on one line
    steps = [("Move", f"traverse under gravity for Δt{nb}={nb}{f['dt']:.2f}{nb}s, {ratio}× shorter than "
                      "a collision time"),
             ("Collide", "random pairs in the same cell; momentum and energy conserved"),
             ("Boundaries", f"wave-driven inflow at {f['z_bot']:.0f}{nb}km; escape above "
                            f"{f['z_top']:.0f}{nb}km")]
    gap, lead = 28.0, 20.5
    col_w = (RIGHT - M - 2 * gap) / 3
    prop, bold = fp(SANS, 16), fp(SANS, 16, "semibold")
    y0 = y_fig + fh + 27
    for k, (verb, rest) in enumerate(steps):
        s.reveal(3 + k)
        x = M + k * (col_w + gap)
        number_badge(s, x + 10.5, y0 - 6, k + 1)
        lines = s.wrap(f"{verb}: {rest}", prop, col_w - 36)
        s.runs(x + 30, y0, [(f"{verb}:", bold, ds.MIDNIGHT), (lines[0][len(verb) + 1:], prop, ds.GRANITE)])
        for i, line in enumerate(lines[1:], 1):
            s.text(x + 30, y0 + i * lead, line, prop)
        if y0 + (len(lines) - 1) * lead > FOOT_RULE_Y - 14:
            raise ValueError(f"slide 4: step '{verb}' runs into the footer")
    footer(s, cite("Bird 1994; HARRAH: Carberry Mogan et al. 2020–2022"))
    return s


def slide5():
    s = Slide(5)
    header(s, "1D RESULT · PUBLISHED",
           "Spherical geometry cuts wave-driven heating by 40–56% compared with slab models.")
    s.reveal(1)                                   # build: growth curves, the numbers, the measure
    s.place("fig05a_amplitude_growth_deck", EVID_X, EVID_Y)
    s.reveal(2)
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
    s.reveal(3)
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Pérez Chávez et al. 2026, ", ("italic", "Icarus"), " 447, 116900"))
    return s


def hero_row(s: Slide, x, y, w, hero, unit, labels, size=36.0):
    """Hero number with its unit, and a two-line caps label flush right on the same row."""
    hw = s.width(hero, fp(SERIF, size))
    s.text(x, y, hero, fp(SERIF, size), color=ds.MIDNIGHT)
    end = x + hw
    if unit:
        s.text(end + 7, y, unit, fp(SANS, 18, 300))
        end += 7 + s.width(unit, fp(SANS, 18, 300))
    prop = fp(SANS, 10.5, "semibold")
    for k, lab in enumerate(labels):
        lw = sum(advance(c, prop) + 0.12 * 10.5 for c in lab) - 0.12 * 10.5
        if x + w - lw < end + 12:
            raise ValueError(f"slide {s.number}: label '{lab}' runs into hero '{hero}'")
        s.tracked(x + w - lw, y - 15 + 15 * k, lab, prop, 0.12)


def slide6():
    """Adopted 6A: tall coherent wedge (FIG 6a) beside the two fits that give the hero numbers."""
    s = Slide(6)
    header(s, "2D RESULT · FIRST 2D DSMC WAVE STUDY",
           "In 2D, wave energy escapes sideways, lowering heating below the 1D prediction.")
    f = facts("slide06_facts")
    y0 = 160.0
    s.reveal(1)                                   # build: wedge, attenuation length, wavelength
    s.place("fig06a_s6_halfwedge_deck", EVID_X, y0)
    s.reveal(2)
    y = y0 + 31
    hero_row(s, WORD_X, y, WORD_W, f"{f['L_att_exobase_km']:.0f} ± {f['sigma_L_exobase_km']:.0f}", "km",
             ["ATTENUATION LENGTH", "AT THE EXOBASE"])
    _, ph = s.place("fig06b_latt_fit_col_deck", WORD_X, y + 9)
    y = y + 9 + ph + 14 + 31
    # rev5 total uncertainty (ODR + resolution); the cache holds the ODR-only error
    s.reveal(3)
    hero_row(s, WORD_X, y, WORD_W, "450 ± 14", "km", ["HORIZONTAL", "WAVELENGTH"])
    s.place("fig06c_lambdah_fit_col_deck", WORD_X, y + 9)
    footer(s, cite("Pérez Chávez et al., to be submitted, JGR Planets"))
    return s


def facts(name: str) -> dict:
    """Numbers cached and checked against their sources by deck/cache_slides78.py."""
    return json.loads((DECK_DATA / f"{name}.json").read_text())


def sci(x: float) -> tuple[str, str]:
    """1.08e9 -> ('1.1 × 10', '9') for a Playfair mantissa with a raised exponent."""
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.1f} × 10", str(e)


def hero_sci(s: Slide, x, y, value: float, size=34.0, color=ds.MIDNIGHT):
    """Hero number in Playfair with a superscript exponent; returns its width."""
    mant, exp = sci(value)
    g = next(s._group)
    w = s.width(mant, fp(SERIF, size))
    s.text(x, y, mant, fp(SERIF, size), color=color, group=g, name=f"{mant}^{exp}")
    s.text(x + w + 1.5, y - 0.42 * size, exp, fp(SERIF, 0.55 * size), color=color, group=g, name=f"{mant}^{exp}")
    return w + 1.5 + s.width(exp, fp(SERIF, 0.55 * size))


def ladder_row(s: Slide, y, color, text, status):
    s.ax.add_patch(Circle((WORD_X + 5, y - 5), 4.0, fc=color, lw=0, zorder=4))
    s.text(WORD_X + 18, y, text, fp(SANS, 14.5))
    s.tracked(RIGHT, y, status, fp(SANS, 10.5), TRACK_LABEL, ha="right")


# slide 7 footer: the former slides 7 and 8 citations, verbatim, one per line
CITE7 = "Full ring: Pérez Chávez et al., to be submitted, JGR Planets · wheel-sector runs: in progress"
CITE8 = ("MAGNUS-P (Srivastava et al. 2022) · viscous theory after Pitteway & Hines 1963; "
         "Vadas & Fritts 2005 · no viscosity: Hines 1960")


def hero_with_unit(s: Slide, x, y, hero, units, size=40.0, unit_size=17.0):
    """Playfair hero number with one or two light unit lines beside it (bottom line on the baseline)."""
    hw = s.width(hero, fp(SERIF, size))
    s.text(x, y, hero, fp(SERIF, size), color=ds.MIDNIGHT)
    for k, u in enumerate(units):
        s.text(x + hw + 10, y - (len(units) - 1 - k) * (unit_size + 2), u, fp(SANS, unit_size, 300))
    return hw


def metric(s: Slide, x, y0, label, hero, units, lines, width, size=40.0):
    """Tracked label, hero number with units, two light lines; returns the last baseline."""
    s.tracked(x, y0, label, fp(SANS, 11), TRACK_LABEL)
    hw = hero_with_unit(s, x, y0 + 40, hero, units, size=size)
    if hw + 10 + max(s.width(u, fp(SANS, 17, 300)) for u in units) > width:
        raise ValueError(f"slide {s.number}: metric '{label}' units run past the column")
    y = y0 + 40
    with s.para():
        for t in lines:
            if s.width(t, fp(SANS, 14, 300)) > width:
                raise ValueError(f"slide {s.number}: metric line too long: {t}")
            y += 18.5 + (4 if y == y0 + 40 else 0)
            s.text(x, y, t, fp(SANS, 14, 300))
    return y


def slide7():
    """Slides 7 (wheel sector) and 8 (fluid coupling) merged. Left column (split_left over split_band):
    the full ring at true scale (step 1), swapped for the wheel sector in the same slot (step 2), over
    the particle budget as 170 sector tiles (steps 1-3). Right column (split_right, step 3): FIG 8a
    redrawn over the COUPLING CHECK card. Figures: deck/fig07_merged.py; numbers from
    deck/data/slide07_facts.json and slide08_facts.json (deck/cache_slides78.py)."""
    s = Slide(7)
    f8 = facts("slide08_facts")
    header(s, "SCALING UP · THE NEXT 2D RUNS",
           "A one-wavelength wheel sector replaces the full ring, and a fluid model drives its waves "
           "from below.")
    y0, y1 = 150.0, 492.0                          # content band; common bottom of the budget and the card
    s.reveal(1)                                    # build: ring + budget, ring -> sector, fluid model
    _, hd = s.place("fig07m_ring_deck", M, y0, until=1)
    lw, hb = s.place("fig07m_budget_deck", M, y0 + hd)
    if abs(y0 + hd + hb - y1) > 0.5:
        raise ValueError(f"slide 7: left column ends at {y0 + hd + hb:.1f} pt, not {y1}")
    s.reveal(2)
    s.place("fig07m_sector_deck", M, y0)

    s.reveal(3)
    rw = ds.SIZES["split_right"][0] * 72
    rx = RIGHT - rw
    if rx - (M + lw) < 0.33 * 72 - 0.5:
        raise ValueError("slide 7: left and right columns closer than the 0.33 in gutter")
    _, hp = s.place("fig07m_profile_deck", rx, y0)
    card = Card("COUPLING CHECK",
                eq=[r"$|\hat w_{\mathrm{fluid}} - \hat w_{\mathrm{lin}}|\,/\,|\hat w_{\mathrm{lin}}| \leq 0.22\%$"],
                lines=[(f"at 100–140 km, so the {f8['n_weak_streams']} weak-forcing", 14.5, ds.GRANITE, 300),
                       ("streams are the first input to DSMC.", 14.5, ds.GRANITE, 300)])
    _, ch = card.layout()
    if y1 - ch < y0 + hp + 6:
        raise ValueError(f"slide 7: card ({ch:.0f} pt) runs into the profile")
    card.draw(s, rx, y1 - ch, w=rw)
    footer(s, lines=(CITE7, CITE8))
    return s


def slide8():
    """Roadmap to MAVEN: the six-node pipeline (FIG 9, strip slot), the hypothesis, then
    acknowledgments beside the close. No measure card; the pipeline is the content."""
    s = Slide(8)
    header(s, "NEXT", "Next: can multi-species NGIMS data pin down the wave that produced it?")
    y_fig = 148.0
    s.reveal(1)                                   # build: nodes 1-3 (FIG 9 b1), nodes 4-6, the close
    _, fh = s.place("fig09_maven_roadmap_deck", M, y_fig, builds=1)

    s.reveal(3)
    # hypothesis: tracked label, Midnight bar, Playfair italic quote
    y = y_fig + fh + 24
    s.tracked(M, y, "HYPOTHESIS", fp(SANS, 11), TRACK_LABEL)
    quote = "\u201cMultiple species break degeneracies that a single species cannot.\u201d"
    qprop = fp(SERIF, 22, style="italic")
    if s.width(quote, qprop) > RIGHT - M - 14:
        raise ValueError("slide 8: hypothesis runs past the margin")
    yq = y + 28
    s.rule(M, yq - 19, 3, 25, ds.MIDNIGHT)
    s.text(M + 14, yq, quote, qprop, color=ds.MIDNIGHT)

    # bottom row: acknowledgments | thank you and contact
    y0 = yq + 30
    s.tracked(M, y0, "ACKNOWLEDGMENTS", fp(SANS, 11), TRACK_LABEL)
    ack = fp(SANS, 13, 300)
    ends = []
    for k, t in enumerate(("NASA FINESST · ACCESS PHY240179 (TACC Stampede3)",
                           "C. Boxe, O. J. Tucker, R. E. Johnson, S. Carberry Mogan")):
        if s.width(t, ack) > WORD_X - M - 24:
            raise ValueError(f"slide 8: acknowledgment line too long: {t}")
        ends.append(y0 + 19 + 17 * k)
        s.text(M, ends[-1], t, ack)
    s.rule(WORD_X - 12, y0 - 10, 0.6, ends[-1] - y0 + 14, ds.GRANITE, alpha=0.35)
    s.text(WORD_X + 4, y0 + 17, "Thank you", fp(SERIF, 30), color=ds.MIDNIGHT)
    ends.append(y0 + 36)
    s.text(WORD_X + 4, ends[-1], "jose.perezchavez@bison.howard.edu", fp(SANS, 15))
    if max(ends) + 8 > FOOT_RULE_Y:
        raise ValueError("slide 8: closing row runs into the footer")
    footer(s, cite("MAVEN NGIMS waves: England et al. 2017; Williamson et al. 2019"))
    return s


# ================================================================= backup slides (Q&A)
def backup(k: int) -> Slide:
    """Backup slide Bk: numbered after the talk so every progress dot reads as done."""
    return Slide(N_SLIDES + k, counter=f"B{k}")


def lines14(s: Slide, x, y, texts, width, lead=18.5):
    """Light 14 pt lines under a hero row; returns the last baseline."""
    with s.para():
        for t in texts:
            if s.width(t, fp(SANS, 14, 300)) > width:
                raise ValueError(f"slide {s.counter}: line too long: {t}")
            y += lead
            s.text(x, y, t, fp(SANS, 14, 300))
    return y


def card_bottom(s: Slide, card: Card, last_y: float):
    """Measure card on the evidence baseline; raises if it would meet the text above it."""
    _, ch = card.layout()
    top = EVID_BOTTOM - ch
    if top < last_y + 10:
        raise ValueError(f"slide {s.counter}: card '{card.label}' meets the text above ({last_y:.0f} vs {top:.0f})")
    card.draw(s, WORD_X, top)


def backup1():
    """S1-S6 resolution runs: the run table (FIG B1) over the growth-convergence plot and two metrics."""
    s = backup(1)
    b = facts("backup_facts")["B1"]
    rows = {r["run"]: r for r in b["rows"]}
    header(s, "BACKUP · RESOLUTION CONVERGENCE",
           f"Peak wave growth converges as azimuthal cells shrink, levelling off near {b['fit']['A']:.1f}.")
    y_fig = 148.0
    _, fh = s.place("figB1_run_table_deck", M, y_fig)
    pw, ph = s.place("figB1b_growth_convergence_deck", M, y_fig + fh + 4)
    if y_fig + fh + 4 + ph > FOOT_RULE_Y:
        raise ValueError("slide B1: growth plot runs into the footer")
    gap = 28.0
    x1 = M + pw + gap
    col_w = (RIGHT - x1 - gap) / 2
    y0 = y_fig + fh + 24
    s4, s6 = rows["S4"], rows["S6"]
    ends = [metric(s, x1, y0, "CELLS PER WAVELENGTH", f"{s6['cells_per_lambda']:.0f}",
                   [r"cells per $\lambda_h$", "in S6"],
                   [f"S1 had {rows['S1']['cells_per_lambda']:.1f}: one cell spanned",
                    f"{rows['S1']['cell_km']:.0f} km at the exobase"], col_w),
            metric(s, x1 + col_w + gap, y0, "TWO PARTICLE BUDGETS",
                   f"{s4['ppc_230'] / rows['S3']['ppc_230']:.0f}×", ["more particles", "per cell"],
                   ["in S4–S6 than in S1–S3;",
                    f"S4 and S6 match to {100 * abs(s4['ppc_230'] / s6['ppc_230'] - 1):.1f}%"], col_w)]
    for x in (x1 - gap / 2, x1 + col_w + gap / 2):
        s.rule(x, y0 - 10, 0.6, 96, ds.GRANITE, alpha=0.35)
    if max(ends) + 8 > FOOT_RULE_Y:
        raise ValueError("slide B1: metrics run into the footer")
    footer(s, cite("Pérez Chávez et al., to be submitted, JGR Planets (Table S1, Table 4)"))
    return s


def backup2():
    """Wheel-sector cone reflection: profiles before/after (FIG B2), the residual T offset, derivation card."""
    s = backup(2)
    b = facts("backup_facts")["B2"]
    n = b["numbers"]
    header(s, "BACKUP · WHEEL-SECTOR BOUNDARY",
           "Reflecting particles off the cone, not a flat plane, removed the wheel sector's density deficit.")
    s.place("figB2_wheel_fix_deck", EVID_X, EVID_Y)
    y = EVID_Y + 31
    neg = lambda v: f"{v:+.1f}".replace("-", "−")
    hero_row(s, WORD_X, y, WORD_W, neg(n['fixNC']), "%", ["DENSITY OFFSET", "AT 392 KM"], size=40)
    y = lines14(s, WORD_X, y + 4, [f"cone wall; plane wall {neg(n['preNC'])}% (model {neg(n['modelNC'])}%)"], WORD_W)
    y += 46
    hero_row(s, WORD_X, y, WORD_W, f"+{n['fixXIITA']:.2f}–{n['fixXIITB']:.2f}", "%", ["TEMPERATURE", "ABOVE 260 KM"],
             size=30)
    y = lines14(s, WORD_X, y + 4, ["Not the estimator, collision frame or band", "width; shared by every test run, so it cancels."],
                WORD_W)
    eq = fp(SANS, 18)
    card = Card("WHY THE PLANE WALL PUSHED GAS DOWN",
                eq=[[(r"$f_R = -p\,\cos^2\!\beta\,/\,R$", eq, ds.MIDNIGHT)],
                    [(r"$d\ln n/dR = -1/H - \cos^2\!\beta\,/\,R$", eq, ds.MIDNIGHT)],
                    [(r"$n_{\mathrm{2D}}/n_{\mathrm{1D}} = (R_0/R)^{\cos^2\beta}$", eq, ds.MIDNIGHT)]],
                lines=[(f"R from Mars' centre; band half-width β = {b['beta_deg']:.1f}°", 14, ds.GRANITE, 300)])
    card_bottom(s, card, y)
    footer(s, cite("Wheel-sector boundary fix: 2-D sector against twelve repeat 1-D DSMC runs"))
    return s


def backup3():
    """One-change test runs of the wheel-sector setup and the pass rule proposed from the measured noise floor (FIG B3)."""
    s = backup(3)
    b = facts("backup_facts")["B3"]
    header(s, "BACKUP · DOMAIN-SIZE TESTS",
           "Four test runs, each changing one setting, will check that the small sector does not change the wave.")
    s.place("figB3_noise_floor_deck", EVID_X, EVID_Y)
    s.tracked(WORD_X, EVID_Y + 10, "FOUR TEST RUNS, ONE CHANGE EACH", fp(SANS, 11), TRACK_LABEL)
    y = EVID_Y + 36
    for name, change in b["twins"]:
        for t, p in ((name, fp(SANS, 15, "semibold")), (change, fp(SANS, 13, 300))):
            if s.width(t, p) > WORD_W:
                raise ValueError(f"slide B3: twin text too long: {t}")
        s.text(WORD_X, y, name, fp(SANS, 15, "semibold"), color=ds.MIDNIGHT)
        s.text(WORD_X, y + 17, change, fp(SANS, 13, 300))
        y += 40
    last = y - 40 + 17
    card = Card("PASS RULE · PROPOSED",
                eq=[[(r"$\delta_q \leq \max(3\,\delta_{\mathrm{floor}}/\sqrt{5},\ 5\%)$", fp(SANS, 18), ds.MIDNIGHT)]],
                lines=[("at each height where the noise is under 10%;", 14, ds.GRANITE, 300),
                       ("a run passes if ≤ 5% of heights fail", 14, ds.GRANITE, 300)])
    card_bottom(s, card, last)
    footer(s, cite(f"Noise floor from {b['n_wave']} repeat runs with the wave and {b['n_nowave']} without; "
                   "pass rule proposed, not yet adopted"))
    return s


def backup4():
    """Plate 1 panel B restyled: the wind a strong wave drives (FIG B4)."""
    s = backup(4)
    b = facts("backup_facts")["B4"]
    header(s, "BACKUP · STRONG FORCING",
           f"At 1 m/s forcing the wave drives its own westward wind and shifts its phase by "
           f"{b['dphase_obs_deg']:.0f}° at 140 km.")
    s.place("figB4_westward_wind_deck", EVID_X, EVID_Y)
    y = EVID_Y + 31
    hero_row(s, WORD_X, y, WORD_W, f"{b['u_peak_m_s']:.1f}".replace("-", "−"), "m/s",
             [f"MEAN WIND AT {b['u_peak_z_km']:.0f} KM", f"AFTER {b['u_peak_t_h']:.0f} HOURS"], size=40)
    y = lines14(s, WORD_X, y + 4, ["westward, and still growing"], WORD_W)
    y += 46
    hero_row(s, WORD_X, y, WORD_W, f"{b['dphase_obs_deg']:.0f}°", None, ["PHASE SHIFT", "AT 140 KM"], size=40)
    y = lines14(s, WORD_X, y + 4, [f"linear theory with the same wind: {b['dphase_lin_deg']:.0f}°"], WORD_W)
    card = Card(f"WEAK FORCING · {b['amp_weak_m_s']:g} M/S",
                eq=[[(r"$|\bar u|$ < " + f"{np.ceil(b['uB_max_m_s'] * 100) / 100:.2f} m/s", fp(SANS, 20), ds.MIDNIGHT)]],
                lines=[(f"so the {b['n_weak_streams']} weak streams are the", 14, ds.GRANITE, 300),
                       ("input to the DSMC runs", 14, ds.GRANITE, 300)])
    card_bottom(s, card, y)
    footer(s, cite(f"MAGNUS-P fluid model, isothermal pure O: {b['period_min']} min, "
                   f"{b['lambda_x_km']:.0f} km wave"))
    return s


def backup5():
    """Plate 3 panel B restyled: coarse/fine radial cells across the H2 exobase (FIG B5)."""
    s = backup(5)
    b = facts("backup_facts")["B5"]
    lo, hi = round(b["res_band_offset_pct_min"]), round(b["res_band_offset_pct_max"])
    header(s, "BACKUP · RADIAL GRID",
           f"Radial cells wider than {b['dr_fine_km']} km across the hydrogen exobase bias the density "
           f"{lo}–{hi}% high.")
    s.place("figB5_radial_grid_deck", EVID_X, EVID_Y)
    y = EVID_Y + 31
    hero_row(s, WORD_X, y, WORD_W, f"+{lo}–{hi}", "%",
             ["DENSITY BIAS", f"{b['res_band_lo_km']:,.0f}–{b['res_band_hi_km']:,.0f} KM"], size=40)
    y = lines14(s, WORD_X, y + 4,
                [f"{b['dr_coarse_km']:,} km against {b['dr_fine_km']} km cells, {b['z_top_ratio_km']:,} km top",
                 f"H₂ exobase at {b['exo_all_min_km']:.0f}–{b['exo_all_max_km']:.0f} km",
                 "Thick lines: pure H₂; thin: coupled O + H₂"], WORD_W)
    card = Card("ADOPTED FOR THE MULTI-SPECIES RUNS",
                eq=[[(rf"$\Delta r \leq {b['dr_fine_km']}$ km,  top ≥ {b['z_top_min_km']:,} km", fp(SANS, 20),
                      ds.MIDNIGHT)]],
                lines=[(f"from the {b['n_runs']}-run upper-boundary sweep", 14, ds.GRANITE, 300)])
    card_bottom(s, card, y)
    footer(s, cite("DSMC upper-boundary sweep: pure H₂ and coupled O + H₂ columns"))
    return s


def backup6():
    """Phase-lag retrieval of lambda_h: the fit and the baseline sweep (FIG B6), error budget, lambda_z."""
    s = backup(6)
    b = facts("backup_facts")["B6"]
    header(s, "BACKUP · HORIZONTAL WAVELENGTH",
           f"The {b['lambda_h_km']:.0f} km horizontal wavelength is one straight-line phase fit over "
           f"{b['n']} columns.")
    s.place("figB6_phase_lag_deck", EVID_X, EVID_Y)
    y = EVID_Y + 31
    hero_row(s, WORD_X, y, WORD_W, f"{b['lambda_h_km']:.0f} ± {b['sigma_total_km']:.0f}", "km",
             ["HORIZONTAL", "WAVELENGTH"], size=40)
    y = lines14(s, WORD_X, y + 4, [f"±{b['sigma_stat_km']:.0f} statistical (ODR, calibrated ×1.4)",
                                   f"±{b['sigma_syst_km']:.0f} fit window and construction"], WORD_W)
    y += 46
    (lv, le), (rv, re_) = b["left"], b["right"]
    hero_row(s, WORD_X, y, WORD_W, f"{lv:.0f} / {rv:.0f}", "km", ["LEFT / RIGHT", "EDGE ALONE"], size=32)
    sig = abs(rv - lv) / np.hypot(le, re_)
    y = lines14(s, WORD_X, y + 4, [f"±{le:.0f} / ±{re_:.0f} km: consistent at {sig:.1f}σ"], WORD_W)
    card = Card("SAME WAVE, VERTICAL",
                eq=[[(rf"$\lambda_z = {b['lambda_z_colmean_km']:.0f} \pm {b['lambda_z_colstd_km']:.0f}$ km",
                      fp(SANS, 20), ds.MIDNIGHT)]],
                lines=[(f"mean over {b['lambda_z_ncol']} columns, so", 14, ds.GRANITE, 300),
                       (r"$\lambda_h \approx \lambda_z$: phase fronts tilt about 45°", 14, ds.GRANITE, 300)])
    card_bottom(s, card, y)
    footer(s, cite("Pérez Chávez et al., to be submitted, JGR Planets; S6 at 231 km"))
    return s


def deck_slides() -> list[Slide]:
    """The 14 slides in order: talk slides 1-8, then backups B1-B6 (also slides/export_pptx.py)."""
    return [slide1(), slide2(), slide3(), slide4(), slide5(), slide6(), slide7(), slide8(),
            backup1(), backup2(), backup3(), backup4(), backup5(), backup6()]


# ================================================================= audit and checks
def figure_size(stem: str):
    """MediaBox of a figure PDF in points."""
    raw = (FIGS / f"{stem}.pdf").read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    return float(m.group(1)), float(m.group(2))


def expected_size(stem: str):
    base = re.sub(r"_b\d+$", "", stem.removesuffix("_deck"))      # build variants share the size
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


def figure_ink_in(stem: str, x, y_top, rect) -> bool:
    """Whether the placed figure has any ink (alpha > 0.03, 300-dpi PNG) inside rect (x0, y0, x1, y1), pt."""
    img = mpimg.imread(FIGS / f"{stem}.png")
    alpha = img[..., 3] if img.shape[-1] == 4 else 1 - img[..., :3].min(axis=-1)
    k = 300.0 / 72.0
    c0, c1 = max(int((rect[0] - x) * k), 0), max(int(np.ceil((rect[2] - x) * k)), 0)
    r0, r1 = max(int((rect[1] - y_top) * k), 0), max(int(np.ceil((rect[3] - y_top) * k)), 0)
    return bool((alpha[r0:r1, c0:c1] > 0.03).any())


def check_layout(s: Slide, k: int | None = None):
    """Text inside the page, no text/text, text/figure-ink or panel/figure-ink collisions, on build
    step k (default: the full slide)."""
    k = s.n_steps if k is None else k
    placed = s.show(k)
    panels = [p[:5] for p in s.panels if p[5] <= k]
    boxes = []
    for t, g, name in s.texts:
        if not t.get_visible():
            continue
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
    for px, py, pw, ph, pname in panels:
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
    for x0, y0, x1, y1, name, first, last in s.images:
        if not first <= k <= (k if last is None else last):
            continue
        if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
            problems.append(f"image outside page: {name}")
        for box, tname in items:
            if hit(box, (x0, y0, x1, y1), pad=-4.0):
                problems.append(f"image over text: {name} / '{tname}'")
        for px, py, pw, ph, pname in panels:
            if hit((px, py, px + pw, py + ph), (x0, y0, x1, y1), pad=-4.0):
                problems.append(f"image over {pname}: {name}")
        for stem, fx, fy, fw, fh in placed:
            if figure_ink_in(stem, fx, fy, (x0 - 4, y0 - 4, x1 + 4, y1 + 4)):
                problems.append(f"image within 4 pt of figure ink: {name} / {stem}")
    for stem, x, y, w, h in placed:
        ink = ink_rect(stem, x, y)
        if ink[0] < 0 or ink[1] < 0 or ink[2] > W or ink[3] > H:
            problems.append(f"figure ink off page: {stem}")
        for box, name in items:
            if hit(box, ink):
                problems.append(f"text over figure ink: '{name}' / {stem}")
        for px, py, pw, ph, pname in panels:
            if "spark" in stem and pname.startswith("card"):
                continue
            if hit((px, py, px + pw, py + ph), ink):
                problems.append(f"{pname} over figure ink: {stem}")
    s.show(s.n_steps)
    if problems:
        raise SystemExit(f"slide {s.counter or s.number} step {k} layout:\n  " + "\n  ".join(problems))


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


def compose(pages, commit: str, out_pdf: Path = DECK_PDF, subject: str | None = None, work: Path = WORK):
    """Lay each page's figure PDFs over its chrome at natural size; one PDF out.

    pages: (slide, build step) pairs; the full deck shows every slide at its last step."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    body = []
    for k, (s, step) in enumerate(pages):
        placed = s.show(step)
        chrome = work / f"chrome{k:02d}.pdf"
        s.fig.savefig(chrome, facecolor=s.field, transparent=False)
        s.show(s.n_steps)
        page = [rf"\AddToShipoutPictureBG*{{\put(0,0){{\includegraphics{{{chrome.name}}}}}}}"]
        for stem, x, y, w, h in placed:
            shutil.copy(FIGS / f"{stem}.pdf", work / f"{stem}.pdf")
            page.append(rf"\AddToShipoutPictureFG*{{\put({x:.3f},{H - y - h:.3f}){{\includegraphics{{{stem}.pdf}}}}}}")
        body.append("\n".join(page) + "\n\\null\\newpage\n")
    subject = subject or f"NSBP/NSHP 2026, slides 1-{N_SLIDES} and backups B1-B6; figures/deck @ {commit}"
    tex = TEX_HEAD % (TITLE, subject, ds.STYLE_VERSION)
    tex += "".join(body) + r"\end{document}" + "\n"
    (work / "deck.tex").write_text(tex)
    subprocess.run(["tectonic", "-X", "compile", "deck.tex"], cwd=work, check=True, capture_output=True)
    shutil.copy(work / "deck.pdf", out_pdf)


def verify(n_pages: int, pdf: Path = DECK_PDF, prev: Path = OUT / "preview", name: str = "slide"):
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
    subprocess.run(["pdftoppm", "-r", "110", "-png", str(pdf), str(prev / name)], check=True)
    return pages, sorted({n for n, _ in fonts})


def check_builds(pages, full_prev: Path, prog_prev: Path, full_index: dict, tol: int = 40):
    """No layout jitter: no build page may put ink where the full slide is blank (an element that moved
    or changed would), and a slide's last build page must equal the full slide. Ink that a later step
    covers (a badge over a grid line) is allowed, and so is ink inside the slot of a figure that is
    swapped out later (Slide.place(..., until=k): the full ring on slide 7).

    Compares the 110 dpi previews; returns the worst share of misplaced inked pixels."""
    worst = 0.0
    bg = np.array(matplotlib.colors.to_rgb(ds.CULTURED)) * 255
    full_png = sorted(full_prev.glob("*.png"))
    prog_png = sorted(prog_prev.glob("*.png"))
    for i, (s, step) in enumerate(pages):
        a = mpimg.imread(prog_png[i])[..., :3] * 255
        b = mpimg.imread(full_png[full_index[id(s)]])[..., :3] * 255
        if a.shape != b.shape:
            raise SystemExit(f"progressive page {i + 1}: preview size differs from the full slide")
        diff = np.abs(a - b).max(axis=-1)
        if step == s.n_steps:
            bad = (diff > tol).mean()
        else:                                                    # build steps exist on light slides only
            ink = np.abs(a - bg).max(axis=-1) > tol
            blank = np.abs(b - bg).max(axis=-1) <= tol
            stray = ink & blank
            k_px = a.shape[1] / W
            for x, y, w, h, first, last in s.swapped:
                if first <= step <= last:
                    stray[int(np.floor(y * k_px)):int(np.ceil((y + h) * k_px)),
                          int(np.floor(x * k_px)):int(np.ceil((x + w) * k_px))] = False
            bad = stray.sum() / max(ink.sum(), 1)
        worst = max(worst, bad)
        if bad > 1e-3:
            raise SystemExit(f"progressive page {i + 1} (slide {s.counter or s.number}, step {step}): "
                             f"{100 * bad:.2f}% of inked pixels are not on the full slide")
    return worst


def main():
    for note in audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    slides = deck_slides()
    full = [(s, s.n_steps) for s in slides]
    prog = [(s, k) for s in slides for k in s.pages()]
    for s in slides:
        for stem in {l[0] for l in s.layers}:
            audit_figure(stem)
        for k in s.pages():
            check_layout(s, k)
    print(f"figures: sizes, fonts, provenance OK; layout: no clipping or collisions on all {len(full)} "
          f"slides and {len(prog)} build pages")
    compose(full, commit)
    pages, fonts = verify(len(full))
    print(f"{DECK_PDF.relative_to(REFACTORED)}: {pages} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")
    compose(prog, commit, PROG_PDF, work=OUT / "_work_progressive",
            subject=f"NSBP/NSHP 2026, progressive builds of slides 2-{N_SLIDES} ({len(prog)} pages); "
                    f"figures/deck @ {commit}")
    pages, fonts = verify(len(prog), PROG_PDF, OUT / "preview_progressive", name="page")
    print(f"{PROG_PDF.relative_to(REFACTORED)}: {pages} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")
    worst = check_builds(prog, OUT / "preview", OUT / "preview_progressive",
                         {id(s): i for i, s in enumerate(slides)})
    print("builds: " + ", ".join(f"{s.counter or s.number} x{len(s.pages())}" for s in slides
                                 if len(s.pages()) > 1)
          + f"; no element moves between steps (worst mismatch {100 * worst:.3f}% of inked pixels)")
    for s in slides:
        plt.close(s.fig)


if __name__ == "__main__":
    main()
