"""Editable PowerPoint export of the 15-slide master deck (slides 1-9, backups B1-B6).

    cd refactored
    conda run -n dsmc python slides/export_pptx.py --render-svg   # figures as SVG (after a figure changes)
    conda run -n dsmc python slides/export_pptx.py --calibrate    # PowerPoint baselines (after a layout change)
    conda run -n dsmc python slides/export_pptx.py --check        # slides/build/NSBP2026_deck_editable.pptx,
                                                                   # then verified in PowerPoint

The slides are not laid out here. build_deck.py's slide functions are called and each Slide's matplotlib
page (points, y down, 960 x 540 pt = 13.333 x 7.5 in) is translated artist by artist, so the PowerPoint
file and NSBP2026_deck_full.pdf share one layout source. Neither PDF is read or written.

  Text       one native text box per text group (kicker caps keep their tracking as character spacing,
             citation runs keep their italics) or per paragraph (Slide.para: headlines, title, light
             lines). Boxes have no insets, no wrap and no autofit; the box top is the baseline minus
             PowerPoint's own first-baseline offset for that line's mix of fonts and spacing, measured
             by --calibrate in PowerPoint (slides/pptx_baselines.json). PowerPoint rounds exact line
             spacing to whole points, so a paragraph whose leading is not whole (the title, 31.5 pt)
             becomes a group of one-line boxes. Text transparency is composited over the field or card
             fill (same colour, no alpha). Fonts are the deck's own, named as their TTFs name them:
             Open Sans (Light, SemiBold) and Playfair Deck LF (Playfair Display with lining figures,
             so numbers stay lining in PowerPoint, which cannot switch OpenType features).
  Math       inline $...$ becomes Unicode runs in STIX Two Text / STIX Two Math (sub/superscripts,
             Greek, accents, TeX operator spacing). Measure-card equation rows (Slide.card_eqs) are
             left out, their space kept, and their LaTeX is written to the slide's speaker notes for
             re-entry as native PowerPoint equations.
  Shapes     background, Crimson rules, card and chip outlines, footer rule and progress dots, as
             native rectangles, rounded rectangles, ellipses and lines.
  Figures    at their exact slots from Slide.placed: SVG (figures/deck/svg, rendered by the figure
             scripts with DECK_SVG=1: text as text, mathtext as outlines) with the 300-dpi PNG as
             fallback; fig01 and fig06a, heavy particle and mesh plots, as their 300-dpi transparent
             PNG. FIG B1, the run table, is a native table built from deck/figB_backups.b1_table(), the
             source figB1 draws.
  --check    exports the file through Microsoft PowerPoint (AppleScript) and compares every text line's
             baseline with the layout (tolerance 1 pt).

The fonts must be installed for PowerPoint (style/fonts/*.ttf; STIX Two ships with macOS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
import zlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont
from lxml import etree
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.spines import Spine
from matplotlib.text import Text
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.oxml.ns import qn
from pptx.util import Emu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402  (style, fonts and deck/ on sys.path)
from build_deck import ds  # noqa: E402

import figB_backups as fb  # noqa: E402

PPTX = bd.OUT / "NSBP2026_deck_editable.pptx"
SVG_DIR = bd.FIGS / "svg"
RASTER = {"fig01_title_wedge", "fig06a_s6_halfwedge"}       # heavy particle / mesh plots: 300-dpi PNG
NATIVE = {"figB1_run_table"}                                # rebuilt as a native table
SVG_SCRIPTS = [("fig02_ngims.py",), ("fig03_regimes.py",), ("fig04_method.py",), ("fig05_growth.py",),
               ("fig06_alternatives.py", "panels"), ("fig07_alternatives.py", "b2"), ("fig08_baselines.py",),
               ("fig09_roadmap.py",), ("figB_backups.py", "B1b", "B2", "B3", "B4", "B5", "B6")]
MATH_FONT = "STIX Two Text"
MATH_SYMBOLS = "STIX Two Math"                              # arrows and relations STIX Two Text lacks
SYSTEM_FONTS = Path("/System/Library/Fonts/Supplemental")
EMU_PT = 12700

# PowerPoint's first baseline below the top of a box with zero insets and exact line spacing L, per
# (font, italic, size, L) used in the deck. It is not linear in L and size, so each combination is
# measured: --calibrate renders probe boxes in PowerPoint itself, exports them to PDF and reads the
# baselines from the text matrices. A combination missing from the file stops the export.
BASELINES = Path(__file__).resolve().parent / "pptx_baselines.json"
WORK = bd.OUT / "_work_pptx"
PPT_DOCS = Path.home() / "Library/Containers/com.microsoft.Powerpoint/Data/Documents"   # sandbox-readable
_OFFSETS: dict = {}
_PITCH: dict = {}
PITCH_TOL = 0.15              # pt; a paragraph is one box only where PowerPoint keeps its leading
_NEEDED: set = set()
_INTENDED: list = []          # (slide index, box left, box right, [baselines]) for --check

THEME_COLORS = {"dk1": ds.GRANITE, "lt1": ds.CULTURED, "dk2": ds.MIDNIGHT, "lt2": bd.CARD_FILL,
                "accent1": ds.MIDNIGHT, "accent2": ds.CRIMSON, "accent3": ds.CELTIC, "accent4": ds.GOLD,
                "accent5": ds.TANGELO, "accent6": ds.GREEN, "hlink": ds.CELTIC, "folHlink": ds.VIOLET}


def emu(pt: float) -> Emu:
    return Emu(round(pt * EMU_PT))


def rgb(color) -> tuple[RGBColor, float]:
    r, g, b, a = matplotlib.colors.to_rgba(color)
    return RGBColor(round(r * 255), round(g * 255), round(b * 255)), a


def set_alpha(fill_fore_color_parent, alpha: float):
    """Add <a:alpha> to the srgbClr under a solidFill element."""
    if alpha >= 0.999:
        return
    clr = fill_fore_color_parent.find(qn("a:srgbClr"))
    a = etree.SubElement(clr, qn("a:alpha"))
    a.set("val", str(round(alpha * 100000)))


# ================================================================= fonts
_FACE = {}


def face(prop) -> tuple[str, bool]:
    """PowerPoint family name and italic flag of the TTF matplotlib uses for prop."""
    path = findfont(prop, fallback_to_default=False)
    if path not in _FACE:
        if Path(path).parent != bd.REFACTORED / "style" / "fonts":
            raise SystemExit(f"{prop}: resolves to {path}, not a deck font")
        name = TTFont(path)["name"]
        _FACE[path] = (name.getDebugName(1), "Italic" in name.getDebugName(2))
    return _FACE[path]


def line_key(segs, line: float) -> str:
    """A line's signature for the offset table: exact spacing and the set of (font, italic, size,
    sub/superscript) of its runs, plain runs first. Mixed lines are measured as such, because
    PowerPoint's first baseline in a line of several fonts follows no simple rule of the single ones."""
    sig = sorted({(f, bool(it), float(sz), sc or "") for _, f, it, sz, sc, *_ in segs}, key=lambda t: (t[3] != "", t))
    return f"{round(line, 2):g}|" + ";".join(f"{f},{'i' if it else 'r'},{sz:g},{sc}" for f, it, sz, sc in sig)


def line_offset(segs, line: float) -> float:
    """PowerPoint's first baseline below the top of a zero-inset box (pt) for this line of runs."""
    key = line_key(segs, line)
    _NEEDED.add(key)
    if key in _OFFSETS:
        return _OFFSETS[key]
    if _OFFSETS.get("_collect"):
        return 0.8 * line
    raise SystemExit(f"no PowerPoint baseline for {key}: run slides/export_pptx.py --calibrate")


# ================================================================= inline math -> runs
GREEK = {n: c for n, c in zip(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon "
    "phi chi psi omega".split(), "αβγδεζηθικλμνξπρστυφχψω")}
GREEK |= {"Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
          "Psi": "Ψ", "Omega": "Ω", "varepsilon": "ε", "varphi": "φ"}
SYMBOL = {"uparrow": "↑", "downarrow": "↓", "to": "→", "rightarrow": "→", "pm": "±", "mp": "∓",
          "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "approx": "≈", "sim": "∼", "times": "×",
          "cdot": "·", "infty": "∞", "langle": "⟨", "rangle": "⟩", "propto": "∝", "partial": "∂",
          "neq": "≠", "degree": "°", "circ": "∘"}
FUNC = {"max", "min", "ln", "log", "exp", "cos", "sin", "tan"}
SPACE = {",": " ", ";": " ", ":": " ", " ": " ", "quad": " ", "qquad": "  ",
         "!": ""}
BIN = set("+−×±∓·∗")                                       # TeX binary operators: medium space
REL = set("=<>≤≥≈∼→←↑↓∝≠")                                  # relations: thick space
MED_SPACE, THICK_SPACE = "\u205f", "\u2005"                  # 4/18 em; 1/4 em (TeX 5/18)
ACCENT = {"bar": "̄", "hat": "̂", "tilde": "̃", "dot": "̇", "vec": "⃗"}


def parse_math(src: str, followed: bool = False):
    """mathtext -> [(text, italic, script)] with script None, 'sub' or 'sup'. Raises on what it cannot
    write faithfully as text."""
    out = []
    pos = 0
    state = {"last": None, "pending": None}     # TeX spacing: class of the last atom, space owed after it

    def group(upright, script, fresh=False):
        nonlocal pos
        saved = dict(state)
        if fresh:                                  # a sub/superscript is its own formula
            state.update(last=None, pending=None)
        if pos < len(src) and src[pos] == "{":
            pos += 1
            atoms(upright, script, stop="}")
            pos += 1
        else:
            atoms(upright, script, single=True)
        if fresh:
            state.update(saved)

    def put(text, italic, script):
        if out and out[-1][1:] == (italic, script):
            out[-1] = (out[-1][0] + text, italic, script)
        else:
            out.append((text, italic, script))

    def emit(text, italic, script, kind=None):
        """kind: None (ordinary; operators are classified by character) or 'space'."""
        if kind == "space":
            if text == "":                         # \! cancels the space owed
                state["pending"] = None
            put(text, False, script)
            return
        if kind == "op":                           # \ln, \max ...: thin space next to ordinary atoms
            if state["pending"]:
                put(state["pending"], False, script)
            elif state["last"] == "ord":
                put("\u2009", False, script)
            put(text, italic, script)
            state.update(last="op", pending="\u2009")
            return
        if kind is None:
            kind = "bin" if text in BIN else "rel" if text in REL else "ord"
        if kind == "bin" and state["last"] in (None, "bin", "rel"):
            kind = "ord"                           # unary, as in TeX
        if kind in ("bin", "rel"):
            sp = MED_SPACE if kind == "bin" else THICK_SPACE
            if state["last"] is not None:
                put(sp, False, script)
            state["pending"] = sp
        elif state["pending"]:
            put(state["pending"], False, script)
            state["pending"] = None
        put(text, italic, script)
        state["last"] = kind

    def atoms(upright, script, stop=None, single=False):
        nonlocal pos
        while pos < len(src):
            c = src[pos]
            if stop and c == stop:
                return
            if c == "\\":
                m = re.match(r"\\([A-Za-z]+|.)", src[pos:])
                cmd = m.group(1)
                pos += len(m.group(0))
                if cmd in SPACE:
                    emit(SPACE[cmd], False, script, kind="space")
                elif cmd in GREEK:
                    emit(GREEK[cmd], not upright and cmd[0].islower(), script)
                elif cmd in SYMBOL:
                    emit(SYMBOL[cmd], False, script)
                elif cmd in FUNC:
                    emit(cmd, False, script, kind="op")
                elif cmd in ("mathrm", "rm", "text"):
                    group(True, script)
                elif cmd in ACCENT:
                    n0 = len(out)
                    if pos < len(src) and src[pos] == " ":
                        pos += 1
                    group(upright, script)
                    if len(out) == n0:
                        raise ValueError(f"accent without a base in {src!r}")
                    t, i, sc = out[-1]
                    out[-1] = (t + ACCENT[cmd], i, sc)
                elif cmd in "{}|%$#&_":
                    emit(cmd, False, script)
                else:
                    raise ValueError(f"unsupported mathtext command \\{cmd} in {src!r}")
            elif c in "_^":
                pos += 1
                group(upright, "sub" if c == "_" else "sup", fresh=True)
            elif c == "{":
                pos += 1
                atoms(upright, script, stop="}")
                pos += 1
            elif c == "}":
                raise ValueError(f"unbalanced braces in {src!r}")
            elif c == " ":                     # mathtext ignores spaces in math
                pos += 1
            else:
                pos += 1
                c = "−" if c == "-" else c
                emit(c, c.isalpha() and c.isascii() and not upright, script)
            if single:
                return

    atoms(False, None)
    if followed and state["pending"] and state["last"] == "rel":   # matplotlib keeps it before text
        put(state["pending"], False, None)
    return out


_CMAP = {}


def math_font(ch: str, italic: bool) -> str:
    """STIX Two Text where it has the character, else STIX Two Math (both ship with macOS)."""
    if not _CMAP:
        _CMAP["text"] = set(TTFont(SYSTEM_FONTS / "STIXTwoText.ttf").getBestCmap())
        _CMAP["italic"] = set(TTFont(SYSTEM_FONTS / "STIXTwoText-Italic.ttf").getBestCmap())
        _CMAP["math"] = set(TTFont(SYSTEM_FONTS / "STIXTwoMath.otf").getBestCmap())
    if ord(ch) in _CMAP["italic" if italic else "text"]:
        return MATH_FONT
    if ord(ch) in _CMAP["math"]:
        return MATH_SYMBOLS
    raise ValueError(f"no STIX Two font has {ch!r} (U+{ord(ch):04X})")


def segments(text: str, prop, color, alpha):
    """[(text, font, italic, size, script, color, alpha)] for one artist's string."""
    font, italic = face(prop)
    size = prop.get_size_in_points()
    parts = text.split("$")
    if len(parts) % 2 == 0:
        raise ValueError(f"unbalanced $ in {text!r}")
    segs = []
    for k, p in enumerate(parts):
        if not p:
            continue
        if k % 2 == 0:
            segs.append((p, font, italic, size, None, color, alpha))
        else:
            for t, it, script in parse_math(p, followed=k < len(parts) - 1 and bool(parts[k + 1])):
                for ch in t:                       # combining accents stay with their base's font
                    f = segs[-1][1] if unicodedata.combining(ch) else math_font(ch, it)
                    if segs and segs[-1][1:] == (f, it, size, script, color, alpha) and k % 2:
                        segs[-1] = (segs[-1][0] + ch, *segs[-1][1:])
                    else:
                        segs.append((ch, f, it, size, script, color, alpha))
    return segs


# ================================================================= text boxes
def add_runs(paragraph, segs, spc=None):
    for text, font, italic, size, script, color, alpha in segs:
        r = paragraph.add_run()
        r.text = text
        f = r.font
        f.name = font
        f.size = emu(size)
        f.italic = italic
        f.bold = False
        c, a = rgb(color)
        f.color.rgb = c
        rpr = r._r.get_or_add_rPr()
        rpr.set("kern", "100")                     # pair kerning on, as matplotlib draws text
        if spc:
            rpr.set("spc", str(round(spc * 100)))
        if script:
            rpr.set("baseline", "-25000" if script == "sub" else "30000")
        set_alpha(rpr.find(qn("a:solidFill")), alpha * a)
        for tag in ("a:ea", "a:cs"):               # same face for every script
            el = rpr.find(qn(tag))
            if el is None:
                el = etree.SubElement(rpr, qn(tag))
            el.set("typeface", font)


def text_box(slide, x, y_base, lines, ha="left", spc=None, name="Text", leading=None, before=None, shapes=None):
    """lines: [segs]; x is the anchor for ha; y_base the first baseline. leading: exact line spacing
    (single-line boxes use 1.2 x size); before[k]: extra space before line k. Where PowerPoint does not
    keep that leading (it has a floor near 1.07 x size), the lines become one box each, grouped."""
    shapes = shapes if shapes is not None else slide.shapes
    main = max(lines[0], key=lambda s: s[3])
    font, italic, size = main[1], main[2], main[3]
    L = leading or round(1.2 * size, 2)
    before = before or [0.0] * len(lines)
    key = line_key(lines[0], L)
    if len(lines) > 1 and not _OFFSETS.get("_collect") and abs(_PITCH[key] - L) > PITCH_TOL:
        grp = shapes.add_group_shape()
        grp.name = name
        y = y_base
        for k, line in enumerate(lines):
            y += before[k] + (L if k else 0)
            text_box(slide, x, y, [line], ha=ha, spc=spc, name=f"{name} · line {k + 1}", leading=L, shapes=grp.shapes)
        return grp
    top = y_base - line_offset(lines[0], L)
    width = max(line_width(l, spc) for l in lines) + 4.0
    left = {"left": x, "right": x - width, "center": x - width / 2}[ha]
    height = L * len(lines) + sum(before)
    _INTENDED.append((len(slide.part.package.presentation_part.presentation.slides) - 1, left, left + width,
                      [y_base + k * L + sum(before[:k + 1]) for k in range(len(lines))]))
    shp = shapes.add_textbox(emu(left), emu(top), emu(width), emu(height))
    shp.name = name
    tf = shp.text_frame
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for k, segs in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT, "center": PP_ALIGN.CENTER}[ha]
        p.line_spacing = emu(L)
        if before[k]:
            p.space_before = emu(before[k])
        add_runs(p, segs, spc)
    return shp


_MEASURE = {}


def line_width(segs, spc=None) -> float:
    """Advance width in pt (matplotlib's measure of the same fonts), for the box width only."""
    key = (tuple(s[:5] for s in segs), spc)
    if key not in _MEASURE:
        fig = Figure(figsize=(1, 1), dpi=72)
        r = FigureCanvasAgg(fig).get_renderer()
        w = 0.0
        for text, font, italic, size, script, *_ in segs:
            fam = {"Open Sans Light": ("Open Sans", 300), "Open Sans SemiBold": ("Open Sans", 600)}.get(
                font, (font, 400))
            t = fig.text(0, 0, text, family=fam[0], weight=fam[1], style="italic" if italic else "normal",
                         size=size * (0.66 if script else 1.0))
            w += t.get_window_extent(r).width + (spc or 0) * len(text)
        _MEASURE[key] = w
    return _MEASURE[key]


# ================================================================= one slide
def opaque(s: bd.Slide, t: Text):
    """The artist's colour composited over what lies under it (slide field or measure-card fill), so no
    text carries transparency: same look, plain colours to edit, and PowerPoint's PDF export keeps it as
    text."""
    alpha = 1.0 if t.get_alpha() is None else t.get_alpha()
    r, g, b, a = matplotlib.colors.to_rgba(t.get_color())
    alpha *= a
    if alpha >= 0.999:
        return t.get_color()
    x, y = t.get_position()
    under = s.field
    for px, py, pw, ph, pname, _ in s.panels:
        if pname.startswith("card") and px <= x <= px + pw and py <= y <= py + ph:
            under = bd.CARD_FILL
    bg = matplotlib.colors.to_rgb(under)
    return matplotlib.colors.to_hex([alpha * c + (1 - alpha) * u for c, u in zip((r, g, b), bg)])


def artist_segs(s: bd.Slide, t: Text):
    return segments(t.get_text(), t.get_fontproperties(), opaque(s, t), 1.0)


def text_units(s: bd.Slide):
    """Text artists grouped into export units, keyed by their first artist: tracked caps, runs, single
    texts, then single-line units of one paragraph merged."""
    groups, order = {}, []
    for t, g, name in s.texts:
        if t in s.card_eqs:
            continue
        if g not in groups:
            groups[g] = (name, [])
            order.append(g)
        groups[g][1].append(t)
    units = []
    for g in order:
        name, arts = groups[g]
        x0, y0 = arts[0].get_position()
        for a in arts:
            if a.get_va() != "baseline" or a.get_rotation() != 0 or a.get_position()[1] != y0:
                raise SystemExit(f"slide {s.counter or s.number}: '{name}' is not a single baseline-aligned line")
        if len(arts) > 1 and all(len(a.get_text()) == 1 for a in arts) and \
                "".join(a.get_text() for a in arts) == name.replace(" ", ""):
            prop = arts[0].get_fontproperties()
            xl = arts[-1].get_position()[0]
            adv = sum(bd.advance(c, prop) for c in name)
            spc = ((xl + bd.advance(name[-1], prop)) - x0 - adv) / max(len(name) - 1, 1)
            segs = segments(name, prop, opaque(s, arts[0]), 1.0)
            units.append(dict(first=arts[0], arts=arts, x=x0, y=y0, ha="left", lines=[segs], spc=spc, name=name))
        elif len(arts) > 1:
            segs = [seg for a in arts for seg in artist_segs(s, a)]
            units.append(dict(first=arts[0], arts=arts, x=x0, y=y0, ha="left", lines=[segs], spc=None, name=name))
        else:
            a = arts[0]
            units.append(dict(first=a, arts=arts, x=x0, y=y0, ha=a.get_ha(), lines=[artist_segs(s, a)], spc=None,
                              name=name, para=s.para_of.get(a)))
    merged, by_para = [], {}
    for u in units:
        p = u.get("para")
        if p is None:
            merged.append(u)
        elif p in by_para:
            m = by_para[p]
            if (u["x"], u["ha"]) != (m["x"], m["ha"]):
                raise SystemExit(f"slide {s.number}: paragraph lines not aligned: {u['name']}")
            m["ys"].append(u["y"])
            m["lines"] += u["lines"]
            m["arts"] += u["arts"]
        else:
            by_para[p] = dict(u, ys=[u["y"]])
            merged.append(by_para[p])
    for m in merged:
        ys = m.pop("ys", [m["y"]])
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        m["leading"] = min(gaps) if gaps else None
        m["before"] = [0.0] + [g - min(gaps) for g in gaps] if gaps else None
    return merged


def export_slide(prs, s: bd.Slide, notes: list):
    s.show(s.n_steps)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    units = {u["first"]: u for u in text_units(s)}
    members = {a for u in units.values() for a in u["arts"]}
    counts = dict(text=0, shape=0)
    children = sorted(enumerate(s.ax.get_children()), key=lambda ia: (ia[1].get_zorder(), ia[0]))
    for _, a in children:
        if isinstance(a, (Spine, Axis)) or not a.get_visible():
            continue
        if isinstance(a, Text):
            if a in units:
                u = units[a]
                text_box(slide, u["x"], u["y"], u["lines"], ha=u["ha"], spc=u["spc"], leading=u["leading"],
                         before=u["before"], name=f"Text · {u['name'][:48]}")
                counts["text"] += 1
            elif a not in members and a not in s.card_eqs and a.get_text():
                raise SystemExit(f"slide {s.number}: unregistered text {a.get_text()!r}")
        elif isinstance(a, FancyBboxPatch):
            x, y, w, h = a.get_x(), a.get_y(), a.get_width(), a.get_height()
            shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, emu(x), emu(y), emu(w), emu(h))
            shp.adjustments[0] = a.get_boxstyle().rounding_size / min(w, h)
            style_fill_line(shp, a.get_facecolor(), a.get_edgecolor(), a.get_linewidth())
            shp.name = "Card" if a.get_facecolor()[3] > 0 else "Chip"
            counts["shape"] += 1
        elif isinstance(a, Rectangle):
            x, y, w, h = a.get_x(), a.get_y(), a.get_width(), a.get_height()
            if a.get_zorder() == -10 and (w, h) == (bd.W, bd.H):
                c, _ = rgb(a.get_facecolor())
                slide.background.fill.solid()
                slide.background.fill.fore_color.rgb = c
                continue
            shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, emu(x), emu(y), emu(w), emu(h))
            style_fill_line(shp, a.get_facecolor(), None, 0)
            shp.name = "Rule"
            counts["shape"] += 1
        elif isinstance(a, Circle):
            (cx, cy), r = a.center, a.radius
            shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, emu(cx - r), emu(cy - r), emu(2 * r), emu(2 * r))
            style_fill_line(shp, a.get_facecolor(), None, 0)
            shp.name = "Dot"
            counts["shape"] += 1
        elif isinstance(a, Line2D):
            xs, ys = a.get_xdata(), a.get_ydata()
            if len(xs) != 2 or a.get_linestyle() != "-":
                raise SystemExit(f"slide {s.number}: only straight solid lines are exported")
            ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(xs[0]), emu(ys[0]), emu(xs[1]), emu(ys[1]))
            c, al = rgb(a.get_color())
            ln.line.color.rgb = c
            ln.line.width = emu(a.get_linewidth())
            set_alpha(ln.line._get_or_add_ln().find(qn("a:solidFill")), (a.get_alpha() or 1.0) * al)
            ln.name = "Rule"
            counts["shape"] += 1
        else:
            raise SystemExit(f"slide {s.number}: no PowerPoint translation for {type(a).__name__}")
    pics = dict(svg=0, png=0, table=0)
    for stem, x, y, w, h in s.placed:
        base = stem.removesuffix("_deck")
        if base in NATIVE:
            b1_native(slide, x, y, w, h)
            pics["table"] += 1
        elif base in RASTER:
            pic = slide.shapes.add_picture(str(bd.FIGS / f"{stem}.png"), emu(x), emu(y), emu(w), emu(h))
            pic.name = f"Figure · {base}"
            pics["png"] += 1
        else:
            add_svg_picture(slide, stem, x, y, w, h)
            pics["svg"] += 1
    eqs = list(dict.fromkeys(s.card_eqs.values()))
    if eqs:
        slide.notes_slide.notes_text_frame.text = "\n".join(
            f"Card {label}: equation for the empty row (LaTeX): {src}" for label, src in eqs)
        notes += [(s.counter or s.number, label) for label, _ in eqs]
    return counts, pics


def style_fill_line(shp, fc, ec, lw):
    c, a = rgb(fc)
    if a > 0:
        shp.fill.solid()
        shp.fill.fore_color.rgb = c
        set_alpha(shp.fill._xPr.find(qn("a:solidFill")), a)
    else:
        shp.fill.background()
    if ec is None or lw == 0 or rgb(ec)[1] == 0:
        shp.line.fill.background()
    else:
        c, a = rgb(ec)
        shp.line.color.rgb = c
        shp.line.width = emu(lw)
        set_alpha(shp.line._get_or_add_ln().find(qn("a:solidFill")), a)
    shp.shadow.inherit = False
    if shp.has_text_frame:
        shp.text_frame.text = ""


# ================================================================= figures
SVG_EXT = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"


def svg_check(path: Path, w: float, h: float):
    root = etree.parse(str(path)).getroot()
    sw, sh = (float(root.get(k).removesuffix("pt")) for k in ("width", "height"))
    if abs(sw - w) > 0.05 or abs(sh - h) > 0.05:
        raise SystemExit(f"{path.name}: {sw} x {sh} pt, slot is {w:.2f} x {h:.2f} pt")
    meta = etree.tostring(root.find("{http://www.w3.org/2000/svg}metadata")).decode()
    if " @ " not in meta or "deckstyle" not in meta:
        raise SystemExit(f"{path.name}: no provenance (Source ... @ commit, Creator deckstyle)")


def add_svg_picture(slide, stem, x, y, w, h):
    svg = SVG_DIR / f"{stem}.svg"
    if not svg.exists():
        raise SystemExit(f"{svg.relative_to(bd.REFACTORED)} missing: run slides/export_pptx.py --render-svg")
    svg_check(svg, w, h)
    pic = slide.shapes.add_picture(str(bd.FIGS / f"{stem}.png"), emu(x), emu(y), emu(w), emu(h))
    pic.name = f"Figure · {stem.removesuffix('_deck')}"
    pkg = slide.part.package
    part = Part(pkg.next_partname("/ppt/media/image%d.svg"), "image/svg+xml", pkg, svg.read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = pic._element.find(".//" + qn("a:blip"))
    ext_lst = blip.find(qn("a:extLst"))
    if ext_lst is None:
        ext_lst = etree.SubElement(blip, qn("a:extLst"))
    ext = etree.SubElement(ext_lst, qn("a:ext"))
    ext.set("uri", SVG_EXT)
    el = etree.SubElement(ext, f"{{{ASVG}}}svgBlip", nsmap={"asvg": ASVG})
    el.set(qn("r:embed"), rid)


# ================================================================= FIG B1 as a native table
def b1_native(slide, x, y, w, h):
    t = fb.b1_table()
    w_in, h_in = t["size"]
    X = lambda xi: x + 72 * xi                     # noqa: E731  figure inches -> slide points
    Y = lambda yi: y + 72 * (h_in - yi)            # noqa: E731
    color = matplotlib.rcParams["text.color"]
    sb = lambda txt, size, weight, c: segments(   # noqa: E731
        txt, bd.fp(bd.SANS, size, weight), c, 1.0)
    xc = t["xc"]
    edges = [X(t["x_span"][0])] + [X((a + b) / 2) for a, b in zip(xc, xc[1:])] + [X(t["x_span"][1])]
    rules = t["rules"]
    head_lead = 72 * (t["header"][1]["lines"][0][0] - t["header"][1]["lines"][1][0])
    y_h1 = Y(t["header"][0]["lines"][0][0])
    hs, rs = t["header_size"], t["row_size"]
    hfont, rfont = face(bd.fp(bd.SANS, hs, "semibold"))[0], face(bd.fp(bd.SANS, rs))[0]
    pad = 2.0
    top = y_h1 - max(line_offset(sb(c["lines"][0][1], hs, "semibold", color), head_lead)
                     for c in t["header"]) - pad
    # row boundaries: header rule, then 9 pt below each baseline unless a rule is there
    rule_y = [Y(r["y"]) for r in rules]
    bounds = [top, rule_y[0]]
    for i, r in enumerate(t["rows"]):
        yb = Y(r["y"])
        nxt = [ry for ry in rule_y[1:] if 0 < ry - yb < 12]
        bounds.append(nxt[0] if nxt else yb + 9.0)
    n_rows, n_cols = len(t["rows"]) + 1, len(xc)
    frame = slide.shapes.add_table(n_rows, n_cols, emu(edges[0]), emu(top), emu(edges[-1] - edges[0]),
                                   emu(bounds[-1] - top))
    frame.name = "Table · FIG B1 run table"
    tbl = frame.table
    tbl_pr = tbl._tbl.tblPr
    for k in ("firstRow", "bandRow", "firstCol", "lastRow", "lastCol", "bandCol"):
        if tbl_pr.get(k) is not None:
            del tbl_pr.attrib[k]
    sid = tbl_pr.find(qn("a:tableStyleId"))
    if sid is not None:
        tbl_pr.remove(sid)
    for j in range(n_cols):
        tbl.columns[j].width = emu(edges[j + 1] - edges[j])
    for i in range(n_rows):
        tbl.rows[i].height = emu(bounds[i + 1] - bounds[i])
    ruled = {0: rules[0]}                          # bottom border of row i (0 = header)
    for r in rules[1:]:
        ry = Y(r["y"])
        i = min(range(1, n_rows), key=lambda i: abs(bounds[i + 1] - ry))
        if abs(bounds[i + 1] - ry) > 0.01:
            raise SystemExit("FIG B1: a rule falls inside a row")
        ruled[i] = r

    def fill_cell(i, j, lines, ha, x_anchor, base_y, size, font, lead):
        cell = tbl.cell(i, j)
        tf = cell.text_frame
        tf.word_wrap = False
        lo, hi = edges[j], edges[j + 1]
        if ha == "left":
            ml, mr = x_anchor - lo, 1.0
        else:
            d = x_anchor - (lo + hi) / 2
            ml, mr = 1.0 + max(0.0, 2 * d), 1.0 + max(0.0, -2 * d)
        mt = base_y - bounds[i] - line_offset(lines[0], lead)
        if (mt < 0 or ml < 0) and not _OFFSETS.get("_collect"):
            raise SystemExit(f"FIG B1: cell ({i}, {j}) does not fit its row")
        mt, ml = max(mt, 0.0), max(ml, 0.0)
        _INTENDED.append((len(slide.part.package.presentation_part.presentation.slides) - 1, lo, hi,
                          [base_y + k * lead for k in range(len(lines))]))
        cell.margin_left, cell.margin_right, cell.margin_top, cell.margin_bottom = emu(ml), emu(mr), emu(mt), 0
        cell.vertical_anchor = MSO_ANCHOR.TOP
        for k, segs in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT if ha == "left" else PP_ALIGN.CENTER
            p.line_spacing = emu(lead)
            add_runs(p, segs)

    for j, c in enumerate(t["header"]):
        fill_cell(0, j, [sb(txt, hs, "semibold", color) for _, txt in c["lines"]], c["ha"], X(c["x"]), y_h1,
                  hs, hfont, head_lead)
    for i, r in enumerate(t["rows"], start=1):
        for j, (xi, ha, v, weight) in enumerate(r["cells"]):
            fill_cell(i, j, [sb(v, rs, weight, r["color"])], ha, X(xi), Y(r["y"]), rs, rfont, rs)
    for i in range(n_rows):
        for j in range(n_cols):
            tc_pr = tbl.cell(i, j)._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tc_pr, qn(tag))
                r = ruled.get(i) if tag == "a:lnB" else ruled.get(i - 1) if tag == "a:lnT" else None
                if r is None:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
                else:
                    ln.set("w", str(round(r["lw"] * EMU_PT)))
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    clr = etree.SubElement(sf, qn("a:srgbClr"))
                    clr.set("val", ds.GRANITE.lstrip("#").upper())
                    if r["alpha"] < 1:
                        etree.SubElement(clr, qn("a:alpha")).set("val", str(round(r["alpha"] * 100000)))
            etree.SubElement(tc_pr, qn("a:noFill"))
    for r in t["rows"]:
        cx, cy, rad, fc = r["dot"]
        shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, emu(X(cx) - 72 * rad), emu(Y(cy) - 72 * rad),
                                     emu(144 * rad), emu(144 * rad))
        style_fill_line(shp, fc, None, 0)
        shp.name = "Dot · run colour"
    for xi, yi, txt in t["notes"]:
        text_box(slide, X(xi), Y(yi), [sb(txt, t["note_size"], t["note_weight"], color)],
                 name=f"Text · {txt[:48]}")


# ================================================================= presentation
def theme(prs):
    master = prs.slide_master.part
    part = master.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    scheme = root.find(".//a:clrScheme", ns)
    scheme.set("name", "Rarefied Order")
    for k, v in THEME_COLORS.items():
        el = scheme.find(f"a:{k}", ns)
        for child in list(el):
            el.remove(child)
        etree.SubElement(el, qn("a:srgbClr")).set("val", v.lstrip("#").upper())
    fonts = root.find(".//a:fontScheme", ns)
    fonts.set("name", "Rarefied Order")
    fonts.find("a:majorFont/a:latin", ns).set("typeface", "Playfair Deck LF")
    fonts.find("a:minorFont/a:latin", ns).set("typeface", "Open Sans")
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


# ================================================================= PowerPoint round trip
APPLESCRIPT = """on run argv
  set outHFS to (POSIX file (item 2 of argv)) as string
  tell application "Microsoft PowerPoint"
    open (POSIX file (item 1 of argv))
    delay 1
    set thePres to active presentation
    save thePres in outHFS as save as PDF
    close thePres saving no
  end tell
end run"""


def powerpoint_pdf(pptx: Path) -> Path:
    """PDF of pptx as PowerPoint renders it (Microsoft PowerPoint via AppleScript, in its sandbox folder)."""
    box = PPT_DOCS / "nsbp_deck_export"
    box.mkdir(parents=True, exist_ok=True)
    src, pdf = box / pptx.name, box / (pptx.stem + ".pdf")
    src.write_bytes(pptx.read_bytes())
    pdf.unlink(missing_ok=True)
    subprocess.run(["osascript", "-e", APPLESCRIPT, str(src), str(pdf)], check=True, timeout=600)
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / pdf.name
    out.write_bytes(pdf.read_bytes())
    return out


def pdf_text_origins(path: Path) -> list[list[tuple[float, float, float]]]:
    """Per page, the (x, baseline y from the top, size) of every text-showing operator, from the content
    streams (PowerPoint's Quartz PDFs: no object streams, Flate content, pages listed in order)."""
    raw = path.read_bytes()
    offs = {int(m.group(1)): m.end() for m in re.finditer(rb"(?<![0-9])(\d+) 0 obj", raw)}

    def obj(n):
        return raw[offs[n]:raw.index(b"endobj", offs[n])]

    def stream(n):
        o = obj(n)
        i = o.index(b"stream") + 6
        i += 2 if o[i:i + 2] == b"\r\n" else 1
        data = o[i:o.rindex(b"endstream")]
        return zlib.decompress(data) if b"FlateDecode" in o[:i] else data

    def kids(n):
        o = obj(n)
        if re.search(rb"/Type\s*/Pages", o):
            refs = re.search(rb"/Kids\s*\[([^\]]*)\]", o).group(1)
            return [p for k in re.findall(rb"(\d+) 0 R", refs) for p in kids(int(k))]
        return [n]

    root = int(re.search(rb"/Root\s+(\d+) 0 R", raw).group(1))
    pages = kids(int(re.search(rb"/Pages\s+(\d+) 0 R", obj(root)).group(1)))
    height = float(re.search(rb"/MediaBox\s*\[\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)", raw).group(1))
    tok = re.compile(rb"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|[-+]?(?:\d+\.?\d*|\.\d+)"
                     rb"|[A-Za-z'\"*]+|<<|>>", re.S)
    mul = lambda m, n: (m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2],  # noqa: E731
                        m[2] * n[1] + m[3] * n[3], m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])
    out = []
    for p in pages:
        refs = re.search(rb"/Contents\s*(\[[^\]]*\]|\d+ 0 R)", obj(p)).group(1)
        data = b"\n".join(stream(int(k)) for k in re.findall(rb"(\d+) 0 R", refs))
        ctm, stack, tm, tf, ops, found = (1, 0, 0, 1, 0, 0), [], (1, 0, 0, 1, 0, 0), 1.0, [], []
        for t in tok.findall(data):
            if t[:1] in b"([<]/" or t[:1].isdigit() or t[:1] in b"-+.":
                ops.append(t)
                continue
            num = lambda k: float(ops[k])  # noqa: E731
            if t == b"q":
                stack.append(ctm)
            elif t == b"Q":
                ctm = stack.pop()
            elif t == b"cm":
                ctm = mul(tuple(num(k) for k in range(-6, 0)), ctm)
            elif t == b"BT":
                tm = (1, 0, 0, 1, 0, 0)
            elif t == b"Tm":
                tm = tuple(num(k) for k in range(-6, 0))
            elif t == b"Td":
                tm = mul((1, 0, 0, 1, num(-2), num(-1)), tm)
            elif t == b"Tf":
                tf = num(-1)
            elif t in (b"Tj", b"TJ", b"'", b'"'):
                m = mul(tm, ctm)
                found.append((m[4], height - m[5], tf * abs(m[3])))
            ops = []
        out.append(found)
    return out


def calibrate():
    """Measure PowerPoint's first-baseline offset and its actual line pitch for every (font, italic,
    size, exact spacing) the deck uses: three-line probe paragraphs, exported to PDF by PowerPoint."""
    _OFFSETS.clear()
    _OFFSETS["_collect"] = True
    WORK.mkdir(parents=True, exist_ok=True)
    build(WORK / "collect.pptx", quiet=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = emu(bd.W), emu(bd.H)
    probes, col, top, slide = [], 3, 0.0, None
    for key in sorted(_NEEDED):
        line, sig = key.split("|")
        line = float(line)
        runs = [(f, it == "i", float(sz), sc or None) for f, it, sz, sc in (e.split(",") for e in sig.split(";"))]
        if slide is None or top + 3 * line + 10 > bd.H - 10:
            col, top = col + 1, 20.0
            if col == 4:
                slide, col = prs.slides.add_slide(prs.slide_layouts[6]), 0
        x = 30 + col * 230
        shp = slide.shapes.add_textbox(emu(x), emu(top), emu(200), emu(3 * line))
        tf = shp.text_frame
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for k in range(3):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.line_spacing = emu(line)
            add_runs(p, [("HH", f, it, sz, sc, ds.GRANITE, 1.0) for f, it, sz, sc in runs])
        probes.append((len(prs.slides) - 1, x, top, line, key))
        top += 3 * line + 10
    probe = WORK / "baseline_probe.pptx"
    prs.save(probe)
    pages = pdf_text_origins(powerpoint_pdf(probe))
    offsets, pitch = {}, {}
    for page, x, top, line, key in probes:
        hits = sorted(y for xx, y, _ in pages[page] if abs(xx - x) < 0.5 and top - 5 < y < top + 3 * line + 5)
        if len(hits) != 3:
            raise SystemExit(f"calibration: {len(hits)} text origins for {key}")
        offsets[key] = round(hits[0] - top, 3)
        pitch[key] = round((hits[2] - hits[0]) / 2, 3)
    version = subprocess.run(["osascript", "-e", 'tell application "Microsoft PowerPoint" to get version'],
                             capture_output=True, text=True).stdout.strip()
    BASELINES.write_text(json.dumps({
        "powerpoint": f"Microsoft PowerPoint for Mac {version}",
        "note": "offsets: first baseline below the box top (pt), zero insets, exact line spacing; pitch: the "
                "line pitch PowerPoint actually draws at that spacing. slides/export_pptx.py --calibrate",
        "offsets": offsets, "pitch": pitch}, indent=1, ensure_ascii=False) + "\n")
    kept = sum(abs(pitch[k] - float(k.split("|")[0])) <= PITCH_TOL for k in pitch)
    print(f"{BASELINES.relative_to(bd.REFACTORED)}: {len(offsets)} combinations; exact spacing kept by "
          f"PowerPoint for {kept}")


def check(pptx: Path = PPTX, tol: float = 1.0):
    """Export pptx through PowerPoint and compare every text line's baseline with the layout."""
    pdf = powerpoint_pdf(pptx)
    pages = pdf_text_origins(pdf)
    worst, bad = 0.0, []
    for page, left, right, ys in _INTENDED:
        for y in ys:
            near = [abs(yy - y) for xx, yy, _ in pages[page] if left - 2 <= xx <= right + 2]
            d = min(near) if near else float("inf")
            worst = max(worst, d)
            if d > tol:
                bad.append((page + 1, round(left), round(y, 1), round(d, 2)))
    print(f"PowerPoint check ({pdf.relative_to(bd.REFACTORED)}): {sum(len(i[3]) for i in _INTENDED)} text lines, "
          f"worst baseline offset {worst:.2f} pt")
    if bad:
        raise SystemExit(f"baselines off by more than {tol} pt (page, x, y, offset): {bad}")
    return pdf


def render_svg():
    env = dict(os.environ, DECK_SVG="1")
    for script, *args in SVG_SCRIPTS:
        subprocess.run([sys.executable, str(bd.REFACTORED / "deck" / script), *args], cwd=bd.REFACTORED,
                       env=env, check=True)


def build(out: Path = PPTX, quiet: bool = False) -> Path:
    slides = [bd.slide1(), bd.slide2(), bd.slide3(), bd.slide4(), bd.slide5(), bd.slide6(), bd.slide7(),
              bd.slide8(), bd.slide9(), bd.backup1(), bd.backup2(), bd.backup3(), bd.backup4(), bd.backup5(),
              bd.backup6()]
    for s in slides:
        for stem, *_ in s.placed:
            bd.audit_figure(stem)
    prs = Presentation()
    prs.slide_width, prs.slide_height = emu(bd.W), emu(bd.H)
    prs.core_properties.title = bd.TITLE
    prs.core_properties.subject = f"NSBP/NSHP 2026, editable export of NSBP2026_deck_full.pdf; figures/deck @ {bd_commit()}"
    theme(prs)
    total, figs, notes = dict(text=0, shape=0), dict(svg=0, png=0, table=0), []
    for s in slides:
        c, p = export_slide(prs, s, notes)
        for k in c:
            total[k] += c[k]
        for k in p:
            figs[k] += p[k]
        plt.close(s.fig)
    prs.save(out)
    if quiet:
        return out
    shown = out.relative_to(bd.REFACTORED) if out.is_relative_to(bd.REFACTORED) else out
    print(f"{shown}: {len(slides)} slides, 13.333 x 7.5 in; {total['text']} text boxes, "
          f"{total['shape']} shapes; figures: {figs['svg']} SVG, {figs['png']} PNG, {figs['table']} native table")
    print("card equations left for native entry (LaTeX in speaker notes): "
          + "; ".join(f"{n}: {label}" for n, label in notes))
    return out


def bd_commit() -> str:
    return subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render-svg", action="store_true", help="re-render the vector figures as SVG and stop")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure PowerPoint's baseline offsets (Microsoft PowerPoint, AppleScript) and stop")
    ap.add_argument("--check", action="store_true",
                    help="after building, export through PowerPoint and compare every baseline with the layout")
    ap.add_argument("--out", type=Path, default=PPTX)
    a = ap.parse_args()
    if a.render_svg:
        render_svg()
    elif a.calibrate:
        calibrate()
    else:
        cal = json.loads(BASELINES.read_text())
        _OFFSETS.update(cal["offsets"])
        _PITCH.update(cal["pitch"])
        build(a.out)
        if a.check:
            check(a.out)
