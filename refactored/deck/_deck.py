"""Shared setup for the NSBP/NSHP 2026 deck figure scripts.

Run every script from ``refactored/``:  ``conda run -n dsmc python deck/<script>.py``

Applies the locked Rarefied Order style (``style/deckstyle.py``; see docs/FIGURE_STYLE.md) and
adds only what the style module leaves to the caller: provenance strings and the wedge mapping.
No colours or sizes are defined here.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REFACTORED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFACTORED / "style"))

import deckstyle as ds  # noqa: E402

ds.use("deck", font_dir=REFACTORED / "style" / "fonts")

OUT = REFACTORED / "figures" / "deck"
# DECK_SVG=1 (slides/export_pptx.py --render-svg): write only figures/deck/svg/<name>_deck.svg, with
# text kept as text (svg.fonttype: none), for the editable PowerPoint export. No PDF or PNG is touched,
# and scripts that also write deck/data files skip those writes.
SVG_ONLY = os.environ.get("DECK_SVG") == "1"
SVG_OUT = OUT / "svg"
# Open Sans weights by the family names their TTFs carry, as PowerPoint and macOS resolve them (a bare
# 'Open Sans' at weight 600 would be synthesised bold from Regular).
SVG_FACES = {"600": "Open Sans SemiBold", "300": "Open Sans Light"}

if SVG_ONLY:
    from matplotlib.backends.backend_svg import RendererSVG

    _svg_draw_text = RendererSVG.draw_text

    def _svg_math_as_path(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """Mathtext as outlines: its sans-serif glyphs are private-use codepoints of matplotlib's
        STIXNonUnicode font, which no other application has. Plain text stays text."""
        if ismath:
            return self._draw_text_as_path(gc, x, y, s, prop, angle, ismath, mtext)
        return _svg_draw_text(self, gc, x, y, s, prop, angle, ismath, mtext)

    RendererSVG.draw_text = _svg_math_as_path


def _svg_faces(path: Path):
    def face(m):
        style = m.group(0)
        w = re.search(r"font-weight: (\d+); ", style)
        if w and w.group(1) in SVG_FACES and "font-family: 'Open Sans'" in style:
            style = style.replace(w.group(0), "").replace("font-family: 'Open Sans'",
                                                          f"font-family: '{SVG_FACES[w.group(1)]}'")
        return style
    text = re.sub(r'style="[^"]*font-family[^"]*"', face, path.read_text())
    # runs of spaces (aligned legend columns) as no-break spaces: SVG renderers collapse plain ones
    text = re.sub(r"(<text[^>]*>)([^<]*)(</text>)",
                  lambda m: m.group(1) + re.sub(r"  +", lambda s: " " + "\u00a0" * (len(s.group(0)) - 1),
                                                m.group(2)) + m.group(3), text)
    path.write_text(text)
MONO = "IBM Plex Mono"
R_MARS_KM = 3389.5  # dsmclib.constants.MARS_RADIUS


def commit() -> str:
    return subprocess.run(["git", "-C", str(REFACTORED), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def source(*inputs) -> str:
    """Provenance for ds.save: '<input>; <input> @ <commit>'."""
    return "; ".join(str(p) for p in inputs) + f" @ {commit()}"


def save(fig, name: str, *inputs, png: bool = True) -> Path:
    """ds.save into refactored/figures/deck with provenance (PNG alongside as a preview)."""
    if SVG_ONLY:
        SVG_OUT.mkdir(parents=True, exist_ok=True)
        path = SVG_OUT / f"{name}_deck.svg"
        fig.savefig(path, metadata={"Creator": f"deckstyle {ds.STYLE_VERSION}", "Source": source(*inputs)})
        _svg_faces(path)
        print(path)
        return path
    path = ds.save(fig, name, target="deck", outdir=OUT, png=png, source=source(*inputs))
    print(path)
    return path


class Build:
    """Build steps of one figure for the progressive deck (slides/build_deck.py places them in turn).

    Draw the whole figure once, calling at(k) before the artists that should first appear at step k
    (k = 1, 2, ...; at() may go back to an earlier step). save() writes <name>_b1 ... <name>_b{n-1}, each
    the finished figure with the later artists hidden, then the finished figure as <name>. Nothing is
    laid out twice, so every artist sits at the same place in every step.
    """

    def __init__(self, fig):
        self.fig, self.step, self.tag = fig, 1, {}
        self._seen = set(fig.findobj())

    def _sweep(self):
        for a in self.fig.findobj():
            if a not in self._seen:
                self._seen.add(a)
                self.tag[a] = self.step

    def at(self, k: int):
        self._sweep()
        self.step = k

    def save(self, name: str, *inputs) -> list[Path]:
        self._sweep()
        n = max(self.tag.values())
        if SVG_ONLY:                     # the editable export uses the finished figure only
            return [save(self.fig, name, *inputs)]
        shown = {a: a.get_visible() for a in self.tag}
        paths = []
        for k in range(1, n):
            for a, s in self.tag.items():
                a.set_visible(shown[a] and s <= k)
            paths.append(save(self.fig, f"{name}_b{k}", *inputs))
        for a in self.tag:
            a.set_visible(shown[a])
        paths.append(save(self.fig, name, *inputs))
        return paths


def r_label(alt_km: float, bound: bool = True) -> str:
    """Shell label on curved-geometry figures. r is the radial altitude above the surface,
    r = R - R_Mars (defined on slide 3, FIG 3): the bounds of a domain read 'r = 100 km' with an
    italic r (mathtext, stixsans), interior shells just '230 km'."""
    return (r"$r$ = " if bound else "") + f"{alt_km:.0f} km"


def wedge_xy(alt_km, phi, phi0: float = 0.0, r0_km: float = R_MARS_KM):
    """Altitude x azimuth -> true-scale (x, y) in km; y = 0 at the surface under phi0, up positive."""
    r = r0_km + np.asarray(alt_km, dtype=float)
    a = np.asarray(phi, dtype=float) - phi0
    return r * np.sin(a), r * np.cos(a) - r0_km
