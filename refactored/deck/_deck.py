"""Shared setup for the NSBP/NSHP 2026 deck figure scripts.

Run every script from ``refactored/``:  ``conda run -n dsmc python deck/<script>.py``

Applies the locked Rarefied Order style (``style/deckstyle.py``; see docs/FIGURE_STYLE.md) and
adds only what the style module leaves to the caller: provenance strings and the wedge mapping.
No colours or sizes are defined here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REFACTORED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFACTORED / "style"))

import deckstyle as ds  # noqa: E402

ds.use("deck", font_dir=REFACTORED / "style" / "fonts")

OUT = REFACTORED / "figures" / "deck"
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
