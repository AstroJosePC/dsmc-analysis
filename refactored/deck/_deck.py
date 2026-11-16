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


def wedge_xy(alt_km, phi, phi0: float = 0.0, r0_km: float = R_MARS_KM):
    """Altitude x azimuth -> true-scale (x, y) in km; y = 0 at the surface under phi0, up positive."""
    r = r0_km + np.asarray(alt_km, dtype=float)
    a = np.asarray(phi, dtype=float) - phi0
    return r * np.sin(a), r * np.cos(a) - r0_km
