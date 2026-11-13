"""Graphite Telemetry figure style for the ubc_hold_carberry report (see ../../oh2_escape_rerun/design_philosophy.md).

The palette, rcParams and font helpers are shared with the simulation cards and the oh2_escape_rerun report
(dsmclib.graphite). H2 orange is the only species ink; upper boundary conditions are told apart by line style,
marker and grey value through the fixed UBC map below, never by a second colour.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from dsmclib.graphite import (BAND_ALPHA, CHARCOAL, COL, FONTS, GRAPHITE, GREY, LABEL, LIGHT,  # noqa: F401
                              PAPER, RAW_ALPHA, RULE, WIDTH, mono, setup, tick_mono)

REPORT = Path(__file__).resolve().parents[1]
FIGDIR = REPORT / "figures"
H2 = COL["H2"]

# Fixed UBC style map. Key = solver UBC code. 'ink' is the stroke colour: H2 orange for the two ballistic
# treatments (the subject of the report), charcoal for reflect (the reference), grey for free escape.
UBC = {
    1: dict(name="reflect", label="reflect ($v<v_\\mathrm{esc}$ specular)", short="reflect",
            ink=CHARCOAL, ls="-", marker="o", mfc=PAPER, lw=1.4),
    2: dict(name="kepler", label="ballistic, Kepler hold", short="Kepler",
            ink=H2, ls="-", marker="s", mfc=H2, lw=1.6),
    5: dict(name="explicit", label="ballistic, explicit RK4", short="explicit",
            ink=H2, ls=(0, (4, 1.8)), marker="D", mfc=PAPER, lw=1.6),
    3: dict(name="free", label="free escape", short="free escape",
            ink=GREY, ls=(0, (1, 1.4)), marker="v", mfc=PAPER, lw=1.3),
}
ANALYTIC_LS = (0, (5, 2.5))   # collisionless analytic references: thin grey dashed
PAPER_MARK = dict(marker="x", ms=4.2, mew=1.0, color=GRAPHITE, ls="none")   # Carberry Mogan et al. (2021) markers


def line_kw(code, **over):
    s = UBC[code]
    kw = dict(color=s["ink"], ls=s["ls"], lw=s["lw"])
    kw.update(over)
    return kw


def marker_kw(code, ms=4.2, **over):
    s = UBC[code]
    kw = dict(color=s["ink"], marker=s["marker"], ms=ms, mfc=s["mfc"], mec=s["ink"], mew=0.9, ls="none")
    kw.update(over)
    return kw


def pending(ax, text, x=0.5, y=0.5, size=7.2):
    """Clearly labelled placeholder for data not yet available."""
    ax.text(x, y, f"PENDING\n{text}", transform=ax.transAxes, ha="center", va="center", fontsize=size,
            color=GREY, linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.5", fc=PAPER, ec=LIGHT, lw=0.6, ls=(0, (3, 2))), zorder=10)


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    return FIGDIR / f"{name}.pdf"
