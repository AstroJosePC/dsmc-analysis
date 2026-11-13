"""Graphite Telemetry figure style for the ubc_hold_carberry report (see ../design_philosophy.md).

The palette, rcParams and font helpers are shared with the simulation cards and the oh2_escape_rerun report
(dsmclib.graphite). H2 orange is the only species ink. Boundary conditions are told apart by ink, dash and marker shape,
datasets by stroke weight and marker fill, and grid resolution by marker size (maps below).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dsmclib.graphite import (BAND_ALPHA, CHARCOAL, COL, FONTS, GRAPHITE, GREY, LABEL, LIGHT,  # noqa: F401
                              PAPER, RAW_ALPHA, RULE, WIDTH, mono, setup, tick_mono)

REPORT = Path(__file__).resolve().parents[1]
FIGDIR = REPORT / "figures"
H2 = COL["H2"]

# Visual grammar (design_philosophy.md, "Visual grammar"). Three orthogonal channels:
#   boundary condition -> ink + dash + marker shape   (UBC, key = solver UBC code)
#   dataset            -> stroke weight + marker fill (DATASET)
#   grid resolution    -> marker size                 (MS)
UBC = {
    1: dict(name="reflect", label="reflect", ink=CHARCOAL, ls="-", marker="o"),
    2: dict(name="kepler", label="Kepler hold", ink=H2, ls="-", marker="s"),
    5: dict(name="explicit", label="explicit RK4", ink=H2, ls=(0, (4, 1.8)), marker="D"),
    3: dict(name="free", label="free escape", ink=GREY, ls=(0, (1, 1.4)), marker="v"),
}
DATASET = {
    "sweep": dict(lw=1.0, fill="full", label="sweep, $3\\times10^5$ s"),                  # collisional, 3e5 s
    "campaign": dict(lw=2.0, fill="none", label="campaign, $2\\times10^6$ s"),            # collisional, 2e6 s
    "verification": dict(lw=1.1, fill="bottom", label="collisionless verification"),     # collisions off, 6e4 s
}
MS = {250: 4.6, 1000: 2.8}    # marker size for the maximum cell width in km
KEY_LW = 1.4                  # stroke of a legend entry that names a boundary condition only
ANALYTIC_LS = (0, (5, 2.5))   # collisionless analytic references: thin grey dashed
ANALYTIC = dict(color=GREY, lw=0.8, ls=ANALYTIC_LS)
PAPER_MARK = dict(marker="x", ms=4.2, mew=1.0, color=GRAPHITE, ls="none")   # Carberry Mogan et al. (2021) markers
LEG_FS = 6.3                  # legend font size
LEG_ROW = 9.0                 # height of one legend row in points


def line_kw(code, ds="sweep", **over):
    """Stroke of boundary condition `code` in dataset `ds`. Explicit RK4 in the campaign is drawn at half weight
    (see plot)."""
    s = UBC[code]
    lw = DATASET[ds]["lw"] if ds else KEY_LW
    if code == 5 and ds == "campaign":
        lw *= 0.5
    kw = dict(color=s["ink"], ls=s["ls"], lw=lw)
    kw.update(over)
    return kw


def marker_kw(code, ds="sweep", ms=4.2, **over):
    """Marker of boundary condition `code` (shape, ink) in dataset `ds` (fill), without a line."""
    s = UBC[code]
    fill = DATASET[ds]["fill"]
    kw = dict(color=s["ink"], marker=s["marker"], ms=ms, mec=s["ink"], mew=0.9, ls="none")
    if fill == "full":
        kw["mfc"] = s["ink"]
    elif fill == "none":
        kw["mfc"] = PAPER
    else:
        kw.update(fillstyle=fill, mfc=s["ink"], markerfacecoloralt=PAPER)
    kw.update(over)
    return kw


def plot(ax, x, y, code, ds, ms=None, markevery=None, **over):
    """Draw one curve in the grammar. The campaign explicit RK4 curve is a slim dashed stroke on a paper-coloured
    core, so that it stays visible when it lies on the heavier Kepler-hold line."""
    kw = line_kw(code, ds, **over)
    if code == 5 and ds == "campaign":
        ax.plot(x, y, color=PAPER, lw=kw["lw"], ls="-", zorder=kw.get("zorder", 3))
    if ms:
        m = marker_kw(code, ds, ms=ms)
        m.pop("ls")
        kw.update(m, markevery=markevery)
    return ax.plot(x, y, **kw)


def handle(code, ds=None, label=None, line=True, ms=None, **over):
    """Legend entry for a boundary condition (ds=None: key stroke) or for a boundary condition in a dataset."""
    kw = line_kw(code, ds) if line else dict(color=UBC[code]["ink"], ls="none")
    if ms:
        m = marker_kw(code, ds or "sweep", ms=ms)
        m.pop("ls")
        kw.update(m)
    kw.update(over)
    return Line2D([], [], label=label or UBC[code]["label"], **kw)


def analytic_handle(label="collisionless, analytic"):
    return Line2D([], [], label=label, **ANALYTIC)


def band_handle(label="$\\pm1\\sigma$", color=GREY):
    return Patch(fc=color, ec="none", alpha=0.25, label=label)


def header(fig, text, left, right=0.985):
    """Setup line at the top of a figure: what experiment the reader is looking at. Only facts true of every panel."""
    h = fig.get_figheight()
    fig.text(left, 1 - 0.05 / h, text, ha="left", va="top", fontsize=6.6, color=GRAPHITE)
    y = 1 - 0.20 / h
    fig.add_artist(Line2D([left, right], [y, y], color=RULE, lw=0.6))


def panel(ax, title, handles=None, ncol=1, rows=0, **kw):
    """Panel title, with an optional legend between the title and the axes. A legend that describes a whole row is
    attached to the left panel; the other panels of the row get the same `rows` so their titles line up."""
    ax.set_title(title, fontsize=8.5, pad=5 + (rows * LEG_ROW + 3 if rows else 0))
    if handles:
        opt = dict(loc="lower left", bbox_to_anchor=(0, 1), ncol=ncol, fontsize=LEG_FS, borderaxespad=0.3,
                   borderpad=0, columnspacing=1.4, handletextpad=0.5, labelspacing=0.32, handlelength=2.1)
        opt.update(kw)
        return ax.legend(handles=handles, **opt)


def inset_legend(ax, handles, loc, **kw):
    """Legend inside an axes, on a paper-coloured box so that grid lines do not run through the text."""
    opt = dict(loc=loc, fontsize=6.0, handletextpad=0.5, labelspacing=0.32, borderaxespad=0.5, borderpad=0.4,
               handlelength=2.1, frameon=True, framealpha=1.0, facecolor=PAPER, edgecolor="none")
    opt.update(kw)
    leg = ax.legend(handles=handles, **opt)
    ax.add_artist(leg)     # keep it when the panel legend is created afterwards
    ax.legend_ = None
    return leg


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
