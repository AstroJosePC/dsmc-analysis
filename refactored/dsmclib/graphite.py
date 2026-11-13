"""Graphite Telemetry: the shared visual system for reports and simulation cards.

Technical monochrome (charcoal and greys) plus one ink per species. Instrument Sans for text,
JetBrains Mono for numbers, Fira Math (tectonic bundle) for equations. The design notes are in
docs/reports/oh2_escape_rerun/design_philosophy.md.

    from dsmclib import graphite as G
    G.setup()                       # matplotlib rcParams + bundled fonts
    G.species_color("H2")
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

ASSETS = Path(__file__).resolve().parent / "assets" / "graphite"
FONTS = ASSETS / "fonts"

# Inks
CHARCOAL = "#1f1f1d"
GRAPHITE = "#4a4a47"
GREY = "#7a7a76"
LIGHT = "#b9b8b3"
RULE = "#e6e5e1"
PAPER = "#ffffff"
COL = {"O": "#2a78d6", "H2": "#eb6834"}  # validated categorical pair; the only colours
LABEL = {"O": "O", "H2": "H$_2$"}
BAND_ALPHA = 0.18
RAW_ALPHA = 0.45

WIDTH = 6.5  # in, full text width of a US Letter page with 1 in margins


def species_color(name: str) -> str:
    """Species ink; species without an assigned ink are drawn in charcoal."""
    return COL.get(name, CHARCOAL)


def species_label(name: str) -> str:
    """Mathtext label with subscript digits (H2 -> H$_2$, CO2 -> CO$_2$)."""
    if name in LABEL:
        return LABEL[name]
    out = ""
    for ch in name:
        out += f"$_{ch}$" if ch.isdigit() else ch
    return out


def setup():
    """Register the bundled fonts and apply the Graphite rcParams."""
    for f in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "Instrument Sans",
        "font.size": 8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Instrument Sans",
        "mathtext.it": "Instrument Sans:italic",
        "mathtext.bf": "Instrument Sans:bold",
        "mathtext.default": "it",
        "mathtext.fallback": "stixsans",
        "text.color": CHARCOAL,
        "axes.labelcolor": CHARCOAL,
        "axes.edgecolor": LIGHT,
        "axes.linewidth": 0.6,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.5,
        "xtick.color": LIGHT,
        "ytick.color": LIGHT,
        "xtick.labelcolor": GRAPHITE,
        "ytick.labelcolor": GRAPHITE,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "legend.handlelength": 1.8,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "errorbar.capsize": 0,
    })


def mono(size=6.6):
    """Font properties for numbers (JetBrains Mono)."""
    return font_manager.FontProperties(family="JetBrains Mono", size=size)


def tick_mono(*axes, size=6.4):
    """Tick labels in the monospaced face (persists through tick regeneration)."""
    for ax in axes:
        ax.tick_params(labelfontfamily="JetBrains Mono", labelsize=size)
