"""Locked visual tokens and helpers for figures that go into the NSBP/NSHP 2026 deck.

Usage
-----
    import deckstyle as ds
    ds.use("deck")                       # or "paper"
    fig, ax = ds.figure("evidence")
    ax.plot(dn, z, color=ds.ROLE["this_work"])
    ds.exobase(ax)
    ds.save(fig, "fig05a_amplitude_growth")
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

STYLE_VERSION = "1.0 (2026-11-16)"  # bump when any token changes; copy to every repo

# ---------------------------------------------------------------- palette
# Howard University OUC primary palette
MIDNIGHT = "#003A63"
CRIMSON = "#E51937"
GRANITE = "#5F6062"
CULTURED = "#F5F5F0"
# Howard University OUC secondary palette
CELTIC = "#2A6EBB"
VIOLET = "#5C3160"
TANGELO = "#EA7125"
GOLD = "#C79316"
SUNGLOW = "#FFC82E"
GREEN = "#5C8727"
SLATE = "#6B818D"

# Semantic roles: plot by role, never by raw hex.
ROLE = {
    "this_work": MIDNIGHT,        # current model result (1D radial, S6, fluid model)
    "prior_work": GRANITE,        # published comparison (e.g. 1D Cartesian, T23), dashed
    "theory": CELTIC,             # analytic reference: open markers, no line
    "exobase": GOLD,              # one dashed line at 230 km
    "forcing": TANGELO,           # lower-boundary forcing, fronts, forced footprint
    "takeaway": CRIMSON,          # at most one annotation per axes, never a data series
    "uncertainty": GRANITE,       # error bars; bands use the series color at 15% alpha
    "placeholder": SLATE,
}
REGIME = {"collisional": CELTIC, "transition": GOLD, "exosphere": VIOLET}  # fill at alpha 0.15
SPECIES = {"O": CELTIC, "Ar": VIOLET, "CO2": TANGELO, "H2": GREEN, "N2": SLATE}
# Resolution ramp, coarse to fine. CIELAB L* = 64, 55, 46, 37, 29, 23 (monotonic).
RUN_RAMP = {"S1": "#8C9EA7", "S2": "#6487AB", "S3": "#2A6EBB",
            "S4": "#1F5A93", "S5": "#124872", "S6": "#003A63"}

LINESTYLE = {"this_work": "-", "prior_work": (0, (3, 2.5)), "exobase": (0, (6, 4)),
             "input_altitude": (0, (1, 2.5))}

EXOBASE_KM = 230.0
DSMC_INPUT_KM = (100.0, 120.0, 140.0)

# Final placed sizes in inches, matched to the deck layout (13.333 x 7.5 in slide).
SIZES = {
    "evidence": (7.22, 4.17),   # left evidence field on content slides
    "full": (12.22, 4.17),      # full-width evidence (pipelines, two-panel strips)
    "narrow": (4.67, 2.60),     # word-column figure above a measure card
    "inset": (2.50, 2.00),
    "spark": (1.22, 0.47),      # measure-card sparkline: no ticks, no labels
}

# ---------------------------------------------------------------- colormaps
# Diverging, lightness-matched halves (L* 23 / 46 / 74 / 96 / 74 / 46 / 23).
# Negative = blue, positive = orange. Always use symmetric limits.
DIVERGING = LinearSegmentedColormap.from_list(
    "rarefied_div",
    ["#003A63", "#2A6EBB", "#90B9E6", "#F5F5F0", "#F1A373", "#B34F11", "#5D2909"], N=256)
# Sequential for magnitudes (density, |amplitude|, wind speed): L* 96 -> 74 -> 46 -> 23.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "rarefied_seq", ["#F5F5F0", "#90B9E6", "#2A6EBB", "#003A63"], N=256)

for _cmap in (DIVERGING, SEQUENTIAL, DIVERGING.reversed(), SEQUENTIAL.reversed()):
    try:
        mpl.colormaps.register(_cmap)
    except ValueError:
        pass  # already registered in this session

_STYLE = Path(__file__).with_name("rarefied.mplstyle")
_PAPER_SCALE = 0.6  # deck sizes x 0.6 = journal-scale text (14 -> 8.4 pt)


# ---------------------------------------------------------------- setup
def register_fonts(font_dir: str | Path | None = None) -> None:
    """Add .ttf/.otf files from font_dir (Open Sans, IBM Plex Mono) to matplotlib."""
    if font_dir is not None:
        for f in Path(font_dir).glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Open Sans" not in names:
        warnings.warn("Open Sans not found; falling back to DejaVu Sans. "
                      "Pass font_dir= or install Open Sans before making deck figures.")


def use(target: str = "deck", font_dir: str | Path | None = None) -> None:
    """Apply the locked style. target: 'deck' (default) or 'paper'."""
    register_fonts(font_dir)
    plt.style.use(_STYLE)
    if target == "paper":
        for key in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                    "ytick.labelsize", "legend.fontsize"):
            mpl.rcParams[key] = round(mpl.rcParams[key] * _PAPER_SCALE, 1)
        mpl.rcParams["lines.linewidth"] = 1.2
        mpl.rcParams["axes.linewidth"] = 0.6
    elif target != "deck":
        raise ValueError("target must be 'deck' or 'paper'")


def figure(size: str = "evidence", **kwargs):
    """Create a figure at an exact deck size (see SIZES)."""
    fig, ax = plt.subplots(figsize=SIZES[size], **kwargs)
    if size == "spark":
        for a in (ax if hasattr(ax, "__iter__") else [ax]):
            a.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------- recurring marks
def exobase(ax, z: float = EXOBASE_KM, label: bool = True, vertical_axis: str = "y") -> None:
    """Draw the Gold Foil dashed exobase line, with a small caps label at the right end."""
    kw = dict(color=ROLE["exobase"], ls=LINESTYLE["exobase"], lw=1.5, zorder=1)
    if vertical_axis == "y":
        ax.axhline(z, **kw)
        if label:
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(0.995, z, "EXOBASE", transform=tr, ha="right", va="bottom",
                    fontsize=mpl.rcParams["legend.fontsize"] - 2, color=GRANITE,
                    fontweight="semibold")
    else:
        ax.axvline(z, **kw)


def input_altitudes(ax, zs=DSMC_INPUT_KM) -> None:
    """Dotted Granite lines at the DSMC coupling altitudes."""
    for z in zs:
        ax.axhline(z, color=GRANITE, ls=LINESTYLE["input_altitude"], lw=0.9, zorder=1)


def regime_bands(ax, edges: dict[str, tuple[float, float]]) -> None:
    """Shade regimes, e.g. {'collisional': (100, 180), 'transition': (180, 280), ...}."""
    for name, (z0, z1) in edges.items():
        ax.axhspan(z0, z1, color=REGIME[name], alpha=0.15, lw=0, zorder=0)


def band(ax, x, lo, hi, color, **kwargs):
    """Uncertainty band in the series color at 15% alpha."""
    return ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, **kwargs)


def takeaway(ax, text: str, xy, xytext, **kwargs):
    """The one Crimson annotation allowed per axes."""
    if getattr(ax, "_rarefied_takeaway", False):
        raise RuntimeError("Only one Crimson takeaway annotation per axes.")
    ax._rarefied_takeaway = True
    return ax.annotate(text, xy=xy, xytext=xytext, color=CRIMSON,
                       fontsize=mpl.rcParams["legend.fontsize"], fontweight="semibold",
                       arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.2), **kwargs)


def label(ax, text: str, xy, color: str = GRANITE, **kwargs):
    """Direct label in caps, used instead of a legend wherever space allows."""
    return ax.text(*xy, text.upper(), color=color, fontweight="semibold",
                   fontsize=mpl.rcParams["legend.fontsize"] - 1, **kwargs)


# ---------------------------------------------------------------- export
def save(fig, name: str, target: str = "deck", outdir: str | Path | None = None,
         png: bool = False, source: str | None = None) -> Path:
    """Save at exact figure size (no bbox_inches='tight', which changes the size).

    source: provenance string, e.g. "data/S6/reduced.h5 @ a1b2c3d", stored in PDF metadata.
    """
    outdir = Path(outdir or f"figures/{target}")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}_{target}.pdf"
    meta = {"Creator": f"deckstyle {STYLE_VERSION}"}
    if source:
        meta["Subject"] = source
    fig.savefig(path, metadata=meta)
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=300)
    return path
