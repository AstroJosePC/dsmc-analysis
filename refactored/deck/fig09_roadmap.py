"""Slide 9 · FIG 9: roadmap from the coupled forward models to a MAVEN NGIMS inversion.

Six nodes on one track: fluid streams -> wheel-sector DSMC -> multi-species runs -> ML emulator ->
synthetic NGIMS -> invert wave properties. Status markers follow Phase2_Slide_Scaffold.md (slide 9):
filled Maximum Green = have, filled Vivid Tangelo = in progress, filled Midnight = next, open Granite
ring = to build, open Crimson ring = the goal (the figure's one Crimson accent). Status words stay
Granite because Green and Tangelo fail text contrast on Cultured. Species are named in text, not
SPECIES colours, so H2/CO2 never share an element with the Green/Tangelo status meanings
(FIGURE_STYLE.md §3). Schematic; the domain top is read from deck/data/slide08_facts.json so slides
8 and 9 quote the same number.
Build steps (progressive deck): 1 the forward models (nodes 1-3, fig09_maven_roadmap_b1), 2 the MAVEN
inversion (nodes 4-6, with the track from node 3 to 4).
"""
import json

from matplotlib.patches import Circle, FancyArrowPatch

from _deck import REFACTORED, Build, ds

FACTS = REFACTORED / "deck" / "data" / "slide08_facts.json"
Z_TOP = json.loads(FACTS.read_text())["z_top_min_km"]
NB = "\u00a0"                       # keeps a value with its unit on one line

# (title, description, status word, marker)  marker: (face, edge, radius_in, edge_lw)
NODES = [
    ("Fluid streams", "MAGNUS-P waves drive the DSMC lower boundary", "HAVE", "green"),
    ("Wheel-sector DSMC", "One-wavelength runs with exact 3D curvature", "IN PROGRESS", "tangelo"),
    ("Multi-species runs", f"CO$_2$, Ar, O, H$_2$; domain top ≥{NB}{Z_TOP:,}{NB}km set", "NEXT", "midnight"),
    ("ML emulator", "Fast surrogate trained on the DSMC ensemble", "TO BUILD", "open"),
    ("Synthetic NGIMS", "Sampled along a spacecraft pass; detrending choices tested", "TO BUILD", "open"),
    ("Invert wave properties", "Recover $\\omega$, amplitude and wavelengths; single- vs multi-species "
                               "uncertainty", "GOAL", "goal"),
]
MARK = {"green": (ds.GREEN, "none", 0.105, 0), "tangelo": (ds.TANGELO, "none", 0.105, 0),
        "midnight": (ds.MIDNIGHT, "none", 0.105, 0), "open": (ds.CULTURED, ds.GRANITE, 0.095, 1.5),
        "goal": (ds.CULTURED, ds.CRIMSON, 0.135, 2.4)}
GROUPS = [("FORWARD MODELS", 0, 2), ("MAVEN INVERSION", 3, 5)]

W_IN, H_IN = ds.SIZES["strip"]
PITCH = W_IN / len(NODES)
TEXT_W = 1.84 * 72                 # column text width in points
Y_GROUP, Y_BRACKET = 2.98, 2.80    # group label baseline, bracket line
Y_TRACK = 2.40
Y_STATUS = 2.03                    # status word baseline
Y_TITLE, TITLE_LEAD = 1.70, 0.245  # first title baseline, two-line block
Y_DESC, DESC_LEAD = 1.135, 0.235    # first description baseline


def wrap(ax, s, size, weight, max_w, max_lines):
    """Greedy word wrap on measured widths (points); raises if more than max_lines are needed."""
    r = ax.figure.canvas.get_renderer()

    def width(t):
        a = ax.text(0, 0, t, fontsize=size, fontweight=weight)
        w = a.get_window_extent(r).width * 72 / ax.figure.dpi
        a.remove()
        return w

    lines, cur = [], ""
    for word in s.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and width(trial) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    if len(lines) > max_lines:
        raise ValueError(f"FIG 9: '{s}' needs {len(lines)} lines (max {max_lines})")
    return lines


def main():
    fig, ax = ds.figure("strip")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    xc = [PITCH * (k + 0.5) for k in range(len(NODES))]
    build = Build(fig)
    step = lambda node: 1 if node <= GROUPS[0][2] else 2       # build step of a node

    # phase brackets
    for label, a, b in GROUPS:
        build.at(step(a))
        x0, x1 = xc[a] - 0.42 * PITCH, xc[b] + 0.42 * PITCH    # 0.16-pitch gap between groups
        ax.plot([x0, x0, x1, x1], [Y_BRACKET - 0.07, Y_BRACKET, Y_BRACKET, Y_BRACKET - 0.07],
                color=ds.GRANITE, lw=0.8, alpha=0.6, solid_joinstyle="miter")
        ax.text(0.5 * (x0 + x1), Y_GROUP, label, ha="center", va="baseline", fontsize=13,
                fontweight="semibold", color=ds.GRANITE)

    # track: built or underway (1-3) solid Midnight, planned (3-6) light Granite
    for k in range(len(NODES) - 1):
        build.at(step(k + 1))
        done = k < 2
        color, alpha, lw = (ds.MIDNIGHT, 1.0, 2.0) if done else (ds.GRANITE, 0.45, 1.5)
        ax.plot([xc[k], xc[k + 1]], [Y_TRACK] * 2, color=color, alpha=alpha, lw=lw,
                solid_capstyle="butt", zorder=1)
        xm = 0.5 * (xc[k] + xc[k + 1])
        ax.add_patch(FancyArrowPatch((xm - 0.09, Y_TRACK), (xm + 0.09, Y_TRACK), arrowstyle="-|>",
                                     mutation_scale=13, color=color, alpha=alpha, lw=0, zorder=2))

    for k, ((title, desc, status, mark), x) in enumerate(zip(NODES, xc)):
        build.at(step(k))
        fc, ec, rad, elw = MARK[mark]
        ax.add_patch(Circle((x, Y_TRACK), rad, fc=fc, ec=ec, lw=elw, zorder=3))
        ax.text(x, Y_STATUS, status, ha="center", va="baseline", fontsize=13, fontweight="semibold",
                color=ds.GRANITE)
        t_lines = wrap(ax, title, 16, "semibold", TEXT_W, 2)
        y0 = Y_TITLE if len(t_lines) == 2 else Y_TITLE - TITLE_LEAD / 2
        for i, line in enumerate(t_lines):
            ax.text(x, y0 - i * TITLE_LEAD, line, ha="center", va="baseline", fontsize=16,
                    fontweight="semibold", color=ds.MIDNIGHT)
        for i, line in enumerate(wrap(ax, desc, 14, "normal", TEXT_W, 5)):
            ax.text(x, Y_DESC - i * DESC_LEAD, line, ha="center", va="baseline", fontsize=14,
                    color=ds.GRANITE)

    build.save("fig09_maven_roadmap", "schematic (no data)", "deck/data/slide08_facts.json: z_top_min_km")


if __name__ == "__main__":
    main()
