"""Slide 4 · FIG 4: the DSMC loop in HARRAH, Move -> Sort -> Collide -> Boundaries -> Sample.

One patch of gas (the same seeded dots) is carried through the five steps, so each pictogram
changes only what that step does. Schematic; no simulation data.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _deck import ds, save

STEPS = [("Move", ["free flight for $\\Delta t$"]),
         ("Sort", ["index into cells"]),
         ("Collide", ["random pairs", "in each cell"]),
         ("Boundaries", ["wave input at base", "escape at top"]),
         ("Sample", ["average to $n$, $T$, $u$"])]
N_DOTS, SEED, NCELL = 44, 7, 3
BOX = 1.45                     # pictogram side, inches
Y_BOX = 2.03                   # bottom of the pictograms
X_FIRST, PITCH = 1.42, 2.345   # centre of the first pictogram, spacing
Y_LABEL, Y_TEXT, Y_RETURN = 3.84, 1.43, 0.50   # Y_TEXT: baseline of the first caption line
LINE_GAP = 0.27


def dots():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0.06, 0.94, N_DOTS)
    u = rng.uniform(0, 1, N_DOTS)
    y = 0.05 + 0.90 * (-np.log(1 - u * (1 - np.exp(-2.0))) / 2.0)     # denser toward the base
    ang = rng.uniform(0, 2 * np.pi, N_DOTS)
    speed = rng.uniform(0.06, 0.13, N_DOTS)
    return x, y, speed * np.cos(ang), speed * np.sin(ang)


def main():
    fig, ax = ds.figure("full")
    fig.set_layout_engine("none")
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    w, h = ds.SIZES["full"]
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)

    x, y, vx, vy = dots()
    cell = np.minimum((y * NCELL).astype(int), NCELL - 1) * NCELL + np.minimum((x * NCELL).astype(int), NCELL - 1)
    counts = np.bincount(cell, minlength=NCELL * NCELL)
    centres = X_FIRST + PITCH * np.arange(len(STEPS))

    def grid(x0, alpha=1.0):
        for k in range(1, NCELL):
            ax.plot([x0 + k * BOX / NCELL] * 2, [Y_BOX, Y_BOX + BOX], color=ds.GRANITE, lw=0.5, alpha=alpha)
            ax.plot([x0, x0 + BOX], [Y_BOX + k * BOX / NCELL] * 2, color=ds.GRANITE, lw=0.5, alpha=alpha)

    for i, ((name, text), xc) in enumerate(zip(STEPS, centres)):
        x0 = xc - BOX / 2
        px, py = x0 + x * BOX, Y_BOX + y * BOX
        ax.text(xc, Y_LABEL, name.upper(), ha="center", va="center", fontsize=14,
                fontweight="semibold", color=ds.MIDNIGHT)
        for row, line in enumerate(text):
            ax.text(xc, Y_TEXT - row * LINE_GAP, line, ha="center", va="baseline", fontsize=14)
        frame = dict(fill=False, ec=ds.GRANITE, lw=0.8, zorder=3)
        dot_alpha = 1.0

        if name == "Move":
            for a, b, c, d in zip(px, py, vx * BOX, vy * BOX):
                ax.plot([a - c, a], [b - d, b], color=ds.MIDNIGHT, lw=0.9, alpha=0.4,
                        solid_capstyle="round", zorder=1)
        elif name == "Sort":
            grid(x0)
        elif name == "Collide":
            grid(x0, alpha=0.45)
            k = int(np.argmax(counts[:NCELL]))                       # busiest base cell
            ax.add_patch(Rectangle((x0 + (k % NCELL) * BOX / NCELL, Y_BOX + (k // NCELL) * BOX / NCELL),
                                   BOX / NCELL, BOX / NCELL, fill=False, ec=ds.MIDNIGHT, lw=1.6, zorder=4))
            members = np.where(cell == k)[0]
            d2 = (px[members][:, None] - px[members]) ** 2 + (py[members][:, None] - py[members]) ** 2
            p, q = members[np.array(np.unravel_index(d2.argmax(), d2.shape))]      # a well-separated pair
            ax.plot(px[[p, q]], py[[p, q]], color=ds.MIDNIGHT, lw=0.9, zorder=4)
            ax.plot(px[[p, q]], py[[p, q]], "o", ms=8.5, mfc="none", mec=ds.MIDNIGHT, mew=1.1, zorder=5)
        elif name == "Boundaries":
            ax.plot([x0, x0 + BOX], [Y_BOX, Y_BOX], color=ds.ROLE["forcing"], lw=2.4, zorder=4,
                    solid_capstyle="butt")
            for xa in x0 + BOX * np.array([0.2, 0.4, 0.6, 0.8]):
                ax.annotate("", xy=(xa, Y_BOX + 0.10), xytext=(xa, Y_BOX - 0.24),
                            arrowprops=dict(arrowstyle="-|>", color=ds.ROLE["forcing"], lw=1.6,
                                            mutation_scale=10, shrinkA=0, shrinkB=0), zorder=5)
            top = int(np.argmax(y))
            ax.annotate("", xy=(px[top] + 0.04, Y_BOX + BOX + 0.15), xytext=(px[top], py[top]),
                        arrowprops=dict(arrowstyle="-|>", color=ds.GRANITE, lw=1.0, mutation_scale=9,
                                        shrinkA=3, shrinkB=0), zorder=5)
        elif name == "Sample":
            for k, c in enumerate(counts):
                ax.add_patch(Rectangle((x0 + (k % NCELL) * BOX / NCELL, Y_BOX + (k // NCELL) * BOX / NCELL),
                                       BOX / NCELL, BOX / NCELL, lw=0, zorder=0,
                                       fc=ds.SEQUENTIAL(0.12 + 0.78 * c / counts.max())))
            grid(x0, alpha=0.45)
            dot_alpha = 0.0
            for k, c in enumerate(counts):
                dark = c / counts.max() > 0.55
                ax.text(x0 + (k % NCELL + 0.5) * BOX / NCELL, Y_BOX + (k // NCELL + 0.5) * BOX / NCELL,
                        str(c), ha="center", va="center", fontsize=12,
                        color=ds.CULTURED if dark else ds.MIDNIGHT)

        ax.add_patch(Rectangle((x0, Y_BOX), BOX, BOX, **frame))
        if dot_alpha:
            ax.plot(px, py, "o", ms=3.4, mew=0, color=ds.MIDNIGHT, alpha=dot_alpha, zorder=2)
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(xc + PITCH - BOX / 2 - 0.16, Y_BOX + BOX / 2),
                        xytext=(xc + BOX / 2 + 0.16, Y_BOX + BOX / 2),
                        arrowprops=dict(arrowstyle="-|>", color=ds.MIDNIGHT, lw=1.6, mutation_scale=13,
                                        shrinkA=0, shrinkB=0))

    # return path: Sample -> Move, once per time step
    xa, xb, r, y_from = centres[-1], centres[0], 0.16, 0.92
    t = np.linspace(0, np.pi / 2, 20)
    path_x = np.r_[xa, xa - r + r * np.cos(-t), xb + r + r * np.cos(-np.pi / 2 - t), xb]
    path_y = np.r_[y_from, Y_RETURN + r + r * np.sin(-t), Y_RETURN + r + r * np.sin(-np.pi / 2 - t), y_from - 0.12]
    ax.plot(path_x, path_y, color=ds.MIDNIGHT, lw=1.6, solid_capstyle="butt", solid_joinstyle="round")
    ax.annotate("", xy=(xb, y_from + 0.02), xytext=(xb, y_from - 0.14),
                arrowprops=dict(arrowstyle="-|>", color=ds.MIDNIGHT, lw=1.6, mutation_scale=13,
                                shrinkA=0, shrinkB=0))
    ax.text(0.5 * (xa + xb), Y_RETURN + 0.11, r"REPEAT EVERY TIME STEP $\Delta t$", ha="center", va="bottom",
            fontsize=12, fontweight="semibold")

    save(fig, "fig04_harrah_loop", "schematic (no data)")
    plt.close(fig)


if __name__ == "__main__":
    main()
