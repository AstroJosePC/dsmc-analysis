"""Slide 4 alternatives (4A, 4B, 4C): what DSMC does physically.

Numbers are read, not typed: particle weight W and time step from the S6 solver parameter file
(Info/Simulation_Parameters.ascii: W = 3.58e23 O atoms per computational particle, dt = 0.131 s),
grid from the run parameters (75 shells x 1,024 columns over 360 deg, 100-795 km), particles per
cell from the pre-forcing transport cache (about 13,000 at the exobase, 1.1e9 in the domain).

  fig04a_timestep_plate  (evidence) schematic: a patch of curved cells, curvature exaggerated, with
                         the three operations of one time step numbered on it
  fig04b_powers_of_ten   (slot)     the S6 ring at true scale -> one exobase cell -> one particle
  fig04c_split_step      (evidence) the same six particles as a real hard-sphere gas (collide on
                         contact, at any instant) and as DSMC (straight flight for dt, then random
                         partners within a cell). Both are small seeded simulations made here.

    conda run -n dsmc python deck/fig04_alternatives.py [a] [b] [c]
"""
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from _deck import R_MARS_KM, ds, r_label, save
from dsmclib import registry
from dsmclib.paths import output_dir
from dsmclib.run_data import DSMCRun
from slots import SLOTS

TRANSPORT = output_dir("s1_s6_azimuthal", "analysis") / "preforcing_transport_S6_azimean_preforcing.npz"


# ================================================================= S6 numbers
def s6_numbers():
    run = DSMCRun(registry.run("S6").path, alias="S6")
    p = run.load_parameters()
    info = (run.data_dir / "Info" / "Simulation_Parameters.ascii").read_text()
    W = float(re.search(r"Particle Weight:\s+([\d.E+-]+)", info).group(1))
    dt = float(re.search(r"Time Step:\s+([\d.E+-]+)", info).group(1))
    t = np.load(TRANSPORT)
    z = t["altitude_km"]
    i_exo = int(np.argmin(np.abs(z - ds.EXOBASE_KM)))
    r_e = p.radial_edges / 1e3
    out = dict(W=W, dt=dt, n_r=p.n_rad_cell, n_phi=p.n_azi_cell,
               z_top=r_e[-1] - p.mars_radius, z_bot=r_e[0] - p.mars_radius,
               n_part=float((t["mean_particles_per_cell"] * p.n_azi_cell).sum()),
               ppc_exo=float(t["mean_particles_per_cell"][i_exo]),
               dr_exo=float(r_e[i_exo + 1] - r_e[i_exo]),
               dphi_exo_km=float(np.deg2rad(p.azi_edges[1] - p.azi_edges[0]) * (p.mars_radius + z[i_exo])),
               tau_100=float(t["mean_free_path_m"][0] / t["thermal_speed_ms"][0]),
               patch_cells=88, source=run.data_dir / "Info" / "Simulation_Parameters.ascii")
    return out


def sci(v, digits=1):
    """3.58e23 -> '3.6 × 10²³' (mathtext)."""
    e = int(np.floor(np.log10(v)))
    return rf"{v / 10 ** e:.{digits}f} × $10^{{{e}}}$"


def canvas(stem=None, size="evidence"):
    if stem in SLOTS:
        fig = plt.figure(figsize=SLOTS[stem])
        w, h = SLOTS[stem]
    else:
        fig, _ = ds.figure(size)
        fig.clf()
        w, h = ds.SIZES[size]
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    return fig, ax


def arrow(ax, p0, p1, color, lw=1.4, ms=10, z=6, style="-|>"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                mutation_scale=ms, shrinkA=0, shrinkB=0), zorder=z)


def badge(ax, x, y, n, r=0.13):
    ax.add_patch(Circle((x, y), r, fc=ds.MIDNIGHT, ec="none", zorder=9))
    ax.text(x, y - 0.005, str(n), ha="center", va="center", fontsize=12, fontweight="semibold",
            color=ds.CULTURED, zorder=10)


# ================================================================= 4A: the plate
def draw_plate(ax, num, short_labels=False, badge_r=0.13):
    """The 4A schematic in its own inch coordinates (7.22 x 4.17 box); 4D embeds it scaled.

    Returns the cell-centre helper and the background particles, so a caller can point at one.
    """
    rng = np.random.default_rng(404)
    cx, cy, R0, DR, NR = 3.55, -5.42, 6.40, 0.58, 4
    NC, DA = 6, 0.105                                   # columns, angular width (rad): curvature exaggerated
    a_edges = (np.arange(NC + 1) - NC / 2) * DA
    r_edges = R0 + DR * np.arange(NR + 1)

    def xy(r, a):
        return cx + r * np.sin(a), cy + r * np.cos(a)

    def cell_poly(i, j, **kw):
        aa = np.linspace(a_edges[j], a_edges[j + 1], 30)
        x0, y0 = xy(r_edges[i], aa)
        x1, y1 = xy(r_edges[i + 1], aa[::-1])
        return Polygon(np.column_stack([np.r_[x0, x1], np.r_[y0, y1]]), closed=True, **kw)

    # grid
    for i, r in enumerate(r_edges):
        aa = np.linspace(a_edges[0], a_edges[-1], 200)
        ax.plot(*xy(r, aa), color=ds.GRANITE, lw=1.6 if i == 0 else 0.7, zorder=2)
    for a in a_edges:
        ax.plot(*xy(np.array([r_edges[0], r_edges[-1]]), a), color=ds.GRANITE, lw=0.7, zorder=2)

    # background computational particles
    pts = []
    for i in range(NR):
        for j in range(NC):
            n = 7 - i
            rr = rng.uniform(r_edges[i] + 0.05, r_edges[i + 1] - 0.05, n)
            aa = rng.uniform(a_edges[j] + 0.012, a_edges[j + 1] - 0.012, n)
            pts.append((i, j, *xy(rr, aa)))
    skip = {(1, 1), (0, 4), (3, 2), (2, 1), (1, 2)}     # cells used by the operations below
    for i, j, x, y in pts:
        if (i, j) not in skip:
            ax.plot(x, y, "o", ms=3.2, mew=0, color=ds.MIDNIGHT, alpha=0.55, zorder=3)

    # 1 MOVE: straight flights for dt; a flight that crosses a curved face changes cell
    def cell_of(x, y):
        r, a = np.hypot(x - cx, y - cy), np.arctan2(x - cx, y - cy)
        return int(np.searchsorted(r_edges, r) - 1), int(np.searchsorted(a_edges, a) - 1)

    starts = [(r_edges[1] + 0.16, a_edges[1] + 0.030), (r_edges[1] + 0.40, a_edges[1] + 0.078),
              (r_edges[1] + 0.20, a_edges[1] + 0.088)]
    moves = [(0.22, 0.62), (0.40, -0.30), (-0.34, 0.06)]   # displacement in inches (x, y)
    for (r, a), (dx, dy) in zip(starts, moves):
        x, y = xy(r, a)
        new = cell_of(x + dx, y + dy)
        if new != cell_of(x, y):
            ax.add_patch(cell_poly(*new, fc=ds.MIDNIGHT, alpha=0.08, lw=0, zorder=1))
        ax.plot(x, y, "o", ms=6, mfc="none", mec=ds.GRANITE, mew=1.0, zorder=5)
        arrow(ax, (x, y), (x + dx, y + dy), ds.MIDNIGHT, lw=1.3, ms=9)
        ax.plot(x + dx, y + dy, "o", ms=4.2, mew=0, color=ds.MIDNIGHT, zorder=7)
    bx, by = xy(r_edges[1] + 0.30, a_edges[1] - 0.005)
    badge(ax, bx - 0.20, by, 1, badge_r)

    # 2 COLLIDE: a random pair in one cell; velocities turn, momentum and energy kept
    i, j = 0, 4
    ax.add_patch(cell_poly(i, j, fc="none", ec=ds.MIDNIGHT, lw=1.8, zorder=6))
    pa = xy(r_edges[0] + 0.22, a_edges[j] + 0.03)
    pb = xy(r_edges[0] + 0.44, a_edges[j] + 0.075)
    for i2, j2, x, y in pts:
        if (i2, j2) == (i, j):
            ax.plot(x, y, "o", ms=3.2, mew=0, color=ds.MIDNIGHT, alpha=0.30, zorder=3)
    va, vb = np.array([0.40, 0.08]), np.array([-0.30, -0.14])
    vcm, g = 0.5 * (va + vb), va - vb
    th = np.deg2rad(115)
    g2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ g
    va2, vb2 = vcm + g2 / 2, vcm - g2 / 2
    for p, v, v2 in ((pa, va, va2), (pb, vb, vb2)):
        arrow(ax, p, (p[0] + v[0], p[1] + v[1]), ds.GRANITE, lw=1.0, ms=8, style="-|>")
        arrow(ax, p, (p[0] + v2[0], p[1] + v2[1]), ds.MIDNIGHT, lw=1.5, ms=10)
        ax.plot(*p, "o", ms=5, mew=0, color=ds.MIDNIGHT, zorder=8)
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=ds.MIDNIGHT, lw=0.8, ls=(0, (1, 2)), zorder=7)
    bx, by = xy(r_edges[1] - 0.05, a_edges[j + 1] + 0.03)
    badge(ax, bx + 0.18, by + 0.02, 2, badge_r)

    # 3 BOUNDARIES: wave-modulated flux in at the base, escape out of the top
    for k, a in enumerate(np.linspace(a_edges[0] + 0.03, a_edges[-1] - 0.03, 11)):
        amp = 0.30 + 0.16 * np.sin(2 * np.pi * k / 10 * 1.0)
        (x0, x1), (y0, y1) = xy(np.array([R0 - amp - 0.06, R0 - 0.04]), a)
        arrow(ax, (x0, y0), (x1, y1), ds.ROLE["forcing"], lw=1.5, ms=8)
    bx, by = xy(R0 - 0.26, a_edges[-1] + 0.03)
    badge(ax, bx + 0.10, by, 3, badge_r)
    # break and the domain top
    r_top = r_edges[-1] + 0.42
    for dd in (-0.04, 0.04):
        x, y = xy(r_edges[-1] + 0.20 + dd, a_edges[0] - 0.03)
    ax.plot(*xy(r_top, np.linspace(a_edges[0], a_edges[-1], 200)), color=ds.GRANITE, lw=0.8,
            ls=(0, (4, 3)), zorder=2)
    for a in (a_edges[0], a_edges[-1]):
        (x0, x1), (y0, y1) = xy(np.array([r_edges[-1] + 0.10, r_top - 0.10]), a)
        ax.plot([x0, x1], [y0, y1], color=ds.GRANITE, lw=0.7, ls=(0, (1, 2)), zorder=2)
    pe = xy(r_top - 0.16, a_edges[2] + 0.04)
    ax.plot(*pe, "o", ms=4.2, mew=0, color=ds.MIDNIGHT, zorder=7)
    arrow(ax, pe, (pe[0] + 0.08, pe[1] + 0.36), ds.MIDNIGHT, lw=1.3, ms=9)
    badge(ax, pe[0] + 0.36, pe[1] + 0.20, 3, badge_r)
    xt, yt = xy(r_top, a_edges[-2])
    top = r_label(num["z_top"]) if short_labels else f"top boundary, {r_label(num['z_top'])}"
    ax.text(xt + 0.05, yt + 0.10, top, ha="left", va="bottom", fontsize=12)
    xb_, yb_ = xy(R0, a_edges[0])
    bot = r_label(num["z_bot"]) if short_labels else f"lower boundary\n{r_label(num['z_bot'])}"
    ax.text(xb_ - 0.10, yb_ + 0.02, bot, ha="right", va="center", fontsize=12, linespacing=1.3)

    # dr, dphi on one cell
    i, j = 2, 4
    rr = np.array([r_edges[i], r_edges[i + 1]])
    (xa, xb), (ya, yb) = xy(rr, a_edges[j + 1] - 0.014)
    arrow(ax, (xa, ya), (xb, yb), ds.GRANITE, lw=0.9, ms=7, style="<|-|>")
    xm, ym = xy(rr.mean(), a_edges[j + 1] - 0.014)
    ax.text(xm - 0.07, ym, r"$\Delta r$", ha="right", va="center", fontsize=15, zorder=8,
            bbox=dict(fc=ds.CULTURED, ec="none", pad=1.5))
    aa = np.linspace(a_edges[j] + 0.006, a_edges[j + 1] - 0.006, 40)
    xs, ys = xy(r_edges[i + 1] + 0.10, aa)
    ax.plot(xs[2:-2], ys[2:-2], color=ds.GRANITE, lw=0.9, zorder=6)
    arrow(ax, (xs[3], ys[3]), (xs[0], ys[0]), ds.GRANITE, lw=0.9, ms=7)
    arrow(ax, (xs[-4], ys[-4]), (xs[-1], ys[-1]), ds.GRANITE, lw=0.9, ms=7)
    xm, ym = xy(r_edges[i + 1] + 0.25, aa.mean())
    ax.text(xm, ym, r"$\Delta\phi$", ha="center", va="center", fontsize=15, zorder=8,
            bbox=dict(fc=ds.CULTURED, ec="none", pad=1.5))
    return dict(xy=xy, r_edges=r_edges, a_edges=a_edges, r_top=r_top, pts=pts)


def fig_a(num):
    fig, ax = canvas(size="evidence")
    draw_plate(ax, num)
    ax.text(0.08, 4.05, "schematic · curvature exaggerated", ha="left", va="center", fontsize=12,
            style="italic")
    save(fig, "fig04a_timestep_plate", num["source"], "schematic; W, dt, grid from S6")
    plt.close(fig)


# ================================================================= 4B: powers of ten
def fig_b(num):
    stem = "fig04b_powers_of_ten"
    fig, ax = canvas(stem)
    rng = np.random.default_rng(10)
    w, h = SLOTS[stem]
    cap_y = 0.42

    # 1 the ring, true scale
    cx, cy, Rm = 1.75, 1.88, 0.86
    k = Rm / R_MARS_KM
    r0, r1 = (R_MARS_KM + num["z_bot"]) * k, (R_MARS_KM + num["z_top"]) * k
    ax.add_patch(Circle((cx, cy), Rm, fc=ds.GRANITE, alpha=0.12, lw=0))
    ax.add_patch(Wedge((cx, cy), r1, 0, 360, width=r1 - r0, fc=ds.MIDNIGHT, alpha=0.18, lw=0))
    for r in (r0, r1):
        ax.add_patch(Circle((cx, cy), r, fc="none", ec=ds.MIDNIGHT, lw=0.6))
    half = np.deg2rad(num["patch_cells"] * 360 / num["n_phi"] / 2)
    aa = np.linspace(-np.pi / 2 - half, -np.pi / 2 + half, 80)
    ax.plot(cx + (r0 - 0.035) * np.cos(aa), cy + (r0 - 0.035) * np.sin(aa), color=ds.ROLE["forcing"], lw=3.0,
            solid_capstyle="butt")
    ax.text(cx, cy, "MARS", ha="center", va="center", fontsize=12, fontweight="semibold")
    # the zoomed spot: exobase, just past the patch edge
    a_s = -np.pi / 2 + half + 0.06
    r_s = (R_MARS_KM + ds.EXOBASE_KM) * k
    sx, sy = cx + r_s * np.cos(a_s), cy + r_s * np.sin(a_s)
    box1 = 0.11
    ax.add_patch(Rectangle((sx - box1 / 2, sy - box1 / 2), box1, box1, fc="none", ec=ds.MIDNIGHT, lw=1.2, zorder=5))
    ax.text(cx, cap_y + 0.02, "THE S6 DOMAIN, TRUE SCALE", ha="center", va="bottom", fontsize=12,
            fontweight="semibold")
    ax.text(cx, cap_y - 0.04, f"{num['n_r']} shells × {num['n_phi']:,} columns · "
            f"{num['z_bot']:.0f}–{num['z_top']:.0f} km", ha="center", va="top", fontsize=13)

    # 2 one cell at the exobase, true aspect, its neighbours
    cw = 0.92
    ch = cw * num["dr_exo"] / num["dphi_exo_km"]
    gx, gy = 4.95, 1.78 - 1.5 * ch
    for i in range(4):
        ax.plot([gx, gx + 3 * cw], [gy + i * ch] * 2, color=ds.GRANITE, lw=0.7)
        ax.plot([gx + i * cw] * 2, [gy, gy + 3 * ch], color=ds.GRANITE, lw=0.7)
    n = int(round(num["ppc_exo"]))
    px = rng.uniform(gx + cw + 0.01, gx + 2 * cw - 0.01, n)
    py = rng.uniform(gy + ch + 0.01, gy + 2 * ch - 0.01, n)
    ax.scatter(px, py, s=0.12, lw=0, color=ds.MIDNIGHT, alpha=0.65, rasterized=True, zorder=3)
    ax.add_patch(Rectangle((gx + cw, gy + ch), cw, ch, fc="none", ec=ds.MIDNIGHT, lw=1.4, zorder=4))
    for xx, yy in ((gx, gy + 3 * ch), (gx, gy)):
        ax.plot([sx + box1 / 2, xx], [sy + (box1 / 2 if yy > gy else -box1 / 2), yy], color=ds.GRANITE,
                lw=0.5, alpha=0.6)
    ax.annotate("", xy=(gx + 3 * cw + 0.10, gy + 2 * ch), xytext=(gx + 3 * cw + 0.10, gy + ch),
                arrowprops=dict(arrowstyle="<|-|>", color=ds.GRANITE, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(gx + 3 * cw + 0.17, gy + 1.5 * ch, rf"$\Delta r$ = {num['dr_exo']:.0f} km", ha="left",
            va="center", fontsize=13)
    ax.annotate("", xy=(gx + 2 * cw, gy + 3 * ch + 0.10), xytext=(gx + cw, gy + 3 * ch + 0.10),
                arrowprops=dict(arrowstyle="<|-|>", color=ds.GRANITE, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.text(gx + 1.5 * cw, gy + 3 * ch + 0.16, rf"$r\Delta\phi$ = {num['dphi_exo_km']:.0f} km", ha="center",
            va="bottom", fontsize=13)
    mid = gx + 1.5 * cw
    ax.text(mid, cap_y + 0.02, "ONE CELL AT THE EXOBASE", ha="center", va="bottom", fontsize=12,
            fontweight="semibold")
    ax.text(mid, cap_y - 0.04, f"≈ {round(n, -3):,.0f} computational particles", ha="center", va="top",
            fontsize=13)

    # 3 one computational particle
    j = int(np.argmin((px - (gx + 1.72 * cw)) ** 2 + (py - (gy + 1.62 * ch)) ** 2))
    p = (px[j], py[j])
    ax.plot(*p, "o", ms=5.5, mfc="none", mec=ds.MIDNIGHT, mew=1.3, zorder=6)
    lx, ly, lr = 10.15, 1.86, 0.98
    for sgn in (1, -1):
        ax.plot([p[0], lx - 0.18], [p[1], ly + sgn * lr * 0.80], color=ds.GRANITE, lw=0.5, alpha=0.6)
    ax.add_patch(Circle((lx, ly), lr, fc=ds.CULTURED, ec=ds.MIDNIGHT, lw=1.2, zorder=4))
    rr = lr * 0.86 * np.sqrt(rng.uniform(0, 1, 2400))
    th = rng.uniform(0, 2 * np.pi, rr.size)
    ax.scatter(lx + rr * np.cos(th), ly + rr * np.sin(th), s=0.5, lw=0, color=ds.GRANITE, alpha=0.75, zorder=5)
    arrow(ax, (lx - 0.30, ly - 0.18), (lx + 0.55, ly + 0.30), ds.MIDNIGHT, lw=2.0, ms=13, z=7)
    ax.text(lx + 0.60, ly + 0.34, r"$\mathbf{v}$", ha="left", va="bottom", fontsize=17, color=ds.MIDNIGHT, zorder=8)
    ax.text(lx, cap_y + 0.02, "ONE COMPUTATIONAL PARTICLE", ha="center", va="bottom", fontsize=12,
            fontweight="semibold")
    ax.text(lx, cap_y - 0.04, f"= {sci(num['W'])} O atoms, one velocity", ha="center", va="top", fontsize=13)
    save(fig, stem, num["source"], TRANSPORT, "true-scale ring; exobase cell; W")
    plt.close(fig)


# ================================================================= 4C: real gas vs DSMC
N_P, BOX = 7, (3.20, 2.45)


def initial(rng):
    pos = np.column_stack([rng.uniform(0.35, BOX[0] - 0.35, N_P), rng.uniform(0.35, BOX[1] - 0.35, N_P)])
    ang = rng.uniform(0, 2 * np.pi, N_P)
    spd = rng.uniform(0.55, 0.95, N_P)
    return pos, np.column_stack([spd * np.cos(ang), spd * np.sin(ang)])


def reflect(p, v, pad=0.06):
    for d in (0, 1):
        lo, hi = pad, BOX[d] - pad
        m = p[:, d] < lo
        p[m, d], v[m, d] = 2 * lo - p[m, d], -v[m, d]
        m = p[:, d] > hi
        p[m, d], v[m, d] = 2 * hi - p[m, d], -v[m, d]


def real_gas(pos, vel, T, diam=0.30, dt=0.002):
    """Hard spheres: free flight, elastic collision on contact. Returns tracks and events."""
    p, v = pos.copy(), vel.copy()
    tracks, events, last = [[q.copy()] for q in p], [], {}
    for step in range(int(T / dt)):
        p += v * dt
        before = v.copy()
        reflect(p, v)
        for i in range(N_P):
            if np.any(before[i] != v[i]):
                tracks[i].append(p[i].copy())
        for i in range(N_P):
            for j in range(i + 1, N_P):
                d = p[j] - p[i]
                dist = np.hypot(*d)
                if dist < diam and np.dot(v[j] - v[i], d) < 0 and step - last.get((i, j), -99) > 5:
                    nrm = d / dist
                    dv = np.dot(v[j] - v[i], nrm) * nrm
                    v[i] += dv
                    v[j] -= dv
                    last[(i, j)] = step
                    tracks[i].append(p[i].copy())
                    tracks[j].append(p[j].copy())
                    events.append((step * dt, i, j, 0.5 * (p[i] + p[j])))
    for i in range(N_P):
        tracks[i].append(p[i].copy())
    return [np.array(t) for t in tracks], events


def dsmc(pos, vel, T, n_steps, rng, p_coll=0.33):
    """Move for dt, reflect at walls, then at most one random pair per cell may collide."""
    p, v = pos.copy(), vel.copy()
    dt = T / n_steps
    tracks, events = [[q.copy()] for q in p], []
    for s in range(n_steps):
        p += v * dt
        reflect(p, v)
        for i in range(N_P):
            tracks[i].append(p[i].copy())
        cell = (p[:, 0] > BOX[0] / 2).astype(int) + 2 * (p[:, 1] > BOX[1] / 2).astype(int)
        for c in range(4):
            members = np.where(cell == c)[0]
            if members.size >= 2 and rng.uniform() < p_coll:
                i, j = rng.choice(members, 2, replace=False)
                vcm, g = 0.5 * (v[i] + v[j]), np.hypot(*(v[i] - v[j]))
                th = rng.uniform(0, 2 * np.pi)
                v[i], v[j] = vcm + 0.5 * g * np.array([np.cos(th), np.sin(th)]), vcm - 0.5 * g * np.array([np.cos(th), np.sin(th)])
                events.append(((s + 1) * dt, i, j, p[i].copy(), p[j].copy()))
    return [np.array(t) for t in tracks], events


def fig_c(num):
    fig, ax = canvas(size="evidence")
    T, n_steps = 3.2, 10
    rng = np.random.default_rng(15)
    pos, vel = initial(rng)
    real_tr, real_ev = real_gas(pos, vel, T)
    ds_tr, ds_ev = dsmc(pos, vel, T, n_steps, np.random.default_rng(11), p_coll=0.18)
    print(f"4C: real gas {len(real_ev)} collisions, DSMC {len(ds_ev)} collisions in {n_steps} steps")
    y_box, strip_y = 1.05, 0.52
    for k, (x_off, title) in enumerate(((0.12, "REAL GAS"), (3.90, "DSMC"))):
        ax.add_patch(Rectangle((x_off, y_box), *BOX, fc="none", ec=ds.GRANITE, lw=0.8, zorder=1))
        ax.text(x_off, y_box + BOX[1] + 0.10, title, ha="left", va="bottom", fontsize=13, fontweight="semibold",
                color=ds.MIDNIGHT)
        tracks = real_tr if k == 0 else ds_tr
        for t in tracks:
            ax.plot(x_off + t[:, 0], y_box + t[:, 1], color=ds.MIDNIGHT, lw=1.2, zorder=3,
                    solid_joinstyle="round")
            ax.plot(x_off + t[0, 0], y_box + t[0, 1], "o", ms=5, mfc=ds.CULTURED, mec=ds.MIDNIGHT, mew=1.0, zorder=5)
            ax.plot(x_off + t[-1, 0], y_box + t[-1, 1], "o", ms=5, mew=0, color=ds.MIDNIGHT, zorder=5)
            if k == 1:
                ax.plot(x_off + t[1:-1, 0], y_box + t[1:-1, 1], "o", ms=2.2, mew=0, color=ds.MIDNIGHT, zorder=4)
        if k == 0:
            for _, i, j, c in real_ev:
                ax.add_patch(Circle((x_off + c[0], y_box + c[1]), 0.09, fc="none", ec=ds.GRANITE, lw=0.9, zorder=6))
        else:
            ax.plot([x_off + BOX[0] / 2] * 2, [y_box, y_box + BOX[1]], color=ds.GRANITE, lw=0.6, ls=(0, (3, 3)), zorder=1)
            ax.plot([x_off, x_off + BOX[0]], [y_box + BOX[1] / 2] * 2, color=ds.GRANITE, lw=0.6, ls=(0, (3, 3)), zorder=1)
            for _, i, j, pi, pj in ds_ev:
                ax.plot(x_off + np.array([pi[0], pj[0]]), y_box + np.array([pi[1], pj[1]]), color=ds.GRANITE,
                        lw=1.0, ls=(0, (1, 1.8)), zorder=6)
        # time strip: when collisions happen
        x0, x1 = x_off, x_off + BOX[0]
        ax.plot([x0, x1], [strip_y, strip_y], color=ds.GRANITE, lw=0.8)
        if k == 1:
            for s in range(n_steps + 1):
                xs = x0 + s / n_steps * (x1 - x0)
                ax.plot([xs, xs], [strip_y - 0.05, strip_y + 0.05], color=ds.GRANITE, lw=0.8)
            xa, xb = x0, x0 + (x1 - x0) / n_steps
            ax.text((xa + xb) / 2, strip_y - 0.10, r"$\Delta t$", ha="center", va="top", fontsize=14)
        evs = [e[0] for e in (real_ev if k == 0 else ds_ev)]
        for te in evs:
            xs = x0 + te / T * (x1 - x0)
            ax.plot([xs, xs], [strip_y, strip_y + 0.22], color=ds.MIDNIGHT, lw=1.6, solid_capstyle="butt")
        ax.text(x1, strip_y - 0.10, "time", ha="right", va="top", fontsize=13)
    ds.label(ax, "collisions", (0.12, strip_y + 0.30), ha="left", va="bottom")
    save(fig, "fig04c_split_step", num["source"], "seeded hard-sphere and DSMC toy runs")
    plt.close(fig)


if __name__ == "__main__":
    num = s6_numbers()
    print({k: v for k, v in num.items() if k != "source"})
    which = sys.argv[1:] or ["a", "b", "c"]
    for key, fn in (("a", fig_a), ("b", fig_b), ("c", fig_c)):
        if key in which:
            fn(num)
