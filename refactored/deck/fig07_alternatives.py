"""Slide 7 alternatives (7A, 7B, 7B-1, 7B-2): why and how the wheel sector truncates the grid.

  fig07e_sector_strip     (deckstyle strip, 7B-2, ADOPTED as FIG 7 of the deck) the full ring at true
                          scale, the sector in 3D, and the full ring's particle budget as 170 sector tiles;
                          build step 1 (fig07e_sector_strip_b1, progressive deck) without the tiles
  fig07d_sector_ring      (deckstyle evidence_tall, 7B-1) the sector in 3D with the true-scale ring

  fig07b_zoom_strip       (slot 12.22 x 3.10, 7A) full ring at true scale, with the one-wavelength
                          slice dark -> the wheel sector (FIG 7a's drawing, periodic seams) -> the polar
                          band edge-on, before and after the cone-reflection fix
  fig07c_sector_3d        (deckstyle evidence_tall, 7B) the wheel sector as one curved 3D brick:
                          seams, cone walls, cells, a particle crossing a seam; insets for the sector
                          in the full ring (true scale) and the cone-wall bounce edge-on

Geometry is read from deck/data/slide07_facts.json (deck/cache_slides78.py); the S6 ring extent from
deck/data/slide04_facts.json (deck/fig04_method.py). Schematic views exaggerate angles and are labelled;
the full-ring views are true scale.

The cone-wall bounce: a wall of constant polar angle has normal e_theta. The fix flips v_theta (specular
reflection in the cone's tangent plane), which keeps the radial speed v_r. The old code flipped v_z
(reflection off a plane of constant z); at latitude +-beta that changes v_r by -2 v_z sin(beta) on the
upper wall and the same sign on the lower one, so every wall hit removed outward speed.

    cd refactored && conda run -n dsmc python deck/fig07_alternatives.py [a] [b] [b1] [b2]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Wedge

import fig07_wheel_sector as w7
from _deck import R_MARS_KM, Build, ds, r_label, save
from slots import SLOTS

DATA = Path(__file__).resolve().parent / "data"
F7 = json.loads((DATA / "slide07_facts.json").read_text())
F4 = json.loads((DATA / "slide04_facts.json").read_text())
G = F7["geometry"]
SOURCES = ("deck/data/slide07_facts.json", "deck/data/slide04_facts.json")
BETA_DRAW = 2.0 * G["beta_deg"]                 # cone angles drawn 2x


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    return fig, ax


def arrow(ax, p0, p1, color, lw=1.4, ms=10, z=6, ls="-"):
    ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p1), arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def caption(ax, x, y, head, sub, ha="center"):
    ax.text(x, y + 0.02, head, ha=ha, va="bottom", fontsize=12, fontweight="semibold")
    ax.text(x, y - 0.04, sub, ha=ha, va="top", fontsize=13)


# ================================================================= shared pieces
def ring(ax, cx, cy, rm, angle_deg=90.0):
    """S6 ring (100-801 km) around Mars at true scale, with the wheel sector (140-505 km, one lambda_h
    wide) drawn dark at angle_deg. Returns the sector's four corners."""
    k = rm / R_MARS_KM
    r0, r1 = (R_MARS_KM + F4["z_bot"]) * k, (R_MARS_KM + F4["z_top"]) * k
    ax.add_patch(Circle((cx, cy), rm, fc=ds.GRANITE, alpha=0.12, lw=0))
    ax.add_patch(Wedge((cx, cy), r1, 0, 360, width=r1 - r0, fc=ds.MIDNIGHT, alpha=0.16, lw=0))
    for r in (r0, r1):
        ax.add_patch(Circle((cx, cy), r, fc="none", ec=ds.MIDNIGHT, lw=0.6))
    s0, s1 = (R_MARS_KM + G["base_km"]) * k, (R_MARS_KM + G["top_edge_km"]) * k
    half = 0.5 * G["sector_deg"]
    ax.add_patch(Wedge((cx, cy), s1, angle_deg - half, angle_deg + half, width=s1 - s0, fc=ds.MIDNIGHT,
                       lw=0, zorder=4))
    ax.text(cx, cy, "MARS", ha="center", va="center", fontsize=12, fontweight="semibold")

    def pt(r, a):
        return np.array([cx + r * np.cos(np.deg2rad(a)), cy + r * np.sin(np.deg2rad(a))])
    return dict(out_l=pt(s1, angle_deg + half), out_r=pt(s1, angle_deg - half),
                in_l=pt(s0, angle_deg + half), in_r=pt(s0, angle_deg - half))


def cone_bounce(ax, apex, length, mode, title=None, takeaway=None, beta_label=True):
    """Edge-on polar band (apex toward Mars' centre) and one particle hitting the upper cone wall.

    mode: "old" (flat-plane bounce, flips v_z), "fix" (cone reflection, flips v_theta) or "both"
    (fix solid, old as a dashed ghost)."""
    b = np.deg2rad(BETA_DRAW)
    apex = np.asarray(apex, float)
    up, dn = np.array([np.cos(b), np.sin(b)]), np.array([np.cos(b), -np.sin(b)])
    ax.add_patch(Polygon([apex, apex + length * up, apex + length * dn], closed=True, fc=ds.CELTIC,
                         alpha=0.10, lw=0, zorder=0))
    for e in (up, dn):
        ax.plot(*zip(apex, apex + length * e), color=ds.GRANITE, lw=1.4, zorder=2)
    ax.plot(*zip(apex, apex + (length + 0.04) * np.array([1.0, 0.0])), color=ds.GRANITE, lw=0.6,
            ls=(0, (1, 2.5)), zorder=1)
    if beta_label:
        ax.add_patch(Arc(apex, 1.1, 1.1, theta1=-BETA_DRAW, theta2=0, color=ds.GRANITE, lw=0.8))
        ax.text(apex[0] + 0.12, apex[1] - 0.36, rf"$\beta = {G['beta_deg']:.1f}^\circ$",
                ha="left", va="top", fontsize=13)
    q = apex + 0.66 * length * up
    v_in = np.array([np.cos(np.deg2rad(62)), np.sin(np.deg2rad(62))])
    n_hat = np.array([-up[1], up[0]])                      # wall normal, the e_theta direction
    v_fix = v_in - 2 * (v_in @ n_hat) * n_hat              # flip v_theta
    v_old = np.array([v_in[0], -v_in[1]])                  # flip v_z
    p0 = q - 0.72 * v_in
    seg = 0.74
    ax.plot(*zip(p0, q), color=ds.MIDNIGHT if mode != "old" else ds.GRANITE, lw=1.6, zorder=4)
    ax.plot(*p0, "o", ms=5, mew=0, color=ds.MIDNIGHT if mode != "old" else ds.GRANITE, zorder=5)
    if mode in ("old", "both"):
        ls = (0, (3, 2.5)) if mode == "both" else "-"
        ax.plot(*zip(q, q + (seg - 0.07) * v_old), color=ds.GRANITE, lw=1.4, ls=ls, zorder=3)
        arrow(ax, q + (seg - 0.08) * v_old, q + seg * v_old, ds.GRANITE, lw=1.2, ms=10, z=3)
    if mode == "old":
        ax.plot([q[0] - 0.34, q[0] + 0.34], [q[1], q[1]], color=ds.GRANITE, lw=0.9, ls=(0, (1, 2)), zorder=3)
    if mode in ("fix", "both"):
        arrow(ax, q, q + seg * v_fix, ds.MIDNIGHT, lw=1.7, ms=11, z=4)
        ax.plot(*zip(q - 0.34 * up, q + 0.34 * up), color=ds.MIDNIGHT, lw=0.7, alpha=0.5, zorder=3)
    if title:
        ax.text(apex[0], apex[1] + length * np.sin(b) + 0.22, title[0], ha="left", va="bottom", fontsize=12,
                fontweight="semibold", color=title[1])
    tip = q + 0.62 * (v_fix if mode != "old" else v_old)
    if takeaway:
        ds.takeaway(ax, takeaway[0], xy=tuple(tip + np.array([0.03, 0.0])), xytext=takeaway[1],
                    ha="left", va="center", linespacing=1.1)
    return q, tip


# ================================================================= 7A: zoom strip
def fig_a():
    stem = "fig07b_zoom_strip"
    w, h = SLOTS[stem]
    fig, ax = canvas(w, h)
    cap_y = 0.30

    # 1 the full ring, true scale; the sector points toward panel 2
    cx, cy, rm = 1.25, 1.86, 0.92
    c = ring(ax, cx, cy, rm, angle_deg=40.0)
    caption(ax, cx, cap_y, "FULL RING (S6), TRUE SCALE", r"dark slice: one $\lambda_h$")

    # 2 the wheel sector (FIG 7a's drawing, scaled into an inset axes)
    nx, ny = (0.95, 4.00), (0.88, 4.12)
    py0, py1 = 0.66, h - 0.03
    s = (py1 - py0) / (ny[1] - ny[0])
    px0 = 3.25
    pw = (nx[1] - nx[0]) * s
    ins = fig.add_axes([px0 / w, py0 / h, pw / w, (py1 - py0) / h])
    ins.set_axis_off()
    ins.patch.set_visible(False)
    ins.set_xlim(*nx)
    ins.set_ylim(*ny)
    ins.set_aspect("equal")
    w7.main_view(ins, notes=False)

    def to_main(xn, yn):
        return np.array([px0 + (xn - nx[0]) * s, py0 + (yn - ny[0]) * s])
    zoom = dict(color=ds.GRANITE, lw=0.5, alpha=0.6, zorder=0)
    for a, b in ((c["out_l"], to_main(*w7.pol(w7.R_OUT, -w7.HALF))),
                 (c["in_r"], to_main(*w7.pol(w7.R_IN, -w7.HALF)))):
        ax.plot([a[0], b[0]], [a[1], b[1]], **zoom)
    sc = to_main(w7.CX, 0)[0]
    caption(ax, sc, cap_y, "WHEEL SECTOR",
            rf"{G['sector_width_km']:.0f} km wide, periodic seams")

    # 3 the polar band edge-on: before and after the fix
    x3 = 6.30
    ax.plot([x3 - 0.30, x3 - 0.30], [0.75, h - 0.15], color=ds.GRANITE, lw=0.5, alpha=0.35)
    L, yc = 2.45, 1.80
    q_old, _ = cone_bounce(ax, (x3, yc), L, "old", title=("BEFORE: FLAT-PLANE BOUNCE", ds.GRANITE),
                           beta_label=False)
    y_lab = yc - L * np.sin(np.deg2rad(BETA_DRAW)) - 0.30
    ax.text(x3 + 0.05, y_lab, "loses outward speed", ha="left", va="center", fontsize=13,
            fontweight="semibold", color=ds.GRANITE)
    x4 = x3 + 2.95
    cone_bounce(ax, (x4, yc), L, "fix", title=("AFTER: REFLECT OFF THE CONE", ds.MIDNIGHT),
                takeaway=("keeps outward speed", (x4 + 0.05, y_lab)))
    caption(ax, 0.5 * (x3 + x4 + L), cap_y, "POLAR BAND, EDGE-ON",
            rf"cone walls at $\beta$ = {G['beta_deg']:.1f}° (angles drawn 2×)")
    save(fig, stem, *SOURCES, f"schematic + true-scale ring; MarsAtmos {F7['sources']['marsatmos_commit']}")
    plt.close(fig)


# ================================================================= 7B: one sector in 3D
A_HALF, B_HALF = np.deg2rad(20.0), np.deg2rad(5.0)      # drawn azimuth and polar half-widths
R0, R1 = 1.85, 2.95                                      # drawn radii of 140 km and the top edge
YAW, PITCH = np.deg2rad(-14.0), np.deg2rad(22.0)
PLACE = {"scale": 1.62, "ox": 2.66, "oy": -1.28}         # world -> figure inches; set per figure
PLACE_7B = dict(PLACE)


def world(r, lat, az):
    r, lat, az = np.broadcast_arrays(r, lat, az)
    return np.stack([r * np.cos(lat) * np.sin(az), r * np.cos(lat) * np.cos(az), r * np.sin(lat)], -1)


def view(p):
    """Orthographic view: yaw about the radial (page-up) axis, then pitch about the page x axis."""
    cy, sy, cp, sp = np.cos(YAW), np.sin(YAW), np.cos(PITCH), np.sin(PITCH)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    x1, z1 = cy * x + sy * z, -sy * x + cy * z
    y2, z2 = cp * y - sp * z1, sp * y + cp * z1
    return np.stack([PLACE["ox"] + PLACE["scale"] * x1, PLACE["oy"] + PLACE["scale"] * y2], -1), z2


def r_of(z_km):
    return R0 + (z_km - G["base_km"]) / (G["top_edge_km"] - G["base_km"]) * (R1 - R0)


def sector_3d(ax):
    def loop(fixed, val):
        """Boundary of one face of the brick, its centre and outward normal (world coordinates)."""
        a, b = np.linspace(-A_HALF, A_HALF, 60), np.linspace(-B_HALF, B_HALF, 30)
        r = np.linspace(R0, R1, 30)
        if fixed == "r":
            pts = [world(val, -B_HALF, a), world(val, b, A_HALF), world(val, B_HALF, a[::-1]),
                   world(val, b[::-1], -A_HALF)]
            centre, normal = world(val, 0, 0), world(1, 0, 0) * (1 if val == R1 else -1)
        elif fixed == "az":
            pts = [world(r, -B_HALF, val), world(R1, b, val), world(r[::-1], B_HALF, val),
                   world(R0, b[::-1], val)]
            centre = world(0.5 * (R0 + R1), 0, val)
            normal = np.array([np.cos(val), -np.sin(val), 0.0]) * np.sign(val)
        else:   # lat
            pts = [world(r, val, -A_HALF), world(R1, val, a), world(r[::-1], val, A_HALF),
                   world(R0, val, a[::-1])]
            centre = world(0.5 * (R0 + R1), val, 0)
            normal = np.array([0.0, -np.sin(val), np.cos(val)]) * np.sign(val)
        return np.concatenate(pts), centre, normal

    # viewer direction in world coordinates: the view z axis pulled back through the rotations
    cy, sy, cp, sp = np.cos(YAW), np.sin(YAW), np.cos(PITCH), np.sin(PITCH)
    eye = np.array([-sy * cp, sp, cy * cp])
    spec = [("r", R1, ds.CULTURED, 1.0, "outer"), ("r", R0, ds.GRANITE, 0.10, "inner"),
            ("az", A_HALF, ds.MIDNIGHT, 0.14, "seam"), ("az", -A_HALF, ds.MIDNIGHT, 0.14, "seam"),
            ("lat", B_HALF, ds.CELTIC, 0.14, "cone"), ("lat", -B_HALF, ds.CELTIC, 0.14, "cone")]
    drawn = []
    for fixed, val, fc, alpha, kind in spec:
        pts, centre, normal = loop(fixed, val)
        xy, _ = view(pts)
        front = float(normal @ eye) > 0
        drawn.append((front, kind, fixed, val, xy, centre))
    for front, kind, fixed, val, xy, _ in drawn:                       # visible faces only
        if not front:
            continue
        fc = {"outer": ds.GRANITE, "inner": ds.GRANITE, "seam": ds.MIDNIGHT, "cone": ds.CELTIC}[kind]
        alpha = {"outer": 0.07, "inner": 0.08, "seam": 0.16, "cone": 0.12}[kind]
        ax.add_patch(Polygon(xy, closed=True, fc=fc, alpha=alpha, lw=0, zorder=1))
        ax.plot(*xy.T, color=ds.MIDNIGHT if kind == "seam" else ds.GRANITE, lw=2.0 if kind == "seam" else 1.1,
                zorder=3)
    fronts = {(d[2], d[3]) for d in drawn if d[0]}

    # cells and the exobase on the front cone wall
    lat_f = B_HALF if ("lat", B_HALF) in fronts else -B_HALF
    for k in range(5, G["n_azi"], 5):
        a = -A_HALF + 2 * A_HALF * k / G["n_azi"]
        xy, _ = view(world(np.array([R0, R1]), lat_f, a))
        ax.plot(*xy.T, color=ds.GRANITE, lw=0.4, alpha=0.35, zorder=2)
    for j in range(8, G["n_radial"], 8):
        xy, _ = view(world(R0 + (R1 - R0) * j / G["n_radial"], lat_f, np.linspace(-A_HALF, A_HALF, 60)))
        ax.plot(*xy.T, color=ds.GRANITE, lw=0.4, alpha=0.35, zorder=2)
    xy, _ = view(world(r_of(230.0), lat_f, np.linspace(-A_HALF, A_HALF, 60)))
    ax.plot(*xy.T, color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5, zorder=2)

    # a particle: straight 3D flight out through one seam, back in through the other
    p0 = world(r_of(270.0), 0.4 * B_HALF, 0.15 * A_HALF)
    d = world(1.0, np.deg2rad(2), np.deg2rad(84)) - world(0.0, 0, 0)    # heading toward +az
    d = d / np.linalg.norm(d)
    n_seam = np.array([np.cos(A_HALF), -np.sin(A_HALF), 0.0])           # +az seam plane through the axis
    tt = -(n_seam @ p0) / (n_seam @ d)
    hit = p0 + tt * d
    c2, s2 = np.cos(2 * A_HALF), np.sin(2 * A_HALF)                     # azimuth -> azimuth - 2A
    rot = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])
    hit2, d2 = rot @ hit, rot @ d
    p1 = hit2 + 0.40 * d2
    seg1, _ = view(np.array([p0, hit]))
    seg2, _ = view(np.array([hit2, p1]))
    ax.plot(*seg1.T, color=ds.MIDNIGHT, lw=1.4, zorder=6)
    ax.plot(*seg2.T, color=ds.MIDNIGHT, lw=1.4, zorder=6)
    arrow(ax, seg2[1] - 0.05 * (seg2[1] - seg2[0]) / np.linalg.norm(seg2[1] - seg2[0]), seg2[1], ds.MIDNIGHT,
          lw=1.4, ms=10, z=6)
    ax.plot(*seg1[0], "o", ms=5, mew=0, color=ds.MIDNIGHT, zorder=7)
    for q in (seg1[1], seg2[0]):
        ax.plot(*q, "o", ms=5.5, mfc=ds.CULTURED, mec=ds.MIDNIGHT, mew=1.1, zorder=7)

    # periodic-seam arrow over the top, labels
    top = view(world(R1 + 0.20, 0, np.linspace(0.92 * A_HALF, -0.92 * A_HALF, 40)))[0]
    ax.plot(*top[:-3].T, color=ds.MIDNIGHT, lw=1.4, zorder=6)
    arrow(ax, top[-4], top[-1], ds.MIDNIGHT, lw=1.4, ms=11)
    mid = view(world(R1 + 0.20, 0, 0.0))[0]
    ax.text(mid[0], mid[1] + 0.14, "PERIODIC SEAMS", ha="center", va="bottom", fontsize=12,
            fontweight="semibold", color=ds.MIDNIGHT)
    return dict(lat_front=lat_f)


def sector_labels(ax, lat_f, notes=True):
    """Altitudes at the right seam, the polar-band bracket at the left end, and (notes) the width
    caption and the 3D-flight note under the sector. Returns the caption's lowest y."""
    # altitude labels beside the right seam's front edge (as in FIG 7a)
    # the italic r leans into the seam's slanted edge, so the r labels sit 0.06 in further out
    for z, lab, dx, kw in ((G["base_km"], r_label(G["base_km"]), 0.18, dict(fontsize=13)),
                           (G["top_edge_km"], r_label(G["top_edge_km"]), 0.18, dict(fontsize=13)),
                           (230.0, "EXOBASE", 0.12,
                            dict(fontsize=11, fontweight="semibold", color=ds.GRANITE))):
        p = view(world(r_of(z), lat_f, A_HALF))[0]
        ax.text(p[0] + dx, p[1], lab, ha="left", va="center", **kw)
    # polar band thickness: bracket across the outer face at its left end
    e0 = view(world(R1, -B_HALF, -A_HALF))[0]
    e1 = view(world(R1, B_HALF, -A_HALF))[0]
    ax.annotate("", xy=tuple(e1 + np.array([-0.10, 0.0])), xytext=tuple(e0 + np.array([-0.10, 0.0])),
                arrowprops=dict(arrowstyle="<|-|>", color=ds.GRANITE, lw=0.9, mutation_scale=8,
                                shrinkA=0, shrinkB=0))
    em = 0.5 * (e0 + e1)
    ax.text(em[0] - 0.18, em[1], "polar band\n" + rf"$\pm${G['beta_deg']:.1f}°", ha="right",
            va="center", fontsize=12, linespacing=1.15)
    base = view(world(R0, lat_f, np.linspace(-A_HALF, A_HALF, 40)))[0]
    bm = np.array([base[len(base) // 2][0], base[:, 1].min()])
    ax.text(bm[0], bm[1] - 0.20, rf"one $\lambda_h$ = {G['sector_width_km']:.0f} km wide, "
            f"{G['n_azi']} cells × {G['azi_cell_km']:.0f} km", ha="center", va="top", fontsize=13)
    if not notes:
        return bm[1] - 0.20
    ax.text(bm[0], bm[1] - 0.50, "particles fly straight in 3D; only the grid is cut", ha="center", va="top",
            fontsize=12, style="italic")
    return bm[1] - 0.70


def fig_b():
    stem = "fig07c_sector_3d"
    w, h = ds.SIZES["evidence_tall"]
    fig, ax = canvas(w, h)
    PLACE.update(PLACE_7B)
    lat_f = sector_3d(ax)["lat_front"]
    sector_labels(ax, lat_f)
    ax.text(0.10, h - 0.10, "NOT TO SCALE", ha="left", va="top", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)

    # inset 1: the sector in the full ring, true scale (top right)
    rx, ry, rm = 6.25, 3.62, 0.62
    ring(ax, rx, ry, rm, angle_deg=90.0)
    ax.text(rx, ry - 0.86, "FULL RING, TRUE SCALE", ha="center", va="top", fontsize=11,
            fontweight="semibold", color=ds.GRANITE)

    # inset 2: the cone wall edge-on, old bounce as a ghost (bottom right)
    x0, y0, L = 4.95, 1.30, 2.00
    _, tip = cone_bounce(ax, (x0, y0), L, "both", title=("CONE WALL, EDGE-ON", ds.GRANITE), beta_label=True)
    y_lab = y0 - L * np.sin(np.deg2rad(BETA_DRAW)) - 0.30
    ds.takeaway(ax, "reflect off the cone", xy=tuple(tip + np.array([0.03, -0.02])), xytext=(x0, y_lab),
                ha="left", va="center")
    ax.text(x0, y_lab - 0.26, "dashed: old bounce", ha="left", va="center", fontsize=12,
            color=ds.GRANITE)
    save(fig, stem, *SOURCES, f"schematic + true-scale ring; MarsAtmos {F7['sources']['marsatmos_commit']}")
    plt.close(fig)


# ================================================================= 7B-1, 7B-2: the sector and the ring only
def sci_parts(x):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.1f} × 10", str(e)


def fig_b1():
    """7B-1: the sector large, the full ring at true scale top right, zoom lines between them."""
    stem = "fig07d_sector_ring"
    w, h = ds.SIZES["evidence_tall"]
    fig, ax = canvas(w, h)
    PLACE.update(scale=1.95, ox=3.22, oy=-2.08)
    lat_f = sector_3d(ax)["lat_front"]
    sector_labels(ax, lat_f)
    rx, ry, rm = 6.45, 1.68, 0.50
    ring(ax, rx, ry, rm, angle_deg=90.0)
    ax.text(w - 0.02, ry - 0.70, "FULL RING, TRUE SCALE", ha="right", va="top", fontsize=11,
            fontweight="semibold", color=ds.GRANITE)
    ax.text(w - 0.02, ry - 0.90, "dark slice: the sector", ha="right", va="top", fontsize=12, color=ds.GRANITE)
    ax.text(0.10, h - 0.10, "SECTOR NOT TO SCALE", ha="left", va="top", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)
    save(fig, stem, *SOURCES, f"schematic + true-scale ring; MarsAtmos {F7['sources']['marsatmos_commit']}")
    plt.close(fig)


def tiles(ax, x0, y0, tile, gap, n, cols):
    """n tiles (the full ring's particle count in sector units), the bottom-left one dark."""
    rows = n // cols
    for i in range(n):
        r, k = divmod(i, cols)
        x, y = x0 + k * (tile + gap), y0 + (rows - 1 - r) * (tile + gap)
        dark = i == n - cols
        ax.add_patch(plt.Rectangle((x, y), tile, tile, fc=ds.MIDNIGHT if dark else ds.GRANITE,
                                   alpha=1.0 if dark else 0.16, lw=0, zorder=2))
    return x0 + cols * (tile + gap) - gap, y0 + rows * (tile + gap) - gap


def fig_b2():
    """7B-2 (adopted FIG 7): full-width strip; the ring (true scale), the sector, and the particle budget
    as tiles. Placed at deckstyle "strip"."""
    stem = "fig07e_sector_strip"
    w, h = ds.SIZES["strip"]
    fig, ax = canvas(w, h)
    build = Build(fig)
    # 1 the full ring
    rx, ry, rm = 1.05, 1.80, 0.80
    ring(ax, rx, ry, rm, angle_deg=90.0)
    ax.text(rx, 0.34, "FULL RING, TRUE SCALE", ha="center", va="bottom", fontsize=12, fontweight="semibold")
    ax.text(rx, 0.30, "dark slice: the sector", ha="center", va="top", fontsize=13)
    # 2 the sector
    PLACE.update(scale=1.42, ox=4.55, oy=-1.62)
    lat_f = sector_3d(ax)["lat_front"]
    sector_labels(ax, lat_f, notes=False)
    ax.text(2.45, h - 0.05, "NOT TO SCALE", ha="left", va="top", fontsize=11, fontweight="semibold",
            color=ds.GRANITE)
    # 3 the particle budget: the full ring = n sector-sized tiles (build step 2)
    build.at(2)
    ratio = F7["s6"]["particles"] / F7["stage1b"]["particles"]
    n = 10 * round(ratio / 10)
    cols = n // 10
    x0, y0, tile, gap = 8.30, 0.78, 0.172, 0.030
    x1, y1 = tiles(ax, x0, y0, tile, gap, n, cols)
    ax.plot([x0 - 0.40, x0 - 0.40], [0.30, h - 0.15], color=ds.GRANITE, lw=0.5, alpha=0.35)
    ax.text(x0, y1 + 0.10, "PARTICLES PER RUN", ha="left", va="bottom", fontsize=12, fontweight="semibold")
    mant, exp = sci_parts(F7["s6"]["particles"])
    ax.text(x0, 0.26, rf"full ring: {mant}$^{{{exp}}}$ = {n} sectors", ha="left", va="top", fontsize=13)
    mant, exp = sci_parts(F7["stage1b"]["particles"])
    ds.takeaway(ax, rf"one sector: {mant}$^{{{exp}}}$", xy=(x0 + tile / 2, y0 - 0.03),
                xytext=(x0 + tile + 0.25, y0 - 0.30), ha="left", va="center")
    build.save(stem, *SOURCES, f"schematic + true-scale ring; MarsAtmos {F7['sources']['marsatmos_commit']}")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "b1", "b2"]
    for name in which:
        {"a": fig_a, "b": fig_b, "b1": fig_b1, "b2": fig_b2}[name]()
