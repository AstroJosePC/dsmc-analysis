"""Slide 3 alternatives (3A, 3B, 3C): how one molecule travels as Kn = lambda_mfp / H grows.

Every number comes from the S6 model atmosphere before forcing (atomic O, T = 270 K, H = 40-45 km),
cached in outputs/s1_s6_azimuthal/analysis/preforcing_transport_S6_azimean_preforcing.npz from
the solver's own Mean Free Path, Scale Height, Knudsen Number and Thermal Velocity outputs:
lambda_mfp = 1.6 km at 100 km and 39 km at 231 km (Kn = 1 at 236 km), mean speed 597 m/s, wave
period P = 661 s. Collisions per wave period = (mean speed / lambda_mfp) P: about 150 at 120 km,
10 at the exobase, below 1 above about 330 km. Regime edges are where this Kn profile crosses
0.1 and 10. The exobase line stays at the deck's 230 km.

Paths are random flights drawn from those numbers: free paths exponential with mean lambda_mfp(z),
directions isotropic in 3D and projected on the page, number of collisions Poisson with mean
nu P. They are seeded illustrations of the statistics, not particle tracks from the run.

  fig03a_regime_windows   (evidence) altitude column + three 40 km windows, one wave period each
  fig03b_kn_profile       (evidence) one molecule's climb beside lambda_mfp(z) and H(z)
  fig03c_collision_clock  (slot)     collisions in one wave period at six altitudes

    conda run -n dsmc python deck/fig03_alternatives.py [a] [b] [c]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from _deck import ds, save
from dsmclib.paths import output_dir
from slots import SLOTS

TRANSPORT = output_dir("s1_s6_azimuthal", "analysis") / "preforcing_transport_S6_azimean_preforcing.npz"
SOURCE = (TRANSPORT, "S6 pre-forcing profile; seeded random flights")


# ================================================================= the model atmosphere
class Profile:
    def __init__(self):
        t = np.load(TRANSPORT)
        self.z = t["altitude_km"]
        self.lam = t["mean_free_path_m"] / 1e3
        self.H = t["scale_height_km"]
        self.kn = t["knudsen_number"]
        self.n = t["density_m3"]
        self.c = t["thermal_speed_ms"] / 1e3
        self.P = float(t["wave_period_s"])
        self.npc = t["mean_particles_per_cell"]

    def at(self, name, z):
        """Log-linear interpolation of a positive profile."""
        return np.exp(np.interp(z, self.z, np.log(getattr(self, name))))

    def coll_per_period(self, z):
        return self.at("c", z) / self.at("lam", z) * self.P

    def z_of_kn(self, kn):
        return float(np.interp(np.log(kn), np.log(self.kn), self.z))

    def regimes(self, z0=100.0, z1=700.0):
        a, b = self.z_of_kn(0.1), self.z_of_kn(10.0)
        return {"collisional": (z0, a), "transition": (a, b), "exosphere": (b, z1)}


def sig2(v, min_dec=0):
    """Two significant figures with thousands separators: 1726 -> '1,700', 0.064 -> '0.064'."""
    if v >= 10:
        r = float(f"{v:.2g}")
        return f"{r:,.0f}"
    d = max(min_dec, 1 - int(np.floor(np.log10(v))))
    return f"{v:.{d}f}"


def directions(rng, n):
    """Isotropic 3D unit vectors projected on the page (horizontal x, vertical z)."""
    mu = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.sqrt(1 - mu ** 2) * np.cos(ph), mu


def flight(rng, lam, n_coll, start=(0.0, 0.0)):
    """Fixed-lambda random flight with n_coll collisions (n_coll + 1 free paths)."""
    ell = rng.exponential(lam, n_coll + 1)
    dx, dz = directions(rng, n_coll + 1)
    x = np.r_[start[0], start[0] + np.cumsum(ell * dx)]
    z = np.r_[start[1], start[1] + np.cumsum(ell * dz)]
    return x, z


def arrow(ax, p0, p1, color, lw=1.4, ms=10, z=5):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                mutation_scale=ms, shrinkA=0, shrinkB=0), zorder=z)


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
    return fig, ax


# ================================================================= 3A: three windows
WIN_KM = 40.0
WINDOWS = [  # altitude, regime, label
    (120.0, "collisional", "COLLISIONAL"),
    (ds.EXOBASE_KM, "transition", "EXOBASE"),
    (400.0, "exosphere", "EXOSPHERE"),
]


def window_path(rng, prof, z, n_tries=400):
    """One molecule's path through a 40 km window over one wave period, chosen so the window shows
    what the regime does: a tangle, a path with one or two collisions, or a straight crossing."""
    lam, nu_p = prof.at("lam", z), prof.coll_per_period(z)
    half = WIN_KM / 2
    if nu_p > 30:                                   # collisional: start in the middle, keep it all
        for _ in range(n_tries):
            x, zz = flight(rng, lam, rng.poisson(nu_p))
            if np.abs(x).max() < half * 0.92 and np.abs(zz).max() < half * 0.92:
                return x, zz, lam, nu_p
        return x, zz, lam, nu_p
    if nu_p > 1:                                    # transition: enter, collide inside, leave
        for _ in range(n_tries):
            x, zz = flight(rng, lam, rng.poisson(nu_p), start=(rng.uniform(-half, half), -half * 1.6))
            inside = (np.abs(x[1:-1]) < half * 0.8) & (np.abs(zz[1:-1]) < half * 0.8)
            seg = np.hypot(np.diff(x), np.diff(zz))
            if inside.sum() == 2 and seg[1:-1].min() > 8:
                return x, zz, lam, nu_p
        return x, zz, lam, nu_p
    d = np.array([np.cos(0.32), np.sin(0.32)])      # exosphere: one straight flight across
    return np.array([-2 * half * d[0], 2 * half * d[0]]), np.array([-2 * half * d[1] - 5, 2 * half * d[1] - 5]), lam, nu_p


def fig_a():
    prof = Profile()
    fig, ax = canvas(size="evidence")
    rng = np.random.default_rng(20261117)

    # altitude column: a slim curved sector, 100-700 km
    cx, yb, R0, Hc = 1.62, 0.62, 8.0, 3.18
    alpha = 0.62 / R0
    ymap = lambda z: (np.asarray(z, float) - 100.0) / 600.0 * Hc       # noqa: E731

    def cxy(z, a):
        r = R0 + ymap(z)
        return cx + r * np.sin(a), yb - R0 + r * np.cos(a)

    def arc(z, n=80):
        return cxy(z, np.linspace(-alpha, alpha, n))

    for name, (za, zb) in prof.regimes().items():
        xa, ya = arc(za)
        xb, yb_ = arc(zb)
        ax.add_patch(Polygon(np.column_stack([np.r_[xa, xb[::-1]], np.r_[ya, yb_[::-1]]]),
                             fc=ds.REGIME[name], alpha=0.15, lw=0, zorder=0))
    zz = 100 + rng.exponential(150.0, 6000)
    zz = zz[zz < 696][:1500]
    aa = rng.uniform(-alpha * 0.98, alpha * 0.98, zz.size)
    ax.scatter(*cxy(zz, aa), s=1.3, color=ds.MIDNIGHT, alpha=0.75, lw=0, zorder=2)
    ax.plot(*arc(100.0), color=ds.GRANITE, lw=0.8, zorder=3)
    ax.plot(*arc(700.0), color=ds.GRANITE, lw=0.5, zorder=3)
    for s in (-1, 1):
        ax.plot(*cxy(np.array([100.0, 700.0]), s * alpha), color=ds.GRANITE, lw=0.5, zorder=3)
    ax.plot(*arc(ds.EXOBASE_KM), color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5, zorder=4)
    for z in (100.0, ds.EXOBASE_KM, 700.0):
        x, y = cxy(z, -alpha)
        ax.text(x - 0.10, y, f"{z:.0f} km", ha="right", va="center", fontsize=13)
    for a in np.linspace(-0.65, 0.65, 5) * alpha:                   # wave input
        (x0, x1), (y0, y1) = cxy(np.array([100.0 - 48, 100.0 - 4]), a)
        arrow(ax, (x0, y0), (x1, y1), ds.ROLE["forcing"], lw=1.6, ms=9)
    ds.label(ax, "wave input", (cx, yb - 0.38), ha="center", va="top")
    for a in np.linspace(-0.6, 0.6, 3) * alpha:                     # escape
        (x0, x1), (y0, y1) = cxy(np.array([672.0, 742.0]), a)
        arrow(ax, (x0, y0), (x1, y1), ds.GRANITE, lw=1.0, ms=8)

    # windows
    S, wx = 1.04, 3.02
    wys = (0.12, 1.45, 2.78)
    k = S / WIN_KM                                                  # inches per km
    for (z, regime, name), wy in zip(WINDOWS, wys):
        x, y = cxy(z, alpha)                                        # zoom lines from the column
        for yy in (wy, wy + S):
            ax.plot([x + 0.04, wx], [y, yy], color=ds.GRANITE, lw=0.5, alpha=0.55, zorder=1)
        ax.plot([x - 0.03, x + 0.04], [y, y], color=ds.GRANITE, lw=1.2, zorder=4)
        box = Rectangle((wx, wy), S, S, fc=ds.CULTURED, ec=ds.GRANITE, lw=0.8, zorder=5)
        ax.add_patch(box)
        ax.add_patch(Rectangle((wx, wy), S, S, fc=ds.REGIME[regime], alpha=0.15, lw=0, zorder=5))
        # neighbours: true relative density (n / n(120 km)) x 240 dots per window
        n_dots = rng.poisson(240 * prof.at("n", z) / prof.at("n", 120.0))
        ax.scatter(wx + rng.uniform(0.04, S - 0.04, n_dots), wy + rng.uniform(0.04, S - 0.04, n_dots),
                   s=2.2, color=ds.GRANITE, alpha=0.55, lw=0, zorder=6)
        px, pz, lam, nu_p = window_path(rng, prof, z)
        X, Y = wx + S / 2 + px * k, wy + S / 2 + pz * k
        line, = ax.plot(X, Y, color=ds.MIDNIGHT, lw=1.0 if nu_p > 30 else 1.4, zorder=7,
                        solid_joinstyle="round")
        line.set_clip_path(box)
        if nu_p > 30:
            ax.plot(X[0], Y[0], "o", ms=4, color=ds.MIDNIGHT, zorder=8)
        else:
            kinks = (np.abs(px[1:-1]) < WIN_KM / 2) & (np.abs(pz[1:-1]) < WIN_KM / 2)
            ax.plot(X[1:-1][kinks], Y[1:-1][kinks], "o", ms=5, mfc=ds.CULTURED, mec=ds.MIDNIGHT,
                    mew=1.2, zorder=8)
        # text beside each window
        tx = wx + S + 0.16
        ds.label(ax, f"{name} · {z:.0f} km", (tx, wy + S - 0.10), color=ds.GRANITE, ha="left", va="top")
        ax.text(tx, wy + S - 0.42, rf"$\lambda_\mathrm{{mfp}}$ ≈ {sig2(lam)} km,  Kn ≈ {sig2(prof.at('kn', z), 1)}",
                ha="left", va="top", fontsize=14)
        coll = f"≈ {sig2(nu_p)} collisions" if nu_p >= 1 else "< 1 collision"
        ax.text(tx, wy + S - 0.70, f"{coll} per wave period", ha="left", va="top", fontsize=14)
    ds.label(ax, "40 km windows · one wave period", (wx, 4.04), ha="left", va="center")
    save(fig, "fig03a_regime_windows", *SOURCE)
    plt.close(fig)


# ================================================================= 3B: one climb + profile
def climb(rng, prof, z0=104.0, z_exit=520.0, width_km=270.0, max_steps=200000):
    """One molecule from near the lower boundary until it leaves the top: free paths drawn from
    the local lambda_mfp(z); side walls reflect. Seeds are tried until one climbs out."""
    for _ in range(200):
        x, z = [width_km * 0.30], [z0]
        for _ in range(max_steps):
            lam = prof.at("lam", z[-1])
            ell = rng.exponential(min(lam, 5000.0))
            dx, dz = directions(rng, 1)
            xn, zn = x[-1] + ell * dx[0], z[-1] + ell * dz[0]
            if zn < 100.0:
                zn = 200.0 - zn
            if xn < 0 or xn > width_km:
                xn = -xn if xn < 0 else 2 * width_km - xn
            x.append(xn)
            z.append(zn)
            if zn > z_exit:
                return np.array(x), np.array(z)
    raise RuntimeError("no climbing path found")


def fig_b():
    prof = Profile()
    fig, ax = canvas(size="evidence")
    W_, H_ = ds.SIZES["evidence"]
    zlo, zhi = 100.0, 520.0
    y0, y1 = 0.80, 3.84
    kz = (y1 - y0) / (zhi - zlo)                              # inches per km (both axes of the path)
    px0, pw = 0.95, 2.05
    width_km = pw / kz

    ax_p = fig.add_axes([px0 / W_, y0 / H_, pw / W_, (y1 - y0) / H_])
    ax_q = fig.add_axes([3.55 / W_, y0 / H_, 3.45 / W_, (y1 - y0) / H_])
    regimes = prof.regimes(zlo, zhi)
    for a in (ax_p, ax_q):
        a.set_ylim(zlo, zhi)
        for name, (za, zb) in regimes.items():
            a.axhspan(za, zb, color=ds.REGIME[name], alpha=0.15, lw=0, zorder=0)
        a.axhline(ds.EXOBASE_KM, color=ds.ROLE["exobase"], ls=ds.LINESTYLE["exobase"], lw=1.5, zorder=1)

    # path panel
    rng = np.random.default_rng(20261121)
    x, z = climb(rng, prof, width_km=width_km)
    print(f"3B climb: {x.size - 1} free paths, {(z < 140).sum()} below 140 km")
    ax_p.set_xlim(0, width_km)
    ax_p.plot(x[:-1], z[:-1], color=ds.MIDNIGHT, lw=0.55, zorder=3, solid_joinstyle="round")
    ax_p.plot(x[-2:], z[-2:], color=ds.MIDNIGHT, lw=1.4, zorder=3)
    arrow(ax_p, (x[-2] + 0.97 * (x[-1] - x[-2]), zhi - 6), (x[-1], z[-1]), ds.MIDNIGHT, lw=1.4)
    ax_p.plot([x[0]], [z[0]], "o", ms=5, color=ds.MIDNIGHT, zorder=4)
    for s_ in ("top", "right", "bottom"):
        ax_p.spines[s_].set_visible(False)
    ax_p.set_xticks([])
    ax_p.set_yticks([100, 200, 300, 400, 500])
    ax_p.set_ylabel("Altitude (km)")
    ax_p.spines["left"].set_bounds(zlo, zhi)
    ax_p.plot([0, width_km], [zlo, zlo], color=ds.GRANITE, lw=0.8, clip_on=False, zorder=2)
    for xa in np.linspace(0.15, 0.85, 5) * width_km:                  # wave input
        ax_p.annotate("", xy=(xa, zlo - 1), xytext=(xa, zlo - 0.36 / kz),
                      arrowprops=dict(arrowstyle="-|>", color=ds.ROLE["forcing"], lw=1.6, mutation_scale=9,
                                      shrinkA=0, shrinkB=0), annotation_clip=False)
    ax_p.text(width_km / 2, zlo - 0.42 / kz, "WAVE INPUT", ha="center", va="top", fontsize=12,
              fontweight="semibold")
    ax_p.text(width_km / 2, zhi + 0.05 / kz, "ONE MOLECULE, TRUE SCALE", ha="center", va="bottom",
              fontsize=12, fontweight="semibold")

    # profile panel
    zz = np.linspace(zlo, zhi, 300)
    lam, H = prof.at("lam", zz), prof.at("H", zz)
    ax_q.set_xscale("log")
    ax_q.set_xlim(1, 3e3)
    ax_q.plot(H, zz, color=ds.GRANITE, ls=ds.LINESTYLE["prior_work"], lw=1.5, zorder=2)
    ax_q.plot(lam, zz, color=ds.ROLE["this_work"], lw=2.4, zorder=3)
    z_kn1 = prof.z_of_kn(1.0)
    ax_q.plot([prof.at("lam", z_kn1)], [z_kn1], "o", ms=7, mfc="none", mec=ds.CRIMSON, mew=1.4, zorder=5)
    ds.takeaway(ax_q, "Kn = 1", xy=(prof.at("lam", z_kn1) * 1.15, z_kn1 - 2),
                xytext=(300, 272), ha="left", va="center")
    ax_q.text(900, 415, r"$\lambda_\mathrm{mfp}$", ha="right", va="center", fontsize=18, color=ds.MIDNIGHT)
    ax_q.text(34, 470, r"$H$", ha="right", va="center", fontsize=18, color=ds.GRANITE)
    za, zb = regimes["collisional"]
    ax_q.text(2.6e3, 0.5 * (za + zb), "COLLISIONAL", ha="right", va="center", fontsize=12,
              fontweight="semibold")
    for name, (xt, y, ha) in {"transition": (2.6e3, 185, "right"), "exosphere": (1.25, 440, "left")}.items():
        ax_q.text(xt, y + 6, name.upper(), ha=ha, va="bottom", fontsize=12, fontweight="semibold")
        kn = {"transition": "0.1 < Kn < 10", "exosphere": "Kn > 10"}[name]
        ax_q.text(xt, y - 4, kn, ha=ha, va="top", fontsize=12)
    ax_q.set_yticks([])
    ax_q.spines["left"].set_visible(False)
    ax_q.set_xticks([1, 10, 100, 1000])
    ax_q.set_xticklabels(["1", "10", "100", "1,000"])
    ax_q.set_xlabel("Length (km)")
    save(fig, "fig03b_kn_profile", *SOURCE)
    plt.close(fig)


# ================================================================= 3C: the collision clock
CLOCK_Z = (120.0, 150.0, 200.0, ds.EXOBASE_KM, 300.0, 400.0)


def fig_c():
    stem = "fig03c_collision_clock"
    prof = Profile()
    fig, ax = canvas(stem)
    rng = np.random.default_rng(661)
    x0, x1 = 1.45, 9.35                       # one wave period
    row_h, pitch, ybase = 0.27, 0.36, 0.66
    edges = prof.regimes()
    for i, z in enumerate(CLOCK_Z):
        y = ybase + i * pitch
        regime = next(n for n, (a, b) in edges.items() if a <= z < b)
        ax.add_patch(Rectangle((x0 - 0.06, y - 0.045), x1 - x0 + 0.12, row_h + 0.09, fc=ds.REGIME[regime],
                               alpha=0.15, lw=0, zorder=0))
        exo = z == ds.EXOBASE_KM
        ax.plot([x0, x1], [y, y], color=ds.ROLE["exobase"] if exo else ds.GRANITE,
                ls=ds.LINESTYLE["exobase"] if exo else "-", lw=1.5 if exo else 0.6, zorder=2)
        nu_p = prof.coll_per_period(z)
        # the mean number of collisions, at random (uniform) times within the period
        t = np.sort(rng.uniform(0, 1, int(round(nu_p))))
        ax.vlines(x0 + t * (x1 - x0), y, y + row_h, color=ds.MIDNIGHT, lw=0.9, zorder=3)
        label = f"{z:.0f} km" if not exo else "230 km · EXOBASE"
        ax.text(x0 - 0.18, y + row_h / 2 - 0.02, f"{z:.0f} km", ha="right", va="center", fontsize=14)
        if exo:
            ax.text(x0 - 0.18, y + row_h / 2 - 0.20, "EXOBASE", ha="right", va="center", fontsize=10.5,
                    fontweight="semibold")
        num = f"≈ {sig2(nu_p)}" if nu_p >= 1 else "< 1"
        ax.text(x1 + 0.30, y + row_h / 2 - 0.02, num, ha="left", va="center", fontsize=16,
                color=ds.MIDNIGHT, fontweight="semibold")
        ax.text(x1 + 1.40, y + row_h / 2 - 0.02, f"Kn ≈ {sig2(prof.at('kn', z), 1)}",
                ha="left", va="center", fontsize=14)
    top = ybase + (len(CLOCK_Z) - 1) * pitch + row_h
    ds.label(ax, "collisions of one molecule during one wave period", (x0, top + 0.16), ha="left",
             va="center")
    ax.text(x1 + 0.30, top + 0.16, "PER PERIOD", ha="left", va="center", fontsize=11, fontweight="semibold")
    # the wave period itself: one cycle of the lower-boundary forcing
    tt = np.linspace(0, 1, 300)
    yf = 0.30
    ax.plot(x0 + tt * (x1 - x0), yf + 0.12 * np.sin(2 * np.pi * tt), color=ds.ROLE["forcing"], lw=2.0, zorder=3)
    ax.text(x0 - 0.18, yf, "WAVE INPUT", ha="right", va="center", fontsize=11, fontweight="semibold")
    ax.text(x1 + 0.30, yf, f"one period, P = {prof.P / 60:.0f} min", ha="left", va="center", fontsize=14)
    save(fig, stem, *SOURCE)
    plt.close(fig)




if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b", "c"]
    for key, fn in (("a", fig_a), ("b", fig_b), ("c", fig_c)):
        if key in which:
            fn()
