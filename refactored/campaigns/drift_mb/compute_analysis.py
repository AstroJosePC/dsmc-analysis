"""drift_mb analysis: 1-D drifting-Maxwellian lower-boundary forcing in pure O and O+H2.

    python -B campaigns/drift_mb/compute_analysis.py [--runs ALIAS ...]

Writes (docs/CONVENTIONS.md, analysis tier):
- outputs/drift_mb/analysis/<alias>_analysis.nc   one per run (registry campaign drift_mb):
    forced runs and s = 0 controls:
      schedule (elapsed, period/phase index, forcing flag), injection-boundary VDFs phase-folded into 16 phases
      plus the ring-down (counts, flux-weighted mixture theory, KS / chi2, binned and unbinned moments vs theory),
      per-output injection moments and mass/momentum/energy rates vs theory, return rates,
      escape series / per-period / phase-folded rates, escape-speed VDFs (fit window, ring-down, phases, control,
      spin-up, Jeans-mapped reference), wave response over the fit window (harmonic amplitude/phase of u_r,
      dn/n0, dT, T_rad, T_tan vs altitude, noise floor from the control), altitude-phase and altitude-time
      perturbation fields, per-period u_r amplitude and cycle-mean dT (ring-down), cycle-mean heating vs control
      and vs spin-up, propagation (phase speed, amplitude growth / damping), ring-down timescales.
    spin-ups: profiles over the last SPIN_WINDOW s, exobase, escape (counts), escape-speed VDF, inventory/escape series.
- outputs/drift_mb/analysis/drift_mb_analysis.nc   cross-run scalars (run x species)
- outputs/drift_mb/tables/{run_summary,vdf_moments,flux_budget,escape_summary,wave_response,heating,ringdown}.csv

Definitions (every setting is also written as an attribute):
- Rates of forced runs are corrected for the reload_inputs restart interval (dsmclib.drift.correct_restart_rates,
  attribute restart_rate_factor).
- Fit window: forcing periods FIT_PERIODS (0-based, [10, 20): the last 10 of 20). Ring-down: periods 20-24.
- Baseline: the s = 0 control of the same atmosphere (O_s000_P330 / OH2_s000_P330) averaged over the same
  elapsed-time window since the restart (rule "time_matched"); when the control covers < MIN_OVERLAP of the window,
  its second half (rule "control_second_half"). The controls themselves use the spin-up's last SPIN_WINDOW s.
  The spin-up's last SPIN_WINDOW s is also a cross-check baseline for every run (suffix _vs_spin).
- Outputs are interval means over P/16; harmonic fits use interval mid-times and divide amplitudes by
  sinc(pi/16) = 0.99359 (attribute interval_sinc).
- Phases are relative to the forcing w'(t) = w' sin(omega t'); a response y = A cos(omega t' + phi) has
  phi_rel = phi + pi/2 (0 = in phase with w').
- Cells whose baseline holds < MIN_PART computational particles (mean per sample) are masked.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit

from dsmclib import drift, profiles, registry, tiers
from dsmclib.constants import G_NEWTON, K_BOLTZMANN
from dsmclib.ncrun import RUN_SUFFIX, open_run_nc
from dsmclib.paths import output_dir
from dsmclib.wave import _amp_phase, _harmonic_fit

warnings.simplefilter("ignore", RuntimeWarning)

CAMPAIGN = "drift_mb"
SCRIPT = "campaigns/drift_mb/compute_analysis.py"
ROOT = Path.home() / "MarsAtmos-DriftMB-sweep"
CONTROL = {"O": "O_s000_P330", "OH2": "OH2_s000_P330"}
SPIN = {"O": "SPIN_O", "OH2": "SPIN_OH2"}
FIT_PERIODS = (10, 20)
N_RING = 5
# Common elapsed-time window (s since the restart): COMMON_WINDOW_P165 x the atmosphere's P165 period, i.e. the P165
# fit window, inside the forcing for every P. Every output cadence is P165/16 times 1, 2 or 4, so the window edges
# fall on output boundaries of all runs and of the control. Escape enhancement and exobase heating are also reported
# here so P is compared at equal forcing duration.
COMMON_WINDOW_P165 = (10, 20)


def common_window(period: float) -> tuple[float, float]:
    p165 = period / round(period / 164.7)
    return COMMON_WINDOW_P165[0] * p165, COMMON_WINDOW_P165[1] * p165
SPIN_WINDOW = 2.0e4
MIN_PART = 10.0
MIN_OVERLAP = 0.5
GAMMA = 5.0 / 3.0
SNR_MIN = 5.0
REF_ALT_KM = (110.0, 150.0, 200.0, 240.0, 300.0, 400.0)
ESC_COARSE = 10                      # escape-speed bins merged for the forced-vs-control chi2 (10 x 25 m/s)
NP = drift.N_PHASE
SINC = math.sin(math.pi / NP) / (math.pi / NP)
PROFILE_FIELDS = ("num_dens", "num_part", "temp_trans_corr", "temp_rad", "temp_tan", "vel_rad", "knudsen",
                  "mean_free_path", "scale_height")
WAVE_FIELDS = {  # name -> (run-file field, relative?)
    "ur": ("vel_rad", False), "dn": ("num_dens", True), "dT": ("temp_trans_corr", False),
    "dTrad": ("temp_rad", False), "dTtan": ("temp_tan", False)}
WAVE_UNITS = {"ur": "m s-1", "dn": "1", "dT": "K", "dTrad": "K", "dTtan": "K"}
WAVE_LONG = {"ur": "radial velocity u_r", "dn": "fractional density perturbation dn/n0",
             "dT": "bulk-flow corrected translational temperature perturbation", "dTrad": "radial temperature perturbation",
             "dTtan": "tangential temperature perturbation"}


# ------------------------------------------------------------------ loading
def matrix() -> dict:
    return {r["id"]: r for r in csv.DictReader(open(ROOT / "matrix.csv"))}


def run_file(alias: str) -> Path:
    return registry.run(alias).path / f"{alias}{RUN_SUFFIX}"


def open_run(alias: str):
    run = open_run_nc(run_file(alias))
    run.alias = alias
    drift.correct_restart_rates(run)
    m = run.macro
    run.macro["temp_trans_corr"] = profiles.temp_trans_corrected(run)
    run.macro["temp_tan"] = (0.5 * (m["temp_zen"] + m["temp_azi"])).where(m["num_part"] > 0, 0.0).assign_attrs(
        units="K", long_name="Tangential temperature (T_zen + T_azi) / 2")
    return run


def avg_dt(run) -> float:
    return float(run.attrs["avg_steps"]) * run.dt


def escape_counts(run) -> np.ndarray:
    """Escaping computational particles per output, (species, time), from the escape-speed histogram."""
    h = run.diagnostics["diagnostics_escape_speed_histogram"]["escape_speed_hist"].sum("boundary_cell")
    return h.sum("speed_bin").transpose("species", "time").values


def escape_hist(run) -> np.ndarray:
    h = run.diagnostics["diagnostics_escape_speed_histogram"]["escape_speed_hist"].sum("boundary_cell")
    return h.transpose("species", "time", "speed_bin").values


# ------------------------------------------------------------------ baselines
@dataclass
class Baseline:
    source: str
    rule: str
    idx: np.ndarray                  # output indices in the source run
    t0: float                        # absolute (t0, t1] window of the source
    t1: float
    prof: xr.Dataset
    esc_count: np.ndarray            # (species,)
    duration: float
    esc_hist: np.ndarray             # (species, speed_bin)
    run: object


def make_baseline(run, idx: np.ndarray, rule: str) -> Baseline:
    t = run.time
    d = avg_dt(run)
    t0, t1 = float(t[idx[0]] - 0.5 * d), float(t[idx[-1]] + 1e-6)
    prof = profiles.time_mean_profiles(run, t0, t1, fields=PROFILE_FIELDS)
    return Baseline(run.name, rule, idx, t0, t1, prof, escape_counts(run)[:, idx].sum(1), len(idx) * d,
                    escape_hist(run)[:, idx].sum(1), run)


def spin_baseline(spin) -> Baseline:
    idx = np.where(spin.time > spin.time[-1] - SPIN_WINDOW)[0]
    return make_baseline(spin, idx, f"spin_last_{SPIN_WINDOW:.0f}s")


def control_baseline(ctrl, csch: xr.Dataset, lo: float, hi: float) -> Baseline:
    """Control outputs inside the elapsed window [lo, hi] s since the restart (time-matched), else its second half."""
    d = avg_dt(ctrl)
    end = csch.elapsed.values
    sel = (end - d >= lo - 1e-6) & (end <= hi + 1e-6)
    frac = sel.sum() * d / (hi - lo)
    if frac >= 0.999:
        rule = "time_matched"
    elif frac >= MIN_OVERLAP:
        rule = f"partial_overlap_{frac:.2f}"
    else:
        sel = end > end[-1] / 2
        rule = "control_second_half"
    return make_baseline(ctrl, np.where(sel)[0], rule)


# ------------------------------------------------------------------ small fits
def harmonic(t, Y, omega):
    """amp (sinc-corrected), amp_se, phase, phase_se of Y(time, ...) at omega."""
    coef, se = _harmonic_fit(t, Y, omega)
    amp, amp_se, ph, ph_se = _amp_phase(coef, se)
    return amp / SINC, amp_se / SINC, ph, ph_se


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def per_period_harmonic(t, Y, omega, period, origin, n_periods=None):
    """Amplitude at omega of each whole period (constant + cos + sin per block); (period, ...) array."""
    blk = np.floor((t - origin) / period + 1e-9).astype(int)
    n = blk.max() + 1 if n_periods is None else n_periods
    out = np.full((n,) + Y.shape[1:], np.nan)
    for p in range(n):
        s = blk == p
        if s.sum() < 6:
            continue
        X = np.column_stack([np.ones(s.sum()), np.cos(omega * t[s]), np.sin(omega * t[s])])
        Yp = Y[s].reshape(s.sum(), -1)
        good = np.all(np.isfinite(Yp), axis=0)
        c = np.full((3, Yp.shape[1]), np.nan)
        if good.any():
            c[:, good] = np.linalg.lstsq(X, Yp[:, good], rcond=None)[0]
        out[p] = (np.hypot(c[1], c[2]) / SINC).reshape(Y.shape[1:])
    return out


def decay_fit(t, y, sigma, t0):
    """y = a exp(-(t - t0)/tau); returns dict(tau, tau_se, a, a_se, resolved)."""
    ok = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    res = {"tau": np.nan, "tau_se": np.nan, "a": np.nan, "a_se": np.nan, "resolved": 0}
    if ok.sum() < 3 or not np.isfinite(y[ok][0]):
        return res
    tt, yy, ss = t[ok] - t0, y[ok], sigma[ok]
    a0 = yy[0] if yy[0] != 0 else np.nanmax(np.abs(yy))
    try:
        p, cov = curve_fit(lambda x, a, tau: a * np.exp(-x / tau), tt, yy, p0=[a0, max(float(np.ptp(tt)), 1.0)],
                           sigma=ss, absolute_sigma=True, bounds=([-np.inf, 1e-3], [np.inf, 1e7]), maxfev=20000)
        e = np.sqrt(np.diag(cov))
        if p[1] >= 0.999e7:            # no decay inside the window: tau unbounded, not a measurement
            res.update(a=p[0], a_se=e[0], at_bound=1)
            return res
        res.update(tau=p[1], tau_se=e[1], a=p[0], a_se=e[0],
                   resolved=int(abs(p[0]) > 3 * e[0] and e[1] < p[1]))
    except (RuntimeError, ValueError):
        pass
    return res


def weighted_line(x, y, sy):
    w = 1.0 / sy ** 2
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    scale = max(chi2 / max(len(x) - 2, 1), 1.0)       # inflate by reduced chi2 when > 1
    return beta, np.sqrt(np.diag(cov) * scale)


# ------------------------------------------------------------------ theory helpers
def jeans_mapped_prob(edges, T, m, r_e, r_t, M, n_sub=40):
    """Bin probabilities of the speed at r_t of Jeans-escaping particles launched at the exobase r_e with a
    flux-weighted Maxwellian at T (v^3 exp(-v^2/vth^2), v > v_esc(r_e)); collisionless energy mapping.
    The last bin is open (overflow)."""
    vth = math.sqrt(2 * K_BOLTZMANN * T / m)
    dphi = 2 * G_NEWTON * M * (1 / r_e - 1 / r_t)

    def dens(vt):
        ve2 = vt ** 2 + dphi
        return ve2 * vt * np.exp(-(ve2 - (2 * G_NEWTON * M / r_e)) / vth ** 2)  # shifted exponent (no underflow)

    lo, hi = edges[:-1], edges[1:].copy()
    hi[-1] = edges[-2] + 40 * vth           # overflow bin integrated over the tail
    x = (np.arange(n_sub) + 0.5) / n_sub
    p = np.array([(dens(a + (b - a) * x) * (b - a) / n_sub).sum() for a, b in zip(lo, hi)])
    return p / p.sum()


def sound(run, z_m):
    """c_s, H, omega_a for the O-dominated isothermal column at altitude z (T0, gamma 5/3)."""
    T0 = float(run.info_surface["T0"].values.ravel()[0])
    mO = float(run.mass.sel(species="O"))
    g = G_NEWTON * run.planet_mass / (run.planet_radius + z_m) ** 2
    cs = math.sqrt(GAMMA * K_BOLTZMANN * T0 / mO)
    H = K_BOLTZMANN * T0 / (mO * g)
    return cs, H, cs / (2 * H)


# ------------------------------------------------------------------ boundary VDFs
def boundary_vdf(run, sch) -> tuple[xr.Dataset, list]:
    hist = run.diagnostics["diagnostics_histogram"]
    dr = run.diagnostics["diagnostics_drift"].isel(boundary_cell=0)
    H = hist["inj_hist_y"].isel(boundary_cell=0).transpose("species", "time", "v_bin").values
    yc = hist["y"].values
    yb = hist["y_bounds"].values
    edges = np.append(yb[:, 0], yb[-1, 1])
    forcing = sch.forcing.values == 1
    phase = sch.phase_index.values
    groups = [f"p{j:02d}" for j in range(NP)] + ["ringdown"]
    nsp = len(run.species)
    shape = (nsp, NP + 1)
    count = np.zeros(shape + (len(yc),), dtype=np.int64)
    prob = np.zeros(shape + (len(yc),))
    st = {k: np.full(shape, np.nan) for k in (
        "N", "s_mean", "s_center", "ks_D", "ks_Dcrit", "chi2", "chi2_dof", "chi2_p",
        "mean_hist", "var_hist", "skew_hist", "exkurt_hist", "mean_th", "var_th", "skew_th", "exkurt_th",
        "mean_unb", "y2_unb", "y2_th", "vt2_unb", "mean_z", "var_z", "skew_z", "exkurt_z", "y2_z", "vt2_z",
        "mean_se", "var_se", "skew_se", "exkurt_se")}
    rows = []
    cnt = dr["inj_count"].values
    svn, svn2, svt2 = dr["inj_sum_vn"].values, dr["inj_sum_vn2"].values, dr["inj_sum_vt2"].values
    s_amp = drift.wave_amplitude_per_species(run) * float(run.attrs.get("wave_amplitude", 1.0))
    for i, sp in enumerate(run.species):
        vth = float(dr["v_th"].sel(species=sp).values[0])
        for g in range(NP + 1):
            sel = (forcing & (phase == g)) if g < NP else ~forcing
            s_steps = drift.phase_s(run, sch, g, sp, vth, periods=[0]) if g < NP else np.zeros(1)
            if g < NP and s_amp[i] == 0:
                s_steps = np.zeros(1)
            c = H[i, sel].sum(0)
            N = c.sum()
            p = drift.mixture_bin_prob(edges, s_steps)
            count[i, g], prob[i, g] = c, p
            rawt = drift.mixture_raw_moments(s_steps, 8)
            th = drift.central_from_raw(rawt[:5])
            rawh = np.array([(c * yc ** k).sum() / N for k in range(5)])
            ob = drift.central_from_raw(rawh)
            emp = np.cumsum(c) / N
            mod = drift.mixture_cdf(edges[1:], s_steps)
            D = float(np.max(np.abs(emp - mod)))
            exp_ = N * p
            use = exp_ >= 5
            lump_o, lump_e = c[~use].sum(), exp_[~use].sum()
            chi2 = float(np.sum((c[use] - exp_[use]) ** 2 / exp_[use]) + ((lump_o - lump_e) ** 2 / lump_e if lump_e > 0 else 0))
            dof = int(use.sum() + (lump_e > 0) - 1)
            n_unb = cnt[i, sel].sum()
            mean_unb = svn[i, sel].sum() / n_unb / vth
            y2_unb = svn2[i, sel].sum() / n_unb / vth ** 2
            vt2_unb = svt2[i, sel].sum() / n_unb / vth ** 2
            mse = drift.sample_moment_se(rawt, N)     # exact asymptotic SEs from E[y^0..8] (delta method)
            se_mean, se_var = mse["mean"], mse["var"]
            se_y2 = math.sqrt(max(rawt[4] - rawt[2] ** 2, 0) / N)
            vals = dict(N=N, s_mean=float(np.sum(s_steps * drift.m_flux(s_steps)) / np.sum(drift.m_flux(s_steps))),
                        s_center=(s_amp[i] / vth * math.sin((g + 0.5) * 2 * math.pi / NP)) if g < NP else 0.0,
                        ks_D=D, ks_Dcrit=1.95 / math.sqrt(N), chi2=chi2, chi2_dof=dof,
                        chi2_p=float(stats.chi2.sf(chi2, dof)),
                        mean_hist=ob["mean"], var_hist=ob["var"], skew_hist=ob["skew"], exkurt_hist=ob["exkurt"],
                        mean_th=th["mean"], var_th=th["var"], skew_th=th["skew"], exkurt_th=th["exkurt"],
                        mean_unb=mean_unb, y2_unb=y2_unb, y2_th=rawt[2], vt2_unb=vt2_unb,
                        mean_z=(mean_unb - th["mean"]) / se_mean, var_z=(ob["var"] - th["var"]) / se_var,
                        skew_z=(ob["skew"] - th["skew"]) / mse["skew"],
                        exkurt_z=(ob["exkurt"] - th["exkurt"]) / mse["exkurt"],
                        mean_se=se_mean, var_se=se_var, skew_se=mse["skew"], exkurt_se=mse["exkurt"],
                        y2_z=(y2_unb - rawt[2]) / se_y2, vt2_z=(vt2_unb - 1.0) * math.sqrt(N))
            for k, v in vals.items():
                st[k][i, g] = v
            rows.append({"run": run.name, "species": sp, "group": groups[g], **{k: float(v) for k, v in vals.items()},
                         "ks_ratio": D / vals["ks_Dcrit"]})
    dims = ("species", "vdf_group")
    ds = xr.Dataset(
        {"vdf_count": (dims + ("v_bin",), count, {"long_name": "Injected computational particles per y bin, folded"}),
         "vdf_model_prob": (dims + ("v_bin",), prob, {"long_name": "Bin probability of the flux-weighted drifting-Maxwellian mixture"}),
         **{f"vdf_{k}": (dims, v) for k, v in st.items()}},
        coords={"species": run.species, "vdf_group": groups, "y": ("v_bin", yc), "y_lo": ("v_bin", yb[:, 0]),
                "y_hi": ("v_bin", yb[:, 1]),
                "phase_angle": ("vdf_group", np.append((np.arange(NP) + 0.5) * 2 * np.pi / NP, np.nan),
                                {"units": "rad", "long_name": "Phase-bin centre of omega t' (forcing w' sin)"})})
    ds["vdf_ks_Dcrit"].attrs["long_name"] = "KS critical distance at the 0.1 % level (1.95 / sqrt N)"
    ds["vdf_vt2_unb"].attrs["long_name"] = "<v_t^2> / v_th^2 of injected particles (theory 1)"
    ds.attrs["vdf_groups"] = "p00..p15: forcing phase bins over all forcing periods; ringdown: w' = 0 periods"
    return ds, rows


# ------------------------------------------------------------------ injection budgets
def injection_series(run, sch) -> xr.Dataset:
    dr = run.diagnostics["diagnostics_drift"].isel(boundary_cell=0)
    d = avg_dt(run)
    avg = int(run.attrs["avg_steps"])
    nt = run.macro.sizes["time"]
    W = run.weight.values[:, None]
    m = run.mass.values[:, None]
    vth = dr["v_th"].transpose("species", "time").values
    N = dr["inj_count"].transpose("species", "time").values
    svn = dr["inj_sum_vn"].transpose("species", "time").values
    svn2 = dr["inj_sum_vn2"].transpose("species", "time").values
    svt2 = dr["inj_sum_vt2"].transpose("species", "time").values
    surf = run.surface.sum("boundary_cell")
    area = float(run.surface["lower_area"].sum())
    phi0 = dr["background_flux"].transpose("species", "time").values
    if "wave_start_step" in run.attrs:
        steps = np.arange(int(sch.step_first[0]), int(sch.step_last[-1]) + 1)
        w = drift.step_drift(run, steps).reshape(len(run.species), nt, avg)
    else:
        w = np.zeros((len(run.species), nt, avg))
    s = w / vth[:, :, None]
    mf = drift.m_flux(s)
    raw = (drift.raw_moments(s, 2) * mf[None]).sum(-1) / mf.sum(-1)[None]
    Ndot_th = phi0 * area * mf.mean(-1)
    out = {
        "inj_npart": (("species", "time"), N, {"units": "1"}),
        "inj_mean_y": (("species", "time"), svn / N / vth, {"units": "1"}),
        "inj_mean_y_th": (("species", "time"), raw[1], {"units": "1"}),
        "inj_y2": (("species", "time"), svn2 / N / vth ** 2, {"units": "1"}),
        "inj_y2_th": (("species", "time"), raw[2], {"units": "1"}),
        "inj_vt2": (("species", "time"), svt2 / N / vth ** 2, {"units": "1", "long_name": "<v_t^2>/v_th^2 (theory 1)"}),
        "Ndot_inj": (("species", "time"), N * W / d, {"units": "s-1", "long_name": "Injected molecules per second"}),
        "Ndot_inj_th": (("species", "time"), Ndot_th, {"units": "s-1", "long_name": "Phi0 A <m(s)> over the interval"}),
        "Pdot_inj": (("species", "time"), W * m * svn / d, {"units": "kg m s-2", "long_name": "Injected normal momentum per second"}),
        "Pdot_inj_th": (("species", "time"), Ndot_th * m * vth * raw[1], {"units": "kg m s-2"}),
        "Edot_inj": (("species", "time"), W * 0.5 * m * (svn2 + svt2) / d, {"units": "W", "long_name": "Injected kinetic energy per second"}),
        "Edot_inj_th": (("species", "time"), Ndot_th * 0.5 * m * vth ** 2 * (raw[2] + 1.0), {"units": "W"}),
        "Ndot_ret": (("species", "time"), surf["return_rate"].transpose("species", "time").values, {"units": "s-1"}),
        "Edot_ret": (("species", "time"), surf["return_energy"].transpose("species", "time").values * W, {"units": "W"}),
    }
    ds = xr.Dataset(out, coords={"species": run.species, "time": run.time})
    ds["Ndot_net"] = (ds.Ndot_inj - ds.Ndot_ret).assign_attrs(units="s-1")
    ds["Edot_net"] = (ds.Edot_inj - ds.Edot_ret).assign_attrs(units="W")
    ds["Mdot_net"] = (ds.Ndot_net * run.mass).assign_attrs(units="kg s-1")
    return ds


BUDGET_VARS = ("Ndot_inj", "Ndot_inj_th", "Pdot_inj", "Pdot_inj_th", "Edot_inj", "Edot_inj_th", "Ndot_ret", "Edot_ret",
               "Ndot_net", "Edot_net", "Mdot_net")


# ------------------------------------------------------------------ per-run analysis (forced and control)
def analyse_forced(alias: str, meta: dict, cache: dict) -> tuple[xr.Dataset, dict]:
    atm = meta["atmosphere"]
    run = open_run(alias)
    sch = drift.schedule(run)
    assert sch.attrs["phase_locked"] == 1, f"{alias}: not phase locked"
    P, omega = sch.attrs["period"], sch.attrs["omega"]
    nF = sch.attrs["n_force_periods"]
    d = avg_dt(run)
    t_mid = sch.elapsed.values - 0.5 * d
    pidx = sch.period_index.values
    is_control = alias == CONTROL[atm]
    spin = cache.setdefault(("spinrun", atm), open_run(SPIN[atm]))
    spin_b = cache.setdefault(("spinbase", atm), spin_baseline(spin))
    ctrl = cache.setdefault(("ctrl", atm), open_run(CONTROL[atm]))
    csch = cache.setdefault(("csch", atm), drift.schedule(ctrl))
    windows = {"forcing": (0.0, nF * P), "fit": (FIT_PERIODS[0] * P, FIT_PERIODS[1] * P),
               "ringdown": (nF * P, (nF + N_RING) * P)}
    base = {k: (spin_b if is_control else control_baseline(ctrl, csch, *w)) for k, w in windows.items()}
    widx = {"forcing": np.where(pidx < nF)[0], "fit": np.where((pidx >= FIT_PERIODS[0]) & (pidx < FIT_PERIODS[1]))[0],
            "ringdown": np.where(pidx >= nF)[0]}
    species = run.species
    nsp = len(species)
    alt = run.macro["altitude"].values
    W = run.weight.values
    summary = {"run": alias, "atmosphere": atm, "s_O": float(meta["s_O"]), "s_H2": float(meta["s_H2"]),
               "w_mps": float(meta["w_mps"]), "period_s": P, "is_control": int(is_control),
               "restart_rate_factor": float(run.attrs["restart_rate_factor"]),
               "baseline_fit_rule": base["fit"].rule, "baseline_ringdown_rule": base["ringdown"].rule,
               "baseline_source": base["fit"].source}
    tables = {k: [] for k in ("vdf", "budget", "escape", "wave", "heating", "ringdown")}

    # ---- schedule and VDFs
    ds = sch.drop_vars([v for v in ("step_first", "step_last") if v in sch]).copy()
    ds["wdrift"] = (("species", "time"),
                    run.diagnostics["diagnostics_drift"]["wdrift"].isel(boundary_cell=0).transpose("species", "time").values,
                    {"units": "m s-1", "long_name": "Drift velocity w' at the output step"})
    vdf, tables["vdf"] = boundary_vdf(run, sch)
    ds = ds.merge(vdf)

    # ---- injection budgets
    inj = injection_series(run, sch)
    ds = ds.merge(inj)
    cinj = cache.setdefault(("cinj", atm), injection_series(ctrl, csch)) if not is_control else None
    for wname, idx in widx.items():
        bidx = base[wname].idx
        for i, sp in enumerate(species):
            row = {"run": alias, "species": sp, "window": wname, "n_outputs": len(idx)}
            for v in BUDGET_VARS:
                row[v] = float(inj[v].isel(species=i).values[idx].mean())
                if cinj is not None:
                    row[f"{v}_control"] = float(cinj[v].sel(species=sp).values[bidx].mean())
            row["inj_over_theory"] = row["Ndot_inj"] / row["Ndot_inj_th"]
            row["mom_over_theory"] = row["Pdot_inj"] / row["Pdot_inj_th"]
            row["energy_over_theory"] = row["Edot_inj"] / row["Edot_inj_th"]
            s_amp = float(meta["s_O"] if sp == "O" else meta["s_H2"])
            row["cycle_mean_flux_ratio_th"] = drift.cycle_mean_flux_ratio(s_amp) if wname != "ringdown" else 1.0
            if cinj is not None:
                for v in ("Ndot_inj", "Pdot_inj", "Edot_inj", "Ndot_ret", "Edot_ret", "Ndot_net", "Edot_net"):
                    row[f"{v}_ratio_control"] = row[v] / row[f"{v}_control"] if row[f"{v}_control"] else np.nan
            tables["budget"].append(row)

    # ---- escape
    n_esc = escape_counts(run)
    hist = escape_hist(run)
    eh = run.diagnostics["diagnostics_escape_speed_histogram"]
    sb = eh["speed_bounds"].values
    sedges = np.append(sb[:, 0], sb[-1, 1])
    rate = n_esc * W[:, None] / d
    per_n = np.array([[n_esc[i, pidx == p].sum() for p in range(nF + N_RING)] for i in range(nsp)])
    per_rate = per_n * W[:, None] / (NP * d)
    per_se = np.sqrt(np.maximum(per_n, 1)) * W[:, None] / (NP * d)
    fold = np.array([[n_esc[i, widx["fit"]][sch.phase_index.values[widx["fit"]] == j].sum() for j in range(NP)]
                     for i in range(nsp)])
    hist_fit = hist[:, widx["fit"]].sum(1)
    hist_ring = hist[:, widx["ringdown"]].sum(1)
    hist_forcing = hist[:, widx["forcing"]].sum(1)
    hist_phase = np.stack([hist[:, widx["fit"]][:, sch.phase_index.values[widx["fit"]] == j].sum(1)
                           for j in range(NP)], axis=1)
    spin_hist = spin_b.esc_hist
    ctrl_hist = base["fit"].esc_hist
    jeans = np.full((nsp, len(sedges) - 1), np.nan)
    esc_rows = []
    for i, sp in enumerate(species):
        def win_rate(b: Baseline):
            k = float(b.esc_count[b.run.species.index(sp)])
            wgt = float(b.run.weight.sel(species=sp))
            return k * wgt / b.duration, math.sqrt(max(k, 1)) * wgt / b.duration, k
        nf = float(n_esc[i, widx["fit"]].sum())
        Rf = nf * W[i] / (len(widx["fit"]) * d)
        Rf_se = math.sqrt(max(nf, 1)) * W[i] / (len(widx["fit"]) * d)
        nr = float(n_esc[i, widx["ringdown"]].sum())
        Rr = nr * W[i] / (len(widx["ringdown"]) * d)
        Rc, Rc_se, kc = win_rate(base["fit"])
        Rs, Rs_se, ks = win_rate(spin_b)
        Rcr, Rcr_se, _ = win_rate(base["ringdown"])
        eps = Rf / Rc - 1 if Rc > 0 else np.nan
        eps_se = (Rf / Rc) * math.hypot(Rf_se / Rf, Rc_se / Rc) if (Rc > 0 and Rf > 0) else np.nan
        eps_s = Rf / Rs - 1 if Rs > 0 else np.nan
        eps_s_se = (Rf / Rs) * math.hypot(Rf_se / Rf, Rs_se / Rs) if (Rs > 0 and Rf > 0) else np.nan
        # modulation at omega of the per-output counts over the fit window (relative)
        mod_amp = mod_amp_se = mod_ph = mod_ph_se = np.nan
        if nf > 0:
            y = n_esc[i, widx["fit"]] / n_esc[i, widx["fit"]].mean() - 1.0
            a, ase, ph, phse = harmonic(t_mid[widx["fit"]], y[:, None], omega)
            mod_amp, mod_amp_se, mod_ph, mod_ph_se = float(a[0]), float(ase[0]), float(wrap(ph[0] + np.pi / 2)), float(phse[0])
        # ring-down relaxation of the excess over the control (ringdown baseline)
        tau = {"tau": np.nan, "tau_se": np.nan, "a": np.nan, "a_se": np.nan, "resolved": 0}
        if Rcr > 0 and nf > 0 and not is_control:
            pp = np.arange(nF - 1, nF + N_RING)
            y = per_rate[i, pp] / Rcr - 1
            sy = per_se[i, pp] / Rcr
            tau = decay_fit((pp + 0.5) * P, y, sy, nF * P)
        # escape-speed VDF comparisons
        mean_v = lambda h: float((h * 0.5 * (sedges[:-1] + sedges[1:])).sum() / h.sum()) if h.sum() > 0 else np.nan  # noqa: E731
        hc = ctrl_hist[base["fit"].run.species.index(sp)]
        p_chi = np.nan
        if hist_fit[i].sum() > 0 and hc.sum() > 0:
            a = hist_fit[i][:-1].reshape(-1, ESC_COARSE).sum(1)
            b = hc[:-1].reshape(-1, ESC_COARSE).sum(1)
            keep = (a + b) >= 10
            if keep.sum() >= 2:
                p_chi = float(stats.chi2_contingency(np.vstack([a[keep], b[keep]]))[1])
        # Jeans-mapped reference at this species' exobase (control/spin baseline profile)
        bp = base["fit"].prof
        z_exo_sp = float(profiles.exobase(bp).sel(species=sp))
        if np.isfinite(z_exo_sp) and hist_fit[i].sum() > 0:
            T_exo = float(profiles.interp_at_altitude(bp, "temp_trans_corr", profiles.exobase(bp)).sel(species=sp))
            jeans[i] = jeans_mapped_prob(sedges, T_exo, float(run.mass.sel(species=sp)),
                                         run.planet_radius + z_exo_sp * 1e3, float(run.radial_edges()[-1]),
                                         run.planet_mass)
        else:
            T_exo = np.nan
        row = {"run": alias, "species": sp, "R_fit": Rf, "R_fit_se": Rf_se, "npart_fit": nf, "R_control": Rc,
               "R_control_se": Rc_se, "npart_control": kc, "enhancement": eps, "enhancement_se": eps_se,
               "R_spin": Rs, "R_spin_se": Rs_se, "enhancement_vs_spin": eps_s, "enhancement_vs_spin_se": eps_s_se,
               "R_ringdown": Rr, "R_control_ringdown": Rcr, "mod_amp": mod_amp, "mod_amp_se": mod_amp_se,
               "mod_phase_rel_forcing": mod_ph, "mod_phase_se": mod_ph_se,
               "tau_ringdown": tau["tau"] if tau["resolved"] else np.nan,
               "tau_ringdown_se": tau["tau_se"] if tau["resolved"] else np.nan, "tau_resolved": tau["resolved"],
               "tau_ringdown_fit": tau["tau"], "tau_ringdown_fit_se": tau["tau_se"], "excess_at_forcing_end": tau["a"],
               "mean_speed_fit": mean_v(hist_fit[i]), "mean_speed_control": mean_v(hc),
               "mean_speed_spin": mean_v(spin_hist[spin.species.index(sp)]) if sp in spin.species else np.nan,
               "vdf_chi2_p_vs_control": p_chi, "exobase_km": z_exo_sp, "T_exobase": T_exo,
               "baseline_rule": base["fit"].rule}
        esc_rows.append(row)
    tables["escape"] = esc_rows
    ds["escape_rate"] = (("species", "time"), rate, {"units": "s-1", "long_name": "DSMC escape rate (restart-corrected)"})
    ds["escape_npart"] = (("species", "time"), n_esc, {"units": "1"})
    ds["escape_rate_period"] = (("species", "period"), per_rate, {"units": "s-1"})
    ds["escape_rate_period_se"] = (("species", "period"), per_se, {"units": "s-1"})
    ds["escape_npart_phase"] = (("species", "phase"), fold, {"long_name": "Escaping computational particles per phase bin, fit window"})
    ds["escape_rate_phase"] = (("species", "phase"), fold * W[:, None] / (len(widx["fit"]) / NP * d), {"units": "s-1"})
    for k, h in (("fit", hist_fit), ("ringdown", hist_ring), ("forcing", hist_forcing)):
        ds[f"escape_vdf_{k}"] = (("species", "speed_bin"), h, {"long_name": f"Escape-speed histogram, {k} window (counts)"})
    ds["escape_vdf_phase"] = (("species", "phase", "speed_bin"), hist_phase)
    ds["escape_vdf_control"] = (("species_base", "speed_bin"), ctrl_hist, {"long_name": f"Baseline ({base['fit'].source}, {base['fit'].rule})"})
    ds["escape_vdf_spin"] = (("species_base", "speed_bin"), spin_hist, {"long_name": f"Spin-up last {SPIN_WINDOW:.0f} s"})
    ds["escape_vdf_jeans_prob"] = (("species", "speed_bin"), jeans,
                                   {"long_name": "Jeans flux at the species exobase mapped collisionlessly to r_top (bin probability)"})
    ds = ds.assign_coords(speed=("speed_bin", eh["speed"].values), speed_lo=("speed_bin", sb[:, 0]),
                          speed_hi=("speed_bin", sb[:, 1]), species_base=base["fit"].run.species)
    for k in ("R_fit", "R_control", "enhancement", "enhancement_se", "enhancement_vs_spin", "tau_ringdown"):
        ds[f"esc_{k}"] = ("species", [r[k] for r in esc_rows])

    # ---- wave response and heating
    bprof = base["fit"].prof
    resolved = (bprof["num_part"] >= MIN_PART).transpose("species", "radial").values
    m = run.macro
    delta = {}
    for name, (f, rel) in WAVE_FIELDS.items():
        x = m[f].transpose("time", "species", "radial").values.astype(float)
        if f not in ("num_dens", "vel_rad"):
            x = np.where(m["num_part"].transpose("time", "species", "radial").values > 0, x, np.nan)
        b = bprof[f].transpose("species", "radial").values
        y = x / b - 1 if rel else x - b
        delta[name] = np.where(resolved[None], y, np.nan)
    fi = widx["fit"]
    # control null at this omega over the control's own baseline window
    null = {}
    if not is_control:
        cb = base["fit"]
        cm = ctrl.macro
        ct = csch.elapsed.values[cb.idx] - 0.5 * avg_dt(ctrl)
        for name, (f, rel) in WAVE_FIELDS.items():
            x = cm[f].transpose("time", "species", "radial").values[cb.idx].astype(float)
            if f not in ("num_dens", "vel_rad"):
                x = np.where(cm["num_part"].transpose("time", "species", "radial").values[cb.idx] > 0, x, np.nan)
            b = cb.prof[f].transpose("species", "radial").values
            y = np.where(resolved[None], x / b - 1 if rel else x - b, np.nan)
            null[name] = harmonic(ct, y, omega)[0]
    z_exo = float(profiles.exobase(bprof).sel(species="O"))
    for name in WAVE_FIELDS:
        a, ase, ph, phse = harmonic(t_mid[fi], delta[name][fi], omega)
        u, ln = WAVE_UNITS[name], WAVE_LONG[name]
        ds[f"{name}_amp"] = (("species", "radial"), a, {"units": u, "long_name": f"Amplitude at omega of the {ln}, fit window"})
        ds[f"{name}_amp_se"] = (("species", "radial"), ase, {"units": u})
        ds[f"{name}_phase"] = (("species", "radial"), wrap(ph + np.pi / 2), {"units": "rad", "long_name": "Phase relative to w' sin(omega t')"})
        ds[f"{name}_phase_se"] = (("species", "radial"), phse, {"units": "rad"})
        ds[f"{name}_null_amp"] = (("species", "radial"), null.get(name, np.full_like(a, np.nan)),
                                  {"units": u, "long_name": "Same fit on the s = 0 control (noise floor at omega)"})
        # altitude-phase (folded over the fit window) and altitude-time fields
        ds[f"{name}_phasefold"] = (("species", "phase", "radial"),
                                   np.stack([np.nanmean(delta[name][fi][sch.phase_index.values[fi] == j], axis=0) for j in range(NP)], axis=1),
                                   {"units": u, "long_name": f"Phase-folded {ln}, fit window"})
        ds[f"{name}_series"] = (("species", "time", "radial"), np.moveaxis(delta[name], 0, 1),
                                {"units": u, "long_name": f"{ln} relative to the baseline profile"})
    # per-period u_r amplitude and cycle-mean fields (all 25 periods)
    per_amp = per_period_harmonic(t_mid, delta["ur"], omega, P, 0.0, nF + N_RING)
    ds["ur_amp_period"] = (("period", "species", "radial"), per_amp, {"units": "m s-1"})
    for name in ("dT", "dn", "dTrad", "dTtan"):
        ds[f"{name}_mean_period"] = (("period", "species", "radial"),
                                     np.stack([np.nanmean(delta[name][pidx == p], axis=0) for p in range(nF + N_RING)]),
                                     {"units": WAVE_UNITS[name], "long_name": f"Cycle-mean {WAVE_LONG[name]}"})
    # control per-period u_r amplitude floor at this omega (rms over the control's whole record)
    if not is_control:
        cm = ctrl.macro
        cy = cm["vel_rad"].transpose("time", "species", "radial").values - spin_b.prof["vel_rad"].transpose("species", "radial").values
        ctm = csch.elapsed.values - 0.5 * avg_dt(ctrl)
        cpa = per_period_harmonic(ctm, np.where(resolved[None], cy, np.nan), omega, P, 0.0)
        floor = np.sqrt(np.nanmean(cpa ** 2, axis=0))
    else:
        floor = np.sqrt(np.nanmean(per_amp ** 2, axis=0))
    ds["ur_amp_period_floor"] = (("species", "radial"), floor, {"units": "m s-1", "long_name": "rms per-period u_r amplitude of the s = 0 control"})
    # cycle-mean heating over the fit window vs baseline and vs spin-up
    prof_fit = profiles.time_mean_profiles(run, float(run.time[fi[0]] - 0.5 * d), float(run.time[fi[-1]] + 1e-6), fields=PROFILE_FIELDS)
    for f, key in (("temp_trans_corr", "dT"), ("temp_rad", "dTrad"), ("temp_tan", "dTtan"), ("num_dens", "dn")):
        for bname, bb in (("", bprof), ("_vs_spin", spin_b.prof)):
            if f == "num_dens":
                v = prof_fit[f] / bb[f] - 1
                se = np.abs(prof_fit[f] / bb[f]) * np.hypot(prof_fit[f + "_sem"] / prof_fit[f], bb[f + "_sem"] / bb[f])
            else:
                v = prof_fit[f] - bb[f]
                se = np.hypot(prof_fit[f + "_sem"], bb[f + "_sem"])
            ok = xr.DataArray(resolved, dims=("species", "radial"))
            ds[f"{key}_cyclemean{bname}"] = v.where(ok).transpose("species", "radial").assign_attrs(
                units=WAVE_UNITS[key], long_name=f"Fit-window mean minus baseline{' (spin-up)' if bname else ''}")
            ds[f"{key}_cyclemean{bname}_se"] = se.where(ok).transpose("species", "radial").assign_attrs(units=WAVE_UNITS[key])
    for f in ("num_dens", "temp_trans_corr", "num_part", "knudsen", "vel_rad"):
        ds[f"baseline_{f}"] = bprof[f].transpose("species", "radial").assign_attrs(long_name=f"Baseline {f} ({base['fit'].rule})")
    ds["resolved"] = (("species", "radial"), resolved.astype(np.int8), {"long_name": f"Baseline holds >= {MIN_PART:g} computational particles"})

    # propagation, damping, heating scalars per species
    cs, H, wa = sound(run, 0.5 * (alt[0] + z_exo * 1e3))
    c_th = cs / math.sqrt(1 - (wa / omega) ** 2) if omega > wa else np.nan
    above = alt >= z_exo * 1e3
    ring_rows, wave_rows, heat_rows = [], [], []
    for i, sp in enumerate(species):
        a, ase, ph, phse = (ds[f"ur_{k}"].values[i] for k in ("amp", "amp_se", "phase", "phase_se"))
        nl = ds["ur_null_amp"].values[i]
        ok = resolved[i] & np.isfinite(a) & (a >= SNR_MIN * ase) & (alt < z_exo * 1e3)
        if not is_control:
            ok &= ~(np.isfinite(nl) & (a < 3 * nl))
        idx = np.where(ok)[0]
        if len(idx):
            br = np.where(np.diff(idx) > 1)[0]
            idx = idx[: br[0] + 1] if len(br) else idx       # contiguous block from the bottom
        prop = dict(c_phase=np.nan, c_phase_se=np.nan, k_z=np.nan, alpha=np.nan, alpha_se=np.nan, n_cells=len(idx),
                    z_lo_km=np.nan, z_hi_km=np.nan)
        if len(idx) >= 4:
            z = alt[idx]
            phu = np.unwrap(ph[idx])
            (_, b), (_, bse) = weighted_line(z, phu, np.maximum(phse[idx], 1e-6))
            (_, al), (_, alse) = weighted_line(z, np.log(a[idx]), np.maximum(ase[idx] / a[idx], 1e-6))
            kz = -b
            prop.update(c_phase=omega / kz, c_phase_se=omega * bse / kz ** 2, k_z=kz, alpha=al, alpha_se=alse,
                        z_lo_km=z[0] / 1e3, z_hi_km=z[-1] / 1e3)
        ipk = int(np.nanargmax(np.where(resolved[i], a, np.nan))) if np.any(np.isfinite(a) & resolved[i]) else 0
        ref = [int(np.argmin(np.abs(alt / 1e3 - zk))) for zk in REF_ALT_KM]
        wrow = {"run": alias, "species": sp, **prop, "c_s": cs, "c_phase_theory": c_th, "H_km": H / 1e3,
                "omega": omega, "omega_acoustic": wa, "alpha_ideal": 1 / (2 * H),
                "damping_rate": (1 / (2 * H) - prop["alpha"]) if np.isfinite(prop["alpha"]) else np.nan,
                "ur_peak_amp": float(a[ipk]), "ur_peak_alt_km": float(alt[ipk] / 1e3), "z_exo_km": z_exo}
        for zk, j in zip(REF_ALT_KM, ref):
            for name in ("ur", "dn", "dT"):
                wrow[f"{name}_amp_{zk:.0f}"] = float(ds[f"{name}_amp"].values[i, j])
                wrow[f"{name}_amp_se_{zk:.0f}"] = float(ds[f"{name}_amp_se"].values[i, j])
                wrow[f"{name}_null_{zk:.0f}"] = float(ds[f"{name}_null_amp"].values[i, j])
                wrow[f"{name}_phase_{zk:.0f}"] = float(ds[f"{name}_phase"].values[i, j])
        wave_rows.append(wrow)
        # heating
        dT, dTse = ds["dT_cyclemean"].values[i], ds["dT_cyclemean_se"].values[i]
        dTs = ds["dT_cyclemean_vs_spin"].values[i]
        exo_ok = above & resolved[i] & np.isfinite(dT)
        okT = resolved[i] & np.isfinite(dT)
        dT_exo = float(np.interp(z_exo, alt[okT] / 1e3, dT[okT], left=np.nan, right=np.nan)) if okT.sum() > 1 else np.nan
        dT_exo_se = float(np.interp(z_exo, alt[okT] / 1e3, dTse[okT], left=np.nan, right=np.nan)) if okT.sum() > 1 else np.nan
        hrow = {"run": alias, "species": sp, "z_exo_km": z_exo, "dT_exo": dT_exo, "dT_exo_se": dT_exo_se,
                "dT_exosphere_mean": float(np.nanmean(dT[exo_ok])) if exo_ok.any() else np.nan,
                "dT_exosphere_mean_se": float(np.sqrt(np.nansum(dTse[exo_ok] ** 2)) / max(exo_ok.sum(), 1)) if exo_ok.any() else np.nan,
                "dT_exosphere_mean_vs_spin": float(np.nanmean(dTs[exo_ok])) if exo_ok.any() else np.nan,
                "n_exo_cells": int(exo_ok.sum()), "baseline_rule": base["fit"].rule}
        for zk, j in zip(REF_ALT_KM, ref):
            hrow[f"dT_{zk:.0f}"] = float(dT[j])
            hrow[f"dT_se_{zk:.0f}"] = float(dTse[j])
            hrow[f"dn_{zk:.0f}"] = float(ds["dn_cyclemean"].values[i, j])
            hrow[f"dTrad_minus_dTtan_{zk:.0f}"] = float(ds["dTrad_cyclemean"].values[i, j] - ds["dTtan_cyclemean"].values[i, j])
        heat_rows.append(hrow)
        # ring-down timescales: u_r amplitude at reference altitudes, exospheric cycle-mean dT
        rrow = {"run": alias, "species": sp, "period_s": P}
        pp = np.arange(nF - 1, nF + N_RING)
        for zk, j in zip(REF_ALT_KM, ref):
            A = per_amp[pp, i, j]
            fl = floor[i, j]
            Adeb = np.sqrt(np.maximum(A ** 2 - fl ** 2, 0))
            rrow[f"ur_ratio_first_ring_{zk:.0f}"] = float(A[1] / A[0]) if A[0] > 0 else np.nan
            if np.isfinite(fl) and A[0] > 3 * fl and not is_control:
                r = decay_fit((pp + 0.5) * P, Adeb, np.full(len(pp), max(fl, 1e-9)), nF * P)
            else:
                r = {"tau": np.nan, "tau_se": np.nan, "resolved": 0}
            rrow[f"tau_ur_{zk:.0f}"] = r["tau"] if r["resolved"] else np.nan
            rrow[f"tau_ur_se_{zk:.0f}"] = r["tau_se"] if r["resolved"] else np.nan
            rrow[f"tau_ur_resolved_{zk:.0f}"], rrow[f"tau_ur_fit_{zk:.0f}"] = r["resolved"], r["tau"]
        dTp = ds["dT_mean_period"].values[:, i]
        ex = above & resolved[i]
        if ex.any() and not is_control:
            yv = np.nanmean(dTp[pp][:, ex], axis=1)
            sv = np.nanstd(delta["dT"][np.isin(pidx, pp)][:, i][:, ex].mean(1)) / math.sqrt(NP) * np.ones(len(pp))
            r = decay_fit((pp + 0.5) * P, yv, sv, nF * P)
        else:
            r = {"tau": np.nan, "tau_se": np.nan, "resolved": 0}
        rrow["tau_dT_exosphere"] = r["tau"] if r["resolved"] else np.nan
        rrow["tau_dT_exosphere_se"] = r["tau_se"] if r["resolved"] else np.nan
        rrow["tau_dT_resolved"], rrow["tau_dT_exosphere_fit"] = r["resolved"], r["tau"]
        er = next(e for e in esc_rows if e["species"] == sp)
        rrow["tau_escape"], rrow["tau_escape_se"], rrow["tau_escape_resolved"] = er["tau_ringdown"], er["tau_ringdown_se"], er["tau_resolved"]
        ring_rows.append(rrow)
    # common elapsed-time window: escape enhancement and exobase heating at equal forcing duration
    end = sch.elapsed.values
    cw = common_window(P)
    ci = np.where((end - d >= cw[0] - 1e-6) & (end <= cw[1] + 1e-6) & (sch.forcing.values == 1))[0]
    cbase = spin_b if is_control else control_baseline(ctrl, csch, *cw)
    prof_c = profiles.time_mean_profiles(run, float(run.time[ci[0]] - 0.5 * d), float(run.time[ci[-1]] + 1e-6),
                                         fields=PROFILE_FIELDS)
    dTc = (prof_c["temp_trans_corr"] - cbase.prof["temp_trans_corr"]).transpose("species", "radial").values
    dTc_se = np.hypot(prof_c["temp_trans_corr_sem"], cbase.prof["temp_trans_corr_sem"]).transpose("species", "radial").values
    for i, sp in enumerate(species):
        okT = resolved[i] & np.isfinite(dTc[i])
        hr = heat_rows[i]
        hr["dT_exo_common"] = float(np.interp(z_exo, alt[okT] / 1e3, dTc[i][okT], left=np.nan, right=np.nan)) if okT.sum() > 1 else np.nan
        hr["dT_exo_common_se"] = float(np.interp(z_exo, alt[okT] / 1e3, dTc_se[i][okT], left=np.nan, right=np.nan)) if okT.sum() > 1 else np.nan
        er = esc_rows[i]
        nf = float(n_esc[i, ci].sum())
        Rf = nf * W[i] / (len(ci) * d)
        kc = float(cbase.esc_count[cbase.run.species.index(sp)])
        Rc = kc * float(cbase.run.weight.sel(species=sp)) / cbase.duration
        er["R_common"], er["R_control_common"] = Rf, Rc
        if Rc > 0 and nf > 0:
            er["enhancement_common"] = Rf / Rc - 1
            er["enhancement_common_se"] = (Rf / Rc) * math.hypot(1 / math.sqrt(nf), 1 / math.sqrt(max(kc, 1)))
        else:
            er["enhancement_common"] = er["enhancement_common_se"] = np.nan
        er["common_baseline_rule"] = cbase.rule
    ds["dT_cyclemean_common"] = (("species", "radial"), np.where(resolved, dTc, np.nan),
                                 {"units": "K", "long_name": "Mean dT over the common elapsed window minus baseline"})
    ds["dT_cyclemean_common_se"] = (("species", "radial"), np.where(resolved, dTc_se, np.nan), {"units": "K"})
    ds["esc_enhancement_common"] = ("species", [r["enhancement_common"] for r in esc_rows])
    ds.attrs.update(common_window_s=f"{cw[0]:.3f},{cw[1]:.3f}", common_baseline_rule=cbase.rule,
                    common_n_outputs=len(ci))
    tables["wave"], tables["heating"], tables["ringdown"] = wave_rows, heat_rows, ring_rows
    for k in ("c_phase", "c_phase_se", "alpha", "damping_rate", "ur_peak_amp", "ur_peak_alt_km"):
        ds[f"prop_{k}"] = ("species", [r[k] for r in wave_rows])
    for k in ("dT_exo", "dT_exo_se", "dT_exosphere_mean", "dT_exosphere_mean_se"):
        ds[f"heat_{k}"] = ("species", [r[k] for r in heat_rows])

    ds = ds.assign_coords(altitude=("radial", alt, {"units": "m"}), radial=run.macro["radial"].values,
                          period=np.arange(nF + N_RING), phase=np.arange(NP),
                          ref_alt_km=("ref_alt", np.array(REF_ALT_KM)))
    srcs = [run_file(alias), run_file(SPIN[atm])] + ([] if is_control else [run_file(CONTROL[atm])])
    ds.attrs.update(tiers.analysis_attrs(f"{alias}: drift_mb drifting-Maxwellian forcing response", srcs, SCRIPT))
    ds.attrs.update(run_alias=alias, atmosphere=atm, s_O=float(meta["s_O"]), s_H2=float(meta["s_H2"]),
                    w_mps=float(meta["w_mps"]), period=P, omega=omega, n_force_periods=nF, n_ringdown_periods=N_RING,
                    fit_periods=f"{FIT_PERIODS[0]}-{FIT_PERIODS[1] - 1}", min_part=MIN_PART, snr_min=SNR_MIN,
                    interval_sinc=SINC, restart_rate_factor=float(run.attrs["restart_rate_factor"]),
                    baseline_source=base["fit"].source, baseline_fit_rule=base["fit"].rule,
                    baseline_fit_window=f"{base['fit'].t0:.3f},{base['fit'].t1:.3f}",
                    baseline_ringdown_rule=base["ringdown"].rule, spin_window_s=SPIN_WINDOW,
                    z_exo_O_km=z_exo, c_s=cs, H_scale_m=H, omega_acoustic=wa, c_phase_theory=c_th,
                    phase_convention="response A cos(omega t' + phi), phi relative to w' sin(omega t')")
    summary.update({"n_outputs": int(run.macro.sizes["time"]), "z_exo_O_km": z_exo})
    return ds, {"summary": summary, **tables}


# ------------------------------------------------------------------ spin-ups
def analyse_spin(alias: str, meta: dict) -> tuple[xr.Dataset, dict]:
    run = open_run(alias)
    b = spin_baseline(run)
    prof = b.prof
    exo = profiles.exobase(prof)
    n_esc = escape_counts(run)
    W = run.weight.values
    d = avg_dt(run)
    inv = (run.macro["num_dens"] * run.cell_volume()).sum("radial").transpose("species", "time").values
    eh = run.diagnostics["diagnostics_escape_speed_histogram"]
    ds = xr.Dataset({
        **{f"profile_{f}": (("species", "radial"), prof[f].transpose("species", "radial").values) for f in PROFILE_FIELDS},
        **{f"profile_{f}_sem": (("species", "radial"), prof[f + "_sem"].transpose("species", "radial").values) for f in PROFILE_FIELDS},
        "exobase_km": ("species", exo.values, {"units": "km"}),
        "escape_rate": (("species", "time"), n_esc * W[:, None] / d, {"units": "s-1"}),
        "escape_npart": (("species", "time"), n_esc),
        "N_phys": (("species", "time"), inv),
        "escape_vdf_window": (("species", "speed_bin"), b.esc_hist),
        "escape_window_rate": ("species", b.esc_count * W / b.duration, {"units": "s-1"}),
        "escape_window_npart": ("species", b.esc_count),
    }, coords={"species": run.species, "time": run.time, "radial": run.macro["radial"].values,
               "altitude": ("radial", run.macro["altitude"].values), "speed": ("speed_bin", eh["speed"].values)})
    ds.attrs.update(tiers.analysis_attrs(f"{alias}: drift_mb spin-up steady state", [run_file(alias)], SCRIPT))
    ds.attrs.update(run_alias=alias, atmosphere=meta["atmosphere"], window_t0=b.t0, window_t1=b.t1,
                    spin_window_s=SPIN_WINDOW, restart_rate_factor=float(run.attrs["restart_rate_factor"]))
    summary = {"run": alias, "atmosphere": meta["atmosphere"], "s_O": 0.0, "s_H2": 0.0, "w_mps": 0.0, "period_s": np.nan,
               "is_control": 0, "restart_rate_factor": float(run.attrs["restart_rate_factor"]),
               "baseline_fit_rule": "", "baseline_ringdown_rule": "", "baseline_source": "",
               "n_outputs": int(run.macro.sizes["time"]), "z_exo_O_km": float(exo.sel(species="O"))}
    for i, sp in enumerate(run.species):
        summary[f"escape_rate_{sp}"] = float(b.esc_count[i] * W[i] / b.duration)
    return ds, {"summary": summary}


# ------------------------------------------------------------------ output
def _attr(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def write(ds: xr.Dataset, path: Path):
    ds.attrs = {k: _attr(v) for k, v in ds.attrs.items() if v is not None}
    for v in ds.variables.values():
        v.attrs = {k: _attr(x) for k, x in v.attrs.items() if x is not None}
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(path)


def cross_run(results: dict, runs: list) -> xr.Dataset:
    """Scalars (run x species) from the per-run tables."""
    sp_all = ["O", "H2"]
    aliases = [a for a in runs if a in results]
    S = pd.DataFrame([results[a]["summary"] for a in aliases]).set_index("run")
    data = {c: ("run", S[c].values) for c in ("s_O", "s_H2", "w_mps", "period_s", "is_control", "restart_rate_factor")}
    data["atmosphere"] = ("run", np.array([str(x) for x in S["atmosphere"]], dtype=object))

    def put(table, cols, prefix, window=None):
        for c in cols:
            arr = np.full((len(aliases), 2), np.nan)
            for i, a in enumerate(aliases):
                for r in results[a].get(table, []):
                    if window is not None and r.get("window", r.get("group")) != window:
                        continue
                    if r["species"] in sp_all:
                        arr[i, sp_all.index(r["species"])] = r.get(c, np.nan)
            data[f"{prefix}{c}"] = (("run", "species"), arr)

    put("escape", ("R_fit", "R_fit_se", "R_control", "enhancement", "enhancement_se", "enhancement_vs_spin",
                   "enhancement_vs_spin_se", "mod_amp", "mod_amp_se", "mod_phase_rel_forcing", "tau_ringdown",
                   "tau_ringdown_se", "mean_speed_fit", "mean_speed_control", "vdf_chi2_p_vs_control",
                   "enhancement_common", "enhancement_common_se"), "esc_")
    put("wave", ("c_phase", "c_phase_se", "c_phase_theory", "alpha", "alpha_se", "alpha_ideal", "damping_rate",
                 "ur_peak_amp", "ur_peak_alt_km", "ur_amp_150", "ur_amp_se_150", "dn_amp_150", "dT_amp_150",
                 "ur_amp_240", "dT_amp_240"), "wave_")
    put("heating", ("dT_exo", "dT_exo_se", "dT_exosphere_mean", "dT_exosphere_mean_se", "dT_exosphere_mean_vs_spin",
                    "dT_150", "dT_300", "dT_exo_common", "dT_exo_common_se"), "heat_")
    put("ringdown", ("tau_ur_150", "tau_ur_se_150", "ur_ratio_first_ring_150", "tau_dT_exosphere", "tau_escape"), "ring_")
    put("budget", ("inj_over_theory", "mom_over_theory", "energy_over_theory", "Ndot_inj_ratio_control",
                   "Edot_inj_ratio_control", "Ndot_net_ratio_control", "Edot_net_ratio_control",
                   "cycle_mean_flux_ratio_th"), "budget_fit_", window="fit")
    put("budget", ("inj_over_theory", "Ndot_net", "Edot_net", "Ndot_net_control", "Edot_net_control"),
        "budget_ringdown_", window="ringdown")
    vdf_cols = ("ks_ratio", "chi2_p")
    for c in vdf_cols:
        arr = np.full((len(aliases), 2), np.nan)
        for i, a in enumerate(aliases):
            rows = [r for r in results[a].get("vdf", []) if r["group"] != "ringdown"]
            for sp in sp_all:
                v = [r[c] for r in rows if r["species"] == sp]
                if v:
                    arr[i, sp_all.index(sp)] = max(v) if c == "ks_ratio" else min(v)
        data[f"vdf_{'max' if c == 'ks_ratio' else 'min'}_{c}"] = (("run", "species"), arr)
    ds = xr.Dataset(data, coords={"run": aliases, "species": sp_all})
    srcs = [tiers.analysis_path(CAMPAIGN, a) for a in aliases]
    ds.attrs.update(tiers.analysis_attrs("drift_mb: cross-run summary", srcs, SCRIPT))
    ds.attrs.update(fit_periods=f"{FIT_PERIODS[0]}-{FIT_PERIODS[1] - 1}", min_part=MIN_PART, spin_window_s=SPIN_WINDOW)
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="aliases (default: the whole campaign)")
    args = ap.parse_args()
    mx = matrix()
    runs = [r.alias for r in registry.campaign(CAMPAIGN, include_alternates=True)]
    todo = args.runs or runs
    cache, results = {}, {}
    for alias in todo:
        meta = mx[alias]
        meta["atmosphere"] = meta["atmosphere"]
        if meta["stage"] == "spin":
            ds, res = analyse_spin(alias, meta)
        else:
            ds, res = analyse_forced(alias, meta, cache)
        path = tiers.analysis_path(CAMPAIGN, alias)
        write(ds, path)
        results[alias] = res
        print(f"{alias}: wrote {path.name}")
    tabdir = output_dir(CAMPAIGN, "tables")
    pd.DataFrame([results[a]["summary"] for a in todo]).to_csv(tabdir / "run_summary.csv", index=False)
    for key, name in (("vdf", "vdf_moments"), ("budget", "flux_budget"), ("escape", "escape_summary"),
                      ("wave", "wave_response"), ("heating", "heating"), ("ringdown", "ringdown")):
        rows = [r for a in todo for r in results[a].get(key, [])]
        if rows:
            pd.DataFrame(rows).to_csv(tabdir / f"{name}.csv", index=False)
    if not args.runs:
        write(cross_run(results, runs), tiers.analysis_path(CAMPAIGN, CAMPAIGN))
        print(f"cross-run: wrote {CAMPAIGN}_analysis.nc; tables in {tabdir}")


if __name__ == "__main__":
    main()
