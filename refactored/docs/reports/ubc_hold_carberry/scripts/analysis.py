"""Data layer of the ubc_hold_carberry report: a read-only loader of analysis-tier files (docs/CONVENTIONS.md).

Both make_numbers.py and make_figures.py import this module; neither opens a run file.

    from analysis import get; A = get()   # A.runs, A.status(), A.reduce(run), A.hold(run), A.analytic(), ...

Inputs (produced by the campaign scripts, never by this report):
  verification + campaign  outputs/h2_hold/analysis/<alias>_analysis.nc, h2_hold_analysis.nc
                           (campaigns/h2_hold/compute_analysis.py; registry campaign h2_hold: E7a..E9 and
                           Hold_H2_U{1,2,5}_z{2500,10000,40000}_dr250)
  sweep                    outputs/h2_ubc_rtop/analysis/<id>_analysis.nc (campaigns/h2_ubc_rtop/compute_analysis.py,
                           fixed 100-300 ks window; H2 runs only)

A run without an analysis file (queued, running, or analysed before its run file was rebuilt) is recorded in
A.status() with the reason, and every number that needs it is written as \\pending by make_numbers.py.
Run-folder metadata (run_meta.txt, BallisticHold file times) is read only for wall times / progress.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from dsmclib import escape, exosphere, registry, tiers
from dsmclib.constants import AMU, G_NEWTON, K_BOLTZMANN
from dsmclib.ncrun import RUN_SUFFIX

warnings.simplefilter("ignore", RuntimeWarning)

REPORT = Path(__file__).resolve().parents[1]
SWEEP = Path.home() / "MarsAtmos-H2UBC-sweep"
CAMP = Path.home() / "MarsAtmos-H2Hold-campaign"
SOLVER_SPECIES = Path("/Users/josepc/MarsAtmos-UBC-explicit/Source/MOD_Species.f90")
HOLD_CAMPAIGN, SWEEP_CAMPAIGN = "h2_hold", "h2_ubc_rtop"

UBC_CODE = {"UBC_REFLECT": 1, "UBC_BALLISTIC": 2, "UBC_FREE_ESCAPE": 3, "UBC_BALLISTIC_EXPLICIT": 5}
VERIF = {  # name -> (UBC code, role)
    "E7a_kepler": (2, "Kepler hold, fixed binary"),
    "E7b_explicit": (5, "explicit RK4, fixed binary"),
    "E7c_explicit_dbg": (5, "explicit RK4, -fcheck=all build"),
    "E7d_explicit_restart": (5, "explicit RK4, restarted at 15 ks"),
    "E8_main_kepler": (2, "Kepler hold, main (bug 2)"),
    "E9_reflect": (1, "reflect, fixed binary"),
}
T0_VER = 3.0e4          # steady window start of the verification runs (campaigns/h2_hold/compute_analysis.py)
RESTART_WIN = (1.5e4, 3.0e4)
SP = "H2"

# ------------------------------------------------------------------ Carberry Mogan et al. (2021), PSS
# Parameters quoted in the text / Table 1 footnote, and key values read off Fig. 1 and Fig. 3 by eye
# (labelled 'digitised' wherever they are shown). Everything derived from them is computed below.
NV_HOLD = 2500  # velocity-grid size for the collisionless held count and escape at each z_top

PAPER = {
    "body": "Callisto", "M": 1.08e23, "r0_km": 2410.0, "n0_cm3": 4e8, "T0": 155.0, "m_amu": 2.02,
    "Kn0": 2.5e-3, "lambda_quoted": 4.7, "phiJ0_quoted": 4.9e29,
    "M_J": 1.898e27, "d_JC_rJ": 26.3, "r_J_km": 71492.0, "hill_quoted": 20.8,
    "t_orb": 1.44e6, "dt_lo": 0.75, "dt_hi": 3.0, "n_steps": 1e7, "avg_steps": 1e5,
    "rmax_lo": 2.0, "rmax_hi": 40.0,
    "fig1_lambda": 10.0, "fig1_Kn0": 0.01,
    # Table 1 / text: Phi at 1 t_orb for UBC (1), (2), (3) [s-1]; exobase radii r_x/r0 legible in Table 1 + Fig. 2
    "Phi_1torb": {1: 1.4e29, "HS": 1.7e29, 3: 1.1e29},
    "rx_Kn": {1: 12.0, "HS": 7.51, 3: 8.85}, "rx_Kn0_UBC1": 6.14,
    # Fig. 3, solid (Phi(UBC3)/Phi(UBC1)) and dashed (ballistic rate / return rate) vs t/t_orb, digitised
    "fig3_t": [0.75, 1.0, 2.0, 3.0, 4.0, 5.0],
    "fig3_phi_ratio": [0.73, 0.745, 0.783, 0.81, 0.83, 0.845],
    "fig3_in_ret": [2.40, 2.05, 1.57, 1.42, 1.33, 1.28],
    "reflect_overestimate": 0.25,
}


def paper_derived():
    p = PAPER
    m = p["m_amu"] * AMU
    r0 = p["r0_km"] * 1e3
    n0 = p["n0_cm3"] * 1e6
    lam = G_NEWTON * p["M"] * m / (K_BOLTZMANN * p["T0"] * r0)
    vbar = np.sqrt(8 * K_BOLTZMANN * p["T0"] / (np.pi * m))
    phi = np.pi * r0**2 * n0 * vbar * (1 + lam) * np.exp(-lam)
    hill = (p["M"] / (3 * p["M_J"])) ** (1 / 3) * p["d_JC_rJ"] * p["r_J_km"] * 1e3
    return {"lambda0": lam, "phiJ0": phi, "hill_r0": hill / r0, "n0_m3": n0, "r0": r0, "m": m}




# ------------------------------------------------------------------ small utilities
_NAME_RE = re.compile(r"Output(\d+)_Proc(\d+)_BallisticHold\.ascii$")


def _hold_files(run_dir: Path):
    """Raw BallisticHold files (only used for progress / timing of runs still going)."""
    d = Path(run_dir) / "Results" / "Diagnostics"
    if not d.is_dir():
        return []
    fs = [f for f in d.glob("Output*_Proc*_BallisticHold.ascii") if _NAME_RE.search(f.name)]
    return sorted(fs, key=lambda f: tuple(int(g) for g in _NAME_RE.search(f.name).groups()))


def read_meta(run_dir: Path) -> dict:
    f = Path(run_dir) / "run_meta.txt"
    if not f.exists():
        return {}
    return dict(re.findall(r"^(\w+):\s+(.*)$", f.read_text(), re.M))


def _iso(s):
    try:
        return datetime.fromisoformat(s.strip())
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ discovery
@dataclass
class Run:
    id: str
    group: str                      # verification | sweep | campaign
    ubc: int
    z_top_km: float                 # nominal
    dr: int
    dir: Path
    alias: str = ""
    campaign: str = ""
    t_end_expected: float = np.nan
    run_file: Optional[Path] = None     # the analysis-tier file this report reads
    status: str = "missing"             # complete | running | queued | missing | stale
    reason: str = ""
    hold_files: list = field(default_factory=list)   # non-empty iff the analysis file has the hold series
    meta: dict = field(default_factory=dict)
    z_top_actual_km: float = np.nan

    @property
    def key(self):
        return f"{self.group}:{self.id}"

    @property
    def has_run(self):
        return self.run_file is not None

    @property
    def has_hold(self):
        return len(self.hold_files) > 0

    @property
    def wall_s(self):
        a, b = _iso(self.meta.get("started", "")), _iso(self.meta.get("finished", ""))
        return (b - a).total_seconds() if a and b else np.nan


def _pick_analysis_file(campaign: str, alias: str, run_dir: Path):
    """(analysis path or None, status, reason, has_hold)."""
    if not run_dir.is_dir():
        return None, "queued", "run directory not created yet (queued)", False
    res = run_dir / "Results"
    run_file = res / f"{alias}{RUN_SUFFIX}"
    finished = (run_dir / "COMPLETE.ascii").is_file()
    f = tiers.analysis_path(campaign, alias)
    if not finished:
        return None, "running", "no COMPLETE.ascii (running)", False
    if not run_file.is_file():
        return None, "stale", f"finished, no {run_file.name} yet (python -m dsmclib.runfile)", False
    if not f.is_file():
        return None, "stale", f"finished, not analysed yet (campaigns/{campaign}/compute_analysis.py)", False
    with xr.open_dataset(f) as ds:
        srcs = str(ds.attrs.get("source_files", "")).split(",")
        has_hold = "hold_time" in ds.dims
    if str(run_file) not in srcs:
        return None, "stale", f"{f.name} was not computed from {run_file.name}", False
    if run_file.stat().st_mtime > f.stat().st_mtime:
        return None, "stale", f"{run_file.name} is newer than {f.name}; re-run compute_analysis.py", False
    return f, "complete", "", has_hold


def discover() -> list[Run]:
    runs = []
    hold_alias = {r.alias: r for r in registry.campaign(HOLD_CAMPAIGN, include_alternates=True)}
    for name, (code, _) in VERIF.items():
        rr = hold_alias[name]
        d = rr.path.parent
        r = Run(name, "verification", code, 2500.0, 250, d, alias=name, campaign=HOLD_CAMPAIGN)
        r.meta = read_meta(d)
        r.run_file, r.status, r.reason, hh = _pick_analysis_file(HOLD_CAMPAIGN, name, d)
        r.hold_files = [r.run_file] if hh else []
        runs.append(r)
    for root, group, campaign, prefix in ((SWEEP, "sweep", SWEEP_CAMPAIGN, ""), (CAMP, "campaign", HOLD_CAMPAIGN, "Hold_")):
        mx = pd.read_csv(root / "matrix.csv")
        mx = mx[mx.atmosphere == "H2"]
        for row in mx.itertuples():
            d = root / "runs" / row.id
            alias = prefix + row.id
            r = Run(row.id, group, int(row.ubc), float(row.z_top_km), int(row.dr_max_km), d, alias=alias,
                    campaign=campaign, t_end_expected=float(row.t_end_s), z_top_actual_km=float(row.z_top_actual_km))
            r.meta = read_meta(d)
            r.run_file, r.status, r.reason, hh = _pick_analysis_file(campaign, alias, d)
            r.hold_files = [r.run_file] if hh else []
            runs.append(r)
    return runs


# ------------------------------------------------------------------ the analysis object
class Analysis:
    def __init__(self):
        self.runs = discover()
        self.by_id = {r.key: r for r in self.runs}
        self.notes = []

    # ---------------- lookup
    def v(self, name) -> Run:
        return self.by_id[f"verification:{name}"]

    def get(self, group, ubc, z, dr=250) -> Optional[Run]:
        for r in self.runs:
            if r.group == group and r.ubc == ubc and r.z_top_km == z and r.dr == dr:
                return r
        return None

    def select(self, group=None, ubc=None, dr=None, need_run=False, need_hold=False):
        out = [r for r in self.runs if (group is None or r.group == group) and (ubc is None or r.ubc == ubc)
               and (dr is None or r.dr == dr)]
        if need_run:
            out = [r for r in out if r.has_run]
        if need_hold:
            out = [r for r in out if r.has_hold]
        return out

    def status(self) -> dict:
        out = {}
        for r in self.runs:
            out[r.key] = {"id": r.id, "alias": r.alias, "group": r.group, "ubc": r.ubc, "z_top_km": r.z_top_km,
                          "dr_km": r.dr, "status": r.status, "reason": r.reason,
                          "analysis_file": str(r.run_file) if r.run_file else None,
                          "has_hold_series": r.has_hold, "started": r.meta.get("started"),
                          "finished": r.meta.get("finished")}
        return out

    @lru_cache(maxsize=None)
    def _open(self, key) -> xr.Dataset:
        with xr.open_dataset(self.by_id[key].run_file) as ds:
            return ds.load()

    def source_params(self) -> dict:
        """Lower-boundary parameters (n0, T0, lambda0, phi_J0, r0, m, M, R, W, dt) of the reference H2 run."""
        a = self.analytic().attrs
        return {k[len("source_"):]: a[k] for k in a if k.startswith("source_") and k not in ("source_files", "source_sha256")}

    # ---------------- per-run reduction in the report's names (p_*, *_t, attrs t0/esc_rate/...)
    def reduce(self, r: Run) -> Optional[xr.Dataset]:
        if not r.has_run:
            return None
        return self._reduce(r.key)

    @lru_cache(maxsize=None)
    def _reduce(self, key) -> xr.Dataset:
        r = self.by_id[key]
        src = self._open(key)
        if r.campaign == SWEEP_CAMPAIGN:
            src = src.sel(species=SP)
            a = src.attrs
            t = src.time.values
            dtav = np.diff(t, prepend=0.0)
            fields = [v[len("profile_"):] for v in src.data_vars if v.startswith("profile_") and not v.endswith("_sem")
                      and f"{v}_sem" in src]
            W = float(src.particle_weight)
            attrs = dict(t0=float(a["profile_window_t0"]), t1=float(a["profile_window_t1"]),
                         esc_rate=float(src.dsmc_escape_rate), esc_npart=int(src.dsmc_escape_npart),
                         esc_ul95=float(src.dsmc_escape_rate_ul95), r0=float(a["r0_m"]), r_top=float(a["r_top_m"]),
                         W=W, m=float(src.species_mass), M=float(a["planet_mass"]), R=float(a["planet_radius"]),
                         steady=int(src.steady), drift_per_ks=float(src.drift_per_ks), ubc_attr=a.get("ubc", ""),
                         window_rule="fixed 100-300 ks (h2_ubc_rtop)")
        else:
            a = src.attrs
            t = src.time.values
            dtav = src.avg_time_interval.values
            fields = [v[len("profile_"):] for v in src.data_vars if v.startswith("profile_") and not v.endswith("_sem")]
            attrs = dict(t0=float(a["window_t0"]), t1=float(a["window_t1"]), esc_rate=float(a["dsmc_escape_rate"]),
                         esc_npart=int(a["dsmc_escape_npart"]), esc_ul95=float(a["dsmc_escape_rate_ul95"]),
                         r0=float(a["r0_m"]), r_top=float(a["r_top_m"]), W=float(a["particle_weight"]),
                         m=float(a["species_mass"]), M=float(a["planet_mass"]), R=float(a["planet_radius"]),
                         steady=int(a["steady"]), drift_per_ks=float(a["drift_per_ks"]), ubc_attr=a["ubc_attr"],
                         window_rule=a["window_rule"], wall_s=float(a["wall_s"]))
        held = src.N_ball_phys.values
        ds = xr.Dataset(
            {f"p_{k}": ("radial", src[f"profile_{k}"].values) for k in fields}
            | {f"p_{k}_sem": ("radial", src[f"profile_{k}_sem"].values) for k in fields}
            | {"altitude": ("radial", src.altitude.values), "radius": ("radial", src.radial.values),
               "esc_rate_t": ("time", src.escape_rate.values), "esc_npart_t": ("time", src.escape_npart.values),
               "held_t": ("time", held), "N_phys_t": ("time", src.N_phys.values), "dt_avg_t": ("time", dtav),
               "inject_t": ("time", src.inject_rate.values), "return_t": ("time", src.return_rate.values)}
            | ({"analytic_num_dens": ("radial", src.analytic_num_dens.values),
                "analytic_temp_trans": ("radial", src.analytic_temp_trans.values)} if "analytic_num_dens" in src else {}),
            coords={"time": t, "radial": np.arange(src.sizes["radial"])})
        ds.attrs.update(attrs, id=r.id, group=r.group, ubc=r.ubc, n_cells=int(src.sizes["radial"]),
                        held_mean=float(np.nanmean(held[t > attrs["t0"]])), source=str(r.run_file))
        return ds

    # ---------------- BallisticHold series (from the h2_hold analysis file)
    def hold(self, r: Run) -> Optional[xr.Dataset]:
        if not r.has_hold:
            return None
        src = self._open(r.key)
        h = src[[v for v in src.data_vars if v.startswith("hold_")]]
        h = h.rename({v: v[len("hold_"):] for v in h.data_vars}).rename({"hold_time": "time"})
        h["part_weight"] = float(src.attrs["hold_part_weight"])
        h.attrs.update(invariant_violation=float(src.attrs["hold_invariant_violation"]), id=r.id,
                       n_files=int(src.attrs["hold_source_count"]))
        return h

    # ---------------- collisionless analytic references (h2_hold cross-run file)
    @lru_cache(maxsize=None)
    def analytic(self) -> xr.Dataset:
        with xr.open_dataset(tiers.analysis_path(HOLD_CAMPAIGN, HOLD_CAMPAIGN)) as ds:
            ds = ds.load()
        ds.attrs["hill"] = float(ds.attrs["hill_m"])
        return ds

    def _ver_top_km(self):
        for name in VERIF:
            r = self.v(name)
            if r.has_run:
                d = self.reduce(r)
                return round((d.attrs["r_top"] - d.attrs["R"]) / 1e3, 1)
        return 2627.3

    def ztop_index(self, an, z_actual_km):
        return int(np.argmin(np.abs(an.ztop.values - z_actual_km)))

    def verification(self) -> dict:
        return json.loads(self.analytic().attrs["verification_json"])

    # ---------------- per-interval escape / hold series in common form
    def escape_series(self, r: Run, source="auto") -> Optional[xr.Dataset]:
        """Per-interval escape counts and rates: from BallisticHold (n_esc_direct + n_esc_orbit) when present,
        otherwise from the run's escape series. Dataset(time): npart, rate, dt."""
        if source in ("auto", "hold") and r.has_hold:
            h = self.hold(r)
            n = (h.n_esc_direct + h.n_esc_orbit).values.astype(float)
            dt = h.avg_time_interval.values
            return xr.Dataset({"npart": ("time", n), "rate": ("time", n * float(h.part_weight) / dt),
                               "dt": ("time", dt)}, coords={"time": h.time.values}, attrs={"src": "hold"})
        if source in ("auto", "run") and r.has_run:
            d = self.reduce(r)
            return xr.Dataset({"npart": ("time", d.esc_npart_t.values), "rate": ("time", d.esc_rate_t.values),
                               "dt": ("time", d.dt_avg_t.values)}, coords={"time": d.time.values},
                              attrs={"src": "run"})
        return None

    # ---------------- cost / progress (run_meta.txt; BallisticHold file times for runs still going)
    def cost(self, r: Run) -> dict:
        out = {"wall_s": r.wall_s, "sim_s": np.nan, "rate": np.nan, "eta_s": np.nan, "status": r.status}
        st = _iso(r.meta.get("started", ""))
        if np.isfinite(r.wall_s):
            out["sim_s"] = r.t_end_expected if np.isfinite(r.t_end_expected) else np.nan
            out["rate"] = out["sim_s"] / r.wall_s
            return out
        files = _hold_files(r.dir)
        if st is not None and files:
            last = max(files, key=lambda f: f.stat().st_mtime)
            try:
                t_last = float(np.loadtxt(last, comments="#", ndmin=2)[0, 0])
            except Exception:  # noqa: BLE001
                return out
            el = last.stat().st_mtime - st.timestamp()
            out["sim_s"], out["elapsed_s"] = t_last, el
            out["rate"] = t_last / el if el > 0 else np.nan
            out["eta_s"] = (r.t_end_expected - t_last) / out["rate"] if out["rate"] > 0 else np.nan
        return out

# ------------------------------------------------------------------ physics helpers
def longest_return_time(r_top, r_hill, mu):
    """Time of flight above r_top of the slowest returning orbit: radial, apoapsis at r_hill (a = r_hill / 2)."""
    a = r_hill / 2
    E_top = np.arccos(1 - r_top / a)
    return float(2 * np.sqrt(a**3 / mu) * (np.pi - (E_top - np.sin(E_top))))


def vhs_ref_diameter() -> float:
    """H2 VHS reference diameter from the solver source (MOD_Species.f90, Thermal_H2)."""
    txt = SOLVER_SPECIES.read_text()
    blk = txt[txt.index("SUBROUTINE Thermal_H2("):]
    return float(re.search(r"ref_diam\s*=\s*([\d.]+)d([-+]?\d+)", blk).expand(r"\1e\2"))


def vhs_visc_exp() -> float:
    txt = SOLVER_SPECIES.read_text()
    blk = txt[txt.index("SUBROUTINE Thermal_H2("):]
    return float(re.search(r"visc_exp\s*=\s*([\d.]+)d0", blk).group(1))


def knudsen_source(n0, T0, m, r0, M, d_ref) -> dict:
    """Carberry's Kn0 = l(r0)/r0 and Kn(r0) = l/H(r0), hard-sphere mean free path at the source density."""
    sigma = np.pi * d_ref**2
    mfp = 1.0 / (np.sqrt(2) * n0 * sigma)
    H = K_BOLTZMANN * T0 * r0**2 / (G_NEWTON * M * m)
    return {"mfp": mfp, "H": H, "Kn0": mfp / r0, "KnH": mfp / H, "sigma": sigma}


def exobase_rows(d: xr.Dataset, sp: dict, Phi: float) -> dict:
    """Table-1 quantities at Kn = 1 (local scale height, solver 'knudsen') and Kn0 = 1 (l / r0)."""
    r0 = sp["r0"]
    alt = d.altitude.values / 1e3
    out = {}
    for key, kn in (("Kn", d.p_knudsen.values), ("KnZero", d.p_mean_free_path.values / r0)):
        ok = np.isfinite(kn) & (kn > 0)
        a, lk = alt[ok], np.log(kn[ok])
        above = np.where(lk >= 0)[0]
        if len(above) == 0:
            out[key] = None
            continue
        i = above[0]
        at_bottom = i == 0
        z = a[0] if at_bottom else float(np.interp(0.0, [lk[i - 1], lk[i]], [a[i - 1], a[i]]))
        n = d.p_num_dens.values
        T = d.p_temp_trans_corr.values
        okn = np.isfinite(n) & (n > 0)
        nx = float(np.exp(np.interp(z, alt[okn], np.log(n[okn]))))
        okT = np.isfinite(T) & (T > 0)
        Tx = float(np.interp(z, alt[okT], T[okT]))
        rx = sp["R"] + z * 1e3
        j = escape.jeans_escape(nx, Tx, sp["m"], rx, sp["M"])
        out[key] = {"z_km": z, "rx_r0": rx / r0, "nx_n0": nx / sp["n0"], "Tx_T0": Tx / sp["T0"],
                    "lam_ratio": float(j["lambda"]) / sp["lambda0"], "phiJx_phiJ0": float(j["rate"]) / sp["phiJ0"],
                    "Phi_phiJx": Phi / float(j["rate"]), "at_bottom": int(at_bottom), "lambda_x": float(j["lambda"])}
    return out


def log_bins(t, per_decade=6, t_min=None):
    t = np.asarray(t)
    lo = np.log10(t_min if t_min else t[0] * 0.999)
    hi = np.log10(t[-1] * 1.001)
    return 10 ** np.arange(lo, hi + 1 / per_decade, 1 / per_decade)


def binned_ratio(ta, na, tb, nb, edges):
    """Ratio of summed counts a/b in time bins over the common span; Poisson 1-sigma. Equal weights and intervals
    are assumed within a bin (true for runs of one campaign)."""
    tc, rr, ee = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sa, sb = na[(ta > lo) & (ta <= hi)], nb[(tb > lo) & (tb <= hi)]
        if len(sa) == 0 or len(sa) != len(sb) or sb.sum() <= 0:
            continue
        a, b = sa.sum(), sb.sum()
        rat = a / b
        tc.append(np.sqrt(lo * hi))
        rr.append(rat)
        ee.append(rat * np.sqrt(1 / max(a, 1) + 1 / b))
    return np.array(tc), np.array(rr), np.array(ee)


def analytic_in_ret(an: xr.Dataset, i: int):
    """Collisionless hold-in / return rate ratio vs t for top i: R_in / (R_in - dN_held/dt) once the hold is fed.
    R_in = bound outward crossings of r_top with apoapsis inside r_Hill = (free escape - ballistic) counted rate."""
    r_in = float(an.escz_free_escape[i] - an.escz_ballistic[i])
    t = an.t.values
    H = an.held_t.values[i]
    dH = np.gradient(H, t)
    ret = r_in - dH
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((H > 0) & (ret > 0), r_in / ret, np.nan)
    return t, ratio, r_in


def window_ratio(sa: xr.Dataset, sb: xr.Dataset, t0: float, t1: float = np.inf):
    """Ratio of mean escape rates a/b over the common outputs in (t0, t1], Poisson 1-sigma from the counts."""
    ta, tb = sa.time.values, sb.time.values
    common = np.intersect1d(ta[(ta > t0) & (ta <= t1)], tb[(tb > t0) & (tb <= t1)])
    if len(common) == 0:
        return np.nan, np.nan, 0, 0
    a = sa.sel(time=common)
    b = sb.sel(time=common)
    ka, kb = float(a.npart.sum()), float(b.npart.sum())
    ra = float((a.rate * a.dt).sum() / a.dt.sum())
    rb = float((b.rate * b.dt).sum() / b.dt.sum())
    if rb <= 0 or kb <= 0:
        return np.nan, np.nan, ka, kb
    r = ra / rb
    return r, r * np.sqrt(1 / max(ka, 1) + 1 / kb), ka, kb


def profile_compare(da: xr.Dataset, db: xr.Dataset, field="num_dens", min_part=10):
    """a/b ratio, its 1-sigma and the z-score per cell (batch-means SEMs), cells with >= min_part particles in both."""
    ya, yb = da[f"p_{field}"].values, db[f"p_{field}"].values
    ea, eb = da[f"p_{field}_sem"].values, db[f"p_{field}_sem"].values
    ok = (da.p_num_part.values >= min_part) & (db.p_num_part.values >= min_part) & (yb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok, ya / yb, np.nan)
        err = np.where(ok, ratio * np.hypot(ea / ya, eb / yb), np.nan)
        z = np.where(ok, (ya - yb) / np.hypot(ea, eb), np.nan)
    return ratio, err, z, ok


A = None


def get() -> Analysis:
    global A
    if A is None:
        A = Analysis()
    return A
