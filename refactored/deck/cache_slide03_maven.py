"""Cache the MAVEN pass drawn on the slide 3 prototype (FIG 3m) into small files in deck/data/.

    cd refactored && conda run -n dsmc python deck/cache_slide03_maven.py

Orbit 2521 (16 Jan 2016, the pass on slide 2) and, for comparison only, the deepest cached deep-dip pass
(orbit 3314, 12 Jun 2016). Sources (read-only here):
  track   spacecraft position (MSO x, y, z) and altitude from the reduced MAVEN in situ Key Parameter files
          (~/maven_data/kp_reduced/mvn_kp_insitu_<date>_v26_r01.parquet, 4 s cadence). The along-track angle
          nu is the angle in the orbit plane from the periapsis position vector, negative inbound.
  column  N_tot(z), the inbound background column density above z (O + Ar + CO2, 5th-order log fits, top
          term n_top * H_top) from williamson-replication/outputs/tables/profiles.parquet: the same column
          whose N_tot = 5e18 m^-2 crossing is the exobase in orbits.parquet and on slide 2.
  orbit   periapsis time, altitude, LST, SZA, latitude and z_exo from orbits.parquet.

Knudsen number. With the column scale height H = N/n (= the scale height of an isothermal column),
  Kn = lambda_mfp / H = 1 / (n sigma) / (N / n) = 1 / (sigma N_tot),
which needs no derivative of a fitted profile and is monotonic in altitude. sigma = 1 / 5e18 m^-2 = 2e-19 m^2
makes Kn = 1 exactly the replication's (and slide 2's) exobase. Sensitivity: hard-sphere cross sections
weighted by the local O / Ar / CO2 mix (Bird 1994 VHS d_ref: CO2 5.62, Ar 4.17 A; O 3.0 A assumed).
The outbound leg has no verified CO2 / O densities (NGIMS quality OU), so Kn on both legs is read from the
inbound N_tot(z) at the spacecraft's altitude (horizontal uniformity over the ~45 deg periapsis arc assumed).

Schematic placement (FIG 3m). The slide 3 sector draws an isothermal column with H = 40 km and Kn = 1 at
230 km, so its radius is a log-Kn axis: z_s = 230 + 40 ln Kn. Each track point goes at the radius of its
measured Kn and at sector angle 0.4 nu (deck/fig03m_regimes_maven.py: the 700 km crossings, nu = +-55 deg,
land on the sector corners). The band a point lies in is therefore the band of its measured Kn; its true
altitude is not the schematic's r scale and is quoted separately. The *_nu_pm22 facts are at nu = +-22 deg,
where the pass is at ~225 km, the top of the NGIMS O profile (above it Kn uses the column's top term).

The replication is not a git repository, so the sources are pinned by SHA-256 in slide03_maven_facts.json.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WREP = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/williamson-replication")
TABLES = WREP / "outputs" / "tables"
CONFIG = WREP / "config.toml"
KP = Path.home() / "maven_data" / "kp_reduced"

ORBITS = {2521: "slide", 3314: "deep dip (comparison only)"}
N_EXO = 5.0e18                      # m^-2, williamson-replication config.toml [column] n_exobase
SIGMA = 1.0 / N_EXO                 # m^2: Kn = 1 at N_tot = N_EXO
D_REF = {"O": 3.0e-10, "Ar": 4.17e-10, "CO2": 5.62e-10}   # m; Bird 1994 (O assumed)
# schematic of slide 3 (deck/fig03_regimes.py): Kn = 1 at 230 km, H = 40 km; ALPHA_DEG: nu of the *_nu_pm22 facts
Z_EXO_S, H_S, ALPHA_DEG = 230.0, 40.0, 22.0
WINDOW_S = 1800.0                   # +-30 min of KP around periapsis


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kp_file(t: float) -> Path:
    day = dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y%m%d")
    return KP / f"mvn_kp_insitu_{day}_v26_r01.parquet"


def track(t_peri: float):
    """KP samples within WINDOW_S of periapsis: t - t_peri [s], altitude [km], nu [deg], SZA [deg]."""
    files = sorted({kp_file(t_peri - WINDOW_S), kp_file(t_peri + WINDOW_S)})
    k = pd.concat([pd.read_parquet(f) for f in files]).sort_values("t_unix")
    k = k[(k.t_unix > t_peri - WINDOW_S) & (k.t_unix < t_peri + WINDOW_S)].dropna(subset=["x", "y", "z", "alt"])
    P = k[["x", "y", "z"]].to_numpy(float)
    i0 = int(np.argmin(np.abs(k.t_unix.to_numpy() - t_peri)))
    u = P[i0] / np.linalg.norm(P[i0])
    h = np.cross(P[0], P[-1])
    h /= np.linalg.norm(h)
    v = np.cross(h, u)
    nu = np.degrees(np.arctan2(P @ v, P @ u))
    t = k.t_unix.to_numpy() - t_peri
    if np.mean(nu[t < 0]) > 0:                  # inbound negative
        nu = -nu
    sza = np.degrees(np.arccos(P[:, 0] / np.linalg.norm(P, axis=1)))
    return t, k.alt.to_numpy(float), nu, sza, float(np.abs(P @ h).max()), files


def column(orbit: int, prof: pd.DataFrame):
    """Inbound N_tot(z) and hard-sphere mean cross section, ascending altitude.

    The replication leaves N_tot NaN within 5 km of periapsis ([column] periapsis_trunc_km); there the
    column is continued down to periapsis by integrating the same fitted backgrounds (n0_O + n0_Ar +
    n0_CO2, trapezoid). Returns z, N, sigma_hs and the altitude of the replication's lowest N_tot."""
    g = prof[prof.orbit == orbit].sort_values("alt")             # profiles.parquet is inbound only
    z = g.alt.to_numpy(float)
    n = {sp: np.nan_to_num(g[f"n0_{sp}"].to_numpy(float)) for sp in D_REF}
    ntot = sum(n.values())
    N = g.N_tot.to_numpy(float).copy()
    ok = np.isfinite(N) & (N > 0)
    i0 = int(np.argmax(ok))                                   # lowest finite replication column
    for i in range(i0 - 1, -1, -1):
        N[i] = N[i + 1] + 0.5 * (ntot[i] + ntot[i + 1]) * (z[i + 1] - z[i]) * 1e3
    keep = np.arange(z.size) < i0
    keep |= ok
    with np.errstate(invalid="ignore", divide="ignore"):       # NaN where no species is fitted
        sig_hs = sum(n[sp] * np.pi * D_REF[sp] ** 2 for sp in D_REF) / ntot
    return z[keep], N[keep], sig_hs[keep], float(z[i0])


def z_at_kn(z, kn, value):
    """Altitude where Kn(z) (monotonic increasing) crosses value; NaN outside the profile."""
    ok = np.isfinite(kn)
    z, kn = z[ok], kn[ok]
    if not kn.min() <= value <= kn.max():
        return float("nan")
    return float(np.interp(np.log(value), np.log(kn), z))


def main():
    prof = pd.read_parquet(TABLES / "profiles.parquet")
    orbits = pd.read_parquet(TABLES / "orbits.parquet").set_index("orbit")
    out, facts, kp_used = {}, {"generated": dt.date.today().isoformat(), "sigma_m2": SIGMA,
                               "n_exobase_m2": N_EXO, "d_ref_m": D_REF,
                               "schematic": {"z_exo_km": Z_EXO_S, "H_km": H_S, "half_angle_deg": ALPHA_DEG},
                               "orbits": {}}, set()
    for orbit, role in ORBITS.items():
        o = orbits.loc[orbit]
        t, alt, nu, sza, plane_km, files = track(float(o.t_peri))
        kp_used.update(files)
        zc, N, sig_hs, z_rep = column(orbit, prof)
        kn_c, kn_hs_c = 1.0 / (SIGMA * N), 1.0 / (sig_hs * N)
        # Kn at the spacecraft: log-interpolate the inbound column in altitude (NaN above / below it)
        # (KP and NGIMS periapsis altitudes differ by < 0.01 km: a 0.5 km tolerance at the bottom)
        lnN = np.interp(np.where(alt >= zc[0] - 0.5, np.maximum(alt, zc[0]), alt), zc, np.log(N),
                        left=np.nan, right=np.nan)
        kn = 1.0 / (SIGMA * np.exp(lnN))
        z_s = Z_EXO_S + H_S * np.log(kn)
        ip = int(np.argmin(alt))
        z_exo_kn = z_at_kn(zc, kn_c, 1.0)
        if abs(z_exo_kn - o.z_exo) > 0.5:
            raise SystemExit(f"orbit {orbit}: Kn = 1 at {z_exo_kn:.2f} km, orbits.parquet z_exo {o.z_exo:.2f}")
        if abs(alt[ip] - o.z_peri) > 0.5:
            raise SystemExit(f"orbit {orbit}: KP periapsis {alt[ip]:.2f} km vs NGIMS {o.z_peri:.2f} km")
        edge = np.abs(nu) <= ALPHA_DEG
        o_nu = np.argsort(nu)
        z10 = z_at_kn(zc, kn_c, 10.0)
        z01 = z_at_kn(zc, kn_c, 0.1)
        trans = (kn >= 0.1) & (kn <= 10.0)
        f = {"role": role, "date": str(o.date)[:10], "t_peri_unix": float(o.t_peri),
             "t_peri_utc": dt.datetime.fromtimestamp(float(o.t_peri), dt.UTC).strftime("%Y-%m-%d %H:%M:%S"),
             "z_peri_km": round(float(o.z_peri), 1), "lst_peri_h": round(float(o.lst_peri), 2),
             "sza_peri_deg": round(float(o.sza_peri), 1), "lat_peri_deg": round(float(o.lat_peri), 1),
             "z_exo_km": round(float(o.z_exo), 1),
             "z_top_O_km": round(float(o.z_top_O), 1), "z_top_CO2_km": round(float(o.z_top_CO2), 1),
             "kn_peri": round(float(kn_c[0]), 3), "kn_peri_hard_sphere": round(float(kn_hs_c[0]), 3),
             "column_bottom_km": round(float(zc[0]), 1),
             "replication_column_bottom_km": round(z_rep, 1),
             "kn_at_replication_column_bottom": round(float(np.interp(z_rep, zc, kn_c)), 3),
             "z_kn10_km": round(z10, 1), "z_kn01_km": None if np.isnan(z01) else round(z01, 1),
             "z_kn10_km_hard_sphere": round(z_at_kn(zc, kn_hs_c, 10.0), 1),
             "z_kn1_km_hard_sphere": round(z_at_kn(zc, kn_hs_c, 1.0), 1),
             "z_kn01_km_hard_sphere": (None if np.isnan(z_at_kn(zc, kn_hs_c, 0.1))
                                       else round(z_at_kn(zc, kn_hs_c, 0.1), 1)),
             "transition_minutes": round(float(np.ptp(t[trans])) / 60.0, 2) if trans.any() else 0.0,
             "transition_nu_deg": [round(float(nu[trans].min()), 1), round(float(nu[trans].max()), 1)]
             if trans.any() else None,
             "alt_at_nu_pm22_km": [round(float(np.interp(a, nu[o_nu], alt[o_nu])), 1)
                                        for a in (-ALPHA_DEG, ALPHA_DEG)],
             "kn_at_nu_pm22": [round(float(np.interp(a, nu[o_nu], kn[o_nu])), 0)
                                    for a in (-ALPHA_DEG, ALPHA_DEG)],
             "z_s_peri_km": round(float(z_s[ip]), 1),
             "z_s_max_nu_pm22_km": round(float(np.nanmax(z_s[edge])), 1),
             "sza_range_nu_pm22_deg": [round(float(sza[edge].min()), 1), round(float(sza[edge].max()), 1)],
             "orbit_plane_residual_km": round(plane_km, 1),
             "minutes_700km_to_peri": round(float(-t[(alt < 700) & (t < 0)].min()) / 60.0, 1)}
        f["nu_at_700km_deg"] = [round(float(nu[(alt < 700)].min()), 1), round(float(nu[(alt < 700)].max()), 1)]
        facts["orbits"][str(orbit)] = f
        out.update({f"o{orbit}_t": t, f"o{orbit}_alt": alt, f"o{orbit}_nu": nu, f"o{orbit}_sza": sza,
                    f"o{orbit}_kn": kn, f"o{orbit}_zs": z_s,
                    f"o{orbit}_col_z": zc, f"o{orbit}_col_kn": kn_c, f"o{orbit}_col_kn_hs": kn_hs_c})
        if orbit == 2521:
            if not np.isfinite(kn[edge]).all():
                raise SystemExit("orbit 2521: Kn undefined somewhere within nu = +-22 deg")
            if not 0.1 < kn_c[0] < 10.0:
                raise SystemExit(f"orbit 2521: periapsis Kn {kn_c[0]:.3f} is not in the transition band")
    facts["sources"] = {str(p): sha256(p) for p in
                        (TABLES / "profiles.parquet", TABLES / "orbits.parquet", CONFIG, *sorted(kp_used))}
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / "slide03_maven.npz", **out)
    (DATA / "slide03_maven_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts["orbits"], indent=2))


if __name__ == "__main__":
    main()
