"""Time-mean altitude profiles per species, isothermal reference, exobase, replicate statistics.

    from dsmclib import profiles
    prof = profiles.time_mean_profiles(run, t0, t1)        # Dataset (species, radial): <x>, <x>_sem
    ref = profiles.isothermal_reference(run, prof)         # barometric n(r) anchored at the lowest cell
    profiles.exobase(prof)                                  # altitude where Kn = 1 per species
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr

from .constants import G_NEWTON, K_BOLTZMANN
from .ncrun import NcRun

PROFILE_FIELDS = ("num_dens", "num_part", "temp_trans", "temp_trans_corr", "temp_rad", "temp_zen", "temp_azi", "temp_kin",
                  "vel_rad", "mean_free_path", "scale_height", "knudsen")



def temp_trans_corrected(run: NcRun) -> xr.DataArray:
    """Bulk-flow corrected temperature temp_trans - m vel_rad^2 / (3 k), per output window (analysis tier).

    In 1-D spherical runs the solver subtracts the cell's Cartesian mean velocity, which is ~0 for
    a spherically symmetric radial flow, so radial bulk-flow energy is counted as translational
    temperature. Subtracting it recovers the temperature about the local radial mean flow.
    Empty cells keep the solver's 0. 1-D runs only (raises otherwise). Attach it explicitly:
        run.macro["temp_trans_corr"] = temp_trans_corrected(run)
    """
    m = run.macro
    if "zenith" in m.dims or "azimuthal" in m.dims:
        raise ValueError("temp_trans_corrected applies to 1-D runs only")
    bias = run.mass * m["vel_rad"] ** 2 / (3.0 * K_BOLTZMANN)
    corr = (m["temp_trans"] - bias).where(m["num_part"] > 0, 0.0)
    return corr.transpose(*m["temp_trans"].dims).assign_attrs(
        units="K", long_name="Translational temperature corrected for radial bulk flow",
        comment="temp_trans - m vel_rad^2/(3k) per output window (dsmclib.profiles.temp_trans_corrected); 1-D runs only")


def time_mean_profiles(run: NcRun, t0: float, t1: Optional[float] = None,
                       fields: Sequence[str] = PROFILE_FIELDS, n_batches: int = 10) -> xr.Dataset:
    """Mean of each field over outputs with t0 < time <= t1, with batch-means standard errors.

    Temperatures and velocities of cells that hold no computational particles in a window are
    written as 0 by the solver; they are masked (NaN) before averaging so they do not bias the mean.
    The default fields include temp_trans_corr, which the caller attaches first
    (run.macro["temp_trans_corr"] = temp_trans_corrected(run)); the run file does not carry it.
    """
    t = run.time
    t1 = float(t[-1]) if t1 is None else float(t1)
    sel = (t > t0) & (t <= t1)
    if sel.sum() < 2 * n_batches:
        raise ValueError(f"only {sel.sum()} outputs in ({t0}, {t1}]; need >= {2 * n_batches}")
    m = run.macro.isel(time=np.where(sel)[0])
    empty = m["num_part"] <= 0
    out = {}
    for f in fields:
        da = m[f]
        if f not in ("num_dens", "num_part"):
            da = da.where(~empty)
        vals = da.transpose("time", ...).values
        with warnings.catch_warnings():  # cells never populated in the window are all-NaN by design
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(vals, axis=0)
            b = np.stack([np.nanmean(c, axis=0) for c in np.array_split(vals[: len(vals) // n_batches * n_batches],
                                                                         n_batches)])
            sem = np.nanstd(b, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(b), axis=0).clip(min=1))
        dims = [d for d in da.dims if d != "time"]
        out[f] = (dims, mean, {**m[f].attrs, "cell_methods": "time: mean"})
        out[f + "_sem"] = (dims, sem, {"units": m[f].attrs.get("units", ""),
                                       "long_name": f"Batch-means standard error of {f} ({n_batches} batches)"})
    ds = xr.Dataset(out, coords={"species": m["species"], "radial": m["radial"], "altitude": m["altitude"]})
    ds.attrs.update(t0=float(t0), t1=t1, n_outputs=int(sel.sum()), n_batches=n_batches, run=run.name)
    return ds


def isothermal_reference(run: NcRun, prof: xr.Dataset, temperature: Optional[float] = None,
                         anchor_index: int = 0) -> xr.DataArray:
    """Barometric number density for each species with 1/r^2 gravity:

        n(r) = n(r0) exp[-(G M m / k T) (1/r0 - 1/r)]

    anchored at cell `anchor_index` of `prof` (DSMC time-mean density there). T defaults to the
    lower-boundary temperature T0 from Info (the isothermal temperature the solver injects at).
    Each species is treated independently (no diffusive coupling), which is what a
    collisionless or diffusively separated isothermal column would give.
    """
    if temperature is None:
        if run.info_surface is None:
            raise ValueError("no temperature given and no /info_surface group to read T0 from")
        temperature = float(run.info_surface["T0"].values.ravel()[0])
    r = prof["radial"]
    r0 = float(r[anchor_index])
    n0 = prof["num_dens"].isel(radial=anchor_index)
    lam = G_NEWTON * run.planet_mass * run.mass / (K_BOLTZMANN * temperature)
    n = n0 * np.exp(-lam * (1.0 / r0 - 1.0 / r))
    return n.assign_attrs(units="m-3", long_name=f"Isothermal barometric density, T = {temperature:g} K",
                          temperature=temperature, anchor_radius=r0)


def exobase(prof: xr.Dataset, knudsen_field: str = "knudsen") -> xr.DataArray:
    """Altitude [km] where the species' Knudsen number (mean free path / scale height) first
    reaches 1, log-linearly interpolated between cell centres. NaN if Kn < 1 everywhere, and
    the lowest altitude if Kn >= 1 already there."""
    alt = prof["altitude"].values / 1e3
    out = []
    for sp in prof["species"].values:
        kn = prof[knudsen_field].sel(species=sp).values
        ok = np.isfinite(kn) & (kn > 0)
        a, k = alt[ok], np.log(kn[ok])
        above = np.where(k >= 0)[0]
        if len(above) == 0:
            out.append(np.nan)
        elif above[0] == 0:
            out.append(a[0])
        else:
            i = above[0]
            out.append(np.interp(0.0, [k[i - 1], k[i]], [a[i - 1], a[i]]))
    return xr.DataArray(out, dims="species", coords={"species": prof["species"]},
                        attrs={"units": "km", "long_name": "Exobase altitude (Kn = 1)"})


def interp_at_altitude(prof: xr.Dataset, field: str, alt_km: xr.DataArray, log: bool = False) -> xr.DataArray:
    """Value of `field` at a per-species altitude (km), linear (or log-linear) in altitude."""
    alt = prof["altitude"].values / 1e3
    out = []
    for sp in prof["species"].values:
        y = prof[field].sel(species=sp).values
        a = float(alt_km.sel(species=sp))
        ok = np.isfinite(y) & ((y > 0) if log else True)
        if not np.isfinite(a) or ok.sum() < 2:
            out.append(np.nan)
            continue
        v = np.interp(a, alt[ok], np.log(y[ok]) if log else y[ok], left=np.nan, right=np.nan)
        out.append(np.exp(v) if log else v)
    return xr.DataArray(out, dims="species", coords={"species": prof["species"]},
                        attrs={"units": prof[field].attrs.get("units", "")})


def regrid(prof: xr.Dataset, alt_km: Sequence[float], fields: Iterable[str] = ("num_dens", "temp_trans"),
           log_fields: Iterable[str] = ("num_dens",)) -> xr.Dataset:
    """Profiles interpolated to common altitudes (km); densities log-linear. NaN outside the grid."""
    alt_km = np.asarray(alt_km, dtype=float)
    src = prof["altitude"].values / 1e3
    log_fields = set(log_fields)
    out = {}
    for f in fields:
        rows = []
        for sp in prof["species"].values:
            y = prof[f].sel(species=sp).values
            ok = np.isfinite(y) & ((y > 0) if f in log_fields else True)
            if ok.sum() < 2:
                rows.append(np.full_like(alt_km, np.nan))
                continue
            yy = np.log(y[ok]) if f in log_fields else y[ok]
            v = np.interp(alt_km, src[ok], yy, left=np.nan, right=np.nan)
            rows.append(np.exp(v) if f in log_fields else v)
        out[f] = (("species", "altitude_km"), np.array(rows), prof[f].attrs)
    return xr.Dataset(out, coords={"species": prof["species"], "altitude_km": alt_km})


def replicate_stats(profs: Sequence[xr.Dataset], fields: Iterable[str] = ("num_dens", "temp_trans")) -> xr.Dataset:
    """Across-replicate mean, standard deviation and standard error (replicates on the same grid)."""
    stack = xr.concat([p[list(fields)] for p in profs], dim="replicate")
    n = stack.sizes["replicate"]
    out = xr.Dataset()
    for f in fields:
        out[f + "_mean"] = stack[f].mean("replicate")
        out[f + "_std"] = stack[f].std("replicate", ddof=1)
        out[f + "_sem"] = out[f + "_std"] / np.sqrt(n)
    out.attrs["n_replicates"] = n
    return out
