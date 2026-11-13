"""One run from its run file (<Results>/<alias>_run.nc), species-aware.

The run file is the run tier of docs/CONVENTIONS.md: written by dsmc_netcdf_webapp/dsmc_netcdf_builder
(>= 1.2.0) from the raw Results/{Macro,Rates,Info} ASCII, bit for bit. It is the single raw-output
parser of the pipeline. This reader adds no derived variables; corrections such as the bulk-flow
temperature live in the analysis tier (dsmclib.profiles.temp_trans_corrected).

    from dsmclib.ncrun import open_run_nc
    run = open_run_nc("OH2_B_r1")               # registry alias -> <Results>/<alias>_run.nc
    run = open_run_nc("/path/to/Results/x.nc")  # or a file
    run.macro["num_dens"].sel(species="H2")      # (time, radial) for 1-D runs

Unlike DSMCRun (single species, .npy/pickle inputs), every field keeps a `species`
dimension labelled by name. 1-D runs (one zenith and one azimuth cell) are squeezed to
(species, time, radial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .constants import MARS_MASS, MARS_RADIUS

RUN_SUFFIX = "_run.nc"
GROUPS = ("macroscopic", "rates_surface", "rates_escape", "info_surface", "info_radial")


def _load_group(path: Path, group: str) -> Optional[xr.Dataset]:
    """Load one group fully and close the file (several open handles crash netCDF4 here)."""
    try:
        with xr.open_dataset(path, group=group) as ds:
            return ds.load()
    except OSError:
        return None


def _root_attrs(path: Path) -> Dict[str, object]:
    import netCDF4

    with netCDF4.Dataset(path) as root:
        return {k: root.getncattr(k) for k in root.ncattrs()}


def _label_species(ds: Optional[xr.Dataset]) -> Optional[xr.Dataset]:
    if ds is None or "species_name" not in ds.variables:
        return ds
    names = [str(n) for n in ds["species_name"].values]
    return ds.drop_vars("species_name").assign_coords(species=names)


def _squeeze_1d(ds: xr.Dataset) -> xr.Dataset:
    for dim in ("zenith", "azimuthal"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.isel({dim: 0}, drop=True)
    return ds


def find_run_file(target: Union[str, Path]) -> tuple[Path, Optional[str]]:
    """Resolve an alias, a Results/run folder or a .nc path to the run file."""
    p = Path(str(target)).expanduser()
    alias = None
    if not p.exists() and p.suffix != ".nc":
        from . import registry

        alias = str(target)
        p = registry.run(alias).path
    if p.is_file():
        return p, alias
    results = p / "Results" if (p / "Results").is_dir() else p
    if alias and (results / f"{alias}{RUN_SUFFIX}").is_file():
        return results / f"{alias}{RUN_SUFFIX}", alias
    found = sorted(results.glob(f"*{RUN_SUFFIX}"))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one *{RUN_SUFFIX} in {results}, found {len(found)}; "
                                "build it with python -m dsmclib.runfile")
    return found[0], alias


@dataclass
class NcRun:
    path: Path
    alias: Optional[str]
    macro: xr.Dataset
    surface: Optional[xr.Dataset]
    escape: Optional[xr.Dataset]
    info_surface: Optional[xr.Dataset]
    info_radial: Optional[xr.Dataset]
    attrs: Dict[str, object] = field(default_factory=dict)

    # ---- run parameters -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.alias or self.path.stem

    @property
    def species(self) -> List[str]:
        return [str(s) for s in self.macro["species"].values]

    @property
    def planet_radius(self) -> float:
        return float(self.attrs.get("planet_radius", MARS_RADIUS))

    @property
    def planet_mass(self) -> float:
        return float(self.attrs.get("planet_mass", MARS_MASS))

    @property
    def dt(self) -> float:
        return float(self.attrs["dt"])

    @property
    def time(self) -> np.ndarray:
        return self.macro["time"].values

    @property
    def t_end(self) -> float:
        return float(self.time[-1])

    @property
    def mass(self) -> xr.DataArray:
        """Particle mass per species [kg]."""
        return self.macro["species_mass"]

    @property
    def weight(self) -> xr.DataArray:
        """Physical molecules per computational particle, per species."""
        return self.macro["particle_weight"]

    @property
    def altitude_km(self) -> xr.DataArray:
        return self.macro["altitude"] / 1.0e3

    @property
    def wave_start_time(self) -> Optional[float]:
        """Time the forcing starts, or None when the run ends before the configured start step."""
        step = self.attrs.get("wave_start_step")
        if step is None:
            return None
        t0 = float(step) * self.dt
        return t0 if t0 < self.t_end else None

    @property
    def wave_omega(self) -> Optional[float]:
        w = self.attrs.get("wave_omega")
        return None if w is None else float(w)

    # ---- geometry -------------------------------------------------------------------
    def cell_volume(self) -> xr.DataArray:
        """Cell volume [m3] on the radial axis (1-D: full spherical shell)."""
        rb = self.macro["radial_bounds"].values
        vol = (rb[:, 1] ** 3 - rb[:, 0] ** 3) / 3.0
        zb = np.deg2rad(np.atleast_2d(self.macro["zenith_bounds"].values))
        ab = np.deg2rad(np.atleast_2d(self.macro["azimuthal_bounds"].values))
        if zb.shape[0] != 1 or ab.shape[0] != 1:
            raise NotImplementedError("cell_volume is implemented for 1-D (single zenith/azimuth cell) runs")
        vol = vol * (np.cos(zb[0, 0]) - np.cos(zb[0, 1])) * (ab[0, 1] - ab[0, 0])
        return xr.DataArray(vol, dims="radial", coords={"radial": self.macro["radial"]},
                            attrs={"units": "m3", "long_name": "Cell volume"})

    def radial_edges(self) -> np.ndarray:
        rb = self.macro["radial_bounds"].values
        return np.append(rb[:, 0], rb[-1, 1])

    def __repr__(self) -> str:
        return (f"NcRun({self.name}: species={self.species}, nt={self.macro.sizes['time']}, "
                f"nr={self.macro.sizes['radial']}, t_end={self.t_end:.1f} s)")


def open_run_nc(target: Union[str, Path], squeeze_1d: bool = True) -> NcRun:
    path, alias = find_run_file(target)
    groups = {g: _label_species(_load_group(path, g)) for g in GROUPS}
    macro = groups["macroscopic"]
    if macro is None:
        raise OSError(f"{path} has no /macroscopic group")
    attrs = _root_attrs(path)
    if "builder_version" not in attrs:
        raise ValueError(f"{path} was built by dsmc_netcdf_builder < 1.1.0 (no per-species weights, "
                         "boundary indices stored as radii); rebuild it")
    if squeeze_1d:
        macro = _squeeze_1d(macro)
    run = NcRun(path=path, alias=alias, macro=macro, surface=groups["rates_surface"],
                escape=groups["rates_escape"], info_surface=groups["info_surface"],
                info_radial=groups["info_radial"], attrs=attrs)
    return run

