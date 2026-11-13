"""DSMCRun: one simulation run's parameters, fields and background, loaded on demand.

    run = DSMCRun(registry.run("S6").path)      # or registry.open_run("S6")
    run.load()                                  # parameters + fields + surface/escape
    bg = run.background()                       # standard MAD-window background
    dn = run.density_perturbation()             # (time, altitude, azimuth)

Attribute names follow the root OptimizedDSMCDataProcessor (params, density_data,
temperature_data, ...) so ported notebook code reads the same.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import background as bg
from .macro import find_macro_pickle, load_field, load_pickle, load_surface_and_escape
from .params import SimulationParameters, params_from_ascii, params_from_npz

# How the growth window's base/top altitudes map to cells: root loaders take the first cell above,
# the 2024 pipeline (and the refactored S1-S6 recipe) the nearest cell (148.3 / 315.4 km in S1-S6)
ALTITUDE_CELL_RULES = ("first_above", "nearest")


class DSMCRun:
    def __init__(self, data_dir, alias: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.alias = alias or self.data_dir.name
        self.macro_path: Optional[Path] = find_macro_pickle(self.data_dir)
        self._macro: Optional[Dict[str, Any]] = None

        self.params: Optional[SimulationParameters] = None
        self.density_data: Optional[np.ndarray] = None
        self.temperature_data: Optional[np.ndarray] = None
        self.radial_vel_data: Optional[np.ndarray] = None
        self.knudsen_data: Optional[np.ndarray] = None
        self.scale_height_data: Optional[np.ndarray] = None
        self.number_particles_data: Optional[np.ndarray] = None
        self.surface_data: Optional[Dict[str, Any]] = None
        self.escape_data: Optional[Dict[str, Any]] = None
        self.background_profiles: Optional[Dict[str, Any]] = None
        self._other_backgrounds: Dict[str, Dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"DSMCRun({self.alias!r}, {self.data_dir})"

    def macro(self) -> Dict[str, Any]:
        """The run's macro pickle, loaded once (raises FileNotFoundError if the run has none)."""
        if self._macro is None:
            if self.macro_path is None:
                raise FileNotFoundError(f"No macro pickle in {self.data_dir}")
            print(f"Loading {self.macro_path.name} ({self.macro_path.stat().st_size / 1e9:.2f} GB)")
            self._macro = load_pickle(self.macro_path)
        return self._macro

    def load_parameters(self, params_file: str = "params_data.npz") -> SimulationParameters:
        if self.params is None:
            if (self.data_dir / params_file).exists():
                self.params = params_from_npz(self.data_dir, params_file)
            else:
                self.params = params_from_ascii(self.data_dir, self.macro() if self.macro_path else None)
        return self.params

    def load_data(self) -> None:
        """Load density, temperature, radial velocity (+ optional fields) and surface/escape data."""
        self.load_parameters()
        self.density_data = load_field(self.data_dir, "density", self.macro)
        self.temperature_data = load_field(self.data_dir, "temperature", self.macro)
        self.radial_vel_data = load_field(self.data_dir, "radial_vel", self.macro)
        self.knudsen_data = load_field(self.data_dir, "knudsen", self.macro)
        self.scale_height_data = load_field(self.data_dir, "scale_height", self.macro)
        self.number_particles_data = load_field(self.data_dir, "number_particles", self.macro)
        self.surface_data, self.escape_data = load_surface_and_escape(self.data_dir, self.macro, self.params.n_azi_cell)

    def load(self) -> "DSMCRun":
        self.load_parameters()
        self.load_data()
        return self

    def release_macro(self) -> None:
        """Drop the cached macro pickle (the S4 one is 3.7 GB) once the fields are extracted."""
        self._macro = None

    # --- Derived quantities -------------------------------------------------

    def wave_time_indices(self):
        return bg.wave_time_indices(self.params)

    def background(self, custom_time_windows=None, ref_altitude: float = 230.0,
                   method: str = "mad_column0") -> Dict[str, Any]:
        """Background profiles (see background.BACKGROUND_METHODS); computed on first call and cached.

        custom_time_windows and ref_altitude only apply to "mad_column0".
        """
        if method == "mad_column0":
            if self.background_profiles is None:
                self.background_profiles = bg.background_profiles(
                    self.params, self.density_data, self.temperature_data, custom_time_windows, ref_altitude
                )
            return self.background_profiles
        builders = {"azimean_snapshot": bg.azimean_snapshot_background,
                    "azimean_preforcing": bg.azimean_preforcing_background}
        if method in builders:
            if method not in self._other_backgrounds:
                self._other_backgrounds[method] = builders[method](self.params, self.density_data, self.temperature_data)
            return self._other_backgrounds[method]
        raise ValueError(f"Unknown background method {method!r}; choose from {bg.BACKGROUND_METHODS}")

    def density_perturbation(self, custom_time_windows=None, background: str = "mad_column0") -> np.ndarray:
        """Relative density perturbation (n - n_bg) / n_bg, shape (time, altitude, azimuth)."""
        return bg.relative_perturbation(self.density_data,
                                        self.background(custom_time_windows, method=background)["density"])

    def forcing_time_mask(self, padding: float) -> np.ndarray:
        """True for output times from forcing start to forcing end + padding [s]."""
        t = self.params.time
        return (self.params.wave_start_time <= t) & (t <= self.params.wave_end_time + padding)

    def altitude_index_above(self, altitude_km: float) -> int:
        """Index of the first cell whose centre is above altitude_km."""
        return int(np.where(self.params.radial_cells > altitude_km)[0][0])

    def altitude_index_nearest(self, altitude_km: float) -> int:
        """Index of the cell whose centre is nearest altitude_km."""
        return int(np.abs(self.params.radial_cells - altitude_km).argmin())

    def altitude_index(self, altitude_km: float, rule: str = "first_above") -> int:
        """Cell index for altitude_km by ALTITUDE_CELL_RULES: "first_above" (root) or "nearest" (2024 pipeline)."""
        if rule == "first_above":
            return self.altitude_index_above(altitude_km)
        if rule == "nearest":
            return self.altitude_index_nearest(altitude_km)
        raise ValueError(f"Unknown altitude cell rule {rule!r}; choose from {ALTITUDE_CELL_RULES}")
