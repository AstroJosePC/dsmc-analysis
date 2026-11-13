"""Reading macroscopic fields and surface/escape data from a run folder.

A run folder holds the fields either as individual .npy files or inside a macro
pickle. S1-S3 keep a 10-member ensemble mean in combined_10runs_macro_grids_data.pkl,
which is picked before the single-run macro_grids_data.pkl (same order as the root
optimized_loader, so every S1-S3 result uses the ensemble).
"""

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

MACRO_PICKLE_GLOBS = [
    "combined_*runs_macro_grids_data.pkl",
    "macro_grids.pkl",
    "macro_grids_data.pkl",
]

SURFACE_BASE_COLUMNS = {"time": 0, "avg_time_interval": 1, "rad": 2, "zen": 3, "azi": 4, "cell_num": 5, "lower_area": 6}
SURFACE_SPECIES_COLUMNS = {
    "num_part_inject": 0,
    "num_part_return": 1,
    "num_part_reemit": 2,
    "num_part_surf": 3,
    "inject_rate": 4,
    "return_rate": 5,
    "reemit_rate": 6,
    "return_time": 7,
    "inject_energy": 8,
    "return_energy": 9,
}
ESCAPE_BASE_COLUMNS = {"time": 0, "avg_time_interval": 1, "rad": 2, "zen": 3, "azi": 4, "cell_num": 5, "upper_area": 6}
ESCAPE_SPECIES_COLUMNS = {
    "num_part_escape": 0,
    "num_part_ball": 1,
    "escape_rate": 2,
    "escape_time": 3,
    "escape_speed": 4,
    "escape_energy": 5,
}

# (primary .npy, alternative .npy, key inside the macro pickle, required)
FIELD_SOURCES = {
    "density": ("density_data.npy", "macro_Number Density.npy", "Number Density", True),
    "temperature": ("temperature_data.npy", "macro_Kinetic Temperature.npy", "Kinetic Temperature", True),
    "radial_vel": ("rad_vel_data.npy", "macro_Radial Velocity.npy", "Radial Velocity", True),
    "knudsen": ("macro_Knudsen Number.npy", None, "Knudsen Number", False),
    "scale_height": ("macro_Scale Height.npy", None, "Scale Height", False),
    "number_particles": ("macro_Number Particles.npy", None, "Number Particles", False),
    "mean_free_path": ("macro_Mean Free Path.npy", None, "Mean Free Path", False),
    "thermal_vel": ("macro_Thermal Velocity.npy", None, "Thermal Velocity", False),
    "temp_trans": ("macro_Translational Temperature.npy", None, "Translational Temperature", False),
}


def find_macro_pickle(data_dir: Path) -> Optional[Path]:
    """First macro pickle in the run folder, ensemble files first; None if there is none."""
    candidates: List[Path] = []
    for pattern in MACRO_PICKLE_GLOBS:
        candidates += list(data_dir.glob(pattern))
    return candidates[0] if candidates else None


def squeeze_species(array: np.ndarray) -> np.ndarray:
    """Drop a leading single-species axis: (1, time, alt, azi) -> (time, alt, azi)."""
    if array.ndim == 4 and array.shape[0] == 1:
        return array.squeeze(0)
    return array


def load_field(
    data_dir: Path,
    field: str,
    get_macro: Callable[[], Dict[str, Any]],
    *,
    mmap_mode: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load one macroscopic field as (time, altitude, azimuth).

    Looks for the .npy files first, then inside the macro pickle (get_macro loads it lazily
    and raises FileNotFoundError when there is none). Optional fields return None when absent.
    """
    primary, alternative, macro_key, required = FIELD_SOURCES[field]
    candidate_paths = []
    for candidate in filter(None, [primary, alternative]):
        path = data_dir / candidate
        candidate_paths.append(str(path))
        if path.exists():
            return squeeze_species(np.load(path, mmap_mode=mmap_mode))

    try:
        macro_data = get_macro()
    except FileNotFoundError:
        if required:
            raise FileNotFoundError(f"{macro_key} data not found in {', '.join(candidate_paths)} or a macro pickle")
        return None

    if macro_key in macro_data:
        return squeeze_species(macro_data[macro_key])
    if required:
        raise FileNotFoundError(f"{macro_key} data not found in {', '.join(candidate_paths)} or the macro pickle")
    return None


def load_surface_and_escape(
    data_dir: Path, get_macro: Callable[[], Dict[str, Any]], n_azi_cell: int
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Surface-injection and top-boundary escape data, from surface_escape_data.npz or the macro pickle.

    Returns (surface_data, escape_data); either can be None when the run folder lacks them.
    The pickle only carries 'Surface Rates', so escape_data is None on that path.
    """
    surface_escape_path = data_dir / "surface_escape_data.npz"
    if surface_escape_path.exists():
        npz = np.load(surface_escape_path, allow_pickle=True)
        surface_data = {
            "base_data": {name: npz["surf_base_data"][:, :, col] for name, col in SURFACE_BASE_COLUMNS.items()},
            "species_data": {name: npz["surf_species_data"][:, :, :, col] for name, col in SURFACE_SPECIES_COLUMNS.items()},
            "species_names": npz["species_names"],
            "flow_rates": npz["flow_rates"],
            "surface_base_data": npz["surface_base_data"],
        }
        escape_data = {
            "base_data": {name: npz["escape_base_data"][:, :, col] for name, col in ESCAPE_BASE_COLUMNS.items()},
            "species_data": {name: npz["escape_species_data"][:, :, :, col] for name, col in ESCAPE_SPECIES_COLUMNS.items()},
            "species_names": npz["species_names"],
        }
        return surface_data, escape_data

    try:
        macro_data = get_macro()
    except FileNotFoundError:
        print(f"Warning: neither surface_escape_data.npz nor a macro pickle in {data_dir.name}; no surface data.")
        return None, None

    if "Surface Rates" not in macro_data:
        print(f"Warning: 'Surface Rates' not in the macro pickle of {data_dir.name}; no surface data.")
        return None, None

    surface_rates = macro_data["Surface Rates"]
    if surface_rates.ndim == 2 and n_azi_cell == 1 and surface_rates.shape[1] == 17:
        surface_rates = surface_rates[:, None, :]
    elif surface_rates.ndim != 3 or surface_rates.shape[2] != 17:
        print(f"Warning: 'Surface Rates' in {data_dir.name} is not (time, azimuth, 17); no surface data.")
        return None, None

    base_data = surface_rates[:, :, 0:7]
    species_data = surface_rates[:, :, 7:17][None, :, :, :]
    surface_data = {
        "base_data": {name: base_data[:, :, col] for name, col in SURFACE_BASE_COLUMNS.items()},
        "species_data": {name: species_data[:, :, :, col] for name, col in SURFACE_SPECIES_COLUMNS.items()},
        "species_names": [],
        "flow_rates": None,
        "surface_base_data": base_data,
    }
    return surface_data, None


def load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
