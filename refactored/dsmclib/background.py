"""Pre-forcing background state and relative perturbations.

Two methods (BACKGROUND_METHODS):
  * "mad_column0" (default; root optimized_loader/loader.get_background_profiles): at a
    reference altitude (230 km), in azimuth column 0, pick the pre-forcing window with the
    lowest median absolute deviation and take the mean over that window at every altitude.
    Exact for 1D runs (one column), but for 2D runs it uses a single column and, for
    S1-S3, only 3 snapshots, so the profile carries particle noise of several % per altitude.
  * "azimean_snapshot" (the lost 2024 pipeline behind the published S1-S6 growth): the
    snapshot 2 outputs before the forcing starts, averaged over all azimuth columns.
    Verified exact against the 2024 growth_peak_data pkls of S1, S2, S3 and S6.
  * "azimean_preforcing" (refactored default for S1-S6): the last PREFORCING_SNAPSHOTS outputs
    before forcing (t = 4908-5012 s in every S run), averaged over time and all azimuth columns.
    S1-S3 have only these 17 pre-forcing outputs; S4-S6 have 767 from t = 7 s, but those include
    the spin-up (+2.9 % at 315 km if all were averaged), so the common, stationary window is used.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .params import SimulationParameters


def find_min_mad_window(data, window_sizes):
    """Window (size, (start, end), MAD) with the lowest rolling median absolute deviation.

    Windows with a median <= 1e-10 are skipped. If none qualifies, returns the first quarter
    of the series with MAD = inf.
    """
    best_mad = float("inf")
    best_info = None

    clean_data = np.nan_to_num(data, nan=0.0)

    fallback_window_size = min(window_sizes[0], len(clean_data) // 4)
    fallback_start = 0
    fallback_end = fallback_start + fallback_window_size - 1

    for window_size in window_sizes:
        if window_size >= len(clean_data):
            continue

        windows = sliding_window_view(clean_data, window_size)  # (n - window_size + 1, window_size)
        if len(windows) == 0:
            continue

        medians = np.median(windows, axis=1)
        valid_windows = medians > 1e-10
        if not np.any(valid_windows):
            continue

        valid_mads = np.median(np.abs(windows[valid_windows] - medians[valid_windows, None]), axis=1)
        if len(valid_mads) > 0:
            idx_min = valid_mads.argmin()
            valid_indices = np.where(valid_windows)[0]
            if valid_mads[idx_min] < best_mad:
                best_mad = valid_mads[idx_min]
                best_info = (window_size, (valid_indices[idx_min], valid_indices[idx_min] + window_size - 1))

    if best_info is None:
        return (fallback_window_size, (fallback_start, fallback_end), float("inf"))

    return (best_info[0], best_info[1], best_mad)


def wave_time_indices(params: SimulationParameters) -> Tuple[int, int]:
    """(first output index after forcing starts, 5 outputs before the first after forcing ends)."""
    start_idx = np.where(params.time > params.wave_start_time)[0]
    start_idx = 0 if len(start_idx) == 0 else start_idx[0]

    end_idx = np.where(params.time > params.wave_end_time)[0]
    end_idx = len(params.time) - 1 if len(end_idx) == 0 else end_idx[0] - 5

    return start_idx, end_idx


def window_sizes(params: SimulationParameters, end_idx: int, custom_time_windows: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate background windows, as (seconds, number of outputs).

    Default: 4 windows from 1000 s to min(half the pre-forcing time, 2500 s).
    """
    dt = np.mean(np.diff(params.time))
    available_time = params.time[end_idx]
    max_window_time = min(available_time * 0.5, 2500)

    if custom_time_windows is not None:
        custom_time_windows = np.asarray(custom_time_windows)
        if np.any(custom_time_windows > available_time * 0.5):
            raise ValueError(f"Window sizes must be less than half the available time ({available_time * 0.5:.1f} s)")
        if np.any(custom_time_windows < 100):
            raise ValueError("Window sizes must be at least 100 seconds")
        time_windows = np.array(custom_time_windows)
    else:
        time_windows = np.linspace(1000, max_window_time, 4)

    sizes = np.maximum((time_windows / dt).astype(int), 3)
    return time_windows, sizes


def background_profiles(
    params: SimulationParameters,
    density: np.ndarray,
    temperature: np.ndarray,
    custom_time_windows: Optional[np.ndarray] = None,
    ref_altitude: float = 230.0,
) -> Dict[str, object]:
    """Background density and temperature per altitude, plus the window used.

    Returns {"density": (n_alt,), "temperature": (n_alt,), "window": {"start", "end"}}.
    """
    start_idx, _ = wave_time_indices(params)
    end_idx = start_idx - 5

    ref_idx = np.abs(params.radial_cells - ref_altitude).argmin()
    density_ref_series = density[:end_idx, ref_idx, 0]
    temperature_ref_series = temperature[:end_idx, ref_idx, 0]

    _, sizes = window_sizes(params, end_idx, custom_time_windows)

    _, (d_start, d_end), _ = find_min_mad_window(density_ref_series, sizes)
    _, (t_start, t_end), _ = find_min_mad_window(temperature_ref_series, sizes)

    window = {"start": (d_start + t_start) // 2, "end": (d_end + t_end) // 2}
    print(f"Background window selected: Start index = {window['start']}, End index = {window['end']}")

    num_altitudes = density.shape[1]
    density_bg = np.empty(num_altitudes)
    temperature_bg = np.empty(num_altitudes)
    for alt in range(num_altitudes):
        density_bg[alt] = np.mean(density[:end_idx, alt, 0][window["start"]: window["end"] + 1])
        temperature_bg[alt] = np.mean(temperature[:end_idx, alt, 0][window["start"]: window["end"] + 1])

    return {"density": density_bg, "temperature": temperature_bg, "window": window}


BACKGROUND_METHODS = ("mad_column0", "azimean_snapshot", "azimean_preforcing")

# Pre-forcing outputs present in every S1-S6 run (S1-S3 start at t = 4908 s, 6.535 s cadence)
PREFORCING_SNAPSHOTS = 17


def forcing_start_index(params: SimulationParameters) -> int:
    """First output with time >= wave_start_time (wave_time_indices uses '>', see below)."""
    return int(np.searchsorted(params.time, params.wave_start_time, side="left"))


def azimean_snapshot_background(params: SimulationParameters, density: np.ndarray, temperature: np.ndarray,
                                offset: int = 2) -> Dict[str, object]:
    """Azimuthal-mean density and temperature of the snapshot `offset` outputs before forcing starts.

    "Forcing starts" = first output with time >= wave_start_time (the 2024 rule; wave_time_indices
    uses '>' and is one output later when an output falls exactly on the start, as in S6).
    """
    idx = forcing_start_index(params) - offset
    print(f"Background snapshot: index {idx} (t = {params.time[idx]:.1f} s), mean over {density.shape[2]} azimuth columns")
    return {"density": density[idx].mean(axis=1), "temperature": temperature[idx].mean(axis=1),
            "window": {"start": idx, "end": idx}}


def azimean_preforcing_background(params: SimulationParameters, density: np.ndarray, temperature: np.ndarray,
                                  n_snapshots: int = PREFORCING_SNAPSHOTS) -> Dict[str, object]:
    """Density and temperature averaged over the last n_snapshots pre-forcing outputs and all azimuth columns."""
    end = forcing_start_index(params)
    start = end - n_snapshots
    if start < 0:
        raise ValueError(f"only {end} pre-forcing outputs, {n_snapshots} requested")
    print(f"Background: outputs {start}-{end - 1} (t = {params.time[start]:.1f}-{params.time[end - 1]:.1f} s), "
          f"mean over time and {density.shape[2]} azimuth columns")
    return {"density": density[start:end].mean(axis=(0, 2)), "temperature": temperature[start:end].mean(axis=(0, 2)),
            "window": {"start": start, "end": end - 1}}


def relative_perturbation(field: np.ndarray, background: np.ndarray) -> np.ndarray:
    """(field - bg) / bg with bg (n_alt,) broadcast over (time, altitude, azimuth)."""
    return (field - background[None, :, None]) / background[None, :, None]
