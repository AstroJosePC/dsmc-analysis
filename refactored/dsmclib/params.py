"""Simulation parameters and grid geometry of one DSMC run.

Two sources, as in the root optimized_loader.py:
  * params_data.npz written by the post-processing (preferred), or
  * Info/Simulation_Parameters.ascii + Info/Radial_Cells.ascii + Info/Grid.ascii.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .constants import MARS_RADIUS


@dataclass
class SimulationParameters:
    """Grid, timing and forcing parameters of one run."""
    mid_azi: np.ndarray
    radial_cells: np.ndarray  # cell-centre altitudes [km]
    r1_distances: np.ndarray
    time: np.ndarray  # output times [s]
    mars_radius: float
    start_step_wave: int
    final_step_wave: int
    timestep: float
    omega_freq: float  # angular frequency [rad/s]
    num_periods: float
    radial_edges: np.ndarray
    azi_edges: np.ndarray
    n_rad_cell: int = 1
    n_azi_cell: int = 1

    @property
    def wave_period(self) -> float:
        return 2 * np.pi / self.omega_freq

    @property
    def wave_start_time(self) -> float:
        return self.start_step_wave * self.timestep

    @property
    def wave_end_time(self) -> float:
        return self.final_step_wave * self.timestep


def extract_grid_edges(grid_ascii_path):
    """Unique sorted radial [m] and azimuthal [deg] cell edges from Info/Grid.ascii."""
    cols = ["cell_num", "i_rad", "i_zen", "i_azi", "rad_lower", "rad_upper",
            "zen_lower", "zen_upper", "azi_lower", "azi_upper",
            "rad_width", "zen_width", "azi_width", "volume"]
    df = pd.read_csv(grid_ascii_path, header=None, names=cols, sep=r"\s+")
    radial_edges = np.unique(np.concatenate([df["rad_lower"].values, df["rad_upper"].values]))
    azimuth_edges = np.unique(np.concatenate([df["azi_lower"].values, df["azi_upper"].values]))
    return radial_edges, azimuth_edges


def params_from_npz(data_dir: Path, params_file: str = "params_data.npz") -> SimulationParameters:
    """Build SimulationParameters from params_data.npz (key aliases as in optimized_loader)."""
    params = np.load(data_dir / params_file, allow_pickle=True)

    mars_radius = params.get("mars_radius", params.get("radius", params.get("R", MARS_RADIUS / 1e3)))
    omega_freq = params.get("omega_freq", params.get("omega", params.get("frequency", 0.0)))
    timestep = params.get("timestep", params.get("dt", params.get("time_step", 0.0)))
    start_step_wave = params.get("start_step_wave", params.get("wave_start", params.get("step0_inject", 0)))
    final_step_wave = params.get("final_step_wave", params.get("wave_end", params.get("stepf_inject", 0)))
    num_periods = params.get("num_periods", 0.0)

    radial_cells = params.get("radial_cells", params.get("radial_distances", np.array([])))
    # Radial cells in m with the planet radius included -> altitude in km
    if len(radial_cells) > 0 and radial_cells[0] > 1e5:
        radial_cells = radial_cells / 1000.0 - mars_radius
    n_rad_cell = len(radial_cells) if len(radial_cells) > 0 else 1
    mid_azi = params.get("mid_azi", np.array([]))
    r1_distances = params.get("r1_distances", np.array([]))
    time = params.get("time", params.get("t", np.array([])))

    radial_edges, azi_edges = extract_grid_edges(f"{data_dir}/Info/Grid.ascii")

    try:
        n_azi_cell = len(mid_azi) if len(mid_azi) > 0 else 1
    except TypeError:  # 1D runs store mid_azi as a scalar
        n_azi_cell = 1

    if mars_radius <= 0:
        raise ValueError("Invalid Mars radius")
    if omega_freq <= 0:
        raise ValueError("Invalid wave frequency")
    if timestep <= 0:
        raise ValueError("Invalid timestep")

    return SimulationParameters(
        radial_cells=radial_cells,
        mid_azi=mid_azi,
        r1_distances=r1_distances,
        time=time,
        mars_radius=mars_radius,
        start_step_wave=start_step_wave,
        final_step_wave=final_step_wave,
        timestep=timestep,
        omega_freq=omega_freq,
        num_periods=num_periods,
        n_azi_cell=n_azi_cell,
        n_rad_cell=n_rad_cell,
        radial_edges=radial_edges,
        azi_edges=azi_edges,
    )


def read_simulation_parameters_ascii(ascii_path: Path) -> Dict[str, Any]:
    """Parse Info/Simulation_Parameters.ascii into {section: {key: value}} (same parser as create_db.py)."""
    parameters = {}
    current_section = None
    num_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    special_keys_no_colon = ["Samples Taken Every", "Averages Taken Every"]

    with open(ascii_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.endswith("Information") or line == "Simulation Parameters":
                current_section = line
                if current_section not in parameters:
                    parameters[current_section] = {}
            elif "Time Step:" in line:
                if "Simulation Parameters" not in parameters:
                    parameters["Simulation Parameters"] = {}
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                match = num_pattern.search(value)
                if match:
                    num_val = float(match.group())
                    if num_val.is_integer() and num_val < 1e15:
                        num_val = int(num_val)
                    value = num_val
                parameters["Simulation Parameters"][key] = value
            elif line and ":" in line and current_section:
                key, value = line.split(":", 1)
                key = key.strip()
                if key not in ["Name", "Species 1", "Species 2"]:
                    match = num_pattern.search(value)
                    if match:
                        num_val = float(match.group())
                        if num_val.is_integer() and num_val < 1e15:
                            num_val = int(num_val)
                        value = num_val
                if current_section in ["Species Information", "Collision Information"]:
                    if not parameters[current_section]:
                        parameters[current_section] = []
                    parameters[current_section].append({key: value})
                else:
                    parameters[current_section][key] = value
            elif current_section == "Simulation Parameters":
                for special_key in special_keys_no_colon:
                    if line.startswith(special_key):
                        key = special_key
                        value_str = line[len(special_key):].strip()
                        match = num_pattern.search(value_str)
                        if match:
                            value = float(match.group())
                            if value.is_integer() and value < 1e15:
                                value = int(value)
                        else:
                            value = value_str
                        parameters["Simulation Parameters"][key] = value
                        break
    return parameters


def radial_cell_midpoints(data_dir: Path) -> np.ndarray:
    """Radial cell midpoints [m from planet centre] from Info/Radial_Cells.ascii (empty if absent)."""
    rad_cells_path = data_dir / "Info" / "Radial_Cells.ascii"
    if not rad_cells_path.exists():
        return np.array([])
    df = pd.read_csv(rad_cells_path, sep=r"\s+", header=None, usecols=[1, 2], names=["R1", "R2"])
    return ((df["R1"] + df["R2"]) / 2.0).values


def params_from_ascii(data_dir: Path, macro_data: Optional[Dict[str, Any]]) -> SimulationParameters:
    """Build SimulationParameters from the Info/*.ascii files.

    Output times come from macro_data["Time Stamps"] when a macro pickle is available;
    otherwise they are regenerated from the step counts (as optimized_loader did).
    """
    ascii_path = data_dir / "Info" / "Simulation_Parameters.ascii"
    if not ascii_path.exists():
        raise FileNotFoundError(f"ASCII parameter file not found: {ascii_path}")

    params_dict = read_simulation_parameters_ascii(ascii_path)
    sim_params = params_dict.get("Simulation Parameters", {})
    planetary = params_dict.get("Planetary Body Information", {})

    timestep = sim_params.get("Time Step", 0.0)
    nsteps = sim_params.get("Number of Total Simulations Steps", 0)
    avg_taken_every = sim_params.get("Averages Taken Every", 1)
    mars_radius = planetary.get("Radius", 0.0)
    omega_freq = sim_params.get("Wave Injection Frequency", 0.0)
    start_step_wave = sim_params.get("Start Step of Wave Injection", 0)
    final_step_wave = sim_params.get("End Step of Wave Injection", 0)
    n_rad_cell = sim_params.get("Number of Radial Cells", 0)
    n_azi_cell = sim_params.get("Number of Azimuth Cells", 1)

    num_periods = round((final_step_wave - start_step_wave - 1) * omega_freq * timestep / (2 * np.pi), 2) if omega_freq > 0 else 0.0

    avg_time_interval = avg_taken_every * timestep
    if macro_data is not None and "Time Stamps" in macro_data:
        time = macro_data["Time Stamps"]
    else:
        print(f"Warning: no macro pickle time stamps for {data_dir.name}. Generating time array.")
        time = np.linspace(avg_time_interval, nsteps * timestep, nsteps // avg_taken_every)

    radial_edges, azi_edges = extract_grid_edges(f"{data_dir}/Info/Grid.ascii")
    radial_cells = (radial_cell_midpoints(data_dir) - MARS_RADIUS) / 1e3
    mid_azi = 0.5 * (azi_edges[:-1] + azi_edges[1:]) if len(azi_edges) > 1 else np.array([])

    if mars_radius <= 0:
        raise ValueError("Invalid Mars radius from ASCII")
    if omega_freq <= 0:
        raise ValueError("Invalid wave frequency from ASCII")
    if timestep <= 0:
        raise ValueError("Invalid timestep from ASCII")

    return SimulationParameters(
        radial_cells=radial_cells,
        mid_azi=mid_azi,
        r1_distances=np.array([]),
        time=time,
        mars_radius=mars_radius,
        start_step_wave=start_step_wave,
        final_step_wave=final_step_wave,
        timestep=timestep,
        omega_freq=omega_freq,
        num_periods=num_periods,
        n_azi_cell=n_azi_cell,
        n_rad_cell=n_rad_cell,
        radial_edges=radial_edges,
        azi_edges=azi_edges,
    )
