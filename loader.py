import os
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from .helper_funcs import (
    compute_heating_rate_slope,
    process_temperature_deviation_peaks,
    process_growth_num_amp,
)
from .helper_funcs import (
    TIAN_GROW,
    TIAN_ALTITUDES,
    SPHERICAL_1D_GROW,
    SPHERICAL_1D_ALTITUDES,
)
from multiprocessing import Pool, cpu_count
from functools import partial

import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(context="paper", font_scale=2.3, style="whitegrid")

PADDING = 700

# compute the scale height
M = 16  # g/mol
R = 8.314  # J/mol/K
R_prime = R / M * 1e3  # J/kg/K
T = 270  # K
g = 3.51  # m/s^2
H = R_prime * T / g / 1e3  # km

# Add these constant dictionaries near the top after other constants
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


@dataclass
class SimulationParameters:
    """Storage class for simulation parameters"""

    mid_azi: np.ndarray
    radial_cells: np.ndarray
    r1_distances: np.ndarray
    time: np.ndarray
    mars_radius: float
    start_step_wave: int
    final_step_wave: int
    timestep: float
    omega_freq: float
    num_periods: float


class DSMCDataProcessor:
    """Processor for DSMC simulation data with spherical geometry considerations"""

    def __init__(self, data_dir: str, background_data_dir: Optional[str] = None):
        """
        Initialize the DSMC data processor

        Args:
            data_dir: Path to directory containing simulation output files
            background_data_dir: Optional path to directory containing background simulation data
        """
        self.data_dir = Path(data_dir)
        self.params: Optional[SimulationParameters] = None
        self.density_data: Optional[np.ndarray] = None
        self.temperature_data: Optional[np.ndarray] = None
        self.radial_vel_data: Optional[np.ndarray] = None
        self.bg_window: Optional[Dict[str, int]] = None
        self.background_profiles: Optional[Dict[str, np.ndarray]] = None
        self.background_profiles_exp: Optional[Dict[str, np.ndarray]] = None
        self.surface_data: Optional[Dict[str, np.ndarray]] = None
        self.escape_data: Optional[Dict[str, np.ndarray]] = None
        self.surface_base_cols = SURFACE_BASE_COLUMNS
        self.surface_species_cols = SURFACE_SPECIES_COLUMNS
        self.escape_base_cols = ESCAPE_BASE_COLUMNS
        self.escape_species_cols = ESCAPE_SPECIES_COLUMNS

        # Initialize background processor if alternative dataset is provided
        if background_data_dir is not None:
            self.background_processor = DSMCDataProcessor(background_data_dir)
            self.background_processor.load_parameters()
            self.background_processor.load_data()
        else:
            self.background_processor = None

    def load_parameters(self, params_file: str = "params_data.npz") -> None:
        """
        Load simulation parameters from NPZ file

        Args:
            params_file: Name of parameter file
        """
        params_path = self.data_dir / params_file
        if not params_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {params_path}")

        params = np.load(params_path, allow_pickle=True)
        print(params)
        omega_val = params["omega"] if "omega" in params else params.get("omega_freq", 0.0)
        self.params = SimulationParameters(
            radial_cells=params.get("radial_cells", np.array([])),
            mid_azi=params.get("mid_azi", np.array([])),
            r1_distances=params.get("r1_distances", np.array([])),
            time=params.get("time", np.array([])),
            mars_radius=params.get("mars_radius", 0.0),
            start_step_wave=params.get("step0_inject", 0),
            final_step_wave=params.get("stepf_inject", 0),
            timestep=params.get("timestep", 0.0),
            omega_freq=omega_val,
            num_periods=params.get("num_periods", 0.0),
        )

    def load_data(self) -> None:
        """Load all simulation data files"""
        # import ipdb; ipdb.set_trace()
        self.density_data = np.load(self.data_dir / "density_data.npy")
        if self.density_data.ndim == 4 and self.density_data.shape[0] == 1:
            self.density_data = self.density_data.squeeze(0)

        self.temperature_data = np.load(self.data_dir / "temperature_data.npy")
        if self.temperature_data.ndim == 4 and self.temperature_data.shape[0] == 1:
            self.temperature_data = self.temperature_data.squeeze(0)

        self.radial_vel_data = np.load(self.data_dir / "rad_vel_data.npy")
        if self.radial_vel_data.ndim == 4 and self.radial_vel_data.shape[0] == 1:
            self.radial_vel_data = self.radial_vel_data.squeeze(0)

        # Load Knudsen number data file
        self.knudsen_data = np.load(self.data_dir / "macro_Knudsen Number.npy")
        if self.knudsen_data.ndim == 4 and self.knudsen_data.shape[0] == 1:
            self.knudsen_data = self.knudsen_data.squeeze(0)

        # Load surface and escape data
        surface_escape_path = self.data_dir / "surface_escape_data.npz"
        if surface_escape_path.exists():
            print("Loading surface and escape data...")
            surface_escape_data = np.load(surface_escape_path, allow_pickle=True)

            # Process surface data with labeled columns - fix indexing to match:
            # surf_base_data shape: (time, azimuthal, property)
            # surf_species_data shape: (species, time, azimuthal, property)
            self.surface_data = {
                "base_data": {
                    name: surface_escape_data["surf_base_data"][:, :, col_idx]
                    for name, col_idx in self.surface_base_cols.items()
                },
                "species_data": {
                    name: surface_escape_data["surf_species_data"][:, :, :, col_idx]
                    for name, col_idx in self.surface_species_cols.items()
                },
                "species_names": surface_escape_data["species_names"],
                "flow_rates": surface_escape_data["flow_rates"],
                "surface_base_data": surface_escape_data["surface_base_data"],
            }

            # Process escape data with labeled columns - fix indexing to match:
            # escape_base_data shape: (time, azimuthal, property)
            # escape_species_data shape: (species, time, azimuthal, property)
            self.escape_data = {
                "base_data": {
                    name: surface_escape_data["escape_base_data"][:, :, col_idx]
                    for name, col_idx in self.escape_base_cols.items()
                },
                "species_data": {
                    name: surface_escape_data["escape_species_data"][:, :, :, col_idx]
                    for name, col_idx in self.escape_species_cols.items()
                },
                "species_names": surface_escape_data["species_names"],
            }

            print("Surface and escape data loaded successfully.")

            # Print data structure summary
            print("\nSurface Data Structure:")
            print("Base properties:", list(self.surface_data["base_data"].keys()))
            print("Species properties:", list(self.surface_data["species_data"].keys()))
            print(f"Number of species: {len(self.surface_data['species_names'])}")

            print("\nEscape Data Structure:")
            print("Base properties:", list(self.escape_data["base_data"].keys()))
            print("Species properties:", list(self.escape_data["species_data"].keys()))
            print(f"Number of species: {len(self.escape_data['species_names'])}")
        else:
            print("Warning: surface_escape_data.npz not found.")

    def get_wave_time_indices(self) -> Tuple[int, int]:
        """
        Calculate start and end time indices for wave analysis

        Returns:
            Tuple of (start_idx, end_idx)
        """
        if self.params is None:
            raise ValueError("Parameters not loaded. Call load_parameters() first.")

        start_idx = np.where(self.params.time > (self.params.start_step_wave * self.params.timestep))[0][0]
        end_idx = np.where(self.params.time > self.params.final_step_wave * self.params.timestep)[0][0] - 5

        return start_idx, end_idx

    def print_background_window_info(self) -> None:
        """Print information about the background window"""
        if self.bg_window is None:
            print("\nWarning: No background window information available.")
            return

        # Check if this is a window from an external processor
        is_external = self.bg_window.get("from_external", False)

        print("\nBackground Window Details:")
        if is_external:
            print("Source: External background processor")

        print(f"Background start index: {self.bg_window['start']}")
        print(f"Background end index: {self.bg_window['end']}")

        # If this is from an external processor, the time might be on a different scale
        if is_external and self.background_processor is not None:
            try:
                bg_time = self.background_processor.params.time
                print(f"Time range: [{bg_time[self.bg_window['start']]:.2f}, {bg_time[self.bg_window['end']]:.2f}] s")
            except:
                print("Time range: [unknown - external time scale]")
        else:
            print(
                f"Time range: [{self.params.time[self.bg_window['start']]:.2f}, {self.params.time[self.bg_window['end']]:.2f}] s"
            )

        print(f"Window size: {self.bg_window['end'] - self.bg_window['start']} steps")

        if hasattr(self.params, "start_step_wave"):
            wave_start = np.where(self.params.time > (self.params.start_step_wave * self.params.timestep))[0][0]
            print(f"Wave start index: {wave_start}")

        # Print additional info about the external processor if available
        if is_external and self.background_processor is not None:
            print("\nExternal Background Processor Information:")
            if hasattr(self.background_processor.params, "start_step_wave"):
                bg_wave_start = np.where(
                    self.background_processor.params.time
                    > (self.background_processor.params.start_step_wave * self.background_processor.params.timestep)
                )[0][0]
                print(f"External wave start index: {bg_wave_start}")
                print(f"External wave start time: {self.background_processor.params.time[bg_wave_start]:.2f} s")

    def _compute_window_sizes(
        self, end_idx: int, custom_time_windows: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the window sizes for background analysis.

        Args:
            end_idx: End index for the analysis period
            custom_time_windows: Optional array of custom window sizes in seconds

        Returns:
            Tuple of (time_windows, window_sizes)
        """
        dt = np.mean(np.diff(self.params.time))
        available_time = self.params.time[end_idx]
        max_window_time = min(available_time * 0.5, 2500)

        if custom_time_windows is not None:
            # Convert to numpy array if it's a list
            if isinstance(custom_time_windows, list):
                custom_time_windows = np.array(custom_time_windows)

            # Validate custom windows
            if np.any(custom_time_windows > available_time * 0.5):
                raise ValueError(
                    f"Window sizes must be less than half the available time ({available_time * 0.5:.1f} s)"
                )
            if np.any(custom_time_windows < 100):
                raise ValueError("Window sizes must be at least 100 seconds")
            time_windows = np.array(custom_time_windows)
        else:
            # Use default window sizes
            time_windows = np.linspace(1000, max_window_time, 4)

        window_sizes = np.maximum((time_windows / dt).astype(int), 3)
        return time_windows, window_sizes

    def _find_optimal_window(self, data: np.ndarray, window_sizes: np.ndarray) -> Tuple[int, Tuple[int, int], float]:
        """
        Find optimal window for a time series using MAD.

        Args:
            data: Time series data
            window_sizes: Array of window sizes to try

        Returns:
            Tuple of (window_size, (start_idx, end_idx), mad_value)
        """
        return find_min_mad_window(data, window_sizes)

    def _compute_background_value(self, data: np.ndarray, window_start: int, window_end: int) -> float:
        """
        Compute background value for a given window.

        Args:
            data: Time series data
            window_start: Start index of window
            window_end: End index of window

        Returns:
            Background value (mean within window)
        """
        return np.mean(data[window_start : window_end + 1])

    def get_background_profiles(self, custom_time_windows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate background profiles using standard method.

        If a background processor is provided, the standard method will be applied
        on the background processor's data instead of the primary processor's data.

        Args:
            custom_time_windows: Optional array of custom window sizes in seconds

        Returns:
            Dictionary containing background profiles
        """
        # If background_profiles already computed, return them
        if self.background_profiles is not None:
            return self.background_profiles

        # Decide which processor to use for background computation
        processor_to_use = self.background_processor if self.background_processor is not None else self

        # If using background processor, verify compatible altitude grid
        if self.background_processor is not None:
            if self.params is None or self.background_processor.params is None:
                raise ValueError("Parameters not loaded for primary or background processor.")
            if not np.allclose(self.params.radial_cells, self.background_processor.params.radial_cells):
                raise ValueError("Background dataset has different altitude grid")
            print("\nUsing external background processor for standard background computation")

        # Check if data is loaded in the processor we're using
        if any(x is None for x in [processor_to_use.density_data, processor_to_use.temperature_data]):
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get wave timing from the processor we're using
        start_idx, _ = processor_to_use.get_wave_time_indices()
        end_idx = start_idx - 5

        # Get reference altitude data
        ref_altitude = 230.0
        ref_idx = np.abs(processor_to_use.params.radial_cells - ref_altitude).argmin()
        density_ref_series = processor_to_use.density_data[:end_idx, ref_idx, 0]
        temperature_ref_series = processor_to_use.temperature_data[:end_idx, ref_idx, 0]

        # Compute window sizes with custom windows if provided
        time_windows, window_sizes = processor_to_use._compute_window_sizes(end_idx, custom_time_windows)

        print(f"\nFinding optimal window using MAD at reference altitude {ref_altitude} km:")
        print("Time windows considered:", time_windows.round(1), "seconds")
        print("Index windows:", window_sizes, "timesteps")
        print(f"(using dt = {np.mean(np.diff(processor_to_use.params.time)):.2f} s)")

        # Find optimal windows - check for None results
        density_result = processor_to_use._find_optimal_window(density_ref_series, window_sizes)
        temp_result = processor_to_use._find_optimal_window(temperature_ref_series, window_sizes)

        # Check if valid windows were found - use fallback if not
        if density_result[0] is None or temp_result[0] is None:
            print("\nWarning: Couldn't find valid background windows with standard method.")
            print("Using fallback window method...")

            # Calculate a fallback window - use first 30% of data before wave start
            fallback_window_end = max(10, int(end_idx * 0.3))
            fallback_window_start = max(0, fallback_window_end - window_sizes[0])

            # Create fallback results
            d_win_size = temp_win_size = window_sizes[0]
            d_win_start = t_win_start = fallback_window_start
            d_win_end = t_win_end = fallback_window_end
            d_mad = t_mad = np.nan  # We don't have a MAD value for the fallback

            print(f"Using fallback window: [{fallback_window_start}, {fallback_window_end}]")
            print(
                f"Time range: [{processor_to_use.params.time[fallback_window_start]:.2f}, {processor_to_use.params.time[fallback_window_end]:.2f}] s"
            )
        else:
            # Unpack results from optimal window finding
            d_win_size, (d_win_start, d_win_end), d_mad = density_result
            t_win_size, (t_win_start, t_win_end), t_mad = temp_result

            # Print window information
            processor_to_use._print_window_info("Density", d_win_size, d_win_start, d_win_end, d_mad)
            processor_to_use._print_window_info("Temperature", t_win_size, t_win_start, t_win_end, t_mad)

        # Use average window
        window_info = {
            "start": (d_win_start + t_win_start) // 2,
            "end": (d_win_end + t_win_end) // 2,
        }

        # Set bg_window in both processors
        processor_to_use.bg_window = window_info

        # If we're using background processor, also set the window in the primary processor
        if self.background_processor is not None:
            self.bg_window = {
                "start": window_info["start"],
                "end": window_info["end"],
                "from_external": True,  # Mark that this window is from external processor
            }

        # Compute background profiles
        num_altitudes = processor_to_use.density_data.shape[1]
        density_bg = np.empty(num_altitudes)
        temperature_bg = np.empty(num_altitudes)

        for alt in range(num_altitudes):
            density_bg[alt] = processor_to_use._compute_background_value(
                processor_to_use.density_data[:end_idx, alt, 0],
                window_info["start"],
                window_info["end"],
            )
            temperature_bg[alt] = processor_to_use._compute_background_value(
                processor_to_use.temperature_data[:end_idx, alt, 0],
                window_info["start"],
                window_info["end"],
            )

        # Store background profiles in the current processor
        self.background_profiles = {
            "density": density_bg,
            "temperature": temperature_bg,
            "source": "external" if self.background_processor is not None else "primary",
        }

        processor_to_use.print_background_window_info()

        if self.background_processor is not None:
            print("\nBackground profiles computed from external processor")

        return self.background_profiles

    def get_background_profiles_experimental(
        self, custom_time_windows: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate background profiles using experimental method with parallel processing.

        If a background processor is provided, the experimental method will be applied
        on the background processor's data instead of the primary processor's data.

        Args:
            custom_time_windows: Optional array of custom window sizes in seconds

        Returns:
            Dictionary containing background profiles with optimal windows
        """
        # Decide which processor to use for background computation
        processor_to_use = self.background_processor if self.background_processor is not None else self

        # If using background processor, verify compatible altitude grid
        if self.background_processor is not None:
            if self.params is None or self.background_processor.params is None:
                raise ValueError("Parameters not loaded for primary or background processor.")
            if not np.allclose(self.params.radial_cells, self.background_processor.params.radial_cells):
                raise ValueError("Background dataset has different altitude grid")
            print("\nUsing external background processor for experimental background computation")

        # Check if data is loaded in the processor we're using
        if any(x is None for x in [processor_to_use.density_data, processor_to_use.temperature_data]):
            raise ValueError("Data must be loaded before computing background profiles")

        # Get wave timing from the processor we're using
        start_idx, end_idx = processor_to_use.get_wave_time_indices()
        end_idx = start_idx - 5

        # Compute window sizes with custom windows if provided
        time_windows, window_sizes = processor_to_use._compute_window_sizes(end_idx, custom_time_windows)

        print("\nParallel computation of optimal windows for each altitude")
        print(f"Time windows considered: {time_windows.round(1)} seconds")
        print(f"Index windows: {window_sizes}")
        print(f"Using dt = {np.mean(np.diff(processor_to_use.params.time)):.2f} s")

        # Initialize storage
        num_altitudes = processor_to_use.density_data.shape[1]
        optimal_windows = np.zeros((num_altitudes, 2), dtype=int)
        density_bg = np.empty(num_altitudes)
        temperature_bg = np.empty(num_altitudes)

        # Prepare data for parallel processing
        density_data_slices = [processor_to_use.density_data[:end_idx, alt, 0] for alt in range(num_altitudes)]
        temp_data_slices = [processor_to_use.temperature_data[:end_idx, alt, 0] for alt in range(num_altitudes)]
        altitudes = processor_to_use.params.radial_cells

        # Create partial functions for parallel processing
        density_worker = partial(_parallel_window_finder, window_sizes=window_sizes)
        temp_worker = partial(_parallel_window_finder, window_sizes=window_sizes)

        # Determine optimal number of processes
        num_processes = min(cpu_count(), num_altitudes)
        print(f"\nUsing {num_processes} processes for parallel computation")

        # Process density and temperature data in parallel
        with Pool(num_processes) as pool:
            # Process density data
            density_args = [(i, altitudes[i], data) for i, data in enumerate(density_data_slices)]
            density_results = pool.starmap(density_worker, density_args)

            # Process temperature data
            temp_args = [(i, altitudes[i], data) for i, data in enumerate(temp_data_slices)]
            temp_results = pool.starmap(temp_worker, temp_args)

        # Sort results by altitude index
        density_results.sort()
        temp_results.sort()

        # Process results for each altitude
        for i in range(num_altitudes):
            d_idx, (d_start, d_end, d_temps, d_mad) = density_results[i]
            t_idx, (t_start, t_end, t_temps, t_mad) = temp_results[i]

            if None in (d_start, t_start):
                # If either density or temperature window finding failed, mark data as invalid
                optimal_windows[i] = [-1, -1]
                density_bg[i] = np.nan
                temperature_bg[i] = np.nan
                print(f"\nWarning: Could not find valid window at altitude {altitudes[i]:.1f} km")
                continue

            # Use average of density and temperature windows
            win_start = (d_start + t_start) // 2
            win_end = (d_end + t_end) // 2
            optimal_windows[i] = [win_start, win_end]

            # Compute background values
            density_bg[i] = np.mean(processor_to_use.density_data[win_start : win_end + 1, i, 0])
            temperature_bg[i] = np.mean(processor_to_use.temperature_data[win_start : win_end + 1, i, 0])

            # Print progress every 10 altitudes
            if i % 10 == 0:
                print(f"\nProcessed altitude {altitudes[i]:.1f} km:")
                print(f"Density MAD: {d_mad:.2e}, Temperature MAD: {t_mad:.2e}")
                print(
                    f"Window: [{win_start}, {win_end}] "
                    f"(Time: [{processor_to_use.params.time[win_start]:.1f}, {processor_to_use.params.time[win_end]:.1f}]s)"
                )

        # Store the background profiles
        background_profiles_exp = {
            "density": density_bg,
            "temperature": temperature_bg,
            "optimal_windows": optimal_windows,
            "source": "external" if self.background_processor is not None else "primary",
        }

        # If we're using the primary processor, save to its attribute
        if processor_to_use == self:
            self.background_profiles_exp = background_profiles_exp

        # If we're using a background processor, we need to copy the profiles to this processor
        if self.background_processor is not None:
            # Store the profiles in this processor too
            self.background_profiles_exp = background_profiles_exp

            # Mark that these were derived from external processor
            print("\nExperimental background profiles computed from external processor")

        # Plot the optimal windows using the processor that generated them
        processor_to_use._plot_optimal_windows(optimal_windows)

        return background_profiles_exp

    def create_stitched_visualization(
        self, data_field: str = "density", altitudes: Optional[list] = None, show_bg_window: bool = True
    ) -> None:
        """
        Create a visualization that shows stitched data from background and primary processors.

        Args:
            data_field: The field to visualize (density, temperature, rad_vel)
            altitudes: List of altitudes to plot [km]. If None, uses 150km and 320km.
            show_bg_window: Whether to show the background window used for computation
        """
        if self.background_processor is None:
            raise ValueError("Background processor not available. Cannot visualize stitched data.")

        if altitudes is None:
            # Default to 150km and 320km
            altitudes = [150, 320]

        # Get cell indices for requested altitudes
        radial_distances = self.params.radial_cells
        altitude_indices = [np.abs(radial_distances - alt).argmin() for alt in altitudes]

        # Get stitched data
        stitched_data = self.stitch_background_data(data_field)

        # Unpack data
        bg_data = stitched_data["bg_data"]
        bg_time = stitched_data["bg_time"]
        primary_data = stitched_data["primary_data"]
        primary_time = stitched_data["primary_time"]
        bg_cut_idx = stitched_data["bg_cut_idx"]
        wave_start_idx = stitched_data["wave_start_idx"]

        # Create figure
        fig, axes = plt.subplots(len(altitude_indices), 1, figsize=(12, 4 * len(altitude_indices)), sharex=True)
        if len(altitude_indices) == 1:
            axes = [axes]  # Make axes iterable for single altitude case

        colors = plt.cm.tab10(np.linspace(0, 1, len(altitude_indices)))

        # Plot for each altitude
        for i, (alt_idx, alt, color) in enumerate(zip(altitude_indices, altitudes, colors)):
            ax = axes[i]

            # Plot background data up to cut point
            bg_line = ax.plot(
                bg_time[:bg_cut_idx],
                bg_data[:bg_cut_idx, alt_idx, 0],
                color=color,
                linestyle="--",
                label=f"Background data ({alt} km)",
            )

            # Plot primary data from wave start
            primary_line = ax.plot(
                primary_time[wave_start_idx:],
                primary_data[wave_start_idx:, alt_idx, 0],
                color=color,
                linestyle="-",
                label=f"Primary data ({alt} km)",
            )

            # Add vertical line at stitch point
            ax.axvline(bg_time[bg_cut_idx - 1], color="magenta", linestyle="-.", label="Stitch point" if i == 0 else "")

            # If requested, show background window used for computation
            assert show_bg_window, "Background window must be shown for stitched visualization"
            if show_bg_window:
                # Check if we have background window info
                bg_processor = self.background_processor
                if bg_processor.bg_window is not None:
                    # Plot background window
                    bg_start = bg_processor.bg_window["start"]
                    bg_end = bg_processor.bg_window["end"]

                    # Make sure indices are within range
                    if bg_start < len(bg_time) and bg_end < len(bg_time):
                        window_patch = ax.axvspan(
                            bg_time[bg_start],
                            bg_time[bg_end],
                            color="gray",
                            alpha=0.2,
                            label="Background window" if i == 0 else "",
                        )

                # Also check if we have experimental windows for this altitude
                # Try to get experimental profiles from either processor
                exp_profiles = None
                if (
                    hasattr(bg_processor, "background_profiles_exp")
                    and bg_processor.background_profiles_exp is not None
                ):
                    exp_profiles = bg_processor.background_profiles_exp
                elif hasattr(self, "background_profiles_exp") and self.background_profiles_exp is not None:
                    exp_profiles = self.background_profiles_exp

                if exp_profiles is not None and "optimal_windows" in exp_profiles:
                    optimal_windows = exp_profiles["optimal_windows"]

                    # Check if this altitude has a valid window
                    if alt_idx < optimal_windows.shape[0]:
                        exp_start, exp_end = optimal_windows[alt_idx]

                        # Only show if valid window
                        if exp_start >= 0 and exp_end >= 0 and exp_start < len(bg_time) and exp_end < len(bg_time):
                            # Different colors for each altitude
                            # color = "C0" if i == 0 else "C1"
                            color = "#FF9AA2" if i == 0 else "#FFDAC1"  # Soft pink for 150km, soft orange for 320km
                            # Use the actual altitude value from the current alt
                            window_label = f"Exp. window ({alt:.2f} km)"

                            exp_window = ax.axvspan(
                                bg_time[exp_start],
                                bg_time[exp_end],
                                color=color,
                                alpha=0.2,  # Increased alpha for better visibility
                                label=window_label if i == 0 else "",
                            )

            # Add title and labels
            ax.set_title(f"{data_field.capitalize()} at {alt} km")
            ax.set_ylabel(f"{data_field.capitalize()}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel("Time (s)")

        # Add overall title
        method_info = ""
        if hasattr(self, "background_profiles") and self.background_profiles is not None:
            method_info = f" (Source: {self.background_profiles.get('source', 'unknown')})"

        fig.suptitle(f"Stitched {data_field.capitalize()} Data Visualization{method_info}", fontsize=16)

        # Adjust layout and show
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def _print_window_info(self, name: str, win_size: int, start: int, end: int, mad: float) -> None:
        """Print information about an optimal window."""
        dt = np.mean(np.diff(self.params.time))
        print(f"\n{name} optimal window - Size: {win_size} steps ({win_size * dt:.1f} s)")
        print(f"Indices: [{start}, {end}], MAD: {mad:.2e}")
        print(f"Time range: [{self.params.time[start]:.1f}, {self.params.time[end]:.1f}] s")

    def _plot_optimal_windows(self, optimal_windows: np.ndarray) -> None:
        """
        Visualize the optimal windows chosen for each altitude with a sequential color palette.

        Args:
            optimal_windows: Array of (start, end) indices for each altitude
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a sequential colormap for the altitude range
        altitudes = self.params.radial_cells
        norm = plt.Normalize(altitudes.min(), altitudes.max())
        cmap = plt.cm.viridis  # Can also try: 'plasma', 'magma', 'inferno'

        # Plot the window ranges for each altitude with color mapping
        for alt_idx, (start, end) in enumerate(optimal_windows):
            color = cmap(norm(altitudes[alt_idx]))
            ax.hlines(
                altitudes[alt_idx],
                self.params.time[start],
                self.params.time[end],
                alpha=0.7,
                color=color,
            )

        # Get ylimits
        ymin, ymax = ax.get_ylim()

        # Create a scatter plot with a single point for the colorbar
        sc = ax.scatter([0], [0], c=[0], cmap=cmap, norm=norm, visible=False)

        # Set older ylimits
        ax.set_ylim(ymin, ymax)

        # Add wave start time reference
        start_idx, _ = self.get_wave_time_indices()
        wave_start = self.params.time[start_idx]
        ax.axvline(wave_start, color="r", linestyle="--", label="Wave Start")

        # Add reference points for 150 km and 320 km
        cell_150km = np.where(self.params.radial_cells > 150)[0][0]
        top_radius_idx = np.where(self.params.radial_cells > 320)[0][0]
        for alt_idx in [cell_150km, top_radius_idx]:
            ax.axhline(
                self.params.radial_cells[alt_idx],
                color="k",
                linestyle=":",
                alpha=0.5,
                label=f"{int(round(self.params.radial_cells[alt_idx], -1))} km",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (km)")
        ax.set_title("Optimal Background Windows by Altitude")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def stitch_background_data(self, data_field: str = "density") -> np.ndarray:
        """
        Stitch data from background processor and primary processor.

        This combines pre-wave data from the background processor with during/post-wave data
        from the primary processor to create a continuous timeline.

        Args:
            data_field: The field to stitch (e.g., "density", "temperature")

        Returns:
            Stitched data array with timeline information
        """
        if self.background_processor is None:
            raise ValueError("Background processor not available. Cannot stitch data.")

        # Get wave timing indices for primary dataset
        start_idx, end_idx = self.get_wave_time_indices()

        # Get the same data field from both processors
        if data_field == "density":
            primary_data = self.density_data
            bg_data = self.background_processor.density_data
        elif data_field == "temperature":
            primary_data = self.temperature_data
            bg_data = self.background_processor.temperature_data
        elif data_field == "rad_vel":
            primary_data = self.radial_vel_data
            bg_data = self.background_processor.radial_vel_data
        else:
            raise ValueError(f"Unsupported data field: {data_field}")

        # Get time arrays from both processors
        primary_time = self.params.time
        bg_time = self.background_processor.params.time

        # Find the cut point in the background data
        # We want to use background data up to when the wave starts in the primary data
        bg_cut_time = primary_time[start_idx]
        bg_cut_idx = np.argmin(np.abs(bg_time - bg_cut_time))

        # Create stitched data dictionary with time and data arrays
        return {
            "primary_data": primary_data,
            "bg_data": bg_data,
            "primary_time": primary_time,
            "bg_time": bg_time,
            "wave_start_idx": start_idx,
            "bg_cut_idx": bg_cut_idx,
        }

    def calculate_perturbations(self) -> Dict[str, np.ndarray]:
        """
        Calculate perturbations from background state

        Returns:
            Dictionary containing perturbation fields
        """
        background = self.get_background_profiles()
        start_idx, end_idx = self.get_wave_time_indices()

        # Add safety check for bg_window
        if self.bg_window is None:
            print("Warning: Background window information is missing. Using wave indices as fallback.")
            self.bg_window = {"start": max(0, start_idx - 100), "end": max(0, start_idx - 10)}

        wave_period = slice(start_idx, end_idx)

        return {
            "density": (self.density_data[wave_period] - background["density"]) / background["density"],
            "temperature": (self.temperature_data[wave_period] - background["temperature"]) / background["temperature"],
        }

    # Add helper method to access surface/escape data
    def get_surface_property(self, property_name: str, species_idx: int = None) -> np.ndarray:
        """
        Get surface property data by name.

        Args:
            property_name: Name of the property to retrieve
            species_idx: Species index for species-specific properties

        Returns:
            Property data as numpy array
        """
        if property_name in self.surface_base_cols:
            return self.surface_data["base_data"][property_name]
        elif property_name in self.surface_species_cols and species_idx is not None:
            return self.surface_data["species_data"][property_name][species_idx]
        else:
            raise ValueError(f"Invalid property name or missing species index: {property_name}")

    def get_escape_property(self, property_name: str, species_idx: int = None) -> np.ndarray:
        """
        Get escape property data by name.

        Args:
            property_name: Name of the property to retrieve
            species_idx: Species index for species-specific properties

        Returns:
            Property data as numpy array
        """
        if property_name in self.escape_base_cols:
            return self.escape_data["base_data"][property_name]
        elif property_name in self.escape_species_cols and species_idx is not None:
            return self.escape_data["species_data"][property_name][species_idx]
        else:
            raise ValueError(f"Invalid property name or missing species index: {property_name}")


def compute_growth(
    data_dir: str,
    background_data_dir: Optional[str] = None,
    alt_range: Tuple[float, float] = (150, 320),
    use_experimental: bool = False,
    use_dynamic: bool = False,
    window_sizes: Optional[np.ndarray] = None,
):
    """
    Analyze wave growth in a specified altitude range.

    This function supports two independent features:
    1. Background computation method (standard vs experimental)
    2. Data source for background (primary data vs external data)

    When an external background dataset is provided, it will be used as the data source
    for both standard and experimental background methods. The visualization will
    show stitched data combining the external background (pre-wave) with the primary data
    (during/post-wave).

    Args:
        data_dir: Directory containing simulation data
        background_data_dir: Optional directory containing background simulation data.
                            When provided, background windows are computed on this external
                            dataset, and data is stitched for visualization.
        alt_range: Tuple of (min_altitude, max_altitude) in km
        use_experimental: Whether to use experimental background method with per-altitude windows.
                         Works with either primary or external data source.
        use_dynamic: Whether to use dynamic peak parameter adjustment
        window_sizes: Optional array of window sizes in seconds for background computation
    """
    # Create exports directory if it doesn't exist
    exports_dir = Path(data_dir) / "exports"
    exports_dir.mkdir(exist_ok=True, parents=True)

    print("\nProcessing data from:", data_dir)
    print(f"Saving visualizations to: {exports_dir}")

    if background_data_dir:
        print("Using background data from:", background_data_dir)

    print(f"Using {'experimental' if use_experimental else 'standard'} background computation")
    print(f"Using {'dynamic' if use_dynamic else 'static'} peak detection parameters")
    print(f"Analyzing altitude range: {alt_range[0]:.1f} - {alt_range[1]:.1f} km")

    if window_sizes is not None:
        print(f"Using custom window sizes: {window_sizes} seconds")

    # Initialize processor with background_data_dir
    processor = DSMCDataProcessor(data_dir, background_data_dir=background_data_dir)
    processor.load_parameters()
    processor.load_data()

    # Get wave timing indices and parameters
    start_idx, end_idx = processor.get_wave_time_indices()
    time = processor.params.time
    radial_distances = processor.params.radial_cells

    # Print simulation parameters
    print("\nSimulation Parameters:")
    print(f"Wave frequency: {processor.params.omega_freq:.4f} Hz")
    print(f"Time step: {processor.params.timestep:.4f} s")
    print(f"Number of periods: {processor.params.num_periods}")
    print(f"Altitude range: {radial_distances[0]:.1f} - {radial_distances[-1]:.1f} km")

    # Get background profiles using both methods and compare
    experimental_background = processor.get_background_profiles_experimental(window_sizes)
    standard_background = processor.get_background_profiles(window_sizes)

    # Compute relative error between methods
    relative_error = (
        np.abs(experimental_background["density"] - standard_background["density"]) / standard_background["density"]
    )
    avg_rel_error = np.mean(relative_error) * 100  # as percentage

    # Compare the backgrounds with error metric
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        standard_background["density"],
        radial_distances,
        "r--",
        label="Standard",
    )
    plt.semilogx(
        experimental_background["density"],
        radial_distances,
        "b-",
        label="Experimental",
    )

    plt.xlabel("Background Density")
    plt.ylabel("Altitude (km)")
    plt.title(f"Comparison of Background Methods\nAvg. Relative Error: {avg_rel_error:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the background comparison figure
    bg_comparison_path = exports_dir / "background_density_comparison.png"
    plt.savefig(bg_comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {bg_comparison_path}")
    plt.close()

    # Use the selected background method for further processing
    background = experimental_background if use_experimental else standard_background

    # Print wave start information since it's not part of the background info
    print(f"Wave start index: {start_idx}")

    # Compute perturbations for both methods
    print("\nComputing perturbations...")
    density_perturbation_exp = (
        processor.density_data - experimental_background["density"][None, :, None]
    ) / experimental_background["density"][None, :, None]

    density_perturbation_std = (
        processor.density_data - standard_background["density"][None, :, None]
    ) / standard_background["density"][None, :, None]

    temp_deviations = processor.temperature_data - 270.0

    plt.figure(figsize=(8, 6))
    plt.axhline(270, color="k", linestyle="--", label="Background Temperature")
#     colors = {
#     '150km': 'green',
#     '200km': 'orange',
#     '250km': 'purple',
#     '300km': 'cyan'
# }
    # instead pick colors from colorblind-friendly palette from seaborn
    colors = sns.color_palette("rocket", 3)
    colors = {
        '150km': colors[0],
        # '200km': colors[1],
        # '250km': colors[3],
        '300km': colors[2]
    }

    # Add temperature evolution visualization at 150 km and 300 km
    for ref_altitude in [150, 300]:
        alt_label_str = f"{int(ref_altitude)}km"
        ref_idx = np.abs(radial_distances - ref_altitude).argmin()
        temp_series = processor.temperature_data[:, ref_idx, 0]
        time = processor.params.time - processor.params.time[start_idx]

        # Calculate window sizes in timesteps for different averaging periods
        # Create temperature evolution visualization
        plt.plot(
            time,
            temp_series,
            color=colors[alt_label_str],
            alpha=0.9,
            linewidth=1.8,
            linestyle="-",
            # marker="^",
            label='_nolegend_'
        )

    # Add reference curves from CSV files
    altitude_files = {
        '150km': '/mnt/performance/data-analysis/exports/150km_altitude_fast.csv',
        # '200km': '/mnt/performance/data-analysis/exports/200km_altitude.csv',
        # '250km': '/mnt/performance/data-analysis/exports/250km_altitude.csv',
        '300km': '/mnt/performance/data-analysis/exports/300km_altitude_fast.csv'
    }

    for alt_label, file_path in altitude_files.items():
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                # Load data from CSV
                data = np.loadtxt(file_path, delimiter=',')

                # Convert normalized temperature back to actual temperature
                # Original data is (T - 270) / 270, so T = 270 * (1 + normalized_value)
                # time_values = data[:, 0]
                time_values = data[:, 0] - 5300
                # time_values = data[:, 0] - 5296
                normalized_temps = data[:, 1]
                actual_temps = 270 * (1 + normalized_temps)

                # Plot the reference curve
                plt.plot(
                    time_values,
                    actual_temps,
                    color=colors[alt_label],
                    # marker='x',
                    linestyle='--',
                    linewidth=1.2,
                    markersize=1,
                    alpha=0.7,
                    label='_nolegend_'
                )
            except Exception as e:
                print(f"Error loading {alt_label} data: {e}")

    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    # plt.title(f"Temperature Evolution at $r=${ref_altitude:.1f} km")
    # plt.title("Temperature Evolution")
    plt.grid(True, alpha=0.3)
    plt.margins(x=0)
    plt.xlim(left=-10, right=2300)
    # plt.legend()
    plt.ylim(216, 405)
    plt.tight_layout()

    # Create custom legend handles
    color_handles = [
        Line2D([], [], color=colors['150km'], linestyle='-', label='150km'),
        # Line2D([], [], color=colors['200km'], linestyle='-', label='200km'),
        # Line2D([], [], color=colors['250km'], linestyle='-', label='250km'),
        Line2D([], [], color=colors['300km'], linestyle='-', label='300km'),
    ]
    marker_handles = [
        Line2D([], [], color='black', linestyle='-', label='Harrah Spherical 1D'),
        Line2D([], [], color='black', linestyle='--', label='Tian et al., 2023'),
    ]
    plt.legend(handles=color_handles + marker_handles, fontsize=16, ncol=2, loc='upper left', frameon=True)
    # plt.xlim(right=1775)

    # Save the temperature evolution figure
    temp_evolution_path = exports_dir / "temperature_evolution_at_multi_alt.png"
    plt.savefig(temp_evolution_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {temp_evolution_path}")
    plt.close()

    # Find cell indices for analysis regions using input altitude range
    cell_min_idx = np.where(radial_distances > alt_range[0])[0][0]
    cell_max_idx = np.where(radial_distances > alt_range[1])[0][0]
    print("\nAnalysis altitude indices:")
    print(f"Lower altitude index: {cell_min_idx} (altitude: {radial_distances[cell_min_idx]:.1f} km)")
    print(f"Upper altitude index: {cell_max_idx} (altitude: {radial_distances[cell_max_idx]:.1f} km)")

    # Process peaks for both methods
    print("\nProcessing density growth peaks")
    base_kwargs = {
        "prominence": 1e-1,
        "distance": 12,
        "sampling_rate": 1 / np.mean(np.diff(processor.params.time)),
        "cutoff_frequency": processor.params.omega_freq * 1.2,
        "filter_order": 2,
        "use_dynamic": use_dynamic,  # Add dynamic flag to kwargs
    }

    time = processor.params.time
    # Process standard method
    growth_peaks_std, baseline_peaks_std = process_growth_num_amp(
        density_perturbation_std[..., 0],
        (processor.params.start_step_wave * processor.params.timestep <= time)
        & (time <= processor.params.final_step_wave * processor.params.timestep + PADDING),
        radial_distances,
        cell_min_idx,
        cell_max_idx,
        processor.params.num_periods,
        **base_kwargs,
    )

    # Process experimental method
    growth_peaks_exp, baseline_peaks_exp = process_growth_num_amp(
        density_perturbation_exp[..., 0],
        (processor.params.start_step_wave * processor.params.timestep <= time)
        & (time <= processor.params.final_step_wave * processor.params.timestep + PADDING),
        radial_distances,
        cell_min_idx,
        cell_max_idx,
        processor.params.num_periods,
        **base_kwargs,
    )

    print("Processing temperature deviations")
    # Process temperature deviations (no experimental comparison needed)
    temp_peaks, temp_last_cycle = process_temperature_deviation_peaks(
        temp_deviations[..., 0],
        time >= processor.params.start_step_wave * processor.params.timestep,
        radial_distances,
        cell_min_idx,
        cell_max_idx,
        processor.params.num_periods,
        **base_kwargs,
    )

    # Visualize density amplitude growth comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    altitudes = radial_distances[cell_min_idx : cell_max_idx + 1]

    # Plot standard method results with native masked array support
    valid_data_std = growth_peaks_std[:, -1].squeeze()
    ax.plot(valid_data_std, altitudes, "r--", label="Standard Method", alpha=0.7)

    # Plot experimental method results with native masked array support
    valid_data_exp = growth_peaks_exp[:, -1].squeeze()
    ax.plot(valid_data_exp, altitudes, "b-", label="Experimental Method", alpha=0.7)

    ax.set_xlabel("Density Amplitude")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Density Amplitude Growth Comparison")

    # Add comparison data
    ax.plot(TIAN_GROW, TIAN_ALTITUDES, "k--", label="Tian et al. (2019)", alpha=0.5)
    ax.plot(
        SPHERICAL_1D_GROW,
        SPHERICAL_1D_ALTITUDES,
        "k:",
        label="Spherical 1D",
        alpha=0.5,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save the density amplitude growth comparison figure
    growth_comparison_path = exports_dir / "density_amplitude_growth_comparison.png"
    plt.savefig(growth_comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {growth_comparison_path}")
    plt.close()

    # Visualize temperature deviations and compute heating rate slope
    # Modify the compute_heating_rate_slope function call to save instead of display
    slope, slope_H = compute_heating_rate_slope(
        temp_last_cycle,
        radial_distances,
        cell_min_idx,
        cell_max_idx,
        H,
        visualize=True,
        save_path=exports_dir / "heating_rate_slope.png",
    )

    # Density evolution visualization with background regions
    fig, (ax1_top, ax2_top) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Create twin axes for 320 km data
    ax1_bottom = ax1_top.twinx()
    ax2_bottom = ax2_top.twinx()

    # Initialize lists to collect all lines and labels for the combined legend
    all_lines = []
    all_labels = []

    # Check if we're using background data from a different dataset
    using_external_bg = background_data_dir is not None

    # For external background case, we'll show stitched data in both subplots
    if using_external_bg and processor.background_processor is not None:
        try:
            # Stitch density data from background and primary processors
            stitched_data = processor.stitch_background_data("density")

            # Unpack stitched data
            bg_data = stitched_data["bg_data"]
            bg_time = stitched_data["bg_time"]
            primary_data = stitched_data["primary_data"]
            primary_time = stitched_data["primary_time"]
            bg_cut_idx = stitched_data["bg_cut_idx"]
            wave_start_idx = stitched_data["wave_start_idx"]

            # Plot data in both subplots with stitched data
            print("\nPlotting stitched data with external background")
            for subplot_idx, (ax_top, ax_bottom) in enumerate([(ax1_top, ax1_bottom), (ax2_top, ax2_bottom)]):
                # Background data (pre-wave)
                bg_line_150 = ax_top.plot(
                    bg_time[:bg_cut_idx], bg_data[:bg_cut_idx, cell_min_idx, 0], color="C0", linestyle="--"
                )
                bg_line_320 = ax_bottom.plot(
                    bg_time[:bg_cut_idx], bg_data[:bg_cut_idx, cell_max_idx, 0], color="C1", linestyle="--"
                )

                # Primary data (during/post-wave)
                primary_line_150 = ax_top.plot(
                    primary_time[wave_start_idx:],
                    primary_data[wave_start_idx:, cell_min_idx, 0],
                    color="C0",
                    linestyle="-",
                )
                primary_line_320 = ax_bottom.plot(
                    primary_time[wave_start_idx:],
                    primary_data[wave_start_idx:, cell_max_idx, 0],
                    color="C1",
                    linestyle="-",
                )

                # Add vertical line at the stitch point
                stitch_line = ax_top.axvline(bg_time[bg_cut_idx - 1], color="magenta", linestyle="-.")
                ax_bottom.axvline(bg_time[bg_cut_idx - 1], color="magenta", linestyle="-.")

                # Set labels
                ax_top.set_ylabel("Density (150 km)", color="C0")
                ax_bottom.set_ylabel("Density (320 km)", color="C1")

                # Add background windows based on which subplot this is
                if subplot_idx == 0:  # Standard method subplot
                    # Add standard background window if available
                    bg_processor = processor.background_processor
                    if bg_processor.bg_window is not None:
                        # Plot background window
                        bg_start = bg_processor.bg_window["start"]
                        bg_end = bg_processor.bg_window["end"]

                        # Make sure indices are within range
                        if bg_start < len(bg_time) and bg_end < len(bg_time):
                            print(
                                f"Drawing standard background window: Time [{bg_time[bg_start]:.1f}, {bg_time[bg_end]:.1f}]"
                            )
                            window_patch = ax_top.axvspan(
                                bg_time[bg_start],
                                bg_time[bg_end],
                                color="gray",
                                alpha=0.2,
                                label="Standard background window",
                            )
                            all_lines.append(window_patch)
                            all_labels.append("Standard background window")
                else:  # Experimental method subplot
                    # Add experimental window if available
                    print("Drawing experimental background windows")

                    # Check for experimental profiles in either processor
                    exp_profiles = None
                    if (
                        hasattr(bg_processor, "background_profiles_exp")
                        and bg_processor.background_profiles_exp is not None
                    ):
                        exp_profiles = bg_processor.background_profiles_exp
                    elif (
                        hasattr(processor, "background_profiles_exp") and processor.background_profiles_exp is not None
                    ):
                        exp_profiles = processor.background_profiles_exp

                    if exp_profiles is not None and "optimal_windows" in exp_profiles:
                        optimal_windows = exp_profiles["optimal_windows"]

                        # Add windows for both altitudes
                        for idx, alt_idx in enumerate([cell_min_idx, cell_max_idx]):
                            ax = ax_top if idx == 0 else ax_bottom

                            # Check if this altitude has a valid window
                            print(f"Checking experimental window for {radial_distances[alt_idx]:.1f} km")
                            if alt_idx < optimal_windows.shape[0]:
                                exp_start, exp_end = optimal_windows[alt_idx]

                                # Only show if valid window
                                print(f"Experimental window indices: [{exp_start}, {exp_end}]")
                                if (
                                    exp_start >= 0
                                    and exp_end >= 0
                                    and exp_start < len(bg_time)
                                    and exp_end < len(bg_time)
                                ):
                                    # Use different colors for different altitudes
                                    altitude_km = radial_distances[alt_idx]
                                    color = (
                                        "#FF9AA2" if idx == 0 else "#FFDAC1"
                                    )  # Soft pink for 150km, soft orange for 320km

                                    print(
                                        f"Drawing external experimental window for {altitude_km:.2f} km: Time [{bg_time[exp_start]:.1f}, {bg_time[exp_end]:.1f}]"
                                    )
                                    exp_window = ax.axvspan(
                                        bg_time[exp_start],
                                        bg_time[exp_end],
                                        color=color,
                                        alpha=0.3,  # Increased alpha for better visibility
                                        label=f"Exp. window ({altitude_km:.2f} km)",
                                        zorder=5,  # Make sure windows are drawn on top
                                    )

                                    # Only add to legend if not already there
                                    window_label = f"Exp. window ({altitude_km:.2f} km)"
                                    if window_label not in all_labels:
                                        all_lines.append(exp_window)
                                        all_labels.append(window_label)

            # Only add these lines once to the legend
            if not all_lines:  # Add them only if we haven't already
                all_lines.extend(
                    [bg_line_150[0], primary_line_150[0], bg_line_320[0], primary_line_320[0], stitch_line]
                )
                all_labels.extend(
                    [
                        "Background data (150 km)",
                        "Primary data (150 km)",
                        "Background data (320 km)",
                        "Primary data (320 km)",
                        "Data stitch point",
                    ]
                )

        except Exception as e:
            # Fallback to traditional visualization if stitching fails
            print(f"Warning: Failed to stitch background data: {e}")
            print("Falling back to traditional visualization")

            # Add text annotation explaining the failure
            ax1_top.text(
                0.5,
                0.5,
                "External background dataset available\nbut data stitching failed",
                transform=ax1_top.transAxes,
                ha="center",
                va="center",
                bbox=dict(facecolor="yellow", alpha=0.2),
            )

            # Use traditional visualization for both subplots
            for ax_top, ax_bottom in [(ax1_top, ax1_bottom), (ax2_top, ax2_bottom)]:
                line1 = ax_top.plot(
                    time,
                    processor.density_data[:, cell_min_idx, 0],
                    color="C0",
                )
                line2 = ax_bottom.plot(
                    time,
                    processor.density_data[:, cell_max_idx, 0],
                    color="C1",
                )

                # Set labels
                ax_top.set_ylabel("Density (150 km)", color="C0")
                ax_bottom.set_ylabel("Density (320 km)", color="C1")

            # Only add density lines once to the legend
            all_lines.extend([line1[0], line2[0]])
            all_labels.extend(["150 km", "320 km"])

            # Add standard background visualization with safety checks
            if processor.bg_window is not None:
                bg_start = min(processor.bg_window["start"], len(time) - 1)
                bg_end = min(processor.bg_window["end"], len(time) - 1)

                bg_patch = ax1_top.axvspan(time[bg_start], time[bg_end], color="gray", alpha=0.2)
                all_lines.append(bg_patch)
                all_labels.append("Standard background window")
    else:
        # Traditional visualization (without stitching)
        # Plot data in both subplots
        for ax_top, ax_bottom in [(ax1_top, ax1_bottom), (ax2_top, ax2_bottom)]:
            line1 = ax_top.plot(
                time,
                processor.density_data[:, cell_min_idx, 0],
                color="C0",
            )
            line2 = ax_bottom.plot(
                time,
                processor.density_data[:, cell_max_idx, 0],
                color="C1",
            )

            # Set labels
            ax_top.set_ylabel("Density (150 km)", color="C0")
            ax_bottom.set_ylabel("Density (320 km)", color="C1")

        # Only add density lines once to the legend
        all_lines.extend([line1[0], line2[0]])
        all_labels.extend(["150 km", "320 km"])

        # Add standard background visualization (top subplot) - with safety checks
        if processor.bg_window is not None:
            bg_start = min(processor.bg_window["start"], len(time) - 1)
            bg_end = min(processor.bg_window["end"], len(time) - 1)

            bg_patch = ax1_top.axvspan(time[bg_start], time[bg_end], color="gray", alpha=0.2)
            all_lines.append(bg_patch)
            all_labels.append("Standard background window")

        # Original behavior for experimental windows in lower subplot
        optimal_windows = experimental_background.get("optimal_windows", None)

        # Check if we actually have optimal windows from experimental method
        if optimal_windows is not None:
            # Draw windows for both 150km and 320km altitudes with different colors
            for idx, alt_idx in enumerate([cell_min_idx, cell_max_idx]):
                if alt_idx >= optimal_windows.shape[0]:
                    continue  # Skip if altitude index is out of range

                win_start, win_end = optimal_windows[alt_idx]

                # Skip invalid windows
                if win_start < 0 or win_end < 0:
                    continue

                ax = ax2_top if idx == 0 else ax2_bottom

                # Use different colors for different altitudes
                color = "#FF9AA2" if idx == 0 else "#FFDAC1"  # Soft pink for 150km, soft orange for 320km
                label = f"Experimental window ({radial_distances[alt_idx]:.1f} km)"

                # Only plot if indices are within time array bounds
                if win_start < len(time) and win_end < len(time):
                    print(
                        f"Drawing experimental window for {radial_distances[alt_idx]:.1f} km: Time [{time[win_start]:.1f}, {time[win_end]:.1f}]"
                    )
                    bg_patch = ax.axvspan(
                        time[win_start],
                        time[win_end],
                        color=color,
                        alpha=0.3,  # Increased alpha for better visibility
                        label=label,
                        zorder=5,  # Make sure windows are drawn on top
                    )
                    all_lines.append(bg_patch)
                    all_labels.append(label)
                else:
                    print(
                        f"WARNING: Window indices [{win_start}, {win_end}] out of range for time array (length {len(time)})"
                    )

    # Add standard background levels to first subplot
    ax1_top.axhline(
        y=standard_background["density"][cell_min_idx],
        color="C0",
        linestyle=":",
        alpha=0.5,
    )
    ax1_bottom.axhline(
        y=standard_background["density"][cell_max_idx],
        color="C1",
        linestyle=":",
        alpha=0.5,
    )
    # Add experimental background levels to second subplot
    line_exp_150 = ax2_top.axhline(
        y=experimental_background["density"][cell_min_idx],
        color="C0",
        linestyle=":",
        alpha=0.5,
    )
    line_exp_320 = ax2_bottom.axhline(
        y=experimental_background["density"][cell_max_idx],
        color="C1",
        linestyle=":",
        alpha=0.5,
    )
    all_lines.extend([line_exp_150, line_exp_320])
    all_labels.extend(
        [
            "Background (150 km)",
            "Background (320 km)",
        ]
    )

    # Add wave start line to both plots and add to legend once
    wave_start_time = time[start_idx]
    wave_line = ax1_top.axvline(x=wave_start_time, color="r", linestyle="--")
    ax2_top.axvline(x=wave_start_time, color="r", linestyle="--")
    all_lines.append(wave_line)
    all_labels.append("Wave start")

    # Set common x-label
    ax2_top.set_xlabel("Time (s)")

    # Add titles
    ax1_top.set_title("Standard Method Background")
    ax2_top.set_title("Experimental Method Background")

    # Add legend inside the first subplot with small text
    ax1_top.legend(all_lines, all_labels, loc="best", fontsize="x-small")
    # Save the density evolution visualization figure
    bg_method_label = "external" if using_external_bg else "internal"
    density_evolution_path = exports_dir / f"density_evolution_{bg_method_label}_background.png"
    plt.savefig(density_evolution_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {density_evolution_path}")
    plt.close()

    # Also save a summary file with key parameters and results
    summary_path = exports_dir / "analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Analysis Summary for {data_dir}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Simulation Parameters:\n")
        f.write(f"- Wave frequency: {processor.params.omega_freq:.4f} Hz\n")
        f.write(f"- Time step: {processor.params.timestep:.4f} s\n")
        f.write(f"- Number of periods: {processor.params.num_periods}\n")
        f.write(f"- Altitude range: {radial_distances[0]:.1f} - {radial_distances[-1]:.1f} km\n\n")

        f.write("Analysis Settings:\n")
        f.write(f"- Analysis altitude range: {alt_range[0]:.1f} - {alt_range[1]:.1f} km\n")
        f.write(f"- Background method: {'experimental' if use_experimental else 'standard'}\n")
        f.write(f"- Peak detection: {'dynamic' if use_dynamic else 'static'}\n")
        if background_data_dir:
            f.write(f"- External background data: {background_data_dir}\n\n")

        f.write("Key Results:\n")
        f.write(f"- Average relative error between background methods: {avg_rel_error:.2f}%\n")
        f.write(f"- Heating rate slope: {slope:.4f} K/km\n")
        f.write(f"- Normalized heating rate slope (H): {slope_H:.4f} K/H\n\n")

        f.write("Generated Figures:\n")
        for fig_path in exports_dir.glob("*.png"):
            f.write(f"- {fig_path.name}\n")

    print(f"Saved analysis summary: {summary_path}")

    return {"slope": slope, "slope_H": slope_H, "exports_dir": exports_dir, "bg_comparison_error": avg_rel_error}


def find_min_mad_window(data, window_sizes):
    """
    Compute the rolling MAD for a range of window sizes using vectorized operations,
    then return the window (its edges) that yields the lowest MAD.
    More robust to handle edge cases and low-density data.
    """
    best_mad = float("inf")
    best_info = None

    # Try to clean data - replace any potential NaN values
    clean_data = np.nan_to_num(data, nan=0.0)

    # As a fallback, use the first 1/4 of data available
    fallback_window_size = min(window_sizes[0], len(clean_data) // 4)
    fallback_start = 0
    fallback_end = fallback_start + fallback_window_size - 1

    for window_size in window_sizes:
        # Skip if window size is larger than the data array
        if window_size >= len(clean_data):
            continue

        windows = sliding_window_view(clean_data, window_size)  # shape: (n-window_size+1, window_size)

        # Handle potential empty windows
        if len(windows) == 0:
            continue

        # Calculate medians for each window
        medians = np.median(windows, axis=1)

        # More lenient condition - accept windows with very low but non-zero median
        # Use a small epsilon value instead of exactly zero
        valid_windows = medians > 1e-10
        if not np.any(valid_windows):
            continue

        # Only compute MAD for valid windows
        valid_mads = np.median(np.abs(windows[valid_windows] - medians[valid_windows, None]), axis=1)

        # Find minimum MAD among valid windows
        if len(valid_mads) > 0:
            idx_min = valid_mads.argmin()
            valid_indices = np.where(valid_windows)[0]  # Get original indices of valid windows
            if valid_mads[idx_min] < best_mad:
                best_mad = valid_mads[idx_min]
                best_info = (window_size, (valid_indices[idx_min], valid_indices[idx_min] + window_size - 1))

    # If no valid windows found, return the fallback window
    if best_info is None:
        return (fallback_window_size, (fallback_start, fallback_end), float("inf"))

    return (best_info[0], best_info[1], best_mad)


def _parallel_window_finder(altitude_index, altitude, data, window_sizes):
    """
    Helper function to find optimal window for a single altitude.
    Designed for parallel processing.

    Args:
        altitude_index: Index of current altitude
        altitude: Actual altitude value in km
        data: Time series data for this altitude
        window_sizes: Array of window sizes to try

    Returns:
        Tuple of (altitude_index, (optimal_window_start, optimal_window_end, temps, min_mad))
    """
    result = find_min_mad_window(data, window_sizes)
    # Check if we got a valid result - all values will be None if no valid window found
    if result[0] is not None:
        win_size, (start, end), mad = result
        return altitude_index, (start, end, data[start : end + 1], mad)
    else:
        # Return a tuple structure expected by the caller with None values
        return altitude_index, (None, None, None, float("inf"))


def visualize_stitched_simulation(
    data_dir: str,
    background_data_dir: str,
    altitudes: list = None,
    data_fields: list = None,
    show_bg_windows: bool = True,
) -> None:
    """
    Create standalone visualization of stitched data from two simulation directories.

    This function creates visualizations of stitched data, combining pre-wave data
    from the background dataset with during/post-wave data from the primary dataset.
    It can also show the background windows used for standard and experimental background
    computation methods.

    Args:
        data_dir: Primary simulation data directory
        background_data_dir: Background simulation data directory
        altitudes: List of altitudes to visualize in km (default: [150, 220, 320])
        data_fields: List of data fields to visualize (default: ["density", "temperature"])
        show_bg_windows: Whether to show background windows used for computation
    """
    print("\nCreating stitched visualization:")
    print(f"Primary data: {data_dir}")
    print(f"Background data: {background_data_dir}")

    # Default values
    if altitudes is None:
        altitudes = [150, 220, 320]
    if data_fields is None:
        data_fields = ["density", "temperature"]

    print(f"Visualizing altitudes: {altitudes} km")
    print(f"Visualizing fields: {data_fields}")

    # Initialize processor with background_data_dir
    processor = DSMCDataProcessor(data_dir, background_data_dir=background_data_dir)
    processor.load_parameters()
    processor.load_data()

    # Create stitched visualizations for each requested data field
    for field in data_fields:
        try:
            processor.create_stitched_visualization(field, altitudes, show_bg_window=show_bg_windows)
        except Exception as e:
            print(f"Error creating stitched visualization for {field}: {e}")


def main():
    """Main function to run various analysis tasks"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMC Data Analysis Tool")

    # Define argument groups
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--growth", action="store_true", help="Run growth analysis")
    mode_group.add_argument("--stitch", action="store_true", help="Create stitched data visualization")

    # Common arguments
    parser.add_argument("--data-dir", required=True, help="Primary data directory")
    parser.add_argument("--bg-data-dir", help="Background data directory")

    # Growth analysis arguments
    parser.add_argument("--alt-min", type=float, default=150, help="Minimum altitude for analysis (km)")
    parser.add_argument("--alt-max", type=float, default=320, help="Maximum altitude for analysis (km)")
    parser.add_argument("--experimental", action="store_true", help="Use experimental background method")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic peak detection")

    # Stitched visualization arguments
    parser.add_argument("--altitudes", type=float, nargs="+", help="Altitudes to visualize (km)")
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["density", "temperature", "rad_vel"],
        default=["density"],
        help="Data fields to visualize",
    )
    parser.add_argument("--hide-windows", action="store_true", help="Hide background windows in visualization")

    # Window sizes
    parser.add_argument("--window-sizes", type=float, nargs="+", help="Custom window sizes (seconds)")

    args = parser.parse_args()

    # Convert window sizes to numpy array if provided
    window_sizes = np.array(args.window_sizes) if args.window_sizes else None

    if args.growth:
        # Run growth analysis
        compute_growth(
            data_dir=args.data_dir,
            background_data_dir=args.bg_data_dir,
            alt_range=(args.alt_min, args.alt_max),
            use_experimental=args.experimental,
            use_dynamic=args.dynamic,
            window_sizes=window_sizes,
        )

    elif args.stitch:
        # Run stitched visualization
        if not args.bg_data_dir:
            parser.error("Background data directory (--bg-data-dir) is required for stitched visualization")

        visualize_stitched_simulation(
            data_dir=args.data_dir,
            background_data_dir=args.bg_data_dir,
            altitudes=args.altitudes,
            data_fields=args.fields,
            show_bg_windows=not args.hide_windows,
        )


if __name__ == "__main__":
    # Use command-line arguments if provided, otherwise run the default example
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # Default example: Run growth analysis with background data directory
        # compute_growth(
        #     "/media/astrojose/AstroBook/ArchiveSims/Results_20250225_224838.674",
        #     alt_range=(150, 320),
        #     use_dynamic=True,
        #     background_data_dir="/media/astrojose/AstroBook/ArchiveSims/Results_20250209_084356.301",
        # )

        compute_growth(
            # "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_133319.916",
            # "/media/astrojose/AstroBook/ArchiveSims/Results_20250514_213720.749",
            "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_135431.191",
            # "/media/astrojose/AstroBook/ArchiveSims/Results_20250209_084356.301",
            alt_range=(150, 320),
            use_dynamic=True,
        )

        # Uncomment to run standalone stitched visualization example
        # visualize_stitched_simulation(
        #    "/media/astrojose/AstroBook/ArchiveSims/Results_20250225_224838.674",
        #    "/media/astrojose/AstroBook/ArchiveSims/Results_20250209_084356.301",
        #    altitudes=[150, 200, 250, 300, 350],
        #    data_fields=["density", "temperature", "rad_vel"]
        # )
