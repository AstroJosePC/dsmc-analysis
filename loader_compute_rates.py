import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
import pickle
from helper_funcs import process_growth_num_amp, compute_heating_rate_slope, process_temperature_deviation_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import stats

PADDING = 40

# compute the scale height
M = 16  # g/mol
num_dens = 1e16  # Oxygen / m^3
R = 8.314  # J/mol/K
R_prime = R / M * 1e3  # J/kg/K
T = 270  # K
g = 3.51  # m/s^2
H = R_prime * T / g / 1e3  # km


@dataclass
class SimulationParameters:
    """Storage class for simulation parameters"""

    mid_azi: np.ndarray
    radial_cells: np.ndarray
    time: np.ndarray
    mars_radius: float
    start_step_wave: int
    final_step_wave: int
    timestep: float
    omega_freq: float
    num_periods: float
    rmin: float
    rmax: float


class DSMCDataProcessor:
    """Processor for DSMC simulation data with spherical geometry considerations"""

    def __init__(self, data_dir: str):
        """
        Initialize the DSMC data processor

        Args:
            data_dir: Path to directory containing simulation output files
        """
        self.data_dir = Path(data_dir)
        self.params: Optional[SimulationParameters | None] = None
        self.density_data: Optional[np.ndarray] = None
        self.temperature_data: Optional[np.ndarray] = None
        self.radial_vel_data: Optional[np.ndarray] = None

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
        omega_val = params["omega"] if "omega" in params else params.get("omega_freq", 0.0)
        radial_cells = params.get("radial_cells", params.get("r1_distances", np.array([])))
        # Compute rmin and rmax from radial_cells
        rmin = round(float(radial_cells[0]), -1) if radial_cells.size > 0 else 0.0
        rmax = round(float(radial_cells[-1]), -1) if radial_cells.size > 0 else 0.0
        self.params = SimulationParameters(
            radial_cells=radial_cells,
            mid_azi=params.get("mid_azi", np.array([])),
            time=params.get("time", np.array([])),
            mars_radius=params.get("mars_radius", 0.0),
            start_step_wave=params.get("step0_inject", 0),
            final_step_wave=params.get("stepf_inject", 0),
            timestep=params.get("timestep", 0.0),
            omega_freq=omega_val,
            num_periods=params.get("num_periods", 0.0),
            rmin=rmin,  # new field
            rmax=rmax,  # new field
        )

    def load_data(self) -> None:
        """Load all simulation data files"""
        # import ipdb; ipdb.set_trace()
        self.density_data = np.load(self.data_dir / "density_data.npy")
        if self.density_data.ndim == 4 and self.density_data.shape == 1:
            self.density_data = self.density_data.squeeze(0)

        self.temperature_data = np.load(self.data_dir / "temperature_data.npy")
        if self.temperature_data.ndim == 4 and self.temperature_data.shape == 1:
            self.temperature_data = self.temperature_data.squeeze(0)

        self.radial_vel_data = np.load(self.data_dir / "rad_vel_data.npy")
        if self.radial_vel_data.ndim == 4 and self.radial_vel_data.shape == 1:
            self.radial_vel_data = self.radial_vel_data.squeeze(0)

    def _find_stable_period(self, data: np.ndarray, window: int = 40, std_threshold: float = 1 / 10) -> int:
        """
        Find start index where time series stabilizes before wave activity.

        Args:
            data: Time series data to analyze
            window: Size of rolling window
            std_threshold: Threshold for std change rate

        Returns:
            Index where data stabilizes
        """
        if self.params is None:
            raise ValueError("Parameters not loaded")

        start_idx, _ = self.get_wave_time_indices()
        analysis_end = start_idx - window  # Leave buffer before wave

        # Calculate rolling std for each altitude
        rolling_std = np.zeros((data.shape[0] - window, data.shape[1]))
        for alt in range(data.shape[1]):
            for i in range(data.shape[0] - window):
                rolling_std[i, alt] = np.std(data[i : i + window, alt, 0])

        # Smooth std curves
        smoothed_std = gaussian_filter1d(rolling_std, sigma=3, axis=0)

        # Find where std stabilizes (rate of change small)
        std_grad = np.gradient(smoothed_std, axis=0)
        stable_idx = np.where(np.all(np.abs(std_grad) < std_threshold, axis=1))[0]

        if len(stable_idx) == 0:
            return window  # Fallback to start

        return stable_idx[0] + window

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

    def get_background_profiles(self) -> Dict[str, np.ndarray]:
        """
        Calculate background density and temperature profiles before wave activity,
        using automated detection of stable period.

        Returns:
            Dictionary containing background profiles
        """
        if any(x is None for x in [self.density_data, self.temperature_data]):
            raise ValueError("Data not loaded. Call load_data() first.")

        start_idx, _ = self.get_wave_time_indices()

        # Find stable period start for density and temperature
        density_start = self._find_stable_period(self.density_data)
        temp_start = self._find_stable_period(self.temperature_data)

        # Use later of two start points for consistency
        bg_start = max(density_start, temp_start)

        # bg_start = start_idx - PADDING
        pre_wave_period = slice(bg_start, start_idx - 5)

        print(f"Stable period: {bg_start} to {start_idx - 5}")

        time_points = np.arange(pre_wave_period.stop - pre_wave_period.start)

        # Initialize output arrays
        density_bg = np.zeros(self.density_data.shape)
        # temp_bg = np.zeros(self.temperature_data.shape)

        # Process each altitude
        for alt_idx in range(self.density_data.shape):
            # Density detrending
            density_time_series = self.density_data[pre_wave_period, alt_idx, 0]
            slope, intercept, _, _, _ = stats.linregress(time_points, density_time_series)
            trend = slope * time_points + intercept
            density_bg[alt_idx] = np.mean(density_time_series - trend) + intercept

        #     # Temperature detrending
        #     temp_time_series = self.temperature_data[pre_wave_period, alt_idx, 0]
        #     slope, intercept, _, _, _ = stats.linregress(time_points, temp_time_series)
        #     trend = slope * time_points + intercept
        #     temp_bg[alt_idx] = np.mean(temp_time_series - trend)

        return {
            "density": np.mean(self.density_data[pre_wave_period], axis=(0, -1)),
            # "density": density_bg,
            "temperature": np.mean(self.temperature_data[pre_wave_period], axis=(0, -1)),
        }

    def calculate_perturbations(self) -> Dict[str, np.ndarray]:
        """
        Calculate perturbations from background state

        Returns:
            Dictionary containing perturbation fields
        """
        background = self.get_background_profiles()
        start_idx, end_idx = self.get_wave_time_indices()

        wave_period = slice(start_idx, end_idx)

        return {
            "density": (self.density_data[wave_period] - background["density"]) / background["density"],
            "temperature": (self.temperature_data[wave_period] - background["temperature"]) / background["temperature"],
        }

    def compute_heating_rate(self, visualize: bool = False, low_alt: float = 150, high_alt: float = 320) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the heating rate (slope from temperature deviation curve) for each simulation.

        Args:
            visualize: Whether to visualize the temperature deviations and the fitted line.
            low_alt: Lower altitude (in km) for heating rate computation.
            high_alt: Upper altitude (in km) for heating rate computation.

        Returns:
            A tuple containing the slope and the slope normalized by the scale height.
        """
        if any(x is None for x in [self.temperature_data, self.params]):
            raise ValueError("Data or parameters not loaded. Call load_data() and load_parameters() first.")

        time = self.params.time
        radial_distances = self.params.radial_cells

        # Compute temperature deviations
        temp_deviations = self.temperature_data - 270.0

        # Find cell indices based on configurable altitudes
        cell_low_alt = np.where(radial_distances > low_alt)[0][0]
        cell_high_alt = np.where(radial_distances > high_alt)[0][0]

        # Process temperature deviation using low_alt and high_alt indices
        temp_peaks, temp_last_cycle = process_temperature_deviation_peaks(
            temp_deviations[..., 0],
            time >= self.params.start_step_wave * self.params.timestep,
            radial_distances,
            cell_low_alt,
            cell_high_alt,
            self.params.num_periods,
            prominence=0.1,
            distance=12,
            sampling_rate=1 / np.mean(np.diff(self.params.time)),
            cutoff_frequency=self.params.omega_freq * 1.2,
        )

        # Compute the slope of the heating rate with configurable indices
        slope, slope_H = compute_heating_rate_slope(
            temp_last_cycle, radial_distances, cell_low_alt, cell_high_alt, H, visualize=visualize
        )

        return slope, slope_H

    def compute_temperature_spectral_energy(self, k_c: float, avg_window: int = 1) -> np.ndarray:
        """
        For each temperature profile T(z,t), compute its Fourier transform and calculate 
        the integrated spectral energy above a given wavenumber k_c, using only the uniform region.
        
        E_high(t) = ∫_|k|>=k_c |T̃(k,t)|^2 dk.
        
        A spatial window (Hanning) is applied to reduce spectral leakage.
        Optionally, average over nearby snapshots by setting avg_window.
        
        Args:
            k_c (float): Wavenumber threshold.
            avg_window (int): Number of snapshots to average over (default: 1, no averaging).
        
        Returns:
            np.ndarray: Array of E_high values for each (or averaged) time snapshot.
        """
        if self.temperature_data is None or self.params is None:
            raise ValueError("Temperature data or simulation parameters are not loaded.")
        
        # Use altitude grid from simulation parameters
        z = self.params.radial_cells  # full altitude grid (non-uniform at low altitudes)
        # Determine the lowest index where spacing becomes uniform
        diff_z = np.diff(z)
        tol = 1e-2  # tolerance on relative std deviation; adjust as needed
        uniform_idx = 0
        for i in range(len(diff_z)):
            if np.std(diff_z[i:]) / np.mean(diff_z[i:]) < tol:
                uniform_idx = i
                break
        
        # Select uniform region of altitude grid.
        z_uniform = z[uniform_idx:]
        n_z_uniform = len(z_uniform)
        dz = np.mean(np.diff(z_uniform))
        
        # Prepare a Hanning window for uniform region
        window_func = np.hanning(n_z_uniform)
        
        n_t = self.temperature_data.shape[0]
        E_high = np.zeros(n_t)
        
        for t in range(n_t):
            # Extract temperature profile and limit to uniform region
            profile = self.temperature_data[t, :, 0] - 270.0
            profile_uniform = profile[uniform_idx:]
            # Apply spatial windowing
            windowed_profile = profile_uniform * window_func
            # Compute Fourier transform along uniform altitude region
            T_fft = np.fft.fft(windowed_profile)
            freqs = np.fft.fftfreq(n_z_uniform, d=dz)
            energy_density = np.abs(T_fft)**2
            
            # Create mask for high wavenumbers
            mask = np.abs(freqs) >= k_c
            if np.any(mask):
                # Estimate dk from the frequency grid
                dk = np.mean(np.diff(np.sort(np.abs(freqs[mask]))))
                # Integrate using the trapezoidal rule
                E_high[t] = np.trapz(energy_density[mask], dx=dk)
            else:
                E_high[t] = 0.0
        
        if avg_window > 1:
            # Simple moving average over time
            kernel = np.ones(avg_window) / avg_window
            E_high = np.convolve(E_high, kernel, mode="valid")
        
        return E_high
