"""
Helper utilities for wave spectral analysis.
Contains core functions for spectral analysis of wave steepening.
"""

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List

# Add default tracking parameters to be used in wave spectral analysis.
DEFAULT_TRACKING_PARAMS = {
    'initial': {
        'max_tracking_gap': 20,         # Keep successful initial phase params
        'min_peak_quality': 0.05,
        'min_temporal_stability': 0.1,
        'wavelength_range': (10, 180),
        'reliability_threshold': 0.15
    },
    'stable': {
        'max_tracking_gap': 30,         # Increased slightly for better continuity
        'min_peak_quality': 0.03,       # More lenient
        'min_temporal_stability': 0.12, # More lenient
        'wavelength_range': (30, 120),
        'reliability_threshold': 0.10
    },
    'unified': {
        "k_range": (0.02, 0.5),         # Broad detection range: ~20-300 km
        "min_power_ratio": 0.1,         # Relative to maximum PSD in window
        "smoothing_window": 15,         # Time steps for moving average
        "reliability_threshold": 0.6,   # Simple consistency check
        "max_gap": 30,                  # Allowed tracking gaps
}}

DEFAULT_SMOOTHING_PARAMS = {
    'cutoff_freq': 0.4,     # Normalized frequency cutoff for Butterworth filter
    'filter_order': 2,      # Filter order for smoothing
    'temp_offset': 270.0,   # Temperature offset to subtract from raw data
}

def resample_to_uniform_grid(z: np.ndarray, data: np.ndarray,
                           n_points: int = None,
                           dz: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample data onto a uniform grid using cubic spline interpolation.

    Args:
        z: Original height coordinates (can be non-uniform)
        data: Data values corresponding to z coordinates
        n_points: Number of points for output grid (optional)
        dz: Desired grid spacing (optional)

    Returns:
        Tuple of (uniform_z, resampled_data)

    Note:
        Either n_points or dz must be provided. If both are provided,
        n_points takes precedence.
    """
    if n_points is None and dz is None:
        raise ValueError("Either n_points or dz must be provided")

    if n_points is None:
        n_points = int((z[-1] - z[0]) / dz) + 1

    # Create uniform grid
    z_uniform = np.linspace(z[0], z[-1], n_points)

    # Create interpolation function
    f = interp1d(z, data, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Resample data onto uniform grid
    data_uniform = f(z_uniform)

    return z_uniform, data_uniform

def detrend_temperature_profile(temp_profile: np.ndarray, z: np.ndarray, polyorder: int = 2) -> np.ndarray:
    """
    Remove polynomial trend from temperature profile.

    Args:
        temp_profile: Temperature values
        z: Height values
        polyorder: Order of polynomial fit

    Returns:
        Detrended temperature profile
    """
    coeffs = np.polyfit(z, temp_profile, polyorder)
    trend = np.polyval(coeffs, z)
    return temp_profile - trend

def smooth_profile(data: np.ndarray, cutoff_freq: float = 0.2, order: int = 3) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to smooth profile data.

    Args:
        data: Input profile data
        cutoff_freq: Normalized cutoff frequency (0 to 1.0)
        order: Filter order

    Returns:
        Smoothed profile data
    """
    b, a = signal.butter(order, cutoff_freq, btype='low')
    return signal.filtfilt(b, a, data)

def compute_spectral_features(temp_profile: np.ndarray, z: np.ndarray,
                            apply_window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of temperature profile with optional windowing.
    """
    # Adaptive window selection
    n = len(temp_profile)
    window = np.hanning(n)  # Better side lobe suppression than Blackman

    # Smart zero-padding (pad to next power of 2 but limit to 4× original)
    n_pad = min(2**np.ceil(np.log2(n*1.5)).astype(int), n*4)

    # Compute FFT with optimized normalization
    fft_result = np.fft.fft(temp_profile * window, n=n_pad)
    psd = np.abs(fft_result)**2 / (n * np.sum(window**2))  # Proper power normalization

    # Frequency axis
    dz = np.mean(np.diff(z))
    k = 2 * np.pi * np.fft.fftfreq(n_pad, dz)

    return k[k>0], psd[k>0]

def find_harmonics(k: np.ndarray, psd: np.ndarray,
                  num_harmonics: int = 3,
                  band_width: float = 0.4,
                  prev_k1: float = None,
                  memory_factor: float = 0.9,
                  height_fraction: float = 1/15,
                  prominence_fraction: float = 1/25,
                  is_stable_phase: bool = False,
                  verbose: bool = False
                  ) -> Dict[str, Dict[str, float]]:
    """
    Find fundamental frequency and harmonics in power spectrum with multi-mode support.
    """
    results = {
        'fundamental': {'k': 0.0, 'power': 0.0},
        'harmonic_2': {'k': 0.0, 'power': 0.0},
        'harmonic_3': {'k': 0.0, 'power': 0.0}
    }

    # Define wave mode ranges based on phase
    if is_stable_phase:
        mode_ranges = [
            {'k_min': 0.06, 'k_max': 0.18},  # Mode 1 (~40-80 km)
            {'k_min': 0.18, 'k_max': 0.35}   # Mode 2 (~25-40 km)
        ]
    else:
        mode_ranges = [
            {'k_min': 0.015, 'k_max': 0.75}  # Wide range for initial phase
        ]

    max_power = 0
    best_mode = None

    # Search each mode range
    for mode in mode_ranges:
        # Create mask for current mode
        mode_mask = (k >= mode['k_min']) & (k <= mode['k_max'])

        if not np.any(mode_mask):
            continue

        # Dynamic thresholds for current mode
        max_psd = np.max(psd[mode_mask])
        base_height = height_fraction * (0.8 if is_stable_phase else 0.6)
        base_prominence = prominence_fraction * (0.6 if is_stable_phase else 0.4)

        # More lenient thresholds for stable phase
        if is_stable_phase:
            height_threshold = max_psd * base_height * 0.7
            prominence_threshold = max_psd * base_prominence * 0.7

        # Scale thresholds based on signal strength
        if max_psd > 1e3:
            height_threshold = max_psd * base_height
            prominence_threshold = max_psd * base_prominence
        elif max_psd > 1e2:
            height_threshold = max_psd * (base_height * 0.6)
            prominence_threshold = max_psd * (base_prominence * 0.6)
        else:
            height_threshold = max_psd * (base_height * 0.4)
            prominence_threshold = max_psd * (base_prominence * 0.4)

        # Find peaks in current mode
        peaks, properties = signal.find_peaks(psd[mode_mask],
                                            height=height_threshold,
                                            prominence=prominence_threshold,
                                            distance=2)

        if len(peaks) > 0:
            peak_powers = psd[mode_mask][peaks]
            k_candidates = k[mode_mask][peaks]

            # Select best peak within mode
            if prev_k1 is not None and len(peaks) > 1:
                distances = np.abs(k_candidates - prev_k1)
                normalized_powers = peak_powers / np.max(peak_powers)
                power_weight = 0.7 if is_stable_phase else 0.8
                distance_weight = 0.3 if is_stable_phase else 0.2
                scores = power_weight * normalized_powers - distance_weight * (distances / np.max(distances))
                best_idx = np.argmax(scores)
            else:
                best_idx = np.argmax(peak_powers)

            if peak_powers[best_idx] > max_power:
                max_power = peak_powers[best_idx]
                best_mode = {
                    'k': k_candidates[best_idx],
                    'power': peak_powers[best_idx],
                    'mode_range': mode
                }

    # Process best mode if found
    if best_mode is not None:
        k1 = best_mode['k']
        p1 = best_mode['power']

        # Validate against previous k1 if available
        if prev_k1 is not None:
            max_k_jump = 0.05 if is_stable_phase else 0.25
            if np.abs(k1 - prev_k1) > max_k_jump:
                # Look for alternative peaks closer to prev_k1
                for mode in mode_ranges:
                    mode_mask = (k >= mode['k_min']) & (k <= mode['k_max'])
                    nearby_k = k[mode_mask]
                    nearby_psd = psd[mode_mask]
                    nearby_peaks = np.abs(nearby_k - prev_k1) <= max_k_jump * 1.5
                    if np.any(nearby_peaks):
                        alt_idx = np.argmax(nearby_psd[nearby_peaks])
                        k1 = nearby_k[nearby_peaks][alt_idx]
                        p1 = nearby_psd[nearby_peaks][alt_idx]
                        break

        results['fundamental']['k'] = k1
        results['fundamental']['power'] = p1

        # Find harmonics for selected mode
        for n in range(2, num_harmonics + 1):
            target_k = k1 * n
            current_width = band_width * (n ** 0.8)
            band_mask = (k >= target_k * (1 - current_width/2)) & (k <= target_k * (1 + current_width/2))

            if np.any(band_mask):
                power = trapezoid(psd[band_mask], k[band_mask])
                results[f'harmonic_{n}']['k'] = target_k
                results[f'harmonic_{n}']['power'] = power

    if verbose:
        if best_mode is not None:
            print(f"Found peak at k = {k1:.4f} (power = {p1:.2e})")
        elif prev_k1 is not None:
            print(f"No new peaks found, using previous k1 = {prev_k1:.4f}")
        else:
            print("No peaks found and no previous k1 available")

    return results

def masked_stats(values: np.ndarray, mask: np.ndarray, reliability: np.ndarray = None) -> Dict:
    """
    Calculate statistics for masked array values.

    Args:
        values: Array of values to compute statistics for
        mask: Boolean mask array
        reliability: Optional reliability scores array

    Returns:
        Dictionary with mean, std, and reliability statistics
    """
    valid = ~np.isnan(values[mask])
    if np.any(valid):
        return {
            'mean': np.mean(values[mask][valid]),
            'std': np.std(values[mask][valid]),
            'reliability': np.mean(reliability[mask][valid]) if reliability is not None else 1.0
        }
    return {'mean': np.nan, 'std': np.nan, 'reliability': 0.0}

def compute_grid_statistics(z: np.ndarray, altitude_range: Tuple[float, float]) -> Tuple[np.ndarray, float, dict]:
    # Compute altitude mask, analysis grid and grid statistics.
    alt_mask = (z >= altitude_range[0]) & (z <= altitude_range[1])
    z_analysis = z[alt_mask]
    desired_dz = np.median(np.diff(z_analysis))
    dz = np.diff(z_analysis)
    rel_diff = np.abs(np.diff(dz) / dz[:-1])
    original_resolution = {
        'mean': np.mean(dz),
        'std': np.std(dz),
        'max_variation': np.max(rel_diff) * 100  # percentage
    }
    return z_analysis, desired_dz, original_resolution

def find_fundamental_mode(k: np.ndarray, psd: np.ndarray,
                        min_k: float = 0.02,
                        max_k: float = 0.3) -> Dict:
    """
    Basic fundamental mode detection with adaptive thresholding.
    Returns the strongest peak in specified k-range.
    """
    # Find global maximum in range
    valid_range = (k >= min_k) & (k <= max_k)
    if not np.any(valid_range):
        return {'k': 0.0, 'power': 0.0}

    max_idx = np.argmax(psd[valid_range])
    return {
        'k': k[valid_range][max_idx],
        'power': psd[valid_range][max_idx]
    }

def basic_spectral_features(temp_profile: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified spectral analysis without windowing or zero-padding.
    """
    dz = np.mean(np.diff(z))
    k = 2 * np.pi * np.fft.fftfreq(len(z), dz)
    psd = np.abs(np.fft.fft(temp_profile)) ** 2 / len(z)
    return k[k>0], psd[k>0]

def process_temperature_snapshot(processor, t: int, alt_mask: np.ndarray,
                                 z_analysis: np.ndarray, desired_dz: float,
                                 smoothing_params: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process temperature profile with configurable smoothing parameters."""
    params = smoothing_params or DEFAULT_SMOOTHING_PARAMS

    # Process temperature profile: detrend, smooth, resample, FFT
    temp_profile = processor.temperature_data[t, alt_mask, 0] - params['temp_offset']
    temp_smoothed_raw = smooth_profile(
        temp_profile,
        cutoff_freq=params['cutoff_freq'],
        order=params['filter_order']
    )
    # Resample smoothed profile to uniform grid
    z_uniform, temp_smoothed = resample_to_uniform_grid(z_analysis, temp_smoothed_raw, dz=desired_dz)
    k, psd = compute_spectral_features(temp_smoothed, z_uniform, apply_window=True)
    return temp_profile, temp_smoothed_raw, temp_smoothed, k, psd

def compute_reliability(peak_quality: float, temporal_stability: float, tracking_quality: float,
                        min_peak_quality: float, min_temporal_stability: float, is_stable_phase: bool) -> float:
    norm_peak = max(0.0, (peak_quality - min_peak_quality) / (1.0 - min_peak_quality))
    norm_temp = max(0.0, (temporal_stability - min_temporal_stability) / (1.0 - min_temporal_stability))
    # Modified weights for stable phase
    if is_stable_phase:
        weights = [0.35, 0.30, 0.35]  # More weight on tracking continuity
    else:
        weights = [0.5, 0.2, 0.3]     # Higher peak quality emphasis

    return np.dot([norm_peak, norm_temp, tracking_quality], weights)

def handle_invalid_measurement(wavelength_stats: dict, t_idx: int) -> None:
    """
    Mark measurement at t_idx as invalid in the wavelength_stats dictionary.
    """
    wavelength_stats['mean'][t_idx] = np.nan
    wavelength_stats['std'][t_idx] = np.nan
    wavelength_stats['reliability'][t_idx] = 0.0
    wavelength_stats['phase_coherence'][t_idx] = 0.0
    wavelength_stats['confidence'][t_idx] = 0.0
    wavelength_stats['is_valid'][t_idx] = False

def initialize_tracking_arrays(length: int) -> Tuple[dict, dict, dict]:
    """
    Initialize and return tracking arrays for wavelength analysis.
    Returns:
        wavelength_stats, diagnostics, harmonic_ratios
    """
    wavelength_stats = {
        'mean': np.full(length, np.nan),
        'std': np.full(length, np.nan),
        'reliability': np.zeros(length),
        'phase_coherence': np.zeros(length),
        'confidence': np.zeros(length),
        'is_valid': np.zeros(length, dtype=bool)
    }
    diagnostics = {
        'fundamental_k': np.zeros(length),
        'fundamental_power': np.zeros(length),
        'harmonic_2_power': np.zeros(length),
        'harmonic_3_power': np.zeros(length),
        'phase': np.full(length, '', dtype='U10'),
        'max_psd': np.zeros(length),
        'tracking_status': np.full(length, '', dtype='U20')
    }
    harmonic_ratios = {
        'R21': np.zeros(length),
        'R31': np.zeros(length)
    }
    return wavelength_stats, diagnostics, harmonic_ratios