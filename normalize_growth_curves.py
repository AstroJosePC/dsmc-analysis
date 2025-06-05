import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import stats

# Import from the oned_analyzer module
from loader import DSMCDataProcessor, PADDING
from helper_funcs import (
    process_growth_num_amp,
    TIAN_GROW,
    TIAN_ALTITUDES,
)

# Set up plotting style
sns.set_theme(context="paper", font_scale=2.5, style="whitegrid")

def normalize_density_growth(data_dir, alt_range=(150, 320), use_experimental=True, use_dynamic=True, window_sizes=None):
    """
    Load a dataset, compare with Tian et al. 2023 reference data, and normalize the curves.
    
    Args:
        data_dir: Directory containing simulation data
        alt_range: Tuple of (min_altitude, max_altitude) in km
        use_experimental: Whether to use experimental background method
        use_dynamic: Whether to use dynamic peak parameter adjustment
        window_sizes: Optional array of window sizes in seconds
        
    Returns:
        Dictionary containing the original and normalized curves
    """
    print(f"\nProcessing data from: {data_dir}")
    output_dir = Path(data_dir) / "exports"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process dataset
    processor = DSMCDataProcessor(data_dir)
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
    
    # Get background profiles using the selected method
    if use_experimental:
        background = processor.get_background_profiles_experimental(window_sizes)
    else:
        background = processor.get_background_profiles(window_sizes)
    
    # Compute perturbations
    density_perturbation = (
        processor.density_data - background["density"][None, :, None]
    ) / background["density"][None, :, None]
    
    # Find cell indices for analysis regions
    cell_min_idx = np.where(radial_distances > alt_range[0])[0][0]
    cell_max_idx = np.where(radial_distances > alt_range[1])[0][0]
    
    # Process peaks
    print("\nProcessing density growth peaks")
    base_kwargs = {
        "prominence": 1e-1,
        "distance": 12,
        "sampling_rate": 1 / np.mean(np.diff(processor.params.time)),
        "cutoff_frequency": processor.params.omega_freq * 1.2,
        "filter_order": 2,
        "use_dynamic": use_dynamic,
    }
    
    growth_peaks, baseline_peaks = process_growth_num_amp(
        density_perturbation[..., 0],
        (processor.params.start_step_wave * processor.params.timestep <= time)
        & (time <= processor.params.final_step_wave * processor.params.timestep + PADDING),
        radial_distances,
        cell_min_idx,
        cell_max_idx,
        processor.params.num_periods,
        **base_kwargs,
    )
    
    # Extract altitudes and growth amplitudes
    harrah_alt = radial_distances[cell_min_idx:cell_max_idx+1]
    harrah_amp = growth_peaks[:, -1].squeeze()
    
    # Apply smoothing
    window_size = min(5, len(harrah_amp) - (len(harrah_amp) % 2 + 1))
    if window_size >= 3:
        poly_order = min(3, window_size - 1)
        harrah_amp_smoothed = savgol_filter(harrah_amp, window_size, poly_order)
    else:
        harrah_amp_smoothed = harrah_amp
    # print out the window used
    print(f"Using Savitzky-Golay smoothing window size: {window_size}")
    
    # Create interpolation function for Tian data
    tian_interp = interp1d(TIAN_ALTITUDES, TIAN_GROW, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Interpolate Tian data to Harrah altitude grid
    tian_amp_interp = tian_interp(harrah_alt)
    
    # Calculate normalized curve
    normalized_amp = harrah_amp / tian_amp_interp
    
    # Compute best fit line (linear regression)
    slope, intercept, r_value, p_value, std_err = stats.linregress(harrah_alt, normalized_amp)
    best_fit_line = intercept + slope * harrah_alt
    
    # Store results
    results = {
        "harrah_alt": harrah_alt,
        "harrah_amp": harrah_amp,
        "harrah_amp_smoothed": harrah_amp_smoothed,
        "tian_alt": TIAN_ALTITUDES,
        "tian_amp": TIAN_GROW,
        "tian_amp_interp": tian_amp_interp,
        "normalized_amp": normalized_amp,
        "best_fit_line": best_fit_line,
        "regression_stats": {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err
        },
        "radial_distances": radial_distances,
        "processor": processor,
    }
    
    # Create visualization
    plot_normalized_growth(results, output_dir)
    
    return results

def plot_normalized_growth(results, output_dir):
    """
    Plot the original and normalized growth curves.
    
    Args:
        results: Dictionary containing the calculation results
        output_dir: Directory to save the output plots
    """
    # Extract data from results
    harrah_alt = results["harrah_alt"]
    harrah_amp_smoothed = results["harrah_amp_smoothed"]
    tian_alt = results["tian_alt"]
    tian_amp = results["tian_amp"]
    normalized_amp = results["normalized_amp"]
    best_fit_line = results["best_fit_line"]
    
    # Get regression statistics
    slope = results["regression_stats"]["slope"]
    intercept = results["regression_stats"]["intercept"]
    r_value = results["regression_stats"]["r_value"]
    r_squared = r_value ** 2
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    
    # Plot 1: Original curves - removed the interpolated Tian curve
    ax1.plot(harrah_amp_smoothed, harrah_alt, lw=4, color='blue', label='Harrah 1D Spherical')
    ax1.plot(tian_amp, tian_alt, lw=4, linestyle='--', color='blue', label='Tian et al. (2023)')
    
    ax1.set_xlabel("n(z) / n(z=150km) Amplitude Peak")
    ax1.set_ylabel("Altitude (km)")
    # ax1.set_title("Original Growth Curves")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='best', frameon=True)
    
    # Plot 2: Normalized curve
    # Show original normalized data as scatter points
    ax2.scatter(normalized_amp, harrah_alt, color='gray', alpha=0.7, s=60, 
             label='Original Data Points')
    
    # Show best fit line
    ax2.plot(best_fit_line, harrah_alt, lw=4, color='red', 
             label=rf'Best Fit Line (r$^2$={r_squared:.3f})')
    
    ax2.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Ratio = 1')
    
    ax2.set_xlabel("Ratio (Harrah / Tian)")
    # ax2.set_ylabel("Altitude (km)")
    # ax2.set_title("Normalized Growth Curve with Trend Line")
    ax2.grid(True, alpha=0.5)
    # ax2.legend(loc='best', frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "normalized_density_growth.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    
    # Create a separate figure for just the normalized curve
    plt.figure(figsize=(10, 8))
    
    # Show original normalized data as scatter points
    plt.scatter(normalized_amp, harrah_alt, color='gray', alpha=0.7, s=60,
             label='Original Data Points')
    
    # Show best fit line
    plt.plot(best_fit_line, harrah_alt, lw=4, color='blue', 
             label=f'Best Fit Line (r²={r_squared:.3f})')
    
    plt.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='Ratio = 1')
    
    plt.xlabel("Amplitude Ratio (Harrah/Tian)")
    plt.ylabel("Altitude (km)")
    # plt.title("Normalized Growth Curve with Trend Line")
    plt.grid(True, alpha=0.3)
    # plt.legend(loc='best', frameon=True)
    
    # Save figure
    output_path = output_dir / "normalized_curve_only.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved normalized-only figure: {output_path}")
    
    # Print regression statistics to console
    print(f"\nRegression Statistics:")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"R-value: {r_value:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    
    plt.close('all')

if __name__ == "__main__":
    # Define the data directory - replace with your simulation directory
    data_dir = "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_133319.916"
    
    # Run normalization
    results = normalize_density_growth(
        data_dir=data_dir,
        alt_range=(150, 300),
        use_dynamic=True,
        use_experimental=False
    )
    
    print("\nNormalization complete!")
