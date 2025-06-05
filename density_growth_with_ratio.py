#!/usr/bin/env python3
# filepath: /mnt/performance/data-analysis/density_growth_with_ratio.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import from the module
from loader import DSMCDataProcessor, find_min_mad_window, PADDING
from helper_funcs import (
    process_growth_num_amp,
    TIAN_GROW,
    TIAN_ALTITUDES,
    SPHERICAL_1D_GROW,
    SPHERICAL_1D_ALTITUDES,
)

# Set up plotting style
sns.set_theme(context="paper", font_scale=2.3, style="whitegrid")

def density_growth_with_ratio(
    data_dirs: List[str],
    output_dir: str = None,
    alt_range: Tuple[float, float] = (150, 320),
    use_experimental: bool = True,
    use_dynamic: bool = True,
    window_sizes: np.ndarray = None,
    dataset_labels: List[str] = None,
    normalize_dataset_idx: int = 0,  # Index of dataset to normalize against Tian
):
    """
    Compare density growth profiles from multiple datasets and show 
    a ratio subplot for one selected dataset.

    Args:
        data_dirs: List of directories containing simulation data
        output_dir: Directory to save output plots (defaults to first data_dir/exports)
        alt_range: Tuple of (min_altitude, max_altitude) in km
        use_experimental: Whether to use experimental background method
        use_dynamic: Whether to use dynamic peak parameter adjustment
        window_sizes: Optional array of window sizes in seconds
        dataset_labels: Optional labels for each dataset
        normalize_dataset_idx: Index of dataset to normalize against Tian data
    """
    if len(data_dirs) < 1:
        raise ValueError("At least one data directory must be provided")

    # Set up default labels if not provided
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(len(data_dirs))]
    elif len(dataset_labels) != len(data_dirs):
        raise ValueError("The number of dataset labels must match the number of datasets")

    # Create exports directory if not specified
    if output_dir is None:
        output_dir = Path(data_dirs[0]) / "exports"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nProcessing data from {len(data_dirs)} datasets:")
    for i, data_dir in enumerate(data_dirs):
        print(f"Dataset {i+1}: {data_dir} (Label: {dataset_labels[i]})")
    print(f"Saving visualization to: {output_dir}")

    # Process each dataset
    processors = []
    growth_results = []

    for data_dir in data_dirs:
        print(f"\nProcessing data from: {data_dir}")
        processor = DSMCDataProcessor(data_dir)
        processor.load_parameters()
        processor.load_data()
        processors.append(processor)

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

        # Store results
        growth_results.append({
            "growth_peaks": growth_peaks,
            "baseline_peaks": baseline_peaks,
            "radial_distances": radial_distances,
            "cell_min_idx": cell_min_idx,
            "cell_max_idx": cell_max_idx,
        })

    # Now create the plot with the growth comparison and ratio inset
    create_growth_plot_with_ratio(
        processors=processors, 
        growth_results=growth_results, 
        dataset_labels=dataset_labels, 
        output_dir=output_dir,
        normalize_dataset_idx=normalize_dataset_idx
    )

    return processors, growth_results


def create_growth_plot_with_ratio(
    processors, 
    growth_results, 
    dataset_labels, 
    output_dir,
    normalize_dataset_idx=0
):
    """
    Create a plot with density growth profiles and a normalized ratio inset.

    Args:
        processors: List of DSMCDataProcessor objects
        growth_results: List of dictionaries containing growth analysis results
        dataset_labels: List of labels for each dataset
        output_dir: Directory to save output plots
        normalize_dataset_idx: Index of dataset to normalize against Tian et al. data
    """
    # Create figure with specific dimensions to leave room for the inset
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set colors based on number of datasets
    if len(dataset_labels) == 2:
        dataset_colors = ["blue", "red"]
    else:
        dataset_colors = plt.cm.viridis(np.linspace(0, 0.9, len(processors)))
    
    # Plot each dataset's growth profile
    for i, (processor, results, label, color) in enumerate(zip(processors, growth_results, dataset_labels, dataset_colors)):
        growth_peaks = results["growth_peaks"]
        radial_distances = results["radial_distances"]
        cell_min_idx = results["cell_min_idx"]
        cell_max_idx = results["cell_max_idx"]
        altitudes = radial_distances[cell_min_idx : cell_max_idx + 1]

        # Get the density amplitude data
        valid_data = growth_peaks[:, -1].squeeze()

        # Apply smoothing
        # Use Savitzky-Golay filter which preserves peaks and valleys better than moving averages
        window_size = min(11, len(valid_data) - (len(valid_data) % 2 + 1))
        if window_size >= 3:  # Ensure we have enough points to smooth
            # Apply smoothing; poly_order should be less than window_size
            poly_order = min(3, window_size - 1)
            smoothed_data = savgol_filter(valid_data, window_size, poly_order)

            # Plot both original and smoothed data
            ax.plot(valid_data, altitudes, linewidth=1.2, color="k", alpha=0.4, linestyle=":")
            ax.plot(smoothed_data, altitudes, linewidth=2.5, color=color, label=f"{label}")
        else:
            # Not enough points for smoothing, plot original data only
            ax.plot(valid_data, altitudes, linewidth=2.5, color=color, label=label)
    
    # Add reference data
    ax.plot(TIAN_GROW, TIAN_ALTITUDES, linestyle="--", color="black", label="Tian et al. (2023)", alpha=0.7, lw=2.5)
    
    # Try to load the additional reference data
    try:
        reference_data_path = "/mnt/performance/data-analysis/exports/tian high freq mode.csv"
        if os.path.exists(reference_data_path):
            reference_data = np.loadtxt(reference_data_path, delimiter=",")
            tian_hf_amp, tian_hf_alt = reference_data[:,0], reference_data[:,1]
            ax.plot(tian_hf_amp, tian_hf_alt, linestyle="--", color="gray", label="Tian HF Mode", alpha=0.7, lw=2.5)
    except Exception as e:
        print(f"Could not load additional reference data: {e}")
    
    # Set labels and title for main plot
    ax.set_xlabel("Density Amplitude")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True)
    
    # Create inset axis for the ratio plot (width_ratio is the fraction of the parent axes width)
    ratio_width = 0.3  # 30% of the main plot width
    axins = inset_axes(
        ax, 
        width=f"{int(ratio_width*100)}%", 
        height="100%",
        loc="right",
        bbox_to_anchor=(0.05, 0, 1, 1),
        bbox_transform=ax.transAxes
    )

    # Calculate normalized ratio for the selected dataset
    # Only do this if we have valid data
    if normalize_dataset_idx < len(growth_results):
        results = growth_results[normalize_dataset_idx]
        harrah_alt = results["radial_distances"][results["cell_min_idx"]:results["cell_max_idx"]+1]
        harrah_amp = results["growth_peaks"][:, -1].squeeze()
        
        # Apply smoothing to the Harrah data
        window_size = min(5, len(harrah_amp) - (len(harrah_amp) % 2 + 1))
        if window_size >= 3:
            poly_order = min(3, window_size - 1)
            harrah_amp_smoothed = savgol_filter(harrah_amp, window_size, poly_order)
        else:
            harrah_amp_smoothed = harrah_amp
            
        # Create interpolation function for Tian data
        tian_interp = interp1d(TIAN_ALTITUDES, TIAN_GROW, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Interpolate Tian data to Harrah altitude grid
        tian_amp_interp = tian_interp(harrah_alt)
        
        # Calculate normalized curve
        normalized_amp = harrah_amp / tian_amp_interp
        
        # Compute best fit line (linear regression)
        slope, intercept, r_value, p_value, std_err = stats.linregress(harrah_alt, normalized_amp)
        best_fit_line = intercept + slope * harrah_alt
        r_squared = r_value ** 2
        
        # Plot the ratio in the inset axis
        axins.scatter(normalized_amp, harrah_alt, color='gray', alpha=0.7, s=40)
        axins.plot(best_fit_line, harrah_alt, lw=2.5, color=dataset_colors[normalize_dataset_idx], 
                  label=f"Ratio fit (r²={r_squared:.2f})")
        axins.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Set inset axis properties
        axins.set_xlabel("Ratio (Harrah/Tian)")
        axins.set_title(f"{dataset_labels[normalize_dataset_idx]}/Tian", fontsize=10)
        axins.tick_params(axis='y', labelleft=False)  # Hide y-axis labels
        axins.grid(True, alpha=0.3)
        
        # Set x-limits to focus on relevant data
        # Find a reasonable range for the ratio data to display
        x_mean = np.mean(normalized_amp)
        x_std = np.std(normalized_amp)
        x_min = max(0, x_mean - 2*x_std)
        x_max = x_mean + 2*x_std
        axins.set_xlim(x_min, x_max)
        
        # Print regression statistics to console
        print(f"\nRegression Statistics for {dataset_labels[normalize_dataset_idx]}:")
        print(f"Slope: {slope:.6f}")
        print(f"Intercept: {intercept:.6f}")
        print(f"R-squared: {r_squared:.6f}")

    # Adjust layout to make room for the inset
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "density_growth_with_ratio.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    
    plt.close()


def main():
    """Parse command line arguments and run the analysis"""
    parser = argparse.ArgumentParser(description="Compare density growth profiles and show normalized ratio")
    parser.add_argument("--data-dirs", nargs="+", required=True, help="Directories containing simulation data")
    parser.add_argument("--output-dir", help="Directory to save output plots")
    parser.add_argument("--alt-min", type=float, default=150, help="Minimum altitude for analysis (km)")
    parser.add_argument("--alt-max", type=float, default=320, help="Maximum altitude for analysis (km)")
    parser.add_argument("--standard-bg", dest="use_experimental", action="store_false",
                        help="Use standard background method (default: experimental)")
    parser.add_argument("--static", dest="use_dynamic", action="store_false",
                        help="Use static peak detection (default: dynamic)")
    parser.add_argument("--window-sizes", type=float, nargs="+", help="Custom window sizes (seconds)")
    parser.add_argument("--labels", nargs="+", help="Labels for each dataset")
    parser.add_argument("--normalize-idx", type=int, default=0, 
                        help="Index of dataset to normalize against Tian data (0-based)")

    args = parser.parse_args()

    # Convert window sizes to numpy array if provided
    window_sizes = np.array(args.window_sizes) if args.window_sizes else None

    # Run analysis
    density_growth_with_ratio(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        alt_range=(args.alt_min, args.alt_max),
        use_experimental=args.use_experimental,
        use_dynamic=args.use_dynamic,
        window_sizes=window_sizes,
        dataset_labels=args.labels,
        normalize_dataset_idx=args.normalize_idx,
    )


if __name__ == "__main__":
    # Use command-line arguments if provided, otherwise run with default paths
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        # Default example with hardcoded paths
        data_dirs = [
            "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_133319.916",
            "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_135431.191",
        ]

        density_growth_with_ratio(
            data_dirs=data_dirs,
            alt_range=(150, 300),
            use_dynamic=True,
            use_experimental=False,
            dataset_labels=["AW1", "AW2"],
            normalize_dataset_idx=0  # Normalize the first dataset (AW1) against Tian
        )
