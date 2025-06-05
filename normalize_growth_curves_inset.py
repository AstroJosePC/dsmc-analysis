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

# Import from the oned_analyzer module
from loader import DSMCDataProcessor, PADDING
from helper_funcs import (
    process_growth_num_amp,
    TIAN_GROW,
    TIAN_ALTITUDES,
)

# Set up plotting style
sns.set_theme(context="paper", font_scale=2.0, style="whitegrid")

def compare_density_growth_with_ratio(
    data_dirs: List[str],
    output_dir: str = None,
    alt_range: Tuple[float, float] = (150, 320),
    use_experimental: bool = True,
    use_dynamic: bool = True,
    window_sizes: np.ndarray = None,
    dataset_labels: List[str] = None,
):
    """
    Compare density growth profiles with an inset plot showing Harrah/Tian ratio.

    Args:
        data_dirs: List of directories containing simulation data
        output_dir: Directory to save output plots (defaults to first data_dir/exports)
        alt_range: Tuple of (min_altitude, max_altitude) in km
        use_experimental: Whether to use experimental background method
        use_dynamic: Whether to use dynamic peak parameter adjustment
        window_sizes: Optional array of window sizes in seconds
        dataset_labels: Optional labels for each dataset
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
    print(f"Saving visualizations to: {output_dir}")

    # Process each dataset
    processors = []
    growth_results = []
    ratio_data = {}

    for i, data_dir in enumerate(data_dirs):
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
        result_data = {
            "growth_peaks": growth_peaks,
            "baseline_peaks": baseline_peaks,
            "radial_distances": radial_distances,
            "cell_min_idx": cell_min_idx,
            "cell_max_idx": cell_max_idx,
        }
        growth_results.append(result_data)
        
        # Calculate ratio for AW1 (first dataset only)
        if i == 0:
            # Extract altitudes and growth amplitudes for the first dataset (AW1)
            harrah_alt = radial_distances[cell_min_idx:cell_max_idx+1]
            harrah_amp = growth_peaks[:, -1].squeeze()
            
            # Apply smoothing (same as in the original script)
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
            
            # Store ratio data
            ratio_data = {
                "harrah_alt": harrah_alt,
                "normalized_amp": normalized_amp,
                "best_fit_line": best_fit_line,
                "regression_stats": {
                    "slope": slope,
                    "intercept": intercept,
                    "r_value": r_value,
                    "p_value": p_value,
                    "std_err": std_err
                }
            }
            
            print(f"\nRegression Statistics:")
            print(f"Slope: {slope:.6f}")
            print(f"Intercept: {intercept:.6f}")
            print(f"R-value: {r_value:.6f}")
            print(f"R-squared: {r_value**2:.6f}")

    # Create visualization with inset ratio plot
    create_growth_comparison_with_ratio(processors, growth_results, ratio_data, dataset_labels, output_dir)

    return processors, growth_results, ratio_data


def create_growth_comparison_with_ratio(processors, growth_results, ratio_data, dataset_labels, output_dir):
    """
    Create comparison visualization with main growth profiles and inset ratio plot.

    Args:
        processors: List of DSMCDataProcessor objects
        growth_results: List of dictionaries containing growth analysis results
        ratio_data: Dictionary containing ratio analysis for AW1
        dataset_labels: List of labels for each dataset
        output_dir: Directory to save output plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Add inset axes for ratio plot (on the right side)
    ratio_ax = inset_axes(ax, width="30%", height="90%", 
                           loc='right', bbox_to_anchor=(0.05, 0, 1, 1),
                           bbox_transform=ax.transAxes)

    # Plot each dataset's growth profile on main plot
    if len(dataset_labels) == 2:
        dataset_colors = ["blue", "red"]
    else:
        dataset_colors = plt.cm.viridis(np.linspace(0, 0.9, len(processors)))

    for i, (processor, results, label, color) in enumerate(zip(processors, growth_results, dataset_labels, dataset_colors)):
        growth_peaks = results["growth_peaks"]
        radial_distances = results["radial_distances"]
        cell_min_idx = results["cell_min_idx"]
        cell_max_idx = results["cell_max_idx"]
        altitudes = radial_distances[cell_min_idx : cell_max_idx + 1]

        # Get the density amplitude data
        valid_data = growth_peaks[:, -1].squeeze()

        # Apply smoothing
        window_size = min(11, len(valid_data) - (len(valid_data) % 2 + 1))
        if window_size >= 3:
            poly_order = min(3, window_size - 1)
            smoothed_data = savgol_filter(valid_data, window_size, poly_order)

            # Plot both original and smoothed data on main plot
            ax.plot(valid_data, altitudes, linewidth=1.2, color="k", alpha=0.4, linestyle=":")
            ax.plot(smoothed_data, altitudes, linewidth=2.5, color=color, label=f"{label}")
        else:
            # Not enough points for smoothing, plot original data only
            ax.plot(valid_data, altitudes, linewidth=2.5, color=color, label=label)

    # Add reference data to main plot
    ax.plot(TIAN_GROW, TIAN_ALTITUDES, linestyle="--", color=dataset_colors[0], 
            label="Tian et al. (2023)", alpha=0.7, lw=2.5)
    
    # Check if we need to add Tian HF Mode data (for AW2 comparison)
    if len(processors) > 1:
        try:
            reference_data = np.loadtxt("/mnt/performance/data-analysis/exports/tian high freq mode.csv", delimiter=",")
            tian_hf_amp, tian_hf_alt = reference_data[:,0], reference_data[:,1]
            ax.plot(tian_hf_amp, tian_hf_alt, linestyle="--", color=dataset_colors[1], 
                    label="Tian HF Mode", alpha=0.7, lw=2.5)
        except:
            print("Tian HF Mode data file not found. Skipping this reference curve.")
    
    # Set labels for main plot
    ax.set_xlabel("Density Amplitude")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, alpha=0.3)
    
    # Plot ratio data in the inset plot (if available)
    if ratio_data:
        # Plot original normalized data as scatter points
        ratio_ax.scatter(ratio_data["normalized_amp"], ratio_data["harrah_alt"], 
                      color='gray', alpha=0.7, s=30, label='Data Points')
        
        # Plot best fit line
        ratio_ax.plot(ratio_data["best_fit_line"], ratio_data["harrah_alt"], 
                   lw=2.5, color='blue', label=f'Best Fit')
        
        # Add reference line at ratio=1
        ratio_ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.5)
        
        # Set labels and styling for ratio plot
        ratio_ax.set_xlabel("Ratio (AW1/Tian)")
        ratio_ax.set_yticklabels([])  # Hide y-axis labels since they're the same as main plot
        ratio_ax.grid(True, alpha=0.3)
        # Calculate r-squared
        r_squared = ratio_data["regression_stats"]["r_value"] ** 2
        
        # Add r-squared annotation
        ratio_ax.text(0.05, 0.05, f"r²={r_squared:.3f}", transform=ratio_ax.transAxes, 
                   fontsize=10, ha='left', va='bottom')
    
    # Create custom legend for main plot
    from matplotlib.lines import Line2D
    
    # Create legend handles
    line_handles = []
    
    # Add dataset lines (AW1, AW2, etc.)
    for i, (label, color) in enumerate(zip(dataset_labels, dataset_colors)):
        line_handles.append(Line2D([0], [0], color=color, lw=2.5, label=label))
    
    # Add reference style lines
    line_handles.append(Line2D([0], [0], color="black", lw=2.5, linestyle="--", 
                               label="Reference Data"))
    
    # Add the legend to main plot
    ax.legend(handles=line_handles, loc='best', frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "density_growth_comparison_with_ratio.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    
    plt.close()


def main():
    """Parse command line arguments and run comparison"""
    parser = argparse.ArgumentParser(description="Compare density growth profiles with ratio inset")
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

    args = parser.parse_args()

    # Convert window sizes to numpy array if provided
    window_sizes = np.array(args.window_sizes) if args.window_sizes else None

    # Run comparison
    compare_density_growth_with_ratio(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        alt_range=(args.alt_min, args.alt_max),
        use_experimental=args.use_experimental,
        use_dynamic=args.use_dynamic,
        window_sizes=window_sizes,
        dataset_labels=args.labels,
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

        compare_density_growth_with_ratio(
            data_dirs=data_dirs,
            alt_range=(150, 300),
            use_dynamic=True,
            use_experimental=False,
            dataset_labels=["AW1", "AW2"]
        )