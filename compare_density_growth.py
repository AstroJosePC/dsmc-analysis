import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter

# Import from the oned_analyzer module
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

def compare_density_growth(
    data_dirs: List[str],
    output_dir: str = None,
    alt_range: Tuple[float, float] = (150, 320),
    use_experimental: bool = True,
    use_dynamic: bool = True,
    window_sizes: np.ndarray = None,
    dataset_labels: List[str] = None,
):
    """
    Compare density growth profiles from multiple datasets.

    Args:
        data_dirs: List of directories containing simulation data
        output_dir: Directory to save output plots (defaults to first data_dir/exports)
        alt_range: Tuple of (min_altitude, max_altitude) in km
        use_experimental: Whether to use experimental background method
        use_dynamic: Whether to use dynamic peak parameter adjustment
        window_sizes: Optional array of window sizes in seconds
        dataset_labels: Optional labels for each dataset
    """
    if len(data_dirs) < 2:
        raise ValueError("At least two data directories must be provided for comparison")

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

    print(f"\nComparing data from {len(data_dirs)} datasets:")
    for i, data_dir in enumerate(data_dirs):
        print(f"Dataset {i+1}: {data_dir} (Label: {dataset_labels[i]})")
    print(f"Saving visualizations to: {output_dir}")

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

    # Create comparison visualization
    compare_growth_profiles(processors, growth_results, dataset_labels, output_dir)

    return processors, growth_results


def compare_growth_profiles(processors, growth_results, dataset_labels, output_dir):
    """
    Create comparison visualization of density growth profiles.

    Args:
        processors: List of DSMCDataProcessor objects
        growth_results: List of dictionaries containing growth analysis results
        dataset_labels: List of labels for each dataset
        output_dir: Directory to save output plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each dataset's growth profile
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
        # Use Savitzky-Golay filter which preserves peaks and valleys better than moving averages
        # Window size needs to be odd and less than the total data points
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
    ax.plot(TIAN_GROW, TIAN_ALTITUDES, linestyle="--", color=dataset_colors[0], label="Tian et al. (2023)", alpha=0.7, lw=2.5)
    # ax.plot(
    #     SPHERICAL_1D_GROW,
    #     SPHERICAL_1D_ALTITUDES,
    #     "k:",
    #     label="Spherical 1D",
    #     alpha=0.7,
    # )

    # Load the new reference data from the CSV
    reference_data = np.loadtxt("/mnt/performance/data-analysis/exports/tian high freq mode.csv", delimiter=",")
    tian_hf_amp, tian_hf_alt = reference_data[:,0], reference_data[:,1]

    ax.plot(tian_hf_amp, tian_hf_alt, linestyle="--", color=dataset_colors[1], label="Tian HF Mode", alpha=0.7, lw=2.5)

    # Set labels and title
    ax.set_xlabel("Density Amplitude")
    ax.set_ylabel("Altitude (km)")

    # Create custom legend with separate color and line style elements
    from matplotlib.lines import Line2D

    # Create legend handles for colors (AW1 and AW2)
    color_handles = [
        Line2D([0], [0], color="blue", lw=2.5, label="AW1"),
        Line2D([0], [0], color="red", lw=2.5, label="AW2")
    ]

    # Create legend handles for line styles (solid and dashed)
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="Harrah 1D Spherical"),
        Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="Tian et al., 2023")
    ]

    # Combine all legend handles
    legend_handles = color_handles + style_handles

    # Add the legend
    ax.legend(handles=legend_handles, loc='best', frameon=True)

    ax.grid(True, alpha=0.3)

    # Draw amplitude points for AW1 and AW2 at 150 and 320 km
    # alt_points = np.array([150, 320])
    breakpoint()

    # Compute amplitude ratios for AW1 and AW2 at 150 and 320 km
    alt_points = np.array([300, ])
    # AW1 (Harrah vs Tian et al. 2023)
    harrah1 = growth_results[0]
    hd1_rad = harrah1["radial_distances"]
    harrah_amp1 = harrah1["growth_peaks"][..., -1].squeeze()
    harrah_alt1 = hd1_rad[harrah1["cell_min_idx"]:harrah1["cell_max_idx"]+1]
    harrah_vals1 = np.interp(alt_points, harrah_alt1, harrah_amp1)
    tian_vals1 = np.interp(alt_points, TIAN_ALTITUDES, TIAN_GROW)
    ratio1 = harrah_vals1 / tian_vals1
    print(f"AW1 (Harrah/Tian) amplitude ratios at {alt_points} km: {ratio1}")

    # AW2 (Harrah vs Tian HF Mode)
    harrah2 = growth_results[1]
    hd2_rad = harrah2["radial_distances"]
    harrah_amp2 = harrah2["growth_peaks"][..., -1].squeeze()
    harrah_alt2 = hd2_rad[harrah2["cell_min_idx"]:harrah2["cell_max_idx"]+1]
    harrah_vals2 = np.interp(alt_points, harrah_alt2, harrah_amp2)
    tian_vals2 = np.interp(alt_points, tian_hf_alt, tian_hf_amp)
    ratio2 = harrah_vals2 / tian_vals2
    print(f"AW2 (Harrah/Tian HF) amplitude ratios at {alt_points} km: {ratio2}")

    # AW1 Harrah and Tian
    # ax.scatter(harrah_vals1, alt_points, color='blue', marker='o')
    # ax.scatter(tian_vals1, alt_points, color='blue', marker='D')
    # AW2 Harrah and Tian HF
    # ax.scatter(harrah_vals2, alt_points, color='red', marker='o')
    # ax.scatter(tian_vals2, alt_points, color='red', marker='D')
    # Annotate values
    # for i, alt in enumerate(alt_points):
    #     ax.text(harrah_vals1[i], alt+3, f"{harrah_vals1[i]:.2f}", color='blue')
    #     ax.text(tian_vals1[i], alt-3, f"{tian_vals1[i]:.2f}", color='blue')
    #     ax.text(harrah_vals2[i], alt+3, f"{harrah_vals2[i]:.2f}", color='red')
    #     ax.text(tian_vals2[i], alt-3, f"{tian_vals2[i]:.2f}", color='red')

    # Save figure
    output_path = output_dir / "density_growth_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_path}")

    plt.close()


def main():
    """Parse command line arguments and run comparison"""
    parser = argparse.ArgumentParser(description="Compare density growth profiles from multiple datasets")
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
    compare_density_growth(
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

        compare_density_growth(
            data_dirs=data_dirs,
            alt_range=(150, 300),
            use_dynamic=True,
            use_experimental=False,
            dataset_labels=["Spherical 1D", "New Simulation"]
        )
