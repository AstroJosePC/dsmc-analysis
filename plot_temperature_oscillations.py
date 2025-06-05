import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

# Import from the oned_analyzer module
from .loader import DSMCDataProcessor, PADDING, H
from .helper_funcs import process_temperature_deviation_peaks

# Set up plotting style
sns.set_theme(context="paper", font_scale=1.5, style="whitegrid")

def plot_temperature_oscillations(data_dir, altitudes_km=None, use_experimental=True, window_sizes=None):
    """
    Plot temperature oscillations with detected peaks at multiple altitudes.
    
    Args:
        data_dir: Directory containing simulation data
        altitudes_km: List of altitudes in km to plot (default: [150, 200, 250, 300])
        use_experimental: Whether to use experimental background method
        window_sizes: Optional array of window sizes in seconds
        
    Returns:
        Dictionary containing the analysis results
    """
    if altitudes_km is None:
        altitudes_km = [150, 200, 250, 300]
    
    print(f"\nProcessing temperature oscillations from: {data_dir}")
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
        
    # Compute temperature perturbations (absolute deviation)
    temperature_perturbation = processor.temperature_data - 270.0  # Assuming 270 K is the reference temperature

    # Define oscillation time mask
    is_oscillating = (
        (processor.params.start_step_wave * processor.params.timestep <= time)
        & (time <= processor.params.final_step_wave * processor.params.timestep + PADDING)
    )
    
    # Get oscillation time array
    oscillation_time = time[is_oscillating]
    
    # Find cell indices for requested altitudes
    altitude_cells = []
    actual_altitudes = []
    for alt_km in altitudes_km:
        cell_idx = np.argmin(np.abs(radial_distances - alt_km))
        altitude_cells.append(cell_idx)
        actual_altitudes.append(radial_distances[cell_idx])
        print(f"Requested {alt_km} km -> Using {radial_distances[cell_idx]:.1f} km (cell {cell_idx})")
    
    # Set up peak detection parameters
    base_kwargs = {
        "prominence": 1.0,  # Temperature peaks typically larger than density
        "distance": 12,
        "sampling_rate": 1 / np.mean(np.diff(processor.params.time)),
        "cutoff_frequency": processor.params.omega_freq * 1.2,
        "filter_order": 2,
    }
    
    # Process temperature peaks for the altitude range covering all requested altitudes
    min_cell = min(altitude_cells)
    max_cell = max(altitude_cells)
    
    print(f"\nProcessing temperature peaks from {radial_distances[min_cell]:.1f} to {radial_distances[max_cell]:.1f} km")
    
    temp_deviation_peaks, temp_deviation_last_cycle = process_temperature_deviation_peaks(
        -temperature_perturbation[..., 0],  # Use first azimuthal cell
        is_oscillating,
        radial_distances,
        min_cell,
        max_cell,
        processor.params.num_periods,
        **base_kwargs,
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette for different altitudes
    colors = plt.cm.viridis(np.linspace(0, 1, len(altitudes_km)))
    
    # Plot each altitude curve
    for i, (alt_km, cell_idx, color) in enumerate(zip(altitudes_km, altitude_cells, colors)):
        # Get temperature oscillation data for this altitude
        temp_oscillation = temperature_perturbation[is_oscillating, cell_idx, 0]
        
        # Plot the temperature curve
        ax.plot(oscillation_time, temp_oscillation, color=color, linewidth=2, 
                label=f'{actual_altitudes[i]:.0f} km', alpha=0.8)
        
        # Get peaks for this altitude if available
        relative_cell_idx = cell_idx - min_cell
        if relative_cell_idx < temp_deviation_peaks.shape[0]:
            peaks_data = -temp_deviation_peaks[relative_cell_idx, :]
            
            # Find peak indices in the oscillation data
            if not np.ma.is_masked(peaks_data):
                # Use a simple peak finding approach to locate peaks in time
                from scipy.signal import find_peaks
                temp_peaks, _ = find_peaks(-temp_oscillation, 
                                         prominence=base_kwargs["prominence"],
                                         distance=base_kwargs["distance"])
                
                if len(temp_peaks) > 0:
                    # Take the most prominent peaks up to num_periods
                    num_peaks_to_show = min(len(temp_peaks), int(processor.params.num_periods))
                    peak_heights = -temp_oscillation[temp_peaks]
                    peak_order = np.argsort(peak_heights)[::-1]
                    selected_peaks = temp_peaks[peak_order[:num_peaks_to_show]]
                    selected_peaks.sort()
                    
                    # Plot the peaks
                    ax.scatter(oscillation_time[selected_peaks], temp_oscillation[selected_peaks], 
                             color=color, s=80, marker='o', edgecolors='black', linewidth=1,
                             zorder=5, alpha=0.9)
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature Perturbation (K)')
    ax.set_title('Temperature Oscillations with Detected Peaks')
    ax.legend(loc='best', frameon=True, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to only show oscillation period
    ax.set_xlim(oscillation_time[0], oscillation_time[-1])
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "temperature_oscillations_with_peaks.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    
    # Store results
    results = {
        "oscillation_time": oscillation_time,
        "altitudes_km": actual_altitudes,
        "altitude_cells": altitude_cells,
        "temp_deviation_peaks": -temp_deviation_peaks,
        "temp_deviation_last_cycle": temp_deviation_last_cycle,
        "processor": processor,
        "temperature_perturbation": temperature_perturbation,
        "is_oscillating": is_oscillating
    }
    
    plt.show()
    
    return results

def plot_temperature_peak_summary(results, output_dir=None):
    """
    Create a summary plot showing peak amplitudes vs altitude.
    
    Args:
        results: Dictionary from plot_temperature_oscillations
        output_dir: Optional output directory (uses results directory if None)
    """
    if output_dir is None:
        output_dir = Path(results["processor"].data_dir) / "exports"
    
    temp_deviation_peaks = results["temp_deviation_peaks"]
    processor = results["processor"]
    altitudes = results["processor"].params.radial_cells

    # Get altitude range that was processed
    min_cell = results["altitude_cells"][0] - (results["altitude_cells"][0] - min(results["altitude_cells"]))
    max_cell = results["altitude_cells"][-1] + (max(results["altitude_cells"]) - results["altitude_cells"][-1])
    
    processed_altitudes = altitudes[min_cell:max_cell+1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each period's peaks
    num_periods = temp_deviation_peaks.shape[1]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, num_periods))
    
    for period in range(num_periods):
        peak_amplitudes = temp_deviation_peaks[:, period]

        ax.plot(peak_amplitudes, processed_altitudes, 
                'o-', color=colors[period], label=f'Period {period+1}', 
                markersize=6, linewidth=2, alpha=0.8)
        
        # Compute and print best-fit line slope
        if np.all(np.isfinite(peak_amplitudes)) and np.all(np.isfinite(processed_altitudes)):
            slope, intercept = np.polyfit(processed_altitudes, peak_amplitudes, 1)
            print(f"Period {period+1}: Best-fit slope = {slope * 50:.2f} K/H0$")
        else:
            print(f"Period {period+1}: Cannot compute slope due to NaN or inf values.")

    ax.set_xlabel('Temperature Peak Amplitude (K)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Temperature Peak Amplitudes vs Altitude')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "temperature_peak_amplitudes.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved peak summary figure: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Define the data directory - replace with your simulation directory
    data_dir = "/media/astrojose/AstroBook/ArchiveSims/Results_20250326_133319.916"
    
    # Run temperature oscillation analysis
    results = plot_temperature_oscillations(
        data_dir=data_dir,
        altitudes_km=[150, 180, 220, 260, 300],
        use_experimental=False
    )
    
    # Create peak summary plot
    plot_temperature_peak_summary(results)
    
    print("\nTemperature oscillation analysis complete!")
