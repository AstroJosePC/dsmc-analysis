import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
from importlib import import_module

# Try relative import first, then fall back to direct import
try:
    from .loader_compute_rates import DSMCDataProcessor
except ImportError:
    # If running as script, modify path to import from same directory
    sys.path.append(str(Path(__file__).parent))
    from loader_compute_rates import DSMCDataProcessor

def analyze_spectral_energy(data_path: str, k_c: float = 0.1, avg_window: int = 5):
    """
    Analyze and plot spectral energy from simulation data
    
    Args:
        data_path: Path to simulation results directory
        k_c: Wavenumber threshold for spectral energy integration
        avg_window: Window size for time averaging
    """
    processor = DSMCDataProcessor(data_path)
    processor.load_parameters()
    processor.load_data()
    
    # Compute spectral energy with both Savitzky-Golay and Gaussian filtering
    results_savgol = processor.compute_temperature_spectral_energy(
        k_c=k_c, 
        avg_window=avg_window,
        filter_type='savgol',
        filter_params={'window_length': 11, 'polyorder': 3}
    )
    
    results_gaussian = processor.compute_temperature_spectral_energy(
        k_c=k_c, 
        avg_window=avg_window,
        filter_type='gaussian',
        filter_params={'sigma': 2.0}
    )
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot spectral energy evolution
    # When avg_window > 1, convolution reduces array length, adjust time array accordingly
    offset = (avg_window - 1) // 2 if avg_window > 1 else 0
    times_savgol = results_savgol['times'][offset:-offset] if avg_window > 1 else results_savgol['times']
    times_gaussian = results_gaussian['times'][offset:-offset] if avg_window > 1 else results_gaussian['times']
    
    # Get wave analysis start and end times and period info
    wave_start_time = processor.params.start_step_wave * processor.params.timestep
    wave_end_time = processor.params.final_step_wave * processor.params.timestep
    wave_period = 2 * np.pi / processor.params.omega_freq  # Convert frequency to period
    num_periods = processor.params.num_periods
    
    # Calculate period markers
    period_times = [wave_start_time + i * wave_period for i in range(int(num_periods) + 1)]
    
    # Plot spectral energy with wave timing indicators
    ax1.plot(times_savgol, results_savgol['spectral_energy'], 
             label='Savitzky-Golay filter', color='blue', alpha=0.7)
    ax1.plot(times_gaussian, results_gaussian['spectral_energy'], 
             label='Gaussian filter', color='red', alpha=0.7)
    ax1.axvline(x=wave_start_time, color='green', linestyle='--', label='Wave Start')
    ax1.axvline(x=wave_end_time, color='orange', linestyle='--', label='Wave End')
    
    # Add period markers
    for i, t in enumerate(period_times):
        if t <= wave_end_time:
            label = f'Period {i}' if i == 0 else None
            ax1.axvline(x=t, color='gray', linestyle=':', alpha=0.5, label=label)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Integrated Spectral Energy')
    ax1.set_title(f'High-k (k ≥ {k_c}) Spectral Energy Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # Plot maximum temperature gradients with wave timing indicators
    ax2.plot(times_savgol, results_savgol['max_gradients'], 
             label='Savitzky-Golay filter', color='blue', alpha=0.7)
    ax2.plot(times_gaussian, results_gaussian['max_gradients'], 
             label='Gaussian filter', color='red', alpha=0.7)
    ax2.axvline(x=wave_start_time, color='green', linestyle='--', label='Wave Start')
    ax2.axvline(x=wave_end_time, color='orange', linestyle='--', label='Wave End')
    
    # Add period markers to second plot
    for i, t in enumerate(period_times):
        if t <= wave_end_time:
            label = f'Period {i}' if i == 0 else None
            ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.5, label=label)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Maximum Temperature Gradient [K/km]')
    ax2.set_title('Maximum Temperature Gradients Evolution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure in the results directory
    out_path = Path(data_path) / 'spectral_energy_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results using savgol results as reference
    np.savez(Path(data_path) / 'spectral_energy_results.npz',
             times=times_savgol,
             spectral_energy_savgol=results_savgol['spectral_energy'],
             spectral_energy_gaussian=results_gaussian['spectral_energy'],
             max_gradients_savgol=results_savgol['max_gradients'],
             max_gradients_gaussian=results_gaussian['max_gradients'],
             altitudes=results_savgol['altitudes'],
             k_c=k_c,
             avg_window=avg_window)

if __name__ == "__main__":
    DATA_PATH = "/media/astrojose/AstroBook/ArchiveSims/Results_20250210_040813.382"
    
    # Run analysis with default parameters
    analyze_spectral_energy(DATA_PATH, 1 / 50)
    print(f"Analysis complete. Results saved in {DATA_PATH}")