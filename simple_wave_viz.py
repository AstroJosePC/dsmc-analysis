import numpy as np
import matplotlib.pyplot as plt
from loader_compute_rates import DSMCDataProcessor
from spectral_analysis_utils import (
    detrend_temperature_profile,
    smooth_profile,
    resample_to_uniform_grid,
    compute_spectral_features,
    find_fundamental_mode,
    DEFAULT_SMOOTHING_PARAMS,
)
from wave_spectral_analysis import _predict_wavelength, analyze_wave_steepening  # new import

def simple_wave_visualization(data_dir, altitude_range=(120,240), num_snapshots=5):
    # ...existing code: load data...
    processor = DSMCDataProcessor(data_dir)
    processor.load_parameters()
    processor.load_data()

    z = processor.params.radial_cells
    alt_mask = (z >= altitude_range[0]) & (z <= altitude_range[1])
    z_analysis = z[alt_mask]
    desired_dz = np.median(np.diff(z_analysis))

    # Process snapshots for temperature profile overlay (sparse profiles)
    start_idx, end_idx = processor.get_wave_time_indices()
    snapshot_indices = np.linspace(start_idx, end_idx-1, num_snapshots, dtype=int)
    snapshot_profiles = []
    for idx in snapshot_indices:
        temp_profile = processor.temperature_data[idx, alt_mask, 0] - DEFAULT_SMOOTHING_PARAMS["temp_offset"]
        temp_detrended = temp_profile
        # temp_detrended = detrend_temperature_profile(temp_profile, z_analysis)
        temp_smoothed = smooth_profile(
            temp_detrended,
            cutoff_freq=2 * np.pi / (_predict_wavelength(processor.params.omega_freq)[0]) * 1.5,
            order=DEFAULT_SMOOTHING_PARAMS["filter_order"]
        )
        z_uniform, temp_uniform = resample_to_uniform_grid(z_analysis, temp_smoothed, dz=desired_dz)
        snapshot_profiles.append((z_uniform, temp_uniform))

    # Run the full wave steepening analysis (without built-in visualization)
    analysis_result = analyze_wave_steepening(
        processor,
        altitude_range=altitude_range,
        stabilization_period=2.5,
        visualize=False,
        num_snapshots=num_snapshots
    )
    # Extract wavelength evolution and corresponding time from analysis results
    analyzed_time = analysis_result["time"]
    wavelength_mean = analysis_result["wavelength_evolution"]["mean"]

    # Create simplified visualization: left subplot for temperature profiles, right for wavelength evolution
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: plot temperature profiles (overlay snapshots)
    ax_rel = ax[0].twiny()  # Create twin axis for relative amplitude

    for i, (z_prof, temp_prof) in enumerate(snapshot_profiles):
        ax[0].plot(temp_prof, z_prof, label=f"T{i+1}", linewidth=2)
        # Calculate and plot relative amplitude
        rel_amp = 100 * (temp_prof) / 270.0  # Relative to 270K
        ax_rel.plot(rel_amp, z_prof, alpha=0)  # Invisible plot to set up the twin axis

    ax[0].set_xlabel("Temperature Perturbation (°C)", fontsize=12)
    ax[0].set_ylabel("Altitude (km)", fontsize=12)
    ax[0].set_title("Temperature Profiles", fontsize=14, pad=20)
    ax_rel.set_xlabel(r"Relative Amplitude % (T/T$_0$ - 1)", fontsize=12)
    ax[0].legend(fontsize=10, loc='upper right')
    ax[0].tick_params(axis='both', labelsize=10)
    ax_rel.tick_params(axis='both', labelsize=10)

    # Right: plot wavelength evolution from analysis
    time_since_onset = analyzed_time - analyzed_time[0]
    ax[1].grid(True, alpha=0.3, linestyle="-", color="gray")
    ax[1].plot(time_since_onset, wavelength_mean, marker="o", linestyle="-", color="C0",
               label="Measured", linewidth=2)

    # Add predicted wavelength
    predicted_wl = _predict_wavelength(processor.params.omega_freq)[0]
    ax[1].axhline(y=predicted_wl, color='r', linestyle='--',
                  label=f"Predicted ({predicted_wl:.1f} km)", linewidth=2)

    ax[1].set_xlabel("Time since wave onset (s)", fontsize=12)
    ax[1].set_ylabel("Wavelength (km)", fontsize=12)
    ax[1].set_title("Wavelength Evolution", fontsize=14)
    ax[1].legend(fontsize=10)
    ax[1].tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    # save figure
    plt.savefig(f"{data_dir}/analysis/simple_wave_visualization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    data_dir = "/media/astrojose/AstroBook/ArchiveSims/Results_20250210_040813.382"
    simple_wave_visualization(data_dir)
