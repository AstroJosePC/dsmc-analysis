"""
Wave steepening analysis using spectral methods.
This script implements spectral analysis techniques to identify and quantify nonlinear wave steepening
in temperature profiles from DSMC simulation data.
"""

# import matplotlib

# matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import seaborn as sns
from loader_compute_rates import DSMCDataProcessor
from spectral_analysis_utils import (
    basic_spectral_features,
    detrend_temperature_profile,
    compute_spectral_features,
    find_fundamental_mode,
    find_harmonics,
    initialize_tracking_arrays,
    masked_stats,
    smooth_profile,
    resample_to_uniform_grid,
    compute_reliability,
    DEFAULT_SMOOTHING_PARAMS,  # new import
    DEFAULT_TRACKING_PARAMS,
)


def analyze_wave_steepening_simple(
    processor: DSMCDataProcessor,
    altitude_range: Tuple[float, float] = (120, 230),
    stabilization_period: int = 2.5,
    time_window: int = 1000,
) -> Dict:
    # Basic tracking state
    wavelength_history = []
    current_wavelength = None

    # Get initial grid
    z = processor.params.radial_cells
    alt_mask = (z >= altitude_range[0]) & (z <= altitude_range[1])
    z_analysis = z[alt_mask]
    desired_dz = np.median(np.diff(z_analysis))
    # Compute predicted wavelength and derive new cutoff frequency
    predicted_wavelength, _, _ = _predict_wavelength(processor.params.omega_freq)
    new_cutoff_freq = 2 * np.pi / (predicted_wavelength)  # in 1/km

    # Get relevant simulation parameters and indices
    start_idx, end_idx = processor.get_wave_time_indices()
    end_idx = min(start_idx + time_window, end_idx)
    time = processor.params.time[start_idx:end_idx]
    time_since_onset = time - time[0]

    # Initialize tracking variables
    wavelength_stats = {
        "mean": np.full(len(time), np.nan),
        "std": np.full(len(time), np.nan),
        "reliability": np.zeros(len(time)),
        "phase_coherence": np.zeros(len(time)),
        "confidence": np.zeros(len(time)),
        "is_valid": np.zeros(len(time), dtype=bool),
    }

    for t_idx, t in enumerate(range(start_idx, end_idx)):
        # Simplified processing pipeline
        # Get temperature profile and resample to uniform grid
        temp_profile = processor.temperature_data[t, alt_mask, 0] - 270.0
        temp_detrended_raw = temp_profile
        # temp_detrended_raw = detrend_temperature_profile(
        #     temp_profile, z_analysis
        # )
        # Use computed cutoff frequency instead of the default value 0.2
        temp_smoothed_raw = smooth_profile(
            temp_detrended_raw, cutoff_freq=new_cutoff_freq, order=2
        )
        z_uniform, temp_uniform = resample_to_uniform_grid(
            z_analysis, temp_smoothed_raw, dz=desired_dz
        )
        k, psd = basic_spectral_features(temp_uniform, z_uniform)

        # Adaptive power threshold
        noise_floor = np.percentile(psd, 10)
        min_power = noise_floor + DEFAULT_TRACKING_PARAMS["unified"][
            "min_power_ratio"
        ] * (np.max(psd) - noise_floor)

        # Find fundamental mode
        mode = find_fundamental_mode(
            k, psd, *DEFAULT_TRACKING_PARAMS["unified"]["k_range"]
        )

        if mode["power"] > min_power:
            wavelength = 2 * np.pi / mode["k"]
            wavelength_history.append(wavelength)

            # Maintain fixed-length history
            if (
                len(wavelength_history)
                > DEFAULT_TRACKING_PARAMS["unified"]["smoothing_window"]
            ):
                wavelength_history.pop(0)

            current_wavelength = np.mean(wavelength_history)
            reliability = min(
                1.0,
                len(wavelength_history)
                / DEFAULT_TRACKING_PARAMS["unified"]["smoothing_window"],
            )

        else:
            reliability = 0.0

        # Store results
        wavelength_stats["mean"][t_idx] = current_wavelength
        wavelength_stats["reliability"][t_idx] = reliability
        wavelength_stats["is_valid"][t_idx] = (
            reliability
            >= DEFAULT_TRACKING_PARAMS["unified"]["reliability_threshold"]
        )

    wave_period = 2 * np.pi / processor.params.omega_freq
    stabilization_time = wave_period * stabilization_period

    # Focus on essential elements
    plt.figure(figsize=(12, 8))

    # Wavelength timeline
    ax1 = plt.subplot(211)
    ax1.plot(time_since_onset, wavelength_stats["mean"], "b-", alpha=0.7)
    ax1.fill_between(
        time_since_onset,
        wavelength_stats["mean"] - wavelength_stats["std"],
        wavelength_stats["mean"] + wavelength_stats["std"],
        color="b",
        alpha=0.1,
    )
    ax1.axvline(stabilization_time, color="r", linestyle="--")

    # Power spectrum
    ax2 = plt.subplot(212)
    ax2.loglog(k, psd, "k-")
    ax2.axvline(
        *[2 * np.pi / λ for λ in [80, 120]], color="gray", linestyle=":"
    )

    plt.tight_layout()
    plt.show()


def analyze_wave_steepening(
    processor: DSMCDataProcessor,
    altitude_range: Tuple[float, float] = (120, 240),
    stabilization_period: int = 2.5,
    visualize: bool = True,
    num_snapshots: int = 5,
) -> Dict:
    """
    Analyze wave steepening using spectral methods.

    Args:
        processor: DSMC data processor instance
        altitude_range: Tuple of (min_alt, max_alt) in km
        stabilization_period: Number of oscillations to wait for wave stabilization
        time_window: Number of time steps to analyze
        visualize: Whether to generate plots
        num_snapshots: Number of temperature profile snapshots to show in visualization
    """
    if not processor.temperature_data is not None:
        raise ValueError("Temperature data not loaded")

    # Get initial grid
    z = processor.params.radial_cells
    alt_mask = (z >= altitude_range[0]) & (z <= altitude_range[1])
    z_analysis = z[alt_mask]
    desired_dz = np.median(np.diff(z_analysis))
    # Compute predicted wavelength and derive new cutoff frequency
    predicted_wavelength, _, _ = _predict_wavelength(processor.params.omega_freq)
    new_cutoff_freq = 2 * np.pi / (predicted_wavelength) * 1.5 # in 1/km

    # Calculate grid statistics for reporting
    dz = np.diff(z_analysis)
    rel_diff = np.abs(np.diff(dz) / dz[:-1])
    original_resolution = {
        "mean": np.mean(dz),
        "std": np.std(dz),
        "max_variation": np.max(rel_diff) * 100,  # as percentage
    }

    print("\nOriginal grid statistics:")
    print(f"Mean spacing: {original_resolution['mean']:.3f} km")
    print(f"Std dev: {original_resolution['std']:.3f} km")
    print(f"Max variation: {original_resolution['max_variation']:.2f}%")

    # Get relevant simulation parameters and indices
    start_idx, end_idx = processor.get_wave_time_indices()
    # end_idx = min(start_idx + time_window, end_idx)

    time = processor.params.time[start_idx:end_idx]
    time_since_onset = time - time[0]

    # Calculate wave period and stabilization
    wave_period = 2 * np.pi / processor.params.omega_freq
    stabilization_time = wave_period * stabilization_period

    # Calculate exact number of wave periods
    wave_periods = (time_since_onset[-1] - time_since_onset[0]) / wave_period
    stable_periods = (
        (time_since_onset[-1] - stabilization_time) / wave_period
        if stabilization_time > 0
        else 0
    )

    print("\nWave period analysis:")
    print(f"Wave period: {wave_period:.1f}s")
    print(f"Total periods: {wave_periods:.1f}")
    print(f"Before stabilization: {wave_periods - stable_periods:.1f}")
    print(f"After stabilization: {stable_periods:.1f}")
    print(f"Stabilization time: {stabilization_time:.1f}s")

    wavelength_stats, diagnostics, harmonic_ratios = (
        initialize_tracking_arrays(len(time))
    )

    prev_k1 = None
    tracking_gap_counter = 0
    wavelength_history = []
    power_history = []

    # Adaptive window sizes with more flexibility
    base_window_size = int(wave_period / np.mean(np.diff(time)))
    min_window_size = max(3, base_window_size // 6)  # Slightly larger minimum

    print("\nProcessing snapshots...")
    valid_measurements = {"initial": 0, "stable": 0}
    total_measurements = {"initial": 0, "stable": 0}

    # Debug variables
    detected_peaks = []
    power_levels = []
    reliability_scores = []

    for t_idx, t in enumerate(range(start_idx, end_idx)):
        if t_idx % 50 == 0:
            print(
                f"Processing time step {t_idx}/{end_idx - start_idx} ({time_since_onset[t_idx]:.1f}s)"
            )

        is_stable_phase = time_since_onset[t_idx] >= stabilization_time
        phase = "stable" if is_stable_phase else "initial"
        params = DEFAULT_TRACKING_PARAMS[phase]

        diagnostics["phase"][t_idx] = phase
        total_measurements[phase] += 1

        # Process temperature profile: detrend, smooth, resample, FFT
        temp_profile = (
            processor.temperature_data[t, alt_mask, 0]
            - DEFAULT_SMOOTHING_PARAMS["temp_offset"]
        )
        # Replace default cutoff with new_cutoff_freq
        temp_smoothed_raw = smooth_profile(
            temp_profile,
            cutoff_freq=new_cutoff_freq,
            order=DEFAULT_SMOOTHING_PARAMS["filter_order"],
        )
        # Resample smoothed profile to uniform grid
        z_uniform, temp_uniform = resample_to_uniform_grid(
            z_analysis, temp_smoothed_raw, dz=desired_dz
        )

        detrend_temp_uniform = temp_uniform
        
        # detrend_temp_uniform = detrend_temperature_profile(
        #     temp_uniform, z_uniform, polyorder=1
        # )

        k, psd = compute_spectral_features(temp_uniform, z_uniform)

        diagnostics["max_psd"][t_idx] = np.max(psd)
        power_levels.append(np.max(psd))

        # Debug plotting: interactive spectral power breakdown with an additional temperature snapshot every 10th snapshot after 100s since onset
        if t_idx % 12 == 0 and (time_since_onset[t_idx] >= 1e6):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            # Spectral power plot
            axs[0].plot(k, psd, "b-", label="PSD")
            axs[0].set_xlabel("Wavenumber (k)")
            axs[0].set_ylabel("Power Spectral Density")
            axs[0].set_title(f"Spectral Power at t = {time_since_onset[t_idx]:.1f} s")
            # Select maximum PSD value and its corresponding k
            max_psd_val = np.max(psd)
            max_idx = np.argmax(psd)
            k_max = k[max_idx]
            axs[0].axvline(k_max, color="orange", linestyle="--", label="Max PSD Position")
            # NEW: Show new_cutoff_freq on the spectral power plot
            axs[0].axvline(new_cutoff_freq, color="magenta", linestyle=":",
                           label=f"Cutoff freq: {new_cutoff_freq:.2f}")
            axs[0].annotate(f"Max PSD: {max_psd_val:.2e}", xy=(k_max, max_psd_val),
                            xytext=(0.7*k[-1], 0.7*max_psd_val),
                            arrowprops=dict(facecolor="orange", shrink=0.05))
            axs[0].legend()
            # Temperature profile plot (uniform grid)
            axs[1].plot(detrend_temp_uniform, z_uniform, "r-", label="Uniform Temperature")
            axs[1].set_xlabel("Temperature")
            axs[1].set_ylabel("Altitude")
            axs[1].set_title(f"Temp vs Altitude at t = {time_since_onset[t_idx]:.1f} s")
            axs[1].legend()
            plt.tight_layout()
            plt.show()

        # Find harmonics with phase-specific parameters
        harmonics = find_harmonics(
            k,
            psd,
            prev_k1=prev_k1,
            is_stable_phase=is_stable_phase,
            height_fraction=1 / 15 if is_stable_phase else 1 / 12,
            prominence_fraction=1 / 25 if is_stable_phase else 1 / 20,
            verbose=(t_idx % 50 == 0),
        )

        # Update tracking and statistics with improved validation
        current_k1 = harmonics["fundamental"]["k"]
        current_power = harmonics["fundamental"]["power"]

        if current_k1 > 0 and current_power > 0:
            current_wavelength = 2 * np.pi / current_k1
            detected_peaks.append(current_wavelength)

            # Dynamic power threshold based on recent history
            recent_max_power = (
                np.max(power_levels[-20:]) if power_levels else current_power
            )
            min_power_threshold = 0.03 * recent_max_power  # Reduced from 0.05

            # Phase-specific wavelength validation
            min_wl, max_wl = params["wavelength_range"]
            is_valid_wavelength = (
                (
                    0.8 * min_wl <= current_wavelength <= 1.2 * max_wl
                )  # 20% buffer
                and current_power >= min_power_threshold
            )

            if is_valid_wavelength:
                if t_idx % 50 == 0:
                    print(
                        f"Valid wavelength at t={time_since_onset[t_idx]:.1f}s: "
                        f"λ={current_wavelength:.1f}km, power={current_power:.2e}"
                    )

                prev_k1 = current_k1
                tracking_gap_counter = 0
                diagnostics["tracking_status"][t_idx] = "tracked"

                # Update wavelength history with phase-specific window
                max_history = (
                    base_window_size if is_stable_phase else min_window_size
                )

                wavelength_history.append(current_wavelength)
                power_history.append(current_power)

                if len(wavelength_history) > max_history:
                    wavelength_history.pop(0)
                    power_history.pop(0)

                # Adaptive window with phase-specific minimums
                base_window_size = int(wave_period / np.mean(np.diff(time)))
                if is_stable_phase:
                    min_window_size = max(
                        3, base_window_size // 5
                    )  # Slightly larger for stable
                else:
                    min_window_size = max(
                        2, base_window_size // 6
                    )  # Keep shorter for initial

                if len(wavelength_history) >= min_window_size:
                    mean_wl = np.mean(wavelength_history)
                    std_wl = np.std(wavelength_history)

                    # Calculate quality metrics
                    peak_quality = min(1.0, current_power / max(power_history))
                    temporal_stability = 1.0 / (1.0 + (std_wl / mean_wl))
                    tracking_quality = 1.0 - (
                        tracking_gap_counter / params["max_tracking_gap"]
                    )

                    # Use compute_reliability from utils instead of inline calculation
                    reliability = compute_reliability(
                        peak_quality=peak_quality,
                        temporal_stability=temporal_stability,
                        tracking_quality=tracking_quality,
                        min_peak_quality=params["min_peak_quality"],
                        min_temporal_stability=params[
                            "min_temporal_stability"
                        ],
                        is_stable_phase=is_stable_phase,
                    )

                    reliability_scores.append(reliability)

                    if reliability >= params["reliability_threshold"]:
                        valid_measurements[phase] += 1
                        if t_idx % 50 == 0:
                            print(
                                f"  Validated: λ={mean_wl:.1f}±{std_wl:.1f}km, r={reliability:.2f}"
                            )
                    elif t_idx % 50 == 0:
                        print(
                            f"  Not validated: reliability {reliability:.2f} < threshold {params['reliability_threshold']:.2f}"
                        )

                    # Store results
                    wavelength_stats["mean"][t_idx] = mean_wl
                    wavelength_stats["std"][t_idx] = std_wl
                    wavelength_stats["reliability"][t_idx] = reliability
                    wavelength_stats["phase_coherence"][t_idx] = (
                        temporal_stability
                    )
                    wavelength_stats["is_valid"][t_idx] = (
                        reliability >= params["reliability_threshold"]
                    )

                # Update harmonic ratios for all tracked measurements
                harmonic_ratios["R21"][t_idx] = (
                    harmonics["harmonic_2"]["power"] / current_power
                )
                harmonic_ratios["R31"][t_idx] = (
                    harmonics["harmonic_3"]["power"] / current_power
                )

                # Store diagnostic information
                for key in [
                    "fundamental_k",
                    "fundamental_power",
                    "harmonic_2_power",
                    "harmonic_3_power",
                ]:
                    component = key.replace("_power", "").replace("_k", "")
                    value_type = "power" if "_power" in key else "k"
                    diagnostics[key][t_idx] = harmonics[component][value_type]
            else:
                if t_idx % 50 == 0:
                    print(
                        f"Invalid wavelength at t={time_since_onset[t_idx]:.1f}s: "
                        f"λ={current_wavelength:.1f}km, power={current_power:.2e}"
                    )
                diagnostics["tracking_status"][t_idx] = "invalid_wavelength"
                _handle_invalid_measurement(wavelength_stats, t_idx)
        else:
            tracking_gap_counter += 1
            diagnostics["tracking_status"][t_idx] = (
                f"gap_{tracking_gap_counter}"
            )

            if tracking_gap_counter > params["max_tracking_gap"]:
                if t_idx % 50 == 0:
                    print(
                        f"Reset tracking at t={time_since_onset[t_idx]:.1f}s after {tracking_gap_counter} gaps"
                    )
                prev_k1 = None
                wavelength_history.clear()
                power_history.clear()

            _handle_invalid_measurement(wavelength_stats, t_idx)

    # Print measurement statistics with additional analysis
    print("\nMeasurement Statistics:")
    if detected_peaks:
        print(
            f"\nDetected wavelengths range: {np.min(detected_peaks):.1f}-{np.max(detected_peaks):.1f} km"
        )
    if reliability_scores:
        print(
            f"Reliability scores range: {np.min(reliability_scores):.2f}-{np.max(reliability_scores):.2f}"
        )

    # Print detailed measurement statistics with improved formatting
    print("\n" + "=" * 50)
    print("Wave Analysis Summary".center(50))
    print("=" * 50)
    print("\nSimulation Parameters:")
    print(f"  Period: {wave_period:.1f}s")
    print(f"  Altitude range: {altitude_range[0]}-{altitude_range[1]} km")
    print(
        f"  Resolution: {np.mean(np.diff(z_analysis)):.2f} ± {np.std(np.diff(z_analysis)):.3f} km"
    )
    print(
        f"  Time analyzed: {time_since_onset[-1]:.1f}s ({wave_periods:.1f} wave periods)"
    )

    print("\nWave Period Analysis:")
    print(f"  Total wave periods: {wave_periods:.1f}")
    print(f"  Before stabilization: {wave_periods - stable_periods:.1f}")
    print(f"  After stabilization: {stable_periods:.1f}")

    print("\nWavelength Analysis:")
    for phase in ["initial", "stable"]:
        total = total_measurements[phase]
        valid = valid_measurements[phase]
        if total > 0:
            print(f"\n{phase.title()} phase:")
            print(
                f"  Valid measurements: {valid}/{total} ({100 * valid / total:.1f}%)"
            )
            valid_mask = (diagnostics["phase"] == phase) & wavelength_stats[
                "is_valid"
            ]
            if np.any(valid_mask):
                mean_wavelength = np.nanmean(
                    wavelength_stats["mean"][valid_mask]
                )
                std_wavelength = np.nanstd(
                    wavelength_stats["mean"][valid_mask]
                )
                mean_reliability = np.nanmean(
                    wavelength_stats["reliability"][valid_mask]
                )
                print(
                    f"  Mean wavelength: {mean_wavelength:.1f} ± {std_wavelength:.1f} km"
                )
                print(f"  Mean reliability: {mean_reliability:.2f}")
                print("  Tracking parameters:")
                print(
                    f"    Wavelength range: {DEFAULT_TRACKING_PARAMS[phase]['wavelength_range']} km"
                )
                print(
                    f"    Reliability threshold: {DEFAULT_TRACKING_PARAMS[phase]['reliability_threshold']:.2f}"
                )

    print("\nHarmonic Ratios (after stabilization):")
    stable_mask = (diagnostics["phase"] == "stable") & wavelength_stats[
        "is_valid"
    ]
    if np.any(stable_mask):
        r21_mean = np.nanmean(harmonic_ratios["R21"][stable_mask])
        r21_std = np.nanstd(harmonic_ratios["R21"][stable_mask])
        r31_mean = np.nanmean(harmonic_ratios["R31"][stable_mask])
        r31_std = np.nanstd(harmonic_ratios["R31"][stable_mask])
        print(f"  R21: {r21_mean:.3f} ± {r21_std:.3f}")
        print(f"  R31: {r31_mean:.3f} ± {r31_std:.3f}")
    else:
        print("  No valid measurements in stable phase")

    print("\n" + "=" * 50 + "\n")

    # Calculate phase-specific wavelength statistics
    stable_mask = diagnostics["phase"] == "stable"
    initial_mask = ~stable_mask

    wavelength_analysis = {
        "initial_phase": masked_stats(
            wavelength_stats["mean"],
            initial_mask,
            wavelength_stats["reliability"],
        ),
        "stable_phase": masked_stats(
            wavelength_stats["mean"],
            stable_mask,
            wavelength_stats["reliability"],
        ),
    }

    wave_period = 2 * np.pi / processor.params.omega_freq

    # Add safe statistics calculation functions
    def safe_nanmean(arr):
        return (
            np.nanmean(arr)
            if len(arr) > 0 and not np.all(np.isnan(arr))
            else np.nan
        )

    def safe_nanstd(arr):
        return (
            np.nanstd(arr)
            if len(arr) > 0 and not np.all(np.isnan(arr))
            else np.nan
        )

    # After the measurement loop, add peak statistics analysis
    wavelength_stats_by_phase = {
        "initial": {"peaks": [], "powers": [], "times": []},
        "stable": {"peaks": [], "powers": [], "times": []},
    }

    for t_idx, t in enumerate(range(start_idx, end_idx)):
        phase = (
            "stable"
            if time_since_onset[t_idx] >= stabilization_time
            else "initial"
        )
        if wavelength_stats["is_valid"][t_idx]:
            wavelength_stats_by_phase[phase]["peaks"].append(
                wavelength_stats["mean"][t_idx]
            )
            wavelength_stats_by_phase[phase]["powers"].append(
                diagnostics["fundamental_power"][t_idx]
            )
            wavelength_stats_by_phase[phase]["times"].append(
                time_since_onset[t_idx]
            )

    # Print peak detection statistics
    print("\nDetailed Wavelength Analysis:")
    for phase in ["initial", "stable"]:
        peaks = wavelength_stats_by_phase[phase]["peaks"]
        times = wavelength_stats_by_phase[phase]["times"]

        print(f"\n{phase.title()} Phase:")
        if peaks:
            print(f"  Number of valid peaks: {len(peaks)}")
            print(
                f"  Wavelength range: {np.min(peaks):.1f}-{np.max(peaks):.1f} km"
            )
            print(
                f"  Mean wavelength: {np.mean(peaks):.1f} ± {np.std(peaks):.1f} km"
            )
            print(f"  Median wavelength: {np.median(peaks):.1f} km")
            print(
                f"  Time coverage: {100 * len(peaks) / total_measurements[phase]:.1f}% of phase"
            )
            if len(peaks) > 1:
                peak_intervals = np.diff(times)
                print(
                    f"  Mean time between peaks: {np.mean(peak_intervals):.1f}s"
                )
        else:
            print("  No valid measurements")

    if visualize:
        # Create analysis directory if it doesn't exist
        results_dir = Path(processor.data_dir) / "analysis"
        results_dir.mkdir(exist_ok=True)

        # Calculate additional metrics for the report with safe statistics
        stable_mask = (diagnostics["phase"] == "stable") & wavelength_stats[
            "is_valid"
        ]
        initial_mask = (diagnostics["phase"] == "initial") & wavelength_stats[
            "is_valid"
        ]

        # Calculate exact number of wave periods
        wave_periods = (
            time_since_onset[-1] - time_since_onset[0]
        ) / wave_period
        stable_periods = (
            (time_since_onset[-1] - stabilization_time) / wave_period
            if stabilization_time > 0
            else 0
        )

        metrics = {
            "wave_period": wave_period,
            "altitude_range": altitude_range,
            "stabilization_time": stabilization_time,
            "total_time": time_since_onset[-1],
            "wave_periods": {
                "total": wave_periods,
                "before_stabilization": wave_periods - stable_periods,
                "after_stabilization": stable_periods,
            },
            "altitude_stats": {
                "min_altitude": z_analysis[0],
                "max_altitude": z_analysis[-1],
                "resolution": {
                    "mean": np.mean(np.diff(z_analysis)),
                    "std": np.std(np.diff(z_analysis)),
                },
                "num_points": len(z_analysis),
            },
            "wavelength_stats": {
                "before_stabilization": {
                    "mean": safe_nanmean(
                        wavelength_stats["mean"][initial_mask]
                    ),
                    "std": safe_nanstd(wavelength_stats["mean"][initial_mask]),
                    "reliability": safe_nanmean(
                        wavelength_stats["reliability"][initial_mask]
                    ),
                    "num_measurements": np.sum(initial_mask),
                    "total_possible": np.sum(
                        diagnostics["phase"] == "initial"
                    ),
                },
                "after_stabilization": {
                    "mean": safe_nanmean(
                        wavelength_stats["mean"][stable_mask]
                    ),
                    "std": safe_nanstd(wavelength_stats["mean"][stable_mask]),
                    "reliability": safe_nanmean(
                        wavelength_stats["reliability"][stable_mask]
                    ),
                    "num_measurements": np.sum(stable_mask),
                    "total_possible": np.sum(diagnostics["phase"] == "stable"),
                },
            },
            "harmonic_ratios": {
                "R21_mean": safe_nanmean(harmonic_ratios["R21"][stable_mask]),
                "R31_mean": safe_nanmean(harmonic_ratios["R31"][stable_mask]),
                "R21_std": safe_nanstd(harmonic_ratios["R21"][stable_mask]),
                "R31_std": safe_nanstd(harmonic_ratios["R31"][stable_mask]),
            },
            "desired_dz": desired_dz,
        }

        # Create figure with metrics panel
        fig = _generate_visualization(
            time,
            harmonic_ratios,
            wavelength_stats,
            diagnostics,
            stabilization_time,
            processor,
            start_idx,
            alt_mask,
            z_analysis,
            num_snapshots,
            metrics,
            desired_dz,
        )

        # Save the figure
        fig.savefig(
            results_dir / "wave_steepening_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        # fallback metrics for non-visualization mode to prevent KeyError downstream
        metrics = {
            "wavelength_stats": {
                "after_stabilization": {"mean": np.nan, "std": np.nan, "reliability": np.nan}
            },
            "harmonic_ratios": {
                "R21_mean": np.nan, "R21_std": np.nan,
                "R31_mean": np.nan, "R31_std": np.nan
            }
        }

    print("\n  After stabilization:")
    if "wavelength_stats" in metrics and not np.isnan(metrics["wavelength_stats"]["after_stabilization"]["mean"]):
        print(
            f"    λ = {metrics['wavelength_stats']['after_stabilization']['mean']:.1f} ± "
            f"{metrics['wavelength_stats']['after_stabilization']['std']:.1f} km"
        )
        print(
            f"    Reliability: {metrics['wavelength_stats']['after_stabilization']['reliability']:.2f}"
        )
    else:
        print("    No valid measurements")

    print("\nHarmonic Ratios (after stabilization):")
    print(
        f"  R21: {metrics['harmonic_ratios']['R21_mean']:.3f} ± {metrics['harmonic_ratios']['R21_std']:.3f}"
    )
    print(
        f"  R31: {metrics['harmonic_ratios']['R31_mean']:.3f} ± {metrics['harmonic_ratios']['R31_std']:.3f}"
    )
    print("\n" + "=" * 50 + "\n")

    return {
        "time": time,
        "harmonic_ratios": harmonic_ratios,
        "altitude_range": altitude_range,
        "diagnostics": diagnostics,
        "stabilization_time": stabilization_time,
        "peak_tracking": {
            "gaps": tracking_gap_counter,
            "max_gap": params["max_tracking_gap"],
        },
        "wavelength_evolution": wavelength_stats,
        "wavelength_analysis": wavelength_analysis,
        "metrics": metrics if visualize else None,
    }


def _handle_invalid_measurement(wavelength_stats: Dict, t_idx: int) -> None:
    """Helper function to handle invalid measurements in wavelength statistics."""
    wavelength_stats["mean"][t_idx] = np.nan
    wavelength_stats["std"][t_idx] = np.nan
    wavelength_stats["reliability"][t_idx] = 0.0
    wavelength_stats["phase_coherence"][t_idx] = 0.0
    wavelength_stats["confidence"][t_idx] = 0.0
    wavelength_stats["is_valid"][t_idx] = False


def _predict_wavelength(omega=9.5e-3):
    """Calculate predicted wavelength based on dispersion relation."""
    # Constants from math.ipynb
    gamma = 5/3
    R = 8.314  # J/mol/K
    T = 270  # K
    g = 3.51  # m/s^2
    M = 16  # g/mol
    R_prime = R / M * 1e3  # J/kg/K

    # Calculate sound speed and acoustic frequency
    c = np.sqrt(gamma * R_prime * T)
    omega_a = (gamma / 2) * (g / c)

    # Calculate vertical wavenumber and wavelength
    kz_1D = np.sqrt((omega / c) ** 2 * (1 - (omega_a / omega) ** 2))
    wavelength = 2 * np.pi / kz_1D / 1e3  # Convert to km

    return wavelength, c, omega_a

def _generate_visualization(
    time,
    harmonic_ratios,
    wavelength_stats,
    diagnostics,
    stabilization_time,
    processor,
    start_idx,
    alt_mask,
    z_analysis,
    num_snapshots,
    metrics,
    desired_dz,
):
    # Set refined theme and color palette
    sns.set_style("whitegrid", {"axes.grid": False})
    sns.set_context("notebook", font_scale=1.1)
    colors = [
        "#4477AA",
        "#EE6677",
        "#228833",
    ]  # Clean and colorblind-friendly palette
    plt.rcParams.update(
        {
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelpad": 10,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "scatter.marker": "o",
            "lines.markersize": 6,
        }
    )

    fig = plt.figure(figsize=(20, 16))
    gs_main = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.15, 0.70, 0.15],
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.08,
        hspace=0.2,
    )

    # Replace the two-column text panel with a table to span the horizontal space.
    ax_metrics = fig.add_subplot(gs_main[0, 0])
    ax_metrics.axis("off")
    # Populate table data for metrics panel with new row for stable mean wavelength including uncertainty
    table_data = [
        ["Wave Analysis Summary", "", "Grid Information", ""],
        [
            "Period (s)",
            f"{metrics['wave_period']:.1f}",
            "Original Points",
            f"{len(z_analysis)}",
        ],
        [
            "Total time (s)",
            f"{metrics['total_time']:.1f}",
            "Desired dz",
            f"{metrics['desired_dz']:.3f} km",
        ],
        [
            "Wave Periods",
            f"{metrics['wave_periods']['total']:.1f}",
            "Original Spacing",
            f"{metrics['altitude_stats']['resolution']['mean']:.3f} ± {metrics['altitude_stats']['resolution']['std']:.3f} km",
        ],
        [
            "Before Stabilization",
            f"{metrics['wave_periods']['before_stabilization']:.1f}",
            "Resampled Spacing",
            f"{metrics['desired_dz']:.3f} km",
        ],
        [
            "After Stabilization",
            f"{metrics['wave_periods']['after_stabilization']:.1f}",
            "",
            "",
        ],
        [
            "Stable Mean λ",
            f"{metrics['wavelength_stats']['after_stabilization']['mean']:.1f} ± {metrics['wavelength_stats']['after_stabilization']['std']:.1f} km",
            "",
            "",
        ],
        ["Stabilization (s)", f"{metrics['stabilization_time']:.1f}", "", ""],
        [
            "Altitude Range (km)",
            f"{metrics['altitude_stats']['min_altitude']:.1f} - {metrics['altitude_stats']['max_altitude']:.1f}",
            "",
            "",
        ],
    ]

    # Calculate predicted wavelength
    predicted_wavelength, c, omega_a = _predict_wavelength(processor.params.omega_freq)

    # Add to metrics table data after the original data list
    table_data.extend([
        ["Predicted λ (km)", f"{predicted_wavelength:.1f}", "", ""],
        ["Sound Speed (m/s)", f"{c:.1f}", "", ""],
        ["Acoustic freq (rad/s)", f"{omega_a:.2e}", "", ""],
    ])

    table = ax_metrics.table(
        cellText=table_data, cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)  # Increase row height for more vertical padding

    # Style header row and first column for enhanced appearance
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.get_text().set_ha("left")
        if row == 0:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("white")
            cell.set_facecolor("#40466e")
        if col == 0 and row != 0:
            cell.get_text().set_fontweight("bold")

    # Bottom panel: create a 2x2 subgridspec for the plots with reduced vertical spacing.
    gs_plots = gs_main[1, 0].subgridspec(2, 2, hspace=0.34, wspace=0.25)

    # 1. Top-left: Harmonic ratios evolution plot
    ax1 = fig.add_subplot(gs_plots[0, 0])
    time_since_onset = time - time[0]
    wave_period = 2 * np.pi / processor.params.omega_freq
    num_periods = int((time[-1] - time[0]) / wave_period)
    period_times = [0 + i * wave_period for i in range(num_periods + 1)]

    ax1.grid(True, alpha=0.1, linestyle="-", color="gray")
    for t in period_times:
        ax1.axvline(t, color="gray", linestyle=":", alpha=0.25)
    ax1.plot(
        time_since_onset,
        harmonic_ratios["R21"],
        color=colors[0],
        alpha=0.15,
        lw=1,
    )
    ax1.plot(
        time_since_onset,
        harmonic_ratios["R31"],
        color=colors[1],
        alpha=0.15,
        lw=1,
    )
    ax1.scatter(
        time_since_onset,
        harmonic_ratios["R21"],
        label="R21",
        color=colors[0],
        s=25,
        alpha=0.7,
        marker="o",
    )
    ax1.scatter(
        time_since_onset,
        harmonic_ratios["R31"],
        label="R31",
        color=colors[1],
        s=25,
        alpha=0.7,
        marker="s",
    )
    ax1.axvline(
        stabilization_time,
        color="gray",
        linestyle="--",
        label="Stabilization",
        lw=1.5,
        alpha=0.8,
    )
    ax1.set_xlabel("Time since wave onset (s)")
    ax1.set_ylabel("Harmonic Ratio")
    ax1.set_title("Wave Steepening Evolution", pad=15)
    leg = ax1.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.9
    )
    leg.get_frame().set_facecolor("white")

    # 2. Top-right: Wavelength evolution plot
    ax2 = fig.add_subplot(gs_plots[0, 1])
    valid_mask = ~np.isnan(wavelength_stats["mean"])
    ax2.grid(True, alpha=0.1, linestyle="-", color="gray")
    for t in period_times:
        ax2.axvline(t, color="gray", linestyle=":", alpha=0.25)
    if np.any(valid_mask):
        ax2.fill_between(
            time_since_onset[valid_mask],
            wavelength_stats["mean"][valid_mask]
            - wavelength_stats["std"][valid_mask],
            wavelength_stats["mean"][valid_mask]
            + wavelength_stats["std"][valid_mask],
            color=colors[0],
            alpha=0.1,
        )
        ax2.plot(
            time_since_onset[valid_mask],
            wavelength_stats["mean"][valid_mask],
            color=colors[0],
            alpha=0.15,
            lw=1,
        )
        scatter = ax2.scatter(
            time_since_onset[valid_mask],
            wavelength_stats["mean"][valid_mask],
            c=wavelength_stats["reliability"][valid_mask],
            cmap="RdYlBu_r",
            s=30,
            marker="o",
        )
        cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Reliability", labelpad=10)

        # Add predicted wavelength reference line
        ax2.axhline(
            predicted_wavelength,
            color='red',
            linestyle=':',
            label=f'Predicted λ = {predicted_wavelength:.1f} km',
            alpha=0.8,
            zorder=1
        )

        # Update legend to include prediction
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(
            handles=handles,
            labels=labels,
            loc='upper left',
            frameon=True,
            fancybox=True,
            framealpha=0.9
        )

    ax2.axvline(
        stabilization_time, color="gray", linestyle="--", lw=1.5, alpha=0.8
    )
    ax2.set_xlabel("Time since wave onset (s)")
    ax2.set_ylabel("Wavelength (km)")
    ax2.set_title("Wavelength Evolution", pad=15)

    # 3. Bottom-left: Spectral power evolution plot
    ax3 = fig.add_subplot(gs_plots[1, 0])
    max_fund_power = np.max(diagnostics["fundamental_power"])
    norm_factor = max_fund_power if max_fund_power > 0 else 1.0
    ax3.grid(True, alpha=0.1, linestyle="-", color="gray")
    for t in period_times:
        ax3.axvline(t, color="gray", linestyle=":", alpha=0.25)
    markers = ["o", "s", "D"]
    for (name, label), marker, color in zip(
        [
            ("fundamental_power", "Fundamental"),
            ("harmonic_2_power", "2nd Harmonic"),
            ("harmonic_3_power", "3rd Harmonic"),
        ],
        markers,
        colors,
    ):
        power = diagnostics[name] / norm_factor
        ax3.semilogy(time_since_onset, power, color=color, alpha=0.15, lw=1)
        ax3.scatter(
            time_since_onset,
            power,
            color=color,
            label=label,
            s=25,
            alpha=0.7,
            marker=marker,
        )
    ax3.axvline(
        stabilization_time, color="gray", linestyle="--", lw=1.5, alpha=0.8
    )
    ax3.set_xlabel("Time since wave onset (s)")
    ax3.set_ylabel("Normalized Power")
    ax3.set_title("Spectral Power Evolution", pad=15)
    leg3 = ax3.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.9
    )
    leg3.get_frame().set_facecolor("white")

    # 4. Bottom-right: Temperature profile evolution with both raw and resampled data
    ax4 = fig.add_subplot(gs_plots[1, 1])
    stable_indices = np.where(time >= time[0] + stabilization_time)[0]
    if len(stable_indices) > 0:
        snapshot_indices = np.linspace(
            stable_indices[0], len(time) - 1, num_snapshots, dtype=int
        )
        profile_colors = plt.cm.viridis(
            np.linspace(0, 1, len(snapshot_indices))
        )
        all_temps = []

        for idx in snapshot_indices:
            temp_profile = (
                processor.temperature_data[start_idx + idx, alt_mask, 0]
                - DEFAULT_SMOOTHING_PARAMS["temp_offset"]
            )
            temp_smoothed_raw = smooth_profile(
                temp_profile,
                cutoff_freq=DEFAULT_SMOOTHING_PARAMS["cutoff_freq"],
                order=DEFAULT_SMOOTHING_PARAMS["filter_order"],
            )
            z_uniform, temp_uniform = resample_to_uniform_grid(
                z_analysis, temp_smoothed_raw, dz=desired_dz
            )

            # ax4.plot(
            #     temp_profile,
            #     z_analysis,
            #     color=profile_colors[np.where(snapshot_indices == idx)[0][0]],
            #     linestyle="--",
            #     alpha=0.6,
            #     label=f"t = {time_since_onset[idx]:.1f}s (raw)",
            #     lw=1.5,
            # )

            ax4.plot(
                temp_uniform,
                z_uniform,
                color=profile_colors[np.where(snapshot_indices == idx)[0][0]],
                linestyle="-",
                alpha=0.8,
                label=f"t = {time_since_onset[idx]:.1f}s (resampled)",
                lw=1.5,
            )

            all_temps.extend(temp_smoothed_raw)

    ax4.set_xlabel("ΔT (K)")
    ax4.set_ylabel("Altitude (km)")
    ax4.set_title(
        "Temperature Profile Evolution\n(Raw: dashed, Resampled: solid)",
        pad=15,
    )

    # Custom legend creation: works even if the raw temperature profile line is disabled
    handles, labels = ax4.get_legend_handles_labels()
    by_time = {}
    for h, l in zip(handles, labels):
        base_time = l.split("s")[0] + "s"  # Extract time string
        if base_time not in by_time:
            by_time[base_time] = []
        by_time[base_time].append((h, l))

    legend_handles = []
    legend_labels = []
    for base_time in sorted(by_time.keys()):
        pairs = by_time[base_time]
        # Always add one handle even if only one line (resampled) is plotted.
        legend_handles.append(pairs[0][0])
        legend_labels.append(base_time)

    # Updated legend placement and styling
    leg4 = ax4.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize="x-small",
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        ncol=1,
    )
    leg4.get_frame().set_facecolor("white")
    leg4.get_frame().set_edgecolor("lightgray")

    # Add a second legend for the line style meanings, also inside the plot
    line_style_handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Raw"),
        plt.Line2D([0], [0], color="gray", linestyle="-", label="Resampled"),
    ]
    ax4.add_artist(leg4)  # Keep the first legend
    leg4_style = ax4.legend(
        handles=line_style_handles,
        loc="upper left",
        fontsize="x-small",
        frameon=True,
        fancybox=True,
        framealpha=0.8,
    )
    leg4_style.get_frame().set_facecolor("white")
    leg4_style.get_frame().set_edgecolor("lightgray")

    ax4.grid(True, alpha=0.2, linestyle=":")

    # NEW: Debug subplot for spectral power vs wavelength with strategic coloring
    ax_debug = fig.add_subplot(gs_main[2, 0])
    ax_debug.set_title("Spectral Power vs Wavelength (Debug Snapshots)")
    ax_debug.set_xlabel("Wavelength (km)")
    ax_debug.set_ylabel("Power Spectral Density")
    ax_debug.grid(True, alpha=0.2, linestyle=":")

    # Calculate new cutoff frequency based on predicted wavelength for debug viz (similar to analyze_wave_steepening)
    predicted_wavelength, _, _ = _predict_wavelength(processor.params.omega_freq)
    new_cutoff_debug = 2 * np.pi / predicted_wavelength * 1.5  # in 1/km

    # Choose snapshot indices similar to temperature profile evolution and map colors via viridis colormap
    stable_indices = np.where(time >= time[0] + stabilization_time)[0]
    if len(stable_indices) > 0:
        snapshot_indices = np.linspace(
            stable_indices[0], len(time) - 1, num_snapshots, dtype=int
        )
        debug_colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))
        for i, idx in enumerate(snapshot_indices):
            # Compute temperature profile and smooth using new_cutoff_debug
            temp_profile = (
                processor.temperature_data[start_idx + idx, alt_mask, 0]
                - DEFAULT_SMOOTHING_PARAMS["temp_offset"]
            )
            temp_smoothed = smooth_profile(
                temp_profile,
                cutoff_freq=new_cutoff_debug,
                order=DEFAULT_SMOOTHING_PARAMS["filter_order"],
            )
            z_uniform, temp_uniform = resample_to_uniform_grid(
                z_analysis, temp_smoothed, dz=desired_dz
            )
            # Compute spectral features
            k_vals, psd_vals = compute_spectral_features(temp_uniform, z_uniform)
            with np.errstate(divide="ignore"):
                wavelengths = 2 * np.pi / k_vals
            label = f"t = {time[idx]-time[0]:.1f}s" if i == 0 else None
            ax_debug.plot(
                wavelengths, psd_vals, alpha=0.7, color=debug_colors[i], label=label
            )
        if label is not None:
            ax_debug.legend(loc="upper right", framealpha=0.8)

    # Apply consistent styling for remaining axes (if any)
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    return fig


if __name__ == "__main__":
    # Example usage
    data_dir = (
        "/media/astrojose/AstroBook/ArchiveSims/Results_20250210_040813.382"
    )
    processor = DSMCDataProcessor(data_dir)
    processor.load_parameters()
    processor.load_data()

    results = analyze_wave_steepening(processor, visualize=True)

    # Print summary statistics
    print("\nWavelength Analysis:")
    for phase in ["initial_phase", "stable_phase"]:
        stats = results["wavelength_analysis"][phase]
        print(f"\n{phase.replace('_', ' ').title()}:")
        if not np.isnan(stats["mean"]):
            print(
                f"  Mean wavelength: {stats['mean']:.1f} ± {stats['std']:.1f} km"
            )
            print(f"  Reliability: {stats['reliability']:.2f}")
        else:
            print("  No valid wavelength measurements")

    # analyze_wave_steepening_simple(processor)
