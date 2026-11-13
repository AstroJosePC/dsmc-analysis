"""Per-cycle peak detection in perturbation time series and the density-growth peak table.

Copied from the root helper_funcs.py (identify_peaks, process_amplitude,
process_growth_num_amp). One behavioural switch was added: when automatic detection finds
fewer than num_periods peaks, the click-to-pick window only opens with interactive=True.
Otherwise that altitude is treated as "cancelled" (NaN peaks), which is what the root code
returns under a non-GUI backend (Agg, inline notebooks).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.signal import find_peaks


def identify_peaks(amplitude_data, num_periods, previous_peaks=None, use_dynamic=False, peak_params=None):
    """Peaks of a 1D series; when at least num_periods are found, keep the num_periods most prominent.

    With use_dynamic and previous_peaks (from the altitude below), distance becomes 80 % of
    the previous mean spacing and prominence 30 % of the previous mean |peak height|.
    Returns (sorted peak indices, find_peaks properties).
    """
    if peak_params is None:
        peak_params = {"prominence": 0.01, "distance": 12, "width": None, "height": None}
    else:
        peak_params = {
            "prominence": peak_params.get("prominence", 0.01),
            "distance": peak_params.get("distance", 12),
            "width": peak_params.get("width", None),
            "height": peak_params.get("height", None),
        }

    if previous_peaks is not None and use_dynamic and not np.any(np.isnan(previous_peaks)):
        peak_spacing = np.diff(previous_peaks)
        if peak_spacing.size > 0:
            peak_params["distance"] = int(np.mean(peak_spacing) * 0.8)
        prev_peak_heights = amplitude_data[previous_peaks]
        if prev_peak_heights.size > 0:
            peak_params["prominence"] = np.mean(np.abs(prev_peak_heights)) * 0.3

    peaks, props = find_peaks(amplitude_data, **peak_params)

    if len(peaks) >= int(num_periods):
        order = np.argsort(props["prominences"])[::-1]
        keep = order[: int(num_periods)]
        return np.sort(peaks[keep]), {k: v[keep] for k, v in props.items()}

    return peaks, props


def process_amplitude(
    amplitude_data,
    baseline_peak_amps,
    altitude=None,
    previous_peaks=None,
    num_periods=5,
    use_dynamic=False,
    prominence=0.01,
    distance=12,
    width=None,
    height=None,
    interactive=False,
):
    """Peak amplitudes of one series normalised by baseline_peak_amps.

    Returns (peak_amps, peak_indices). When fewer than num_periods peaks are found, returns
    (NaN * num_periods, []) unless interactive=True, in which case a matplotlib window lets
    you add/remove peaks by hand (needs a GUI backend such as %matplotlib qt or macosx).
    """
    num_periods = int(num_periods)
    peaks, _ = identify_peaks(
        amplitude_data,
        num_periods,
        previous_peaks=previous_peaks,
        use_dynamic=use_dynamic,
        peak_params={"prominence": prominence, "distance": distance, "width": width, "height": height},
    )

    if len(peaks) >= num_periods:
        return amplitude_data[peaks] / baseline_peak_amps, peaks

    if interactive:
        peaks = _pick_peaks_by_hand(amplitude_data, peaks, baseline_peak_amps, altitude, previous_peaks)
        if peaks is not None and len(peaks) == num_periods:
            return amplitude_data[peaks] / baseline_peak_amps, peaks

    return np.full(num_periods, np.nan), np.array([])


def _pick_peaks_by_hand(amplitude_data, peaks, baseline_peak_amps, altitude, previous_peaks):
    """Blocking matplotlib picker: left-click adds, right-click removes, 'Cancel' returns None."""
    fig, ax = plt.subplots()
    line, = ax.plot(amplitude_data)
    scatter = ax.scatter(peaks, amplitude_data[peaks], color="r")

    if np.iterable(baseline_peak_amps):
        ax.axhline(baseline_peak_amps[-1], color="k", linestyle="--")
    if altitude is not None:
        ax.text(0.01, 0.99, f"Altitude: {altitude:.2f} km", transform=ax.transAxes, va="top", fontsize=12)

    has_legend_elements = False
    if previous_peaks is not None and not np.any(np.isnan(previous_peaks)):
        ax.axvspan(previous_peaks[0], previous_peaks[-1], color="y", alpha=0.1, label="Previous peak range")
        ax.plot(previous_peaks, amplitude_data[previous_peaks], "y*", label="Previous altitude peaks")
        has_legend_elements = True

    cancel_box = ax.text(
        0.99, 0.01, "Cancel peak detection", transform=ax.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.3), picker=True, fontsize=8,
    )
    ax.text(
        0.99, 0.99, "Left-click to add, right-click to remove\nClick 'Cancel' for noisy data",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=8,
    )

    cancelled = False

    def on_pick(event):
        nonlocal cancelled
        if event.artist == cancel_box:
            cancelled = True
            plt.close(fig)

    def on_click(event):
        nonlocal peaks
        if event.button is MouseButton.LEFT:
            new_peak = np.argmin(np.abs(line.get_xdata() - event.xdata))
            peaks = np.sort(np.append(peaks, new_peak))
        elif event.button is MouseButton.RIGHT:
            dist = np.hypot(event.xdata - line.get_xdata()[peaks], event.ydata - line.get_ydata()[peaks])
            closest_peak = np.argmin(dist)
            if dist[closest_peak] < 5:
                peaks = np.delete(peaks, closest_peak)
        scatter.set_offsets(np.c_[line.get_xdata()[peaks], line.get_ydata()[peaks]])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("pick_event", on_pick)
    if has_legend_elements:
        ax.legend()

    try:
        plt.show()
    finally:
        plt.close(fig)

    return None if cancelled else peaks


def process_growth_num_amp(
    num_density_amplitude_azi,
    is_oscillating_extended,
    mid_rad_km,
    cell_150km,
    top_radius_idx,
    num_periods,
    use_dynamic=False,
    interactive=False,
    return_indices=False,
    **peak_kwargs,
):
    """Per-cycle density peak amplitudes from cell_150km up to top_radius_idx, normalised to the base cell.

    num_density_amplitude_azi: relative density perturbation of one column, (time, altitude).
    is_oscillating_extended:   boolean time mask of the forcing window (+ padding).
    Returns (growth, baseline_peak_amps): growth is a masked (n_altitudes, num_periods) array
    whose first row is 1. When detection fails at an altitude, that altitude and all above
    it are masked.
    peak_kwargs: prominence, distance, width, height (see process_amplitude).
    return_indices: also return the peak positions per altitude (indices into the masked time
    window; an empty array where detection failed), for diagnostic plots.
    """
    num_periods = int(num_periods)

    num_den_amp_150km = num_density_amplitude_azi[is_oscillating_extended, cell_150km]
    baseline_peaks_amps, baseline_peak_indices = process_amplitude(
        num_den_amp_150km, 1, altitude=mid_rad_km[cell_150km], num_periods=num_periods,
        interactive=interactive, **peak_kwargs,
    )

    if np.all(np.isnan(baseline_peaks_amps)) or len(baseline_peak_indices) == 0:
        print("Peak detection cancelled at baseline altitude (150 km). Cannot proceed with growth analysis.")
        result = (
            np.ma.masked_all((top_radius_idx - cell_150km + 1, num_periods)),
            np.ma.masked_array(np.full(num_periods, np.nan), mask=True),
        )
        return result + ([np.array([])] * (top_radius_idx - cell_150km + 1),) if return_indices else result

    growth_array_size = top_radius_idx - cell_150km + 1
    growth_num_amp_peaks = np.ma.zeros((growth_array_size, num_periods))
    growth_num_amp_peaks[0] = 1.0
    all_indices = [np.asarray(baseline_peak_indices)] + [np.array([])] * (growth_array_size - 1)

    previous_peaks = baseline_peak_indices
    for i in range(1, growth_array_size):
        current_altitude = mid_rad_km[cell_150km + i]
        num_density_amp_slice = num_density_amplitude_azi[is_oscillating_extended, cell_150km + i]

        peaks, peak_indices = process_amplitude(
            num_density_amp_slice,
            baseline_peaks_amps,
            altitude=current_altitude,
            previous_peaks=previous_peaks if use_dynamic else None,
            use_dynamic=use_dynamic,
            num_periods=num_periods,
            interactive=interactive,
            **peak_kwargs,
        )

        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            growth_num_amp_peaks[i:] = np.ma.masked
            print(f"Peak detection cancelled at altitude {current_altitude:.1f} km. Masking this and higher altitudes.")
            break

        growth_num_amp_peaks[i] = peaks
        all_indices[i] = np.asarray(peak_indices)
        if len(peak_indices) > 0:
            previous_peaks = peak_indices

    if return_indices:
        return growth_num_amp_peaks, baseline_peaks_amps, all_indices
    return growth_num_amp_peaks, baseline_peaks_amps
