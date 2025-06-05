import pprint
import re
from glob import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, ifft
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy as sp
import os
import pickle
from typing import Tuple


mars_radius = 3389.5e3  # m

num_part_inject_idx = 7
inject_rate_idx = 11


# extract timestep from simulation parameters file using regex
def extract_from_info(res_path):
    with open(f"{res_path}/Info/Simulation_Parameters.as    cii", "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove spaces before and after the string
            line = line.strip()
            if re.match(r"^Time Step", line):
                timestep = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if re.match(r"^Number of Total Simulations", line):
                nsteps = int(re.findall(r"\d+", line)[0])
            if re.match(r"Particle Weight", line):
                # extract number in exponential notation
                part_weight = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])
            if re.match(r"^Number of Radial Cells", line):
                num_rad_cells = int(re.findall(r"\d+", line)[0])
    return timestep, nsteps, part_weight, num_rad_cells


def extract_from_info_updated(res_path):
    info_dict = {}
    # extract timestep from simulation parameters file using regex
    with open(f"{res_path}/Info/Simulation_Parameters.ascii", "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove spaces before and after the string
            line = line.strip()
            if re.match(r"^Time Step", line):
                info_dict["timestep"] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if re.match(r"^Number of Total Simulations", line):
                info_dict["nsteps"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Particle Weight", line):
                # extract number in exponential notation
                info_dict["part_weight"] = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])
            if re.match(r"Number of Radial Cells", line):
                info_dict["num_radial_cells"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Number of Azimuth Cells", line):
                info_dict["num_azi_cells"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Start Step of Wave Injection", line):
                info_dict["step0_inject"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"End Step of Wave Injection", line):
                info_dict["stepf_inject"] = int(re.findall(r"\d+", line)[0])
            if re.match(r"Wave Injection Frequency", line):
                # extract number in exponential notation
                info_dict["OMEGA"] = float(re.findall(r"[-+]?\d*\.\d+[eE][-+]?\d+", line)[0])

    num_periods = (
        (info_dict["stepf_inject"] - info_dict["step0_inject"] - 1)
        * info_dict["OMEGA"]
        * info_dict["timestep"]
        / (2 * np.pi)
    )
    info_dict["num_periods"] = num_periods

    return info_dict


macro_headers = header = [
    "Time",
    "Avg Time Interval",
    "Radius",
    "Zenith",
    "Azimuth",
    "Cell Number",
    "Number of Particles",
    "Number Density",
    "Radial Velocity",
    "Zenithal Velocity",
    "Azimuthal Velocity",
    "Thermal Velocity",
    "Radial Temperature",
    "Zenithal Temperature",
    "Azimuthal Temperature",
    "Kinetic Temperature",
    "Translational Temperature",
    "Internal Temperature",
    "Overpopulation Temperature",
    "Theoretical Collision Frequency",
    "Mean Free Path",
    "Scale Height",
    "Knudsen Number",
    "Thermal Conductivity Up",
    "Thermal Conductivity Down",
]

macro_properties = macro_headers[6:]

# create dictionary map of macro header names to their index
macro_header_map = {}
for i in range(len(macro_headers)):
    macro_header_map[macro_headers[i]] = i

# create dictionary map of macro property names to their index
macro_property_map = {}
for i in range(len(macro_properties)):
    macro_property_map[macro_properties[i]] = i


def open_macro(path):
    macro_dataframe = pd.read_csv(path, sep=r"\s+", header=None).drop(columns=[19])
    macro_dataframe.columns = macro_headers
    return macro_dataframe


def open_surface(path):
    surf_dataframe = pd.read_csv(path, sep=r"\s+", header=None)
    surf_dataframe.columns = [
        "Time",
        "Average Time Interval",
        "Radius",
        "Zenith",
        "Azimuth",
        "Cell Number",
        "Area",
        "Number of Particles Injected",
        "Number of Particles Returned",
        "Number of Particles Reemitted",
        "Number of Particles on Surface",
        "Injection Rate",
        "Return Rate",
        "Reemission Rate",
    ]
    return surf_dataframe


def open_surface(path) -> pd.DataFrame:
    surf_dataframe = pd.read_csv(path, sep=r"\s+", header=None)
    surf_dataframe.columns = [
        "Time",
        "Avg TimeInterval",
        "Rad",
        "Zen",
        "Azi",
        "Cell Num",
        "Lower Area",
        "Num Part Inject",
        "Num Part Return",
        "Num Part Reemit",
        "Num Part Surf",
        "Inject Rate",
        "Return Rate",
        "Reemit Rate",
        "Return Time",
        "Inject Energy",
        "Return Energy",
    ]
    return surf_dataframe


def extract_macro_data(res_path, n_properties, num_rad_cells, window_size):
    macro_files = glob(f"{res_path}/Macro/Output*_Proc*_Macroscopic.ascii")
    macro_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])

    # extract output number from the last file
    number_of_outputs = int(re.split(r"(\d+)", macro_files[-1])[1])

    macro_data = np.zeros((n_properties, number_of_outputs, num_rad_cells))
    timestamps = np.zeros(number_of_outputs)

    for k, i in enumerate(range(0, len(macro_files), window_size)):
        macro_dfs = [open_macro(macro_files[i]) for i in range(i, i + window_size)]
        df = pd.concat(macro_dfs, axis=0, ignore_index=True).sort_values(by="Cell Number")

        df_reshaped = df.values[:, 6:].T
        macro_data[:, k, :] = df_reshaped
        timestamps[k] = df["Time"][0]

    return macro_data, timestamps, number_of_outputs


macro_props = ["Kinetic Temperature", "Number Density", "Radial Velocity", "Azimuthal Velocity"]


def extract_macro_data(res_path, num_radial_cells, num_azi_cells):
    """
    Extract macroscopic data from the macroscopic output files.
    :param res_path: path to the simulation results
    :param number_of_outputs: number of outputs
    :param num_processes: number of processes used in the simulation
    :param num_radial_cells: number of radial cells
    :param num_azi_cells: number of zenithal cells
    :return: macroscopic data, timestamps, number of particles, number of particles per cell
    """
    macro_files = glob(f"{res_path}/Macro/Output*_Proc*_Macroscopic.ascii")
    macro_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])

    number_of_outputs = int(re.split(r"(\d+)", macro_files[-1])[-4])
    num_processes = int(re.split(r"(\d+)", macro_files[-1])[-2]) + 1

    time_stamps = np.zeros(number_of_outputs)
    num_particles = np.zeros((number_of_outputs, num_radial_cells))
    num_particle_data = np.zeros((number_of_outputs, num_radial_cells, num_azi_cells))

    macro_grids_data = {prop: np.zeros((number_of_outputs, num_radial_cells, num_azi_cells)) for prop in macro_props}

    for k, i in enumerate(range(0, len(macro_files), num_processes)):
        macro_dfs = [open_macro(macro_files[i]) for i in range(i, i + num_processes)]
        df = pd.concat(macro_dfs, axis=0, ignore_index=True)
        df_sort_rad = df.sort_values(by="Radius")
        df_sort_cell = df.sort_values(by="Cell Number")

        grouped_df = df_sort_rad.groupby("Radius")
        time_stamps[k] = macro_dfs[0]["Time"].values[0]

        num_particles[k, :] = grouped_df.sum()["Number of Particles"].values
        num_particle_data[k, :, :] = df_sort_cell["Number of Particles"].values.T.reshape(
            (num_radial_cells, num_azi_cells)
        )

        data = df_sort_cell[macro_props].values.T.reshape(len(macro_props), num_radial_cells, num_azi_cells)

        for j, prop in enumerate(macro_props):
            macro_grids_data[prop][k, :, :] = data[j, :, :]

    return macro_grids_data, time_stamps, num_particles, num_particle_data


def extract_surf_data():
    surf_rates = np.zeros((time_stamps.size, 2))
    surf_rates_azi = np.zeros((time_stamps.size, 2))
    surf_rates_surf = np.zeros((time_stamps.size, info_dict["num_azi_cells"]))

    for k, i in enumerate(range(0, len(surf_files), 12)):
        surf_dfs = [open_surface(surf_files[i]) for i in range(i, i + 12)]
        df = pd.concat(surf_dfs, axis=0, ignore_index=True)
        surf_rates[k, 0] = df["Num Part Inject"].sum()
        surf_rates[k, 1] = df["Inject Rate"].sum()

        # sort df by Azi and sum only first 2 rows
        df = df.sort_values(by=["Azi"])
        surf_rates_azi[k, 0] = df["Num Part Inject"].iloc[0:2].sum()
        surf_rates_azi[k, 1] = df["Inject Rate"].iloc[0:2].sum()

        surf_rates_surf[k, :] = df["Num Part Inject"].values


def process_temperature_deviation_peaks(
    temperature_deviation_azi, is_oscillating_extended, mid_rad_km, cell_150km, top_radius_idx, num_periods, **kwargs
):
    """Process temperature deviation peaks, handling cancelled peak detection with masked arrays"""
    temp_deviation_peaks = np.ma.zeros((top_radius_idx - cell_150km + 1, int(num_periods)))
    temp_deviation_last_cycle = np.ma.zeros(top_radius_idx - cell_150km + 1)

    for i in range(0, top_radius_idx - cell_150km + 1):
        temperature_deviation_slice = temperature_deviation_azi[is_oscillating_extended, cell_150km + i]
        peaks, peak_indices = process_amplitude(
            -temperature_deviation_slice,
            1,
            num_periods=num_periods,
            altitude=mid_rad_km[cell_150km + i],
            return_peaks=True,
            **kwargs,
        )

        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            # If peaks were cancelled or no peaks found, mask this altitude and all higher altitudes
            temp_deviation_peaks[i:] = np.ma.masked
            temp_deviation_last_cycle[i:] = np.ma.masked
            print(f"Temperature peak detection cancelled at altitude {mid_rad_km[cell_150km + i]:.1f} km. "
                  f"Masking this and higher altitudes.")
            # Break out of the loop immediately
            break

        temp_deviation_peaks[i] = -peaks

        # Only compute last cycle if we have enough peaks
        if len(peak_indices) >= 2:
            temperature_deviations_last_cycle = temperature_deviation_slice[peak_indices[-2]:peak_indices[-1]]
            temp_deviation_last_cycle[i] = temperature_deviations_last_cycle.mean()
        else:
            temp_deviation_last_cycle[i] = np.ma.masked

    return temp_deviation_peaks, temp_deviation_last_cycle


def process_growth_num_amp(
    num_density_amplitude_azi, is_oscillating_extended, mid_rad_km, cell_150km, top_radius_idx, num_periods, use_dynamic=False, **kwargs
):
    """Process growth in density amplitude, handling cancelled peak detection with masked arrays

    Args:
        use_dynamic: Whether to use dynamic parameter adjustment based on previous peaks
    """
    # Convert num_periods to integer if it's a numpy array or float
    num_periods = int(num_periods)

    num_den_amp_150km = num_density_amplitude_azi[is_oscillating_extended, cell_150km]
    baseline_peaks_amps, baseline_peak_indices = process_amplitude(
        num_den_amp_150km, 1, altitude=mid_rad_km[cell_150km],
        num_periods=num_periods, return_peaks=True, **kwargs
    )

    # Check if baseline peaks were cancelled or none found
    if np.all(np.isnan(baseline_peaks_amps)) or len(baseline_peak_indices) == 0:
        print("Peak detection cancelled at baseline altitude (150 km). Cannot proceed with growth analysis.")
        return (
            np.ma.masked_all((top_radius_idx - cell_150km + 1, num_periods)),
            np.ma.masked_array(np.full(num_periods, np.nan), mask=True)
        )

    growth_array_size = top_radius_idx - cell_150km + 1
    growth_num_amp_peaks = np.ma.zeros((growth_array_size, int(num_periods)))
    growth_num_amp_peaks[0] = 1.0  # Baseline altitude is always 1.0

    # Store previous peaks for dynamic adjustment
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
            return_peaks=True,
            **kwargs
        )

        if np.all(np.isnan(peaks)) or len(peak_indices) == 0:
            # If peaks were cancelled or none found, mask this altitude and all higher altitudes
            growth_num_amp_peaks[i:] = np.ma.masked
            print(f"Peak detection cancelled at altitude {current_altitude:.1f} km. "
                  f"Masking this and higher altitudes.")
            # Break out of the loop immediately
            break

        growth_num_amp_peaks[i] = peaks

        # Update previous peaks only if they were successfully found
        if len(peak_indices) > 0:
            previous_peaks = peak_indices

    return growth_num_amp_peaks, baseline_peaks_amps


def read_simulation_parameters(file_path):
    parameters = {}
    current_section = None
    num_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")  # compile regex pattern

    with open(file_path, "r") as file:
        for line in file:  # iterate over file object directly
            line = line.strip()
            if line.endswith("Information") or line == "Simulation Parameters":
                current_section = line
                if current_section not in parameters:
                    parameters[current_section] = []
                parameters[current_section].append({})
            elif line and ":" in line and current_section:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Try to extract a numerical value
                if key not in ["Name", "Species 1", "Species 2"]:
                    match = num_pattern.search(value)  # use compiled regex pattern
                    if match:
                        num_value = float(match.group())
                        if (
                            num_value.is_integer() and num_value < 1e15
                        ):  # check if num_value can be converted to int without exception
                            num_value = int(num_value)
                        value = num_value
                if key == "Time Step":
                    if "Simulation Parameters" not in parameters:
                        parameters["Simulation Parameters"] = [{}]
                    parameters["Simulation Parameters"][-1][key] = value
                elif key in ["Name", "Collision Type"] and parameters[current_section][-1]:
                    parameters[current_section].append({key: value})
                else:
                    parameters[current_section][-1][key] = value
    return parameters


def process_amplitude(amplitude_data, baseline_peak_amps, altitude=None, previous_peaks=None, **kwargs):
    """Process amplitudes in time series data with optional dynamic parameter adjustment.

    Args:
        amplitude_data: Array of amplitude values
        baseline_peak_amps: Reference amplitudes for normalization
        altitude: Current altitude for display
        previous_peaks: Information about peaks from previous altitude for dynamic adjustment
        **kwargs: Additional parameters for peak detection
    """
    # Get required parameters with defaults
    num_periods = int(kwargs.pop("num_periods", 5))
    return_peaks = kwargs.pop("return_peaks", False)
    use_dynamic = kwargs.pop("use_dynamic", False)

    # Extract peak finding parameters
    peak_params = {
        'prominence': kwargs.pop("prominence", 0.01),
        'distance': kwargs.pop("distance", 12),
        'width': kwargs.pop("width", None),
        'height': kwargs.pop("height", None)
    }

    # If we have previous peak information and dynamic adjustment is enabled, adjust parameters
    if previous_peaks is not None and use_dynamic and not np.any(np.isnan(previous_peaks)):
        peak_spacing = np.diff(previous_peaks)
        peak_params['distance'] = int(np.mean(peak_spacing) * 0.8)  # 80% of average spacing

        # Get the amplitudes at previous peaks
        prev_peak_heights = amplitude_data[previous_peaks]
        # Set prominence as 30% of the mean peak height
        peak_params['prominence'] = np.mean(np.abs(prev_peak_heights)) * 0.3

        print(f"\nDynamically adjusted parameters at altitude {altitude:.1f} km:")
        print(f"Distance: {peak_params['distance']}")
        print(f"Prominence: {peak_params['prominence']:.2e}")

    # Apply filtering as before
    fs = kwargs.pop("sampling_rate", 1000)
    cutoff = kwargs.pop("cutoff_frequency", 50)
    order = kwargs.pop("filter_order", 5)
    b, a = butter(order, cutoff / (0.5 * fs), btype="high", analog=False)

    # Apply filtering in frequency domain
    freq_domain_signal = fft(amplitude_data)
    filtered_freq_signal = filtfilt(b, a, freq_domain_signal)
    filtered_time_signal = np.real(ifft(filtered_freq_signal))

    # Find peaks with current parameters
    peaks, props = find_peaks(amplitude_data, **peak_params)

    if len(peaks) >= num_periods:
        # Sort peaks by prominence
        peak_order = np.argsort(props["prominences"])[::-1]
        peaks = peaks[peak_order[:num_periods]]
        peaks.sort()

        # Validate peak spacing
        # peak_spacing = np.diff(peaks)
        # expected_spacing = len(amplitude_data) / (num_periods + 1)
        # spacing_ok = np.all(
        #     (peak_spacing > 0.5 * expected_spacing) &
        #     (peak_spacing < 1.5 * expected_spacing)
        # )

        # if spacing_ok:
        peak_amps = amplitude_data[peaks]
        peak_amps = peak_amps / baseline_peak_amps
        if return_peaks:
            return peak_amps, peaks
        return peak_amps

    # Make sure any existing plots are cleared
    plt.close('all')

    # Interactive plotting for manual peak selection
    fig, ax = plt.subplots()
    line, = ax.plot(amplitude_data)
    line2, = ax.plot(filtered_time_signal)
    scatter = ax.scatter(peaks, amplitude_data[peaks], color="r")

    if np.iterable(baseline_peak_amps):
        hline = ax.axhline(baseline_peak_amps[-1], color="k", linestyle="--")

    if altitude is not None:
        ax.text(0.01, 0.99, f"Altitude: {altitude:.2f} km", transform=ax.transAxes, va="top", fontsize=12)

    # Show previous peaks if available
    has_legend_elements = False  # Track if we have any labeled elements
    if previous_peaks is not None and not np.any(np.isnan(previous_peaks)):
        ax.axvspan(previous_peaks[0], previous_peaks[-1], color='y', alpha=0.1,
                  label='Previous peak range')
        ax.plot(previous_peaks, amplitude_data[previous_peaks], 'y*',
                label='Previous altitude peaks')
        has_legend_elements = True

    cancel_box = ax.text(
        0.99, 0.01, "Cancel peak detection",
        transform=ax.transAxes, va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
        picker=True
    )

    text_box = ax.text(
        0.99, 0.99,
        "Left-click to add, right-click to remove\nClick 'Cancel' for noisy data",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    cancelled = False

    def on_pick(event):
        nonlocal cancelled
        if event.artist == cancel_box:
            cancelled = True
            # Close the current figure and clear all plots
            plt.close('all')

    def on_click(event):
        nonlocal peaks
        if event.button is MouseButton.LEFT:
            x, y = event.xdata, event.ydata
            new_peak = np.argmin(np.abs(line.get_xdata() - x))
            peaks = np.append(peaks, new_peak)
            peaks = np.sort(peaks)
            scatter.set_offsets(np.c_[line.get_xdata()[peaks], line.get_ydata()[peaks]])
        elif event.button is MouseButton.RIGHT:
            x, y = event.xdata, event.ydata
            dist = np.hypot(x - line.get_xdata()[peaks], y - line.get_ydata()[peaks])
            closest_peak = np.argmin(dist)
            if dist[closest_peak] < 5:  # Tolerance for click proximity
                peaks = np.delete(peaks, closest_peak)
                scatter.set_offsets(np.c_[line.get_xdata()[peaks], line.get_ydata()[peaks]])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect('pick_event', on_pick)
    if has_legend_elements:
        plt.legend()

    try:
        plt.show()
    finally:
        # Ensure all plots are closed even if an exception occurs
        plt.close('all')

    if cancelled:
        # Return immediately when cancelled
        if return_peaks:
            return np.full(num_periods, np.nan), np.array([])
        return np.full(num_periods, np.nan)

    # Process peaks after manual selection
    if len(peaks) == num_periods:
        peak_amps = amplitude_data[peaks]
        peak_amps = peak_amps / baseline_peak_amps
        if return_peaks:
            return peak_amps, peaks
        return peak_amps
    else:
        if return_peaks:
            return np.full(num_periods, np.nan), np.array([])
        return np.full(num_periods, np.nan)


def plot_density_amplitude(
    num_density_amplitude_azi,
    is_oscillating_extended,
    rad_cells_df_r1,
    time_stamps,
    altitude,
    color,
    label,
    alpha,
    **kwargs,
) -> None:
    prominence = kwargs.pop("prominence", 0.01)
    filter_window = kwargs.pop("filter_window", 9)
    width = kwargs.pop("width", 3)

    cell = np.argmin(np.abs(rad_cells_df_r1 - (altitude + mars_radius)))
    convolved = savgol_filter(num_density_amplitude_azi[is_oscillating_extended, cell], filter_window, 3)
    peaks, _ = find_peaks(convolved, prominence=prominence, width=width)

    plt.scatter(time_stamps[is_oscillating_extended][peaks], convolved[peaks], color="k", alpha=0.5)
    plt.plot(time_stamps[is_oscillating_extended], convolved, color, alpha=alpha)

    # Check if the label already exists in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if label not in labels:
        plt.plot([], [], color=color, label=label)


def plot_density_amp_examples(num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps):
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 150e3, "g--", "150km", 0.5
    )
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 230e3, "b--", "230km", 0.99
    )
    plot_density_amplitude(
        num_density_amplitude_azi, is_oscillating_extended, rad_cells_df_r1, time_stamps, 310e3, "r--", "310km", 0.5
    )

    plt.legend(fontsize=14)
    plt.xlabel("Time [secs]")
    plt.ylabel("$n$ Amplitude")
    plt.tight_layout()
    plt.show()


def plot_heating_rate(valid_temps, valid_alts, full_temps, full_alts, slope, slope_H, stats_data, title=None, save_path=None):
    """
    Plot heating rate data with best fit line, confidence bands, and slope uncertainty.

    Args:
        valid_temps: Temperature values used for fitting
        valid_alts: Altitude values used for fitting
        full_temps: All temperature values including masked/invalid points
        full_alts: All altitude values including masked/invalid points
        slope: Computed slope in K/km
        slope_H: Computed slope in K/H
        stats_data: Dictionary containing statistical data:
            - r_value: R-value from linear regression
            - p_value: p-value from linear regression
            - std_err: Standard error of the slope
            - intercept: Y-intercept of the fit line
        title: Optional title for the plot
    """
    from scipy import stats
    import os
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    H = slope_H / slope
    
    # Generate points for confidence bands
    x_pred = np.linspace(valid_alts.min(), valid_alts.max(), 100)
    y_pred = slope * x_pred + stats_data['intercept']
    
    # Number of observations and statistics
    n = len(valid_alts)
    x_mean = np.mean(valid_alts)
    x_dev_sq = np.sum((valid_alts - x_mean)**2)
    
    # Compute residual standard error
    y_pred_data = slope * valid_alts + stats_data['intercept']
    residuals = valid_temps - y_pred_data
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))

    # Compute standard error of regression line
    x_new_dev_sq = (x_pred - x_mean)**2
    SE = residual_std * np.sqrt(1/n + x_new_dev_sq/x_dev_sq)

    # Compute confidence intervals for slope
    alpha = 0.05  # 95% confidence level
    df = n - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    slope_ci = (slope - t_crit * stats_data['std_err'],
                slope + t_crit * stats_data['std_err'])
    slope_H_ci = (slope_ci[0] * H, slope_ci[1] * H)

    # Compute pointwise confidence bands (95%)
    ci_pointwise = t_crit * SE

    # Compute simultaneous confidence bands (Working-Hotelling)
    f_value = stats.f.ppf(1 - alpha, 2, n - 2)
    ci_simultaneous = np.sqrt(2 * f_value) * SE

    # Plot all points, showing masked/invalid points differently
    # ax.plot(full_temps, full_alts, 'o', alpha=0.5)
    ax.scatter(full_temps, full_alts, color='C0', alpha=0.8)

    # Plot valid points and best fit line with both confidence bands
    # ax.plot(valid_temps, valid_alts, 'go', label='Valid Points', alpha=0.7)
    ax.plot(y_pred, x_pred, 'r-', label='Best Fit')

    # Plot confidence bands
    # ax.fill_betweenx(x_pred, y_pred - ci_pointwise, y_pred + ci_pointwise,
    #                     color='red', alpha=0.1, label='95% Pointwise CI')
    ax.fill_betweenx(x_pred, y_pred - ci_simultaneous, y_pred + ci_simultaneous,
                        color='blue', alpha=0.1, label='95% Simultaneous CI')
    
    # Use math text for uncertainty in the slopes
    stats_text = (
        f"$s = {slope:.2f} \\pm {t_crit * stats_data['std_err']:.2f}\\,\\mathrm{{K/km}}$" "\n"
        f"$\\;\\;\\;\\;\\;\\; [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]$" "\n\n"
        f"$\\bf{{s_H = {slope_H:.2f} \\pm {t_crit * stats_data['std_err'] * H:.2f}\\,\\mathrm{{K/H}}}}$" "\n"
        f"$\\bf{{[ {slope_H_ci[0]:.2f}, {slope_H_ci[1]:.2f} ]}}$" "\n\n"
        f"$R^2 = {stats_data['r_value']**2:.3f}$" "\n"
        f"$p = {stats_data['p_value']:.3e}$" "\n"
        f"$\\sigma = {stats_data['std_err']:.3f}$"
    )

    ax.text(0.98, 0.02, stats_text,
            ha='right', va='bottom', transform=ax.transAxes,
            bbox=dict(edgecolor='black', facecolor='white', alpha=0.8), fontsize=12)

    ax.set_xlabel(r'$T(r) - T_0$ (K)')
    ax.set_ylabel('Altitude (km)')
    # ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    # ax.margins(y=0)

    if title:
        ax.set_title(title)

    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

    return fig, ax

def compute_heating_rate_slope(
    temp_deviation_last_cycle, mid_rad_km, cell_150km, top_radius_idx, H, visualize=True, save_path=None
) -> Tuple[float, float]:
    """
    Compute heating rate slope with both pointwise and simultaneous confidence bands.

    Returns:
        Tuple of (slope in K/km, slope in K/H)
    """
    from scipy import stats

    # Get valid (unmasked) data points
    if isinstance(temp_deviation_last_cycle, np.ma.MaskedArray):
        valid_mask = ~temp_deviation_last_cycle.mask
    else:
        valid_mask = ~np.isnan(temp_deviation_last_cycle)

    if not np.any(valid_mask):
        print("No valid temperature deviation data points found for slope calculation")
        return np.nan, np.nan

    altitudes = mid_rad_km[cell_150km:top_radius_idx + 1]
    valid_temps = temp_deviation_last_cycle[valid_mask].ravel()
    valid_alts = altitudes[valid_mask].ravel()

    # Ensure we have enough points for fitting
    if len(valid_alts) < 2:
        print("Not enough valid points for slope calculation")
        return np.nan, np.nan

    # Print debug info
    print(f"Computing slope with {len(valid_alts)} valid points")
    print(f"Altitude range: {valid_alts.min():.1f} - {valid_alts.max():.1f} km")

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_alts, valid_temps)
    slope_H = slope * H

    if visualize:
        stats_data = {
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'intercept': intercept
        }
        plot_heating_rate(
            valid_temps, valid_alts,
            temp_deviation_last_cycle, altitudes,
            slope, slope_H, stats_data,
            # title='Temperature Deviation Profile',
            save_path=save_path
        )

    return slope, slope_H


def smoothed_last_peak_growth_num(growth_num_amp_peaks, mid_rad_km, cell_150km, top_radius_idx, visualize=True) -> None:
    growth_num_amp_last_peak = growth_num_amp_peaks[:, -1]

    # growth_num_amp_last_peak_smooth = savgol_filter(growth_num_amp_last_peak, 5, 3)
    # try a butter filter
    b, a = sp.signal.butter(1, 0.25)
    growth_num_amp_last_peak_smooth = sp.signal.filtfilt(b, a, growth_num_amp_last_peak)

    if visualize:
        plt.plot(
            growth_num_amp_last_peak_smooth,
            mid_rad_km[cell_150km : top_radius_idx + 1],
            label="mode " + str(i),
            color="#a367c1",
        )

        plt.plot(growth_num_amp_last_peak, mid_rad_km[cell_150km : top_radius_idx + 1], color="k", alpha=0.25)

        plt.show()

    return growth_num_amp_last_peak_smooth


def plot_overview_of_simulation(time_stamps, num_particle_data, mid_rad_km, surf_rates, surface_flow_rate, azi_center):
    total_particles = num_particle_data.sum(axis=(1, 2))
    average_particles = num_particle_data.mean(axis=(0, 2))
    altitude_at_10_particles = mid_rad_km[np.argmin(np.abs(average_particles - 10.0))]

    # Create the GridSpec layout
    fig = plt.figure(figsize=(12, 8))  # Adjust figure size as needed
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # First subplot: Total Particles over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_stamps[:], total_particles, label="CO2", lw=5, marker="o", markersize=5)
    # ax1.set_ylim(top=915000)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Total # of Particles")
    ax1.set_title("Total Number of Particles in Domain")

    # Second subplot: Average Particles per Altitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(average_particles, mid_rad_km)
    ax2.set_xlabel("Avg # of Particles")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Average # of Particles per Altitude")
    ax2.axvline(x=10, color="red", linestyle="--")
    ax2.axhline(y=altitude_at_10_particles, color="green", linestyle="--", label="Altitude at 10 particles")

    ax2.annotate(
        f"Altitude at 10 particles: {altitude_at_10_particles:.2f} km",
        xy=(10, altitude_at_10_particles),
        xytext=(10, -10),  # Offset from the xy point by 20 points right and 20 points up
        textcoords="offset points",  # Use offset in points for xytext
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=12,
        ha="left",
        va="top",
    )

    # Third subplot: Injection Rate over Time
    ax3 = fig.add_subplot(gs[1, :])  # Span both columns in the second row
    # ax3.plot(time_stamps, surf_rates[:, :, inject_rate_idx].sum(axis=-1), label='Live Injection Rate')
    ax3.plot(time_stamps, surf_rates[:, azi_center, inject_rate_idx].mean(axis=1), label="Live Injection Rate")
    # plot surf_rates for every azi index
    # for i in range(num_azi_cells):
    #     ax3.plot(time_stamps, surf_rates[:, i, inject_rate_idx], label=f'Azi: {i}', alpha=0.5)
    ax3.plot(time_stamps, surface_flow_rate, "k--", label="Surface Flow Rate", lw=3)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Injection Rate")
    ax3.set_title("Injection Rate as a function of time")
    ax3.legend(fontsize=12)

    # Adjust overall layout for better spacing
    plt.tight_layout()
    plt.show()


def extract_pickle_data(
    results_path="/mnt/i/ArchiveSims/Results_20240709_173728", pattern="growth_peak_data_3[0-5]azi_*.pkl"
):
    pickle_files = glob(os.path.join(results_path, pattern))
    basenames = [os.path.basename(f) for f in pickle_files]
    pickle_data = {}
    for bname, pfile in zip(basenames, pickle_files):
        with open(pfile, "rb") as f:
            pickle_data[bname] = pickle.load(f)

    return pickle_data, basenames


def filter_temperature_profiles(temp_profiles, filter_type="savgol", **kwargs):
    """
    Apply a Savitzky–Golay or Gaussian filter to each temperature profile to reduce high-frequency noise.

    Args:
        temp_profiles (np.ndarray): A 1D or 2D array of temperature profiles.
        filter_type (str): Filter type: "savgol" or "gaussian".
        **kwargs: For "savgol": window_length (int, default=11), polyorder (int, default=3).
                  For "gaussian": sigma (float, default=2), mode (str, default="reflect").

    Returns:
        np.ndarray: Filtered temperature profiles.
    """
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d

    single_profile = False
    if temp_profiles.ndim == 1:
        temp_profiles = temp_profiles[np.newaxis, :]
        single_profile = True

    filtered_profiles = []
    if filter_type == "savgol":
        window_length = kwargs.get("window_length", 11)
        polyorder = kwargs.get("polyorder", 3)
        for profile in temp_profiles:
            filtered_profiles.append(savgol_filter(profile, window_length, polyorder))
    elif filter_type == "gaussian":
        sigma = kwargs.get("sigma", 2)
        mode = kwargs.get("mode", "reflect")
        for profile in temp_profiles:
            filtered_profiles.append(gaussian_filter1d(profile, sigma, mode=mode))
    else:
        raise ValueError("filter_type must be either 'savgol' or 'gaussian'")

    filtered_profiles = np.array(filtered_profiles)
    if single_profile:
        return filtered_profiles[0]
    return filtered_profiles


# Wave growth data from Tian et al. (2023)
TIAN_GROWTH_DATA = '''
1.0008130081300812, 150
0.9595528455284552, 151.19565217391303
0.9359756097560976, 152.73291925465838
0.9182926829268292, 154.44099378881987
0.9182926829268292, 156.49068322981367
0.9477642276422764, 158.3695652173913
1.0008130081300812, 159.0527950310559
1.0479674796747966, 159.90683229813664
1.0833333333333333, 161.1024844720497
1.0833333333333333, 161.1024844720497
1.1010162601626017, 161.95652173913044
1.106910569105691, 163.15217391304347
1.1186991869918699, 164.17701863354037
1.0951219512195123, 165.54347826086956
1.0951219512195123, 165.54347826086956
1.136382113821138, 168.1055900621118
1.136382113821138, 168.1055900621118
1.159959349593496, 169.30124223602485
1.1658536585365855, 171.1801242236025
1.1304878048780487, 171.86335403726707
1.0951219512195123, 172.37577639751552
1.1245934959349593, 173.22981366459626
1.1776422764227643, 173.74223602484471
1.1953252032520325, 174.42546583850933
1.2130081300813007, 174.93788819875778
1.2130081300813007, 174.93788819875778
1.2660569105691057, 176.4751552795031
1.2660569105691057, 176.4751552795031
1.3132113821138212, 176.8167701863354
1.366260162601626, 178.01242236024845
1.4075203252032522, 179.37888198757764
1.4252032520325204, 180.91614906832297
1.4134146341463416, 182.11180124223603
1.3957317073170732, 183.47826086956522
1.3839430894308942, 185.01552795031057
1.3957317073170732, 187.2360248447205
1.4075203252032522, 188.09006211180125
1.4075203252032522, 188.09006211180125
1.4428861788617886, 188.77329192546583
1.4782520325203252, 190.13975155279502
1.5077235772357724, 190.82298136645963
1.5430894308943088, 191.33540372670808
1.584349593495935, 192.70186335403727
1.5961382113821139, 193.38509316770185
1.5961382113821139, 193.38509316770185
1.5902439024390245, 194.9223602484472
1.6315040650406503, 196.80124223602485
1.6550813008130083, 198.50931677018633
1.7022357723577235, 199.87577639751552
1.8083333333333336, 200.90062111801242
1.8083333333333336, 200.90062111801242
1.7729674796747967, 200.72981366459626
1.8083333333333336, 202.9503105590062
1.8495934959349594, 203.9751552795031
1.896747967479675, 205.17080745341616
1.9497967479674796, 206.70807453416148
1.9674796747967482, 207.73291925465838
1.9733739837398376, 211.14906832298135
2.0441056910569104, 212.3447204968944
2.1089430894308943, 215.24844720496895
2.1678861788617887, 216.78571428571428
2.203252032520325, 217.81055900621118
2.226829268292683, 219.51863354037266
2.226829268292683, 220.88509316770185
2.2916666666666665, 222.0807453416149
2.338821138211382, 223.1055900621118
2.4272357723577236, 226.00931677018633
2.5215447154471544, 228.22981366459626
2.5215447154471544, 228.22981366459626
2.4920731707317074, 227.54658385093165
2.391869918699187, 224.81366459627327
2.5569105691056913, 229.4254658385093
2.5922764227642277, 230.79192546583852
2.6453252032520327, 232.15838509316768
2.733739837398374, 233.69565217391303
2.798577235772358, 235.40372670807454
2.8869918699186994, 237.96583850931677
2.98130081300813, 240.8695652173913
3.099186991869919, 244.45652173913044
3.211178861788618, 248.04347826086956
3.276016260162602, 250.77639751552795
3.3113821138211383, 252.82608695652175
3.452845528455285, 255.9006211180124
3.5825203252032525, 258.9751552795031
3.700406504065041, 261.0248447204969
3.835975609756098, 263.5869565217391
3.965650406504065, 265.9782608695652
4.12479674796748, 268.5403726708074
4.3134146341463415, 271.6149068322981
4.460772357723577, 274.17701863354034
4.584552845528456, 276.7391304347826
4.696544715447154, 279.81366459627327
4.808536585365854, 282.5465838509317
4.920528455284552, 284.7670807453416
5.056097560975609, 287.67080745341616
5.173983739837398, 290.2329192546584
5.291869918699187, 292.4534161490683
5.462804878048781, 295.3571428571429
5.627845528455285, 298.43167701863354
5.828252032520325, 301.1645962732919
5.952032520325203, 303.72670807453414
6.140650406504065, 306.97204968944095
'''

# Convert Tian data to numpy arrays
tian_data_array = np.array([list(map(float, line.split(",")))
                           for line in TIAN_GROWTH_DATA.strip().split("\n")])
TIAN_GROW = tian_data_array[:, 0]
TIAN_ALTITUDES = tian_data_array[:, 1]

# Wave growth data from spherical 1D model
SPHERICAL_1D_DATA = '''
4.994565217391305, 313.91616766467064
4.899456521739131, 312.23952095808386
4.759963768115942, 309.38922155688624
4.607789855072464, 306.3712574850299
4.385869565217392, 302.17964071856284
4.0815217391304355, 296.47904191616766
3.916666666666667, 293.4610778443114
3.5552536231884058, 284.07185628742513
3.3967391304347827, 274.01197604790417
2.921195652173913, 263.9520958083832
2.33786231884058, 244
2.261775362318841, 234.4431137724551
1.9891304347826089, 223.88023952095807
1.7798913043478262, 213.31736526946108
1.5769927536231885, 200.5748502994012
1.4248188405797102, 189.34131736526945
1.3677536231884058, 184.1437125748503
1.2092391304347827, 175.4251497005988
1.190217391304348, 171.73652694610777
1.1331521739130435, 163.8562874251497
1.0570652173913044, 158.99401197604791
1, 150.61077844311376
'''

# Convert spherical 1D data to numpy arrays
spherical_1d_array = np.array([list(map(float, line.split(",")))
                              for line in SPHERICAL_1D_DATA.strip().split("\n")])
SPHERICAL_1D_GROW = spherical_1d_array[:, 0]
SPHERICAL_1D_ALTITUDES = spherical_1d_array[:, 1]