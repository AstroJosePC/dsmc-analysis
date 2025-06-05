import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm

import re
from glob import glob

# set seaborn style
# sns.set(style="ticks", context="talk")
sns.set_theme(style="ticks", context="talk")

# compute the scale height
M = 16  # g/mol
num_dens = 1e16  # Oxygen / m^3
R = 8.314  # J/mol/K
R_prime = R / M * 1e3  # J/kg/K
T = 270  # K
g = 3.51  # m/s^2
H = R_prime * T / g / 1e3  # km

# mars_radius = 3389.5e3


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


def open_macro(macro_file):
    return pd.read_csv(macro_file, sep=r"\s+", header=None, names=column_headers, usecols=usecols)


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


results_path = "/media/astrojose/AstroBook/ArchiveSims/Results_June2024_Experiments_6"

parameters = read_simulation_parameters(f"{results_path}/Info/Simulation_Parameters.ascii")
print(parameters)

timestep = parameters["Simulation Parameters"][0]["Time Step"]
nsteps = parameters["Simulation Parameters"][1]["Number of Total Simulations Steps"]
part_weight = parameters["Species Information"][0]["Particle Weight"]
num_radial_cells: int = parameters["Simulation Parameters"][1]["Number of Radial Cells"]
num_azi_cells: int = parameters["Simulation Parameters"][1]["Number of Azimuth Cells"]

if "Start Step of Wave Injection" in parameters["Simulation Parameters"][1]:
    step0_inject = parameters["Simulation Parameters"][1]["Start Step of Wave Injection"]
    stepf_inject = parameters["Simulation Parameters"][1]["End Step of Wave Injection"]
    OMEGA = parameters["Simulation Parameters"][1]["Wave Injection Frequency"]
    # num_periods = round(parameters["Simulation Parameters"][1]["num_periods"])
    period_length = 2 * np.pi / OMEGA

    print("\nStep0 Inject: ", step0_inject)
    print("Stepf Inject: ", stepf_inject)
    print("Omega: ", OMEGA)
    WAVE_ACTION = True
else:
    WAVE_ACTION = False
    num_periods = 0
    step0_inject = None
    stepf_inject = None
    OMEGA = None


# print every parameter
for key, value in parameters.items():
    print(key)
    if isinstance(value, list):
        for i in value:
            if isinstance(i, dict):
                for k, v in i.items():
                    print(f"\t{k}: {v}")
            else:
                print(i)
    else:
        for k, v in value.items():
            print(f"\t{k}: {v}")

mars_radius = parameters["Planetary Body Information"][0]["Radius"]

num_periods = round((stepf_inject - step0_inject - 1) * OMEGA * timestep / (2 * np.pi), 2)

print("Number of periods: ", num_periods)

num_species_total = len(parameters["Species Information"])
print("Number of species: ", num_species_total)

macro_files = glob(f"{results_path}/Macro/Output*_Proc*_Macroscopic.ascii")
macro_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])
column_headers = (
    ["time", "avg_time_interval", "grid_rad", "grid_zen", "grid_azi", "cell_num"]
    + [f"num_part_{i}" for i in range(1, num_species_total + 1)]
    + [f"num_dens_{i}" for i in range(1, num_species_total + 1)]
    + [f"vel_rad_{i}" for i in range(1, num_species_total + 1)]
    + [f"vel_zen_{i}" for i in range(1, num_species_total + 1)]
    + [f"vel_azi_{i}" for i in range(1, num_species_total + 1)]
    + [f"vel_therm_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_rad_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_zen_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_azi_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_kin_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_trans_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_int_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_ov_{i}" for i in range(1, num_species_total + 1)]
    # + [f"q_rad_{i}" for i in range(1, num_species_total + 1)]
    # + [f"q_zen_{i}" for i in range(1, num_species_total + 1)]
    # + [f"q_azi_{i}" for i in range(1, num_species_total + 1)]
    + [f"coll_freq_theo_{i}_{j}" for i in range(1, num_species_total + 1) for j in range(1, num_species_total + 2)]
    + [f"mean_free_path_{i}" for i in range(1, num_species_total + 1)]
    + [f"scale_height_{i}" for i in range(1, num_species_total + 1)]
    + [f"knudsen_{i}" for i in range(1, num_species_total + 1)]
    + [f"therm_conduct_up_{i}" for i in range(1, num_species_total + 1)]
    + [f"therm_conduct_down_{i}" for i in range(1, num_species_total + 1)]
)


number_of_outputs = int(re.split(r"(\d+)", macro_files[-1])[-4]) - int(re.split(r"(\d+)", macro_files[0])[-4]) + 1
num_processes = int(re.split(r"(\d+)", macro_files[-1])[-2]) + 1

# limit numb of outputs to 10
# number_of_outputs = 10
# macro_files = macro_files[:number_of_outputs*num_processes]

macro_props = ["Kinetic Temperature", "Number Density", "Radial Velocity"]
macro_grids_data = {
    prop: np.zeros((num_species_total, number_of_outputs, num_radial_cells, num_azi_cells)) for prop in macro_props
}

# columns we are actually interested in:
usecols = (
    ["time", "grid_rad", "grid_zen", "grid_azi", "cell_num"]
    + [f"num_part_{i}" for i in range(1, num_species_total + 1)]
    + [f"num_dens_{i}" for i in range(1, num_species_total + 1)]
    + [f"vel_rad_{i}" for i in range(1, num_species_total + 1)]
    + [f"temp_kin_{i}" for i in range(1, num_species_total + 1)]
)


def open_macro(macro_file):
    return pd.read_csv(macro_file, sep=r"\s+", header=None, names=column_headers, usecols=usecols)


time_stamps = np.zeros(number_of_outputs, dtype=np.float32)
num_particle_data = np.zeros((num_species_total, number_of_outputs, num_radial_cells, num_azi_cells), dtype=np.float32)

macro_props = ["Kinetic Temperature", "Number Density", "Radial Velocity"]

# for k, i in enumerate(range(0, len(macro_files), num_processes)):
for k, i in tqdm(enumerate(range(0, len(macro_files), num_processes)), total=len(macro_files) // num_processes):
    macro_dfs = [open_macro(macro_files[i]) for i in range(i, i + num_processes)]
    df = pd.concat(macro_dfs, axis=0, ignore_index=True)
    df_sort_cell = df.sort_values(by="cell_num")
    time_stamps[k] = macro_dfs[0]["time"].values[0]

    for j in range(num_species_total):
        num_particle_data[j, k, :, :] = df_sort_cell[f"num_part_{j + 1}"].values.T.reshape(
            (num_radial_cells, num_azi_cells)
        )

        spec_macro_props = [f"temp_kin_{j + 1}", f"num_dens_{j + 1}", f"vel_rad_{j + 1}"]
        data = df_sort_cell[spec_macro_props].values.T.reshape(len(macro_props), num_radial_cells, num_azi_cells)

        for h, prop in enumerate(macro_props):
            macro_grids_data[prop][j, k, :, :] = data[h, :, :]

# store all data in macro_grids_data and pickle dump
macro_grids_data["Time Stamps"] = time_stamps
macro_grids_data["Number Particles"] = num_particle_data

rad_cells_headers = (
    ["num_rad_cells", "R1", "R2", "CW"]
    + [f"n_{i}" for i in range(1, num_species_total + 1)]
    + [f"MFP_{i}" for i in range(1, num_species_total + 1)]
    + [f"H_{i}" for i in range(1, num_species_total + 1)]
    + [f"Kn_{i}" for i in range(1, num_species_total + 1)]
)

rad_cells_df = pd.read_csv(f"{results_path}/Info/Radial_Cells.ascii", sep=r"\s+", header=None, names=rad_cells_headers)

mid_bound = (rad_cells_df["R1"] + rad_cells_df["R2"]) / 2.0
rad_cells_df["Scaled Radius"] = mid_bound / rad_cells_df["R1"][0]
rad_cells_df["Mid Radius"] = mid_bound

mid_rad_km = mid_bound.values / 1e3 - mars_radius
r2_rad_km = rad_cells_df["R2"].values / 1e3 - mars_radius
r1_rad_km = rad_cells_df["R1"].values / 1e3 - mars_radius

# store rad_cells_df in macro_grids_data
macro_grids_data["Radial Cells"] = rad_cells_df

surface_headers = [
    "i_surf",
    "i_zen",
    "i_azi",
    "zen_lower_bound",
    "zen_upper_bound",
    "azi_lower_bound",
    "azi_upper_bound",
    "area",
    "T0",
    "num_dens",
    "scale_height",
    "speed",
    "flux",
    "flow_rate",
    "Jeans_parameter",
    "Jeans_escape_flux",
    "Jeans_escape_rate",
]
surface_df = pd.read_csv(f"{results_path}/Info/Surface_Cells.ascii", sep=r"\s+", header=None, names=surface_headers)

# rebuild the time grid
time_bounds = np.zeros(time_stamps.size + 1)
time_bounds[1:] = time_stamps

surface_headers = [
    "i_surf",
    "i_zen",
    "i_azi",
    "zen_lower_bound",
    "zen_upper_bound",
    "azi_lower_bound",
    "azi_upper_bound",
    "area",
    "T0",
    "num_dens",
    "scale_height",
    "speed",
    "flux",
    "flow_rate",
    "Jeans_parameter",
    "Jeans_escape_flux",
    "Jeans_escape_rate",
]
surface_df = pd.read_csv(f"{results_path}/Info/Surface_Cells.ascii", sep=r"\s+", header=None, names=surface_headers)
surf_speed = surface_df["speed"][0]
surf_n0 = 1e16

# stepf_inject = int(step0_inject + 2 * np.pi * num_periods / OMEGA / timestep + 1)
if WAVE_ACTION:
    # store step0_inject, stepf_inject, timestep to macro_grids_data
    macro_grids_data["Wave Injection"] = {
        "Start Step": step0_inject,
        "End Step": stepf_inject,
        "Timestep": timestep,
        "omega": OMEGA,
        "num_periods": num_periods,
    }

    # number_density = np.array([open_macro(file)['Number Density'] for file in macro_files])
    is_oscillating = (step0_inject * timestep <= time_stamps) & (time_stamps <= stepf_inject * timestep)

    surface_flux = np.zeros(len(time_stamps))
    surface_flux[:] = 0.25 * surf_n0 * surf_speed
    surface_flux[is_oscillating] = (
        0.25 * surf_n0 * surf_speed * (1 + 0.25 * np.sin(OMEGA * (time_stamps[is_oscillating] - step0_inject * timestep)))
    )
    surface_flow_rate = surface_flux * surface_df["area"][0]

# Save the macro_grids_data to a file
with open(f"{results_path}/macro_grids_data.pkl", "wb") as f:
    pickle.dump(macro_grids_data, f)

params_output_path = f"{results_path}/params_data.npz"
np.savez(
    params_output_path,
    time=macro_grids_data["Time Stamps"],
    step0_inject=step0_inject,
    stepf_inject=stepf_inject,
    timestep=timestep,
    omega=OMEGA,
    num_periods=num_periods,
    mid_rad_km=mid_rad_km,  # radial distances in km
    radial_cells=mid_rad_km,
    mars_radius=mars_radius,
    wave_action=WAVE_ACTION,
)

density_data = macro_grids_data["Number Density"]
temperature_data = macro_grids_data["Kinetic Temperature"]
radial_vel_data = macro_grids_data["Radial Velocity"]

density_output_path = f"{results_path}/density_data.npy"
temperature_output_path = f"{results_path}/temperature_data.npy"
rad_vel_output_path = f"{results_path}/rad_vel_data.npy"

np.save(density_output_path, density_data)
np.save(temperature_output_path, temperature_data)
np.save(rad_vel_output_path, radial_vel_data)

# open surface files and plot number of particles on surface as a function of time
surf_files = sorted(glob(f"{results_path}/Rates/Output*_Proc*_Surface.ascii"))
surf_files.sort(key=lambda x: [int(d) for d in re.split(r"(\d+)", x) if d.isdigit()])

# surf_files = surf_files[:number_of_outputs*num_processes]


def extract_surface_data(surf_files, num_processes):
    """
    Extracts the surface data from multiple files
    """
    final_array = []  # Step 1: Initialize the final array list
    num_files = len(surf_files)

    for i in tqdm(range(0, num_files, num_processes), total=num_files // num_processes):
        # Step 2: Loop over surf_files in steps of num_processes
        # for i in range(0, num_files, num_processes):
        sample_surf_rates = []  # List to hold data arrays for the current time stamp

        # Step 3: Loop over the range (i, i + num_processes)
        for j in range(i, min(i + num_processes, num_files)):
            # Open the surface data file and append its data to sample_surf_rates
            df = open_surface(surf_files[j])
            sample_surf_rates.append(df)

        # Step 4: Concatenate and convert to numpy array
        if sample_surf_rates:
            df_concat = pd.concat(sample_surf_rates, axis=0, ignore_index=True)
            final_array.append(df_concat.values)

    # Step 5: Stack the arrays to get the final shape (time_stamps, num_azi_cells, 17)
    return np.stack(final_array, axis=0)


# surf_rates = np.asarray([open_surface(file).values[0] for file in surf_files])
# surf_rates = extract_surface_data(surf_files, num_processes)
# num_part_inject_idx = 7
# inject_rate_idx = 11

# azi_center = np.where(~np.isclose(surf_rates[:, :, num_part_inject_idx].std(axis=0), 0.0))[0]
# print("Azi Center Index: ", azi_center)

# store the surface data in macro_grids_data
# macro_grids_data["Surface Rates"] = surf_rates
if WAVE_ACTION:
    macro_grids_data["Computed flow rate"] = surface_flow_rate

# Save the macro_grids_data to a file
with open(f"{results_path}/macro_grids_data.pkl", "wb") as f:
    pickle.dump(macro_grids_data, f)
