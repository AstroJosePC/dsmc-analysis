import math  # For math.sqrt, math.exp, math.pi

# --- Martian Atmosphere and Atomic Oxygen Properties ---

# Physical constants (Corrected grav to gravitational constant)
grav_constant = 6.674e-11  # Gravitational constant [N m^2/kg^2]
boltz = 1.380649e-23  # Boltzmann constant
pi = math.pi
amu = 1.66054e-27  # Atomic mass unit [kg]

# Mars properties
planet_mass_mars = 6.4171e23  # Mass of Mars [kg]
planet_radius_mars = 3389.5e3  # Radius of Mars [m]

# Atomic Oxygen (O) properties
species_info = [
    {
        "name": "O",  # Atomic Oxygen
        "n0": 1.0e16,  # Example number density
        "mass": 15.999 * amu,  # Mass of Atomic Oxygen [kg]
        "energy": "THERMAL",
        "ref_xsec": 4.5e-20,  # Reference cross-section for O [m^2]
        "visc_exp": 0.67,  # Viscosity exponent for O
        "int_DOF": 3.0,  # Internal degrees of freedom (for atom)
    }
]
num_species_total = len(species_info)

num_surf_cells = 1
surface = [{"T0": 270.0}]  # Surface temperature 270K for Mars

num_coll_total = 1  # Now we have 1 collision type: O-O self-collision
collision = [
    {
        "consider": "CONSIDER_COLLISIONS",
        "energy": "THERMAL",
        "species1": 0,
        "species2": 0,  # O-O self-collision
        "reduced_mass": (species_info[0]["mass"] * species_info[0]["mass"])
        / (species_info[0]["mass"] + species_info[0]["mass"]),  # Reduced mass for O-O
    }
]


# --- Modified Radial Boundaries ---
altitude_lower_bound = 100.0e3  # 100 km above surface [m]
altitude_upper_bound = 1100.0e3  # 1100 km above surface [m] # Adjusted to 1100km to match Fortran test case
grid_lower_rad_bound_global = planet_radius_mars + altitude_lower_bound  # Lower bound at 100km above Mars radius
grid_upper_rad_bound_global = planet_radius_mars + altitude_upper_bound  # Upper bound at 1100km above Mars radius


min_rad_cell_width_global = 100.0  # Example minimum cell width (m) - can be adjusted
T_ref_VHS = 273.0  # Reference Temperature for VHS model (K)
grav_constant_for_code = grav_constant  # Renaming for clarity in code, to match Fortran 'grav'


def Calculate_VHS_CollCrossSection_python(collision_info, species_info_spec, rel_speed):
    """
    Python implementation of Calculate_VHS_CollCrossSection (Bird 1994).
    """
    ref_xsec = species_info_spec["ref_xsec"]
    visc_exp = species_info_spec["visc_exp"]
    reduced_mass = collision_info["reduced_mass"]

    xsec = ref_xsec * (
        math.sqrt(8.0 * boltz * T_ref_VHS / pi / reduced_mass)
        * math.sqrt(8.0 * boltz * T_ref_VHS / pi / reduced_mass)
        / (rel_speed * rel_speed)
    ) ** (visc_exp - 0.5)
    return xsec


def Collision_Frequency_python(n, T, collision_info, species_info_list):
    """
    Python implementation of Collision_Frequency function.
    """
    rel_speed = math.sqrt(8.0 * boltz * T / pi / collision_info["reduced_mass"])
    coll_xsec = 0.0

    if collision_info["energy"].upper() == "THERMAL":
        xsec1 = 0.0
        xsec2 = 0.0

        spec1_name = species_info_list[collision_info["species1"]]["name"]
        spec2_name = species_info_list[collision_info["species2"]]["name"]

        if spec1_name == "H2O":
            xsec1 = Calculate_VHS_CollCrossSection_python(
                collision_info, species_info_list[collision_info["species1"]], rel_speed
            )  # Using VHS as default for now
        else:  # Default to VHS for other species
            xsec1 = Calculate_VHS_CollCrossSection_python(
                collision_info, species_info_list[collision_info["species1"]], rel_speed
            )

        if spec2_name == "H2O":
            xsec2 = Calculate_VHS_CollCrossSection_python(
                collision_info, species_info_list[collision_info["species2"]], rel_speed
            )  # Using VHS as default for now
        else:  # Default to VHS for other species
            xsec2 = Calculate_VHS_CollCrossSection_python(
                collision_info, species_info_list[collision_info["species2"]], rel_speed
            )

        coll_xsec = 0.25 * (math.sqrt(xsec1) + math.sqrt(xsec2)) ** 2

    elif collision_info["energy"].upper() == "NONTHERMAL":
        coll_xsec = 1.0e-19  # Example value for nonthermal
        pass  # Implement NONTHERMAL cross-section logic here if required

    coll_freq = n * coll_xsec * rel_speed
    return coll_freq


def calculate_cell_width_python(r0, min_rad_cell_width, processes, num_rad_cells_current):
    """
    Python implementation of Calculate_Cell_Width, now with Martian properties and Atomic Oxygen, WITH DEBUG PRINTS.
    """

    if processes.upper() in ("NONTHERMAL",):
        cw = min_rad_cell_width
    elif processes.upper() in ("THERMAL", "THERMAL_AND_NONTHERMAL"):
        num_dens = [0.0] * num_species_total
        coll_freq = [0.0] * num_species_total
        mean_free_path = [0.0] * num_species_total
        scale_height = [0.0] * num_species_total
        Knudsen = [0.0] * num_species_total

        T0 = max(surf["T0"] for surf in surface)  # MAXVAL(surface(1:num_surf_cells)%T0)

        if (
            num_rad_cells_current == 1
        ):  # Corrected: Match Fortran BOTTOM condition (num_rad_cells == BOTTOM, where BOTTOM is 1)
            num_dens[:] = [spec["n0"] for spec in species_info]  # species_info(1:num_species_total)%n0
        else:
            for i_spec in range(num_species_total):
                exponent = -(r0 - grid_lower_rad_bound_global) / (
                    boltz
                    * T0
                    * r0
                    * grid_lower_rad_bound_global
                    / grav_constant_for_code
                    / planet_mass_mars
                    / species_info[i_spec]["mass"]
                )
                num_dens[i_spec] = species_info[i_spec]["n0"] * math.exp(
                    exponent
                )  # Use grav_constant_for_code and planet_mass_mars

        for i_coll in range(num_coll_total):
            if collision[i_coll]["consider"].upper() == "CONSIDER_COLLISIONS":
                if collision[i_coll]["energy"].upper() == "THERMAL":
                    spec1 = collision[i_coll]["species1"]
                    spec2 = collision[i_coll]["species2"]
                    cf_val_spec1 = Collision_Frequency_python(num_dens[spec2], T0, collision[i_coll], species_info)
                    coll_freq[spec1] += cf_val_spec1
                    if spec1 != spec2:
                        cf_val_spec2 = Collision_Frequency_python(num_dens[spec1], T0, collision[i_coll], species_info)
                        coll_freq[spec2] += cf_val_spec2

        for i_spec in range(num_species_total):
            if species_info[i_spec]["energy"].upper() == "THERMAL":
                if coll_freq[i_spec] > 0:  # Avoid division by zero if coll_freq is zero
                    mean_free_path[i_spec] = (
                        math.sqrt(8.0 * boltz * T0 / pi / species_info[i_spec]["mass"]) / coll_freq[i_spec]
                    )
                else:
                    mean_free_path[i_spec] = float("inf")  # Or a very large number if coll_freq is zero
                scale_height[i_spec] = (
                    boltz * T0 * r0 * r0 / grav_constant_for_code / planet_mass_mars / species_info[i_spec]["mass"]
                )  # Use grav_constant_for_code and planet_mass_mars
                if scale_height[i_spec] > 0:  # Avoid division by zero if scale_height is zero (though unlikely)
                    Knudsen[i_spec] = mean_free_path[i_spec] / scale_height[i_spec]
                else:
                    Knudsen[i_spec] = float("inf")  # Or a very large number if scale_height is zero

        cw_calculated_thermal = (
            sum(num_dens[i] * mean_free_path[i] for i in range(num_species_total)) / sum(num_dens)
            if sum(num_dens) > 0
            else float("inf")
        )  # Changed to inf if sum_dens is zero

        cw = cw_calculated_thermal

        if math.isinf(cw):  # Handle inf case, set cw to min_rad_cell_width if CW_thermal is inf
            cw = min_rad_cell_width
        elif min_rad_cell_width > 0.0 and cw > min_rad_cell_width:  # Normal capping logic
            cw = min_rad_cell_width

    else:  # Default case if processes is not recognized
        cw = min_rad_cell_width

    return cw


def define_radial_cells_python(
    grid_lower_rad_bound, grid_upper_rad_bound, min_rad_cell_width, processes, max_num_rad_cells_limit=10000
):
    """
    Python implementation of Define_Radial_Cells subroutine.
    """

    num_rad_cells = 1  # Corrected: Initialize num_rad_cells to 1, like Fortran
    radial_cells = []  # In Python, we can use a list to store cell bounds

    lower_bound = grid_lower_rad_bound
    cell_width = calculate_cell_width_python(
        lower_bound, min_rad_cell_width, processes, num_rad_cells
    )  # Pass num_rad_cells
    upper_bound = lower_bound + cell_width

    radial_cells.append({"lower_bound": lower_bound, "upper_bound": upper_bound})  # Store first cell bounds

    cell_counter = 2  # Start cell counter from 2 to align with Fortran output

    while upper_bound <= grid_upper_rad_bound:
        num_rad_cells += 1

        if num_rad_cells > max_num_rad_cells_limit:  # Simulate MAX_NUM_RAD_CELLS check
            print("Error_Max_Num_Radial_Cells: Maximum number of radial cells exceeded.")
            break  # Or raise an exception

        lower_bound = upper_bound  # Update lower bound for next cell
        cell_width = calculate_cell_width_python(
            lower_bound, min_rad_cell_width, processes, num_rad_cells
        )  # Pass updated num_rad_cells
        upper_bound = lower_bound + cell_width

        radial_cells.append({"lower_bound": lower_bound, "upper_bound": upper_bound})  # Store cell bounds
        cell_counter += 1

    final_upper_rad_bound = upper_bound  # Fortran code updates grid_upper_rad_bound

    return num_rad_cells, final_upper_rad_bound


def find_min_rad_cell_width_for_cells_python(
    target_cells=60,
    initial_min_width=1.0,
    tolerance=2,
    max_iterations=25,
    grid_lower_rad_bound=grid_lower_rad_bound_global,
    grid_upper_rad_bound=grid_upper_rad_bound_global,
    processes="THERMAL",
    fixed_min_rad_cell_width=None,
):
    if fixed_min_rad_cell_width is not None:
        num_cells, _ = define_radial_cells_python(
            grid_lower_rad_bound, grid_upper_rad_bound, fixed_min_rad_cell_width, processes
        )
        print(f"Testing fixed min_rad_cell_width = {fixed_min_rad_cell_width:.6f}, num_rad_cells = {num_cells}")
        return fixed_min_rad_cell_width, num_cells

    current_min_width = initial_min_width
    if initial_min_width <= 0:
        current_min_width = (grid_upper_rad_bound - grid_lower_rad_bound) / target_cells

    # Track best result
    best_width = current_min_width
    best_num_cells = float('inf')
    plateau_count = 0
    last_num_cells = float('inf')

    for iteration in range(max_iterations):
        num_cells, _ = define_radial_cells_python(
            grid_lower_rad_bound, grid_upper_rad_bound, current_min_width, processes
        )

        print(f"Iteration {iteration + 1}: min_rad_cell_width = {current_min_width:.6f}, num_rad_cells = {num_cells}")

        # Update best result if we found fewer cells
        if num_cells < best_num_cells:
            best_num_cells = num_cells
            best_width = current_min_width
            plateau_count = 0
        
        # Check if we've hit a plateau
        if num_cells >= last_num_cells:
            plateau_count += 1
        else:
            plateau_count = 0

        # If we've been on a plateau for 5 iterations, return best result
        if plateau_count >= 10:
            print(f"\nHit minimum possible cells at {best_num_cells}.")
            print(f"Best achievable min_rad_cell_width: {best_width:.6f}")
            return best_width, best_num_cells

        if abs(num_cells - target_cells) <= tolerance:
            return current_min_width, num_cells

        last_num_cells = num_cells
        
        if num_cells < target_cells:
            current_min_width *= 0.9
        else:
            current_min_width *= 1.1

        if current_min_width <= 0:
            print("min_rad_cell_width became non-positive. Iteration stopped.")
            return None, None

    print(f"\nMaximum iterations reached. Best achievable result: {best_num_cells} cells")
    print(f"Best min_rad_cell_width: {best_width:.6f}")
    return best_width, best_num_cells


if __name__ == "__main__":
    # --- Configuration ---
    desired_radial_cells = 60
    initial_width_guess = 10e3  # Initial guess, not used in fixed width test
    acceptable_tolerance = 0  # Tolerance, not used in fixed width test
    max_iter = 100  # Max iterations, not used in fixed width test
    lower_bound = grid_lower_rad_bound_global  # Use global lower bound (Mars radius + 100km)
    upper_bound = grid_upper_rad_bound_global  # Use global upper bound (Mars radius + 1100km) # Adjusted to 1100km
    process_type = "THERMAL"  # Set to THERMAL for Martian atmosphere
    min_rad_cell_width_input = min_rad_cell_width_global  # Use global min_rad_cell_width
    fixed_width_test_value = 16581.189770  # Fixed width to test

    # --- Find min_rad_cell_width ---
    optimal_min_width, achieved_cells = find_min_rad_cell_width_for_cells_python(
        target_cells=desired_radial_cells,
        initial_min_width=initial_width_guess,
        tolerance=acceptable_tolerance,
        max_iterations=max_iter,
        grid_lower_rad_bound=lower_bound,
        grid_upper_rad_bound=upper_bound,
        processes=process_type,
    )

    if optimal_min_width is not None:
        print(f"\nOptimal min_rad_cell_width: {optimal_min_width:.6f} meters")
        print(f"Achieved number of cells: {achieved_cells}")
    else:
        print("\nError in finding optimal min_rad_cell_width.")
