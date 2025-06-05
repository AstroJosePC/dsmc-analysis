import argparse
from calc_rad import find_min_rad_cell_width_for_cells_python

# Mars parameters
MARS_RADIUS = 3389.5e3  # Mars radius in meters


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate optimal radial cell width for Mars atmosphere")
    parser.add_argument("atmosphere_height", type=float, help="Atmosphere height in km (e.g., 1100)")
    parser.add_argument("target_cells", type=int, help="Number of desired radial cells (e.g., 60)")
    parser.add_argument(
        "--boundary_file_path",
        type=str,
        help="Path to the boundary file (e.g., Input/Boundaries/RadialCellWidth.input)",
    )
    args = parser.parse_args()

    # Convert atmosphere height from km to meters
    atmosphere_height = args.atmosphere_height * 1000

    params = {
        "target_cells": args.target_cells,
        "initial_min_width": 10e3,  # 10 km initial guess
        "tolerance": 0,  # 0.1 meter tolerance
        "max_iterations": 100,
        "grid_lower_rad_bound": MARS_RADIUS + 100e3,  # Lower bound: Mars radius + 100km
        "grid_upper_rad_bound": MARS_RADIUS + atmosphere_height,
        "processes": "THERMAL",
    }

    optimal_width, achieved_cells = find_min_rad_cell_width_for_cells_python(**params)

    if optimal_width is not None:
        print(f"Optimal minimum radial cell width: {optimal_width:.2f} meters")
        print(
            f"For {achieved_cells} cells from "
            f"{params['grid_lower_rad_bound'] / 1000:.1f} km to "
            f"{params['grid_upper_rad_bound'] / 1000:.1f} km"
        )
        if achieved_cells == args.target_cells and args.boundary_file_path:
            with open(args.boundary_file_path, "r+") as file:
                lines = file.readlines()
                lines[0] = f"{optimal_width:.2f}\n"
                file.seek(0)
                file.writelines(lines)
        elif achieved_cells != args.target_cells:
            raise ValueError(f"Achieved cells ({achieved_cells}) do not match target cells ({args.target_cells})")
    else:
        print("Could not find optimal width within given parameters")


if __name__ == "__main__":
    main()
