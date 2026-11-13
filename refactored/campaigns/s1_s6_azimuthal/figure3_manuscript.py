"""Generate combined manuscript Figure 3 (a: density-amplitude growth, b: <dT> and heating slopes) + Table 2 summary.

    python campaigns/s1_s6_azimuthal/figure3_manuscript.py
    python campaigns/s1_s6_azimuthal/figure3_manuscript.py --individual-panels
    python campaigns/s1_s6_azimuthal/figure3_manuscript.py --background mad_column0 --altitude-cells first_above

Defaults to the refactored S1-S6 recipe (dsmclib.resolution.STAGING_RECIPE: background='azimean_preforcing',
altitude_cells='nearest') and median over the last 2 forcing cycles (Text S2).
Reads caches from outputs/s1_s6_azimuthal/cache/ and outputs/radial_1d/cache/ (computing 1D_5p if absent).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dsmclib import constants, growth, heating, plotting, resolution
from dsmclib.background import BACKGROUND_METHODS
from dsmclib.paths import output_dir
from dsmclib.run_data import ALTITUDE_CELL_RULES

CAMPAIGN = "s1_s6_azimuthal"


def build_table2(growth_summary: dict, temp_s6: dict, temp_1d: dict) -> pd.DataFrame:
    """Table 2 metrics: growth factor c = ln G(230 km), exponential slope, and d<dT>/dz in K/H0."""
    g_tian = growth.exponential_growth_slope(constants.TIAN_FIG3_ALTITUDES, constants.TIAN_FIG3_GROW)
    g_1d = growth.exponential_growth_slope(constants.SPHERICAL_1D_ALTITUDES, constants.SPHERICAL_1D_GROW)
    g_center = growth.exponential_growth_slope(growth_summary["altitudes"], growth_summary["center"])
    g_edge = growth.exponential_growth_slope(growth_summary["altitudes"], growth_summary["edge"])

    s6_heat = heating.group_heating_slopes(temp_s6)
    col_1d = int(temp_1d["azimuthal_indices"][0])
    heat_1d = heating.heating_slope(temp_1d["altitudes"], temp_1d["temp_last_cycle"][col_1d])

    c_ref = g_1d["c_at_target"]
    h_ref = heat_1d["slope_K_H"]

    rows = [
        {
            "geometry": "1D Cartesian (T23)",
            "columns": "-",
            "c_150_230": g_tian["c_at_target"],
            "c_vs_radial_pct": 100.0 * (g_tian["c_at_target"] / c_ref - 1.0),
            "exp_slope_km": g_tian["slope_km"],
            "exp_r2": g_tian["r2"],
            "dTdz_K_per_H0": plotting.T23_SLOPE_K_H,
            "ci95_K_per_H0": np.nan,
            "dTdz_vs_radial_pct": 100.0 * (plotting.T23_SLOPE_K_H / h_ref - 1.0),
            "n_points": np.nan,
        },
        {
            "geometry": "1D radial (P26)",
            "columns": str([col_1d]),
            "c_150_230": c_ref,
            "c_vs_radial_pct": 0.0,
            "exp_slope_km": g_1d["slope_km"],
            "exp_r2": g_1d["r2"],
            "dTdz_K_per_H0": h_ref,
            "ci95_K_per_H0": heat_1d["ci_half_K_H"],
            "dTdz_vs_radial_pct": 0.0,
            "n_points": heat_1d["n"],
        },
        {
            "geometry": "2D center (S6)",
            "columns": str(growth_summary["columns"]["center"]),
            "c_150_230": g_center["c_at_target"],
            "c_vs_radial_pct": 100.0 * (g_center["c_at_target"] / c_ref - 1.0),
            "exp_slope_km": g_center["slope_km"],
            "exp_r2": g_center["r2"],
            "dTdz_K_per_H0": s6_heat["center"]["slope_K_H"],
            "ci95_K_per_H0": s6_heat["center"]["ci_half_K_H"],
            "dTdz_vs_radial_pct": 100.0 * (s6_heat["center"]["slope_K_H"] / h_ref - 1.0),
            "n_points": s6_heat["center"]["n"],
        },
        {
            "geometry": "2D edge (S6)",
            "columns": str(growth_summary["columns"]["edge"]),
            "c_150_230": g_edge["c_at_target"],
            "c_vs_radial_pct": 100.0 * (g_edge["c_at_target"] / c_ref - 1.0),
            "exp_slope_km": g_edge["slope_km"],
            "exp_r2": g_edge["r2"],
            "dTdz_K_per_H0": s6_heat["edge"]["slope_K_H"],
            "ci95_K_per_H0": s6_heat["edge"]["ci_half_K_H"],
            "dTdz_vs_radial_pct": 100.0 * (s6_heat["edge"]["slope_K_H"] / h_ref - 1.0),
            "n_points": s6_heat["edge"]["n"],
        },
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="S6", help="2D run alias (default: S6)")
    parser.add_argument("--radial-run", default=resolution.P26_ALIAS, help="1D radial reference alias (default: 1D_5p)")
    parser.add_argument("--background", default=resolution.STAGING_RECIPE["background"], choices=BACKGROUND_METHODS)
    parser.add_argument("--altitude-cells", default=resolution.STAGING_RECIPE["altitude_cells"],
                        choices=ALTITUDE_CELL_RULES)
    parser.add_argument("--n-last-peaks", type=int, default=2,
                        help="number of final forcing cycles for the median growth curve (default: 2, Text S2)")
    parser.add_argument("--individual-panels", action="store_true",
                        help="also save standalone Figure_3_left_panel.png and Figure_3_right_panel.png")
    args = parser.parse_args()

    fig_dir = output_dir(CAMPAIGN, "figures")
    tab_dir = output_dir(CAMPAIGN, "tables")

    growth_data = growth.cached_growth(
        args.run,
        background=args.background,
        altitude_cells=args.altitude_cells,
    )
    growth_summary = growth.centre_and_edge_growth(growth_data, n_last_peaks=args.n_last_peaks)
    temp_s6 = heating.cached_temperature_last_cycle(args.run)
    temp_1d = heating.cached_temperature_last_cycle(args.radial_run)

    table2 = build_table2(growth_summary, temp_s6, temp_1d)
    suffix = "" if (args.background == resolution.STAGING_RECIPE["background"]
                    and args.altitude_cells == resolution.STAGING_RECIPE["altitude_cells"]) else f"_{args.background}_{args.altitude_cells}"
    table_path = tab_dir / f"table2_growth_and_heating_{args.run}{suffix}.csv"
    table2.to_csv(table_path, index=False, float_format="%.4f")
    print(table2.to_string(index=False, float_format="%.4f"))
    print(f"\nSaved {table_path}")

    fig = plotting.plot_figure3_manuscript(growth_summary, temp_s6, temp_1d)
    combined_path = fig_dir / f"Figure_3_combined{suffix}.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined_path}")

    if args.individual_panels:
        rc = {
            **sns.axes_style("whitegrid"),
            **sns.plotting_context("paper", font_scale=1.8),
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
        with plt.rc_context(rc):
            fig_l, ax_l = plt.subplots(figsize=(10, 8))
            plotting.plot_figure3_left_panel(ax_l, growth_summary, panel_label="(a)")
            fig_l.tight_layout()
            left_path = fig_dir / f"Figure_3_left_panel{suffix}.png"
            fig_l.savefig(left_path, dpi=300, bbox_inches="tight")
            plt.close(fig_l)
            print(f"Saved {left_path}")

            fig_r, ax_r = plt.subplots(figsize=(10, 8))
            plotting.plot_figure3_right_panel(ax_r, temp_s6, temp_1d, panel_label="(b)")
            fig_r.tight_layout()
            right_path = fig_dir / f"Figure_3_right_panel{suffix}.png"
            fig_r.savefig(right_path, dpi=300, bbox_inches="tight")
            plt.close(fig_r)
            print(f"Saved {right_path}")


if __name__ == "__main__":
    main()
