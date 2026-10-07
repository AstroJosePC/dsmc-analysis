#!/usr/bin/env python3
"""
Generate updated Figure 3 (Left: Density Amplitude Growth, Right: Temperature Increase / Heating Rate)
Mirroring the notebook ('density and heat manuscript plots.ipynb') and zenodo panel with updates:
- Nominal exobase (~230 km) marked with a horizontal line
- 1D-spherical yellow curve (#dccd7d) replaced with a darker, high-contrast color (dark goldenrod #996515 / #b8860b)
- Added legend to the left panel
- Maintained exact physical calculations and least-squares fits
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import data processor and helper functions
from optimized_loader import OptimizedDSMCDataProcessor
from helper_funcs import (
    SPHERICAL_1D_GROW,
    SPHERICAL_1D_ALTITUDES,
    measure_exponential_slopes,
)

# ---------------------------------------------------------
# Path resolution
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Simulation paths (try local Mac mount first, then Windows fallback)
S6_CANDIDATE_PATHS = [
    Path("/Volumes/MyBook/ArchiveSims/Results_20240722_041906.318"),
    Path(r"Z:\ArchiveSims\Results_20240722_041906.318"),
    Path(r"I:\ArchiveSims\Results_20240722_041906.318"),
]
S6_DIR = next((p for p in S6_CANDIDATE_PATHS if p.exists()), None)
if S6_DIR is None:
    raise FileNotFoundError("Could not find S6 directory in candidate paths.")

# 1D Spherical heating rate dataset (AW1)
AW1_CANDIDATE_PATHS = [
    Path("/Volumes/MyBook/ArchiveSims/Results_20250326_133319.916/exports/heating_rate_data.npz"),
    Path(r"Z:\ArchiveSims\Results_20250326_133319.916\exports\heating_rate_data.npz"),
    Path(r"I:\ArchiveSims\Results_20250326_133319.916\exports\heating_rate_data.npz"),
    BASE_DIR / "heating_rate_data_1d_radial.npz",
    BASE_DIR / "zenodo_release" / "heating_rate_data_1d_radial.npz",
]
AW1_PATH = next((p for p in AW1_CANDIDATE_PATHS if p.exists()), None)
if AW1_PATH is None:
    raise FileNotFoundError("Could not find AW1 heating rate dataset in candidate paths.")

# ---------------------------------------------------------
# Color Definitions
# ---------------------------------------------------------
COLOR_1D_CART = "r"               # Red (Tian et al.)
COLOR_1D_SPH = "#a66a00"          # Dark goldenrod / dark amber (replaces pale #dccd7d)
COLOR_2D_CENTER = "#337538"       # Green (Center columns)
COLOR_2D_EDGES = "#2e2585"        # Indigo / Navy (Edge columns)
COLOR_EXOBASE = "#555555"         # Subtle charcoal for exobase line
PANEL_LABEL_KW = dict(fontsize=20, fontweight='bold', va='top')

# ---------------------------------------------------------
# Scale Height & Physics Setup
# ---------------------------------------------------------
M = 16.0       # g/mol
R = 8.314      # J/mol/K
R_prime = (R / M) * 1e3  # J/kg/K
T = 270.0      # K
g = 3.51       # m/s^2
H = R_prime * T / g / 1e3  # Scale height in km (~39.9 km)
EXOBASE_ALT = 230.0        # km

# ---------------------------------------------------------
# 1D Cartesian (Tian et al.) Reference Dataset
# ---------------------------------------------------------
RAW_TIAN_DATA = """
1.017909439016246, 150.57948303858404
1.0725718607400097, 158.34256714001563
1.153737274814689, 167.49702251229667
1.3956875091515908, 183.8869581036869
1.5444907682885036, 192.30683573243584
1.6975455491150422, 200.0915730440512
1.900201416320553, 208.11311908950768
2.0950628270950817, 214.73318838517866
2.316673395166841, 222.20356338421942
2.561819172591888, 232.49030338887752
2.7731805365904414, 240.47209281915315
2.9900081427613685, 248.34692768945735
3.2238418356908025, 255.45978889620045
3.487436180447623, 261.37687258602926
3.751030525204437, 268.0377253735611
4.014624869961253, 273.5069481013321
4.537562037785263, 284.1529770150856
6.182840387849028, 310.40257288221983
5.958000815962361, 307.02798218271016
5.663979837341338, 302.8090097893674
5.439140265454671, 299.0982323194296
5.173944872972962, 294.5447043088934
4.776151784250398, 287.2941788300541
4.2803517026541655, 278.1890801928631
1.253665158025981, 174.640377196619
6.5806334765715855, 315.63597773849114
6.3903846080521, 313.1038113489329
"""
tian_arr = np.array([list(map(float, line.split(','))) for line in RAW_TIAN_DATA.strip().split('\n')])
tian_arr = tian_arr[np.argsort(tian_arr[:, 1])]
tian_growth = tian_arr[:, 0]
tian_altitudes = tian_arr[:, 1]

# ---------------------------------------------------------
# Load 2D S6 Simulation and Process Growth & Heating
# ---------------------------------------------------------
print("Loading S6 simulation from:", S6_DIR)
high_res_processor = OptimizedDSMCDataProcessor(str(S6_DIR))
high_res_processor.load_parameters()
high_res_processor.load_data()

# Identify oscillating columns (0-indexed; forced range 511-598 inclusive, 88 cells):
# middle pair (554, 555) and outer edges (511, 598). 1-indexed (Fortran/text): 555/556 and 512/599.
osc_cols = high_res_processor.compute_osc_cols()
mid_idx = len(osc_cols) // 2
cols_center = [int(osc_cols[mid_idx - 1]), int(osc_cols[mid_idx])]  # [554, 555]
cols_outer = [int(osc_cols[0]), int(osc_cols[-1])]                 # [511, 598]
cols_to_compute = sorted(cols_center + cols_outer)

print(f"Center columns: {cols_center}, Outer edge columns: {cols_outer}")

# 1. Density Growth Curves
print("Computing density growth curves...")
growth_data = high_res_processor.compute_density_growth(azimuthal_indices=cols_to_compute)

combined_growth_middle = {}
for col in cols_center:
    altitudes, mean_growth, _, _ = high_res_processor.get_density_growth_curves(growth_data, col, n_last_peaks=2)
    combined_growth_middle[col] = mean_growth
avg_growth_middle = np.mean(list(combined_growth_middle.values()), axis=0)

combined_growth_outer = {}
for col in cols_outer:
    altitudes, mean_growth, _, _ = high_res_processor.get_density_growth_curves(growth_data, col, n_last_peaks=2)
    combined_growth_outer[col] = mean_growth
avg_growth_outer = np.mean(list(combined_growth_outer.values()), axis=0)

# Measure slopes for printing
slope_inner, r2_inner = measure_exponential_slopes(altitudes, avg_growth_middle, "2D Spherical Center")
slope_outer, r2_outer = measure_exponential_slopes(altitudes, avg_growth_outer, "2D Spherical Edges")
slope_tian, r2_tian = measure_exponential_slopes(tian_altitudes, tian_growth, "Tian et al. 2023")
slope_1d, r2_1d = measure_exponential_slopes(SPHERICAL_1D_ALTITUDES, SPHERICAL_1D_GROW, "1D Spherical")

print("\n--- Density Growth Slopes ---")
print(f"2D Spherical Center: {slope_inner:.6f} km⁻¹ (R² = {r2_inner:.4f})")
print(f"2D Spherical Edges:  {slope_outer:.6f} km⁻¹ (R² = {r2_outer:.4f})")
print(f"Tian et al. 2023:    {slope_tian:.6f} km⁻¹ (R² = {r2_tian:.4f})")
print(f"1D Spherical Model:  {slope_1d:.6f} km⁻¹ (R² = {r2_1d:.4f})")

# 2. Temperature Heating Rates
print("\nComputing temperature heating rates...")
heating_results = high_res_processor.compute_temperature_heating_rate(H=H, azimuthal_indices=cols_to_compute)

hr = heating_results['heating_rate_results']
avg_center_temp = 0.5 * (hr[cols_center[0]]['temp_last_cycle'] + hr[cols_center[1]]['temp_last_cycle'])
avg_edge_temp = 0.5 * (hr[cols_outer[0]]['temp_last_cycle'] + hr[cols_outer[1]]['temp_last_cycle'])

ref = hr[cols_center[0]]
x_alt = ref['radial_distances'][ref['cell_min_idx']:ref['cell_max_idx'] + 1]  # ~153 to 327 km
y1_center = avg_center_temp
y2_edges = avg_edge_temp


def fit_slope_ci(x, y):
    """OLS slope with 95% t-interval half-width, as in helper_funcs.compute_heating_rate_slope."""
    lr = stats.linregress(x, y)
    return lr.slope, lr.intercept, stats.t.ppf(0.975, len(x) - 2) * lr.stderr


slope1_center, intercept1_center, ci1_center = fit_slope_ci(x_alt, y1_center)
slope2_edges, intercept2_edges, ci2_edges = fit_slope_ci(x_alt, y2_edges)

# Tian reference line (slope 9.4 K/H)
line_94_y = 9.4 * (x_alt - x_alt[0]) / H
line_94_intercept = np.mean([y1_center[0], y2_edges[0]])

# 1D Spherical AW1 data
aw1_data = np.load(AW1_PATH)

print("\n--- Temperature Heating Slopes (95% t-interval) ---")
print(f"Slope for center: {slope1_center * H:.4f} +- {ci1_center * H:.4f} K/H")
print(f"Slope for edges:  {slope2_edges * H:.4f} +- {ci2_edges * H:.4f} K/H")

# Table 2 temperature-gradient column, all rows with the same estimator and interval
aw1_ci = 0.5 * (float(aw1_data['slope_H_ci_upper']) - float(aw1_data['slope_H_ci_lower']))
table2 = pd.DataFrame([
    dict(geometry='1D Cartesian (T23)', dTdz_K_per_H0=9.4, ci95_K_per_H0=np.nan, n_points=np.nan),
    dict(geometry='1D Radial (P26)', dTdz_K_per_H0=float(aw1_data['slope_H']), ci95_K_per_H0=aw1_ci, n_points=int(aw1_data['n_points'])),
    dict(geometry='2D centerline (S6)', dTdz_K_per_H0=slope1_center * H, ci95_K_per_H0=ci1_center * H, n_points=len(x_alt)),
    dict(geometry='2D edges (S6)', dTdz_K_per_H0=slope2_edges * H, ci95_K_per_H0=ci2_edges * H, n_points=len(x_alt)),
])
table2_path = BASE_DIR / "exports" / "table2_temperature_gradients.csv"
table2.to_csv(table2_path, index=False, float_format='%.4f')
print(table2.to_string(index=False, float_format='%.2f'))
print(f"Saved {table2_path}")


# ---------------------------------------------------------
# Plotting Configuration
# ---------------------------------------------------------
sns.set_theme(context="paper", font_scale=1.8, style="whitegrid")
plt.rcParams.update({
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def plot_left_panel(ax):
    """Plot Left Panel: Relative density-amplitude growth vs. altitude."""
    # Curves
    ax.plot(tian_growth, tian_altitudes, ls='--', color=COLOR_1D_CART, linewidth=3, label='1D Cartesian')
    ax.plot(SPHERICAL_1D_GROW, SPHERICAL_1D_ALTITUDES, ls='-', color=COLOR_1D_SPH, linewidth=3, label='1D Spherical Model')
    ax.plot(avg_growth_middle, altitudes, ls='-', color=COLOR_2D_CENTER, linewidth=3.2, label='2D Spherical Center')
    ax.plot(avg_growth_outer, altitudes, ls='-', color=COLOR_2D_EDGES, linewidth=3.2, label='2D Spherical Edges')

    # Nominal exobase horizontal line
    ax.axhline(EXOBASE_ALT, color=COLOR_EXOBASE, linestyle=':', linewidth=2.2, alpha=0.9, label=f'Exobase (~{int(EXOBASE_ALT)} km)', zorder=2)

    # Styling & limits
    ax.set_ylabel("Altitude (km)", fontsize=20, fontweight='medium')
    ax.set_xlabel(r"$n(z)$ Amplitude Peak / $n(\mathrm{150\ km})$ Amplitude Peak", fontsize=18, fontweight='medium')
    ax.set_xscale('log')
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(149.0, 330.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # Legend on the left panel (lower right: the top-left corner carries the panel letter)
    ax.legend(loc='lower right', frameon=True, framealpha=0.92, facecolor='white', edgecolor='#cccccc', fontsize=14)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, **PANEL_LABEL_KW)


def plot_right_panel(ax):
    """Plot Right Panel: Associated temperature increase vs. altitude."""
    # 1D Cartesian reference line (9.4 K/H)
    ax.plot(line_94_y + line_94_intercept, x_alt, color=COLOR_1D_CART, linestyle='--', linewidth=3, label='1D Cartesian')

    # 1D Spherical (AW1) scatter + fit
    ax.scatter(aw1_data['full_temps'], aw1_data['full_alts'], color=COLOR_1D_SPH, alpha=0.35, edgecolors='k', linewidths=0.5, s=40, zorder=3)
    ax.plot(aw1_data['y_pred'], aw1_data['x_pred'], color=COLOR_1D_SPH, linewidth=3.5, label='1D Spherical', zorder=4)

    # 2D Spherical Center scatter + linear fit
    ax.scatter(avg_center_temp, x_alt, color=COLOR_2D_CENTER, alpha=0.35, edgecolors='k', linewidths=0.5, s=45, zorder=3)
    ax.plot(slope1_center * x_alt + intercept1_center, x_alt, color=COLOR_2D_CENTER, linestyle='-', linewidth=3.5, label='2D Spherical Center', zorder=4)

    # 2D Spherical Edges scatter + linear fit
    ax.scatter(avg_edge_temp, x_alt, color=COLOR_2D_EDGES, alpha=0.35, edgecolors='k', linewidths=0.5, s=45, zorder=3)
    ax.plot(slope2_edges * x_alt + intercept2_edges, x_alt, color=COLOR_2D_EDGES, linestyle='-', linewidth=3.5, label='2D Spherical Edges', zorder=4)

    # Nominal exobase horizontal line
    ax.axhline(EXOBASE_ALT, color=COLOR_EXOBASE, linestyle=':', linewidth=2.2, alpha=0.9, label=f'Exobase (~{int(EXOBASE_ALT)} km)', zorder=2)

    # Styling & limits
    ax.set_ylabel("Altitude (km)", fontsize=20, fontweight='medium')
    ax.set_xlabel(r"$\langle T \rangle - T_0$ (K)", fontsize=20, fontweight='medium')
    ax.set_ylim(149.0, 330.0)

    # Legend (lower right to avoid overlap with high-altitude data)
    ax.legend(loc='lower right', frameon=True, framealpha=0.92, facecolor='white', edgecolor='#cccccc', fontsize=18)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, **PANEL_LABEL_KW)


# ---------------------------------------------------------
# 1. Combined 2-Panel Figure 3
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), sharey=True)

plot_left_panel(ax1)
plot_right_panel(ax2)

# Adjust right panel y-axis label since shared
ax2.set_ylabel("")

plt.tight_layout()
combined_path = BASE_DIR / "Figure_3_combined.png"
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"\nSaved combined Figure 3 to: {combined_path}")
plt.close(fig)

# ---------------------------------------------------------
# 2. Individual Panels (Matching original standalone dimensions)
# ---------------------------------------------------------
# Left Panel
fig_l, ax_l = plt.subplots(figsize=(10, 8))
plot_left_panel(ax_l)
plt.tight_layout()
left_path = BASE_DIR / "Figure_3_left_panel.png"
plt.savefig(left_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 3 Left Panel to: {left_path}")
plt.close(fig_l)

# Right Panel
fig_r, ax_r = plt.subplots(figsize=(10, 8))
plot_right_panel(ax_r)
plt.tight_layout()
right_path = BASE_DIR / "Figure_3_right_panel.png"
plt.savefig(right_path, dpi=300, bbox_inches='tight')
print(f"Saved Figure 3 Right Panel to: {right_path}")
plt.close(fig_r)

print("\nDone!")
