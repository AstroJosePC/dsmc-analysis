"""Figures for growth results. Functions draw on an Axes and return it; saving is up to the caller.

Call use_style() once at the top of a notebook for the manuscript look (it is no longer
applied on import as in the root loaders).
"""

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .constants import (
    SPHERICAL_1D_ALTITUDES,
    SPHERICAL_1D_GROW,
    TIAN_ALTITUDES,
    TIAN_FIG3_ALTITUDES,
    TIAN_FIG3_GROW,
    TIAN_GROW,
)
from .growth import growth_curve, growth_matrix
from .heating import H0_KM, centre_and_edge_columns, group_heating_slopes, heating_slope
from .resolution import ALTITUDE, GROWTH, fit_band

FIG3_COLORS = {
    "1d_cart": "r",
    "1d_sph": "#a66a00",
    "2d_center": "#337538",
    "2d_edges": "#2e2585",
    "exobase": "#555555",
}
EXOBASE_ALT_KM = 230.0
T23_SLOPE_K_H = 9.4


def use_style(font_scale: float = 2.3) -> None:
    """Seaborn 'paper' + whitegrid theme used by the root scripts."""
    sns.set_theme(context="paper", font_scale=font_scale, style="whitegrid")


def plot_reference_growth(ax, tian: bool = True, spherical_1d: bool = True):
    """Overlay the Tian et al. and spherical-1D reference growth curves."""
    if tian:
        ax.plot(TIAN_ALTITUDES, TIAN_GROW, "k--", linewidth=2, label="Tian et al. (2023)")
    if spherical_1d:
        ax.plot(SPHERICAL_1D_ALTITUDES, SPHERICAL_1D_GROW, "k:", linewidth=2, label="Spherical 1D")
    return ax


def plot_growth_curves(
    growth_data: Dict[str, Any],
    ax=None,
    n_last_peaks: int = 5,
    references: bool = True,
    log_y: bool = True,
):
    """Growth vs altitude, one viridis line per azimuth column (from optimized_loader.plot_amplitude_growths)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    params = growth_data["params"]
    columns = growth_data["azimuthal_indices"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))
    valid = []
    for color, azi in zip(colors, columns):
        try:
            altitudes, curve = growth_curve(growth_data, azi, n_last_peaks)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        ax.plot(altitudes, curve, color=color, linewidth=2, marker="o", markersize=4, label=f"Azimuthal Index {azi}")
        valid.append(curve)

    if references:
        plot_reference_growth(ax)

    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Density Amplitude Growth")
    ax.set_title(
        f"{growth_data.get('alias', '')} density amplitude growth\n"
        f"Wave frequency: {params.omega_freq:.4f} rad/s, period: {params.wave_period:.1f} s"
    )
    ax.grid(True, alpha=0.3)
    if log_y and valid and np.all(np.array(valid) > 0):
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
    return ax


def plot_growth_heatmap(growth_data: Dict[str, Any], ax=None, n_last_peaks: int = 5, contour_levels=(2, 3, 4, 5)):
    """Growth over (azimuth, altitude) with contour lines (from optimized_loader.plot_amplitude_growth_heatmap)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    params = growth_data["params"]
    altitudes, columns, matrix = growth_matrix(growth_data, n_last_peaks)
    C = np.ma.masked_invalid(matrix)

    # the cells the growth was computed on (rule "first_above" or "nearest", see growth.compute_density_growth)
    first = next(r for r in growth_data["growth_results"].values() if r is not None)
    cell_min_idx, cell_max_idx = first["cell_min_idx"], first["cell_max_idx"]
    # Same edge convention as the root heatmap: cell centres as y edges
    azi_edges = params.azi_edges[columns[0]: columns[-1] + 2]
    radial_edges = params.radial_cells[cell_min_idx: cell_max_idx + 2]
    azi_mids = params.mid_azi[columns]
    radial_mids = params.radial_cells[cell_min_idx: cell_max_idx + 1]

    pcm = ax.pcolormesh(azi_edges, radial_edges, C, cmap="viridis", shading="auto",
                        vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    cs = ax.contour(azi_mids, radial_mids, C, levels=list(contour_levels), colors="white", linewidths=1.5, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=10, fmt="%d")

    ax.set_xlabel("Azimuthal Position (degrees)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(f"{growth_data.get('alias', '')} density amplitude growth")
    ax.figure.colorbar(pcm, ax=ax, label="Density Amplitude Growth")
    ax.grid(True, alpha=0.3)
    return ax


def plot_resolution_fit(ax, gmax, popt, pcov, dist: str = "t", level: float = 0.95,
                        color: str = "r", point_color: Optional[str] = None, label: str = "",
                        references: Optional[Dict[str, float]] = None):
    """Fig 7: G_max vs N_phi, Eq. 10 fit with its band, optional horizontal reference lines.

    references maps a legend label to a growth value, e.g. {"1D Cartesian (T23)": 6.14};
    call twice on one Axes (references only once) to overlay two point sets.
    """
    styles = [dict(color="k", ls="--", lw=2.5), dict(color="green", ls="--", lw=2.5)]
    for (ref_label, value), style in zip((references or {}).items(), styles):
        ax.axhline(y=value, label=ref_label, **style)

    ax.scatter(gmax["N_phi"], gmax["G_max"], s=90, color=point_color, label=f"{label} $G_{{max}}$".strip(), zorder=5)
    x_fit = np.linspace(gmax["N_phi"].min(), gmax["N_phi"].max(), 200)
    y_fit, y_lo, y_hi = fit_band(x_fit, popt, pcov, level, dist, n_points=len(gmax))
    ax.plot(x_fit, y_fit, color=color, ls="--", label=rf"{label} fit: $A - B/N_\phi$".strip())
    band = f"{int(level * 100)}% CI ({'t, dof=' + str(len(gmax) - 2) if dist == 't' else 'normal'})"
    ax.fill_between(x_fit, y_lo, y_hi, color=color, alpha=0.2, label=f"{label} {band}".strip())

    ax.set(xlabel=r"Number of Azi Cells $N_\phi$", ylabel=r"$A_\rho(r=320\,\mathrm{km}) / A_\rho(r=150\,\mathrm{km})$")
    return ax


def plot_resolution_fit_manuscript(gmax, popt, pcov, t23: float, p26: float, dist: str = "t",
                                   level: float = 0.95, figsize=(10, 6.92)):
    """Fig 7 in the manuscript's (rev4) look: plain matplotlib, boxed axes, legend lower right.

    Draws without the seaborn theme, whatever use_style() set. Returns the Figure.
    """
    rc = {"font.family": "DejaVu Sans", "font.size": 17, "axes.edgecolor": "black", "axes.linewidth": 1.3,
          "axes.facecolor": "white", "figure.facecolor": "white", "xtick.direction": "out",
          "ytick.direction": "out", "xtick.major.size": 5, "ytick.major.size": 5,
          "xtick.color": "black", "ytick.color": "black", "axes.labelcolor": "black", "text.color": "black"}
    with plt.style.context("default"), plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        x_fit = np.linspace(gmax["N_phi"].min(), gmax["N_phi"].max(), 200)
        y_fit, y_lo, y_hi = fit_band(x_fit, popt, pcov, level, dist, n_points=len(gmax))
        ax.scatter(gmax["N_phi"], gmax["G_max"], s=200, color="#1f77b4", label="2D growth maxima", zorder=5)
        ax.plot(x_fit, y_fit, color="red", ls="--", lw=2.5, label=r"Empirical fit: $A - B/N_\phi$")
        ax.fill_between(x_fit, y_lo, y_hi, color="red", alpha=0.17, label=f"Conditional {int(level * 100)}% fit band")
        ax.axhline(t23, color="black", ls="--", lw=2.5, label="1D Cartesian (T23)")
        ax.axhline(p26, color="green", ls="--", lw=2.5, label="1D radial (P26)")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"Number of azimuthal cells $N_\phi$")
        ax.set_ylabel(r"$A_\rho(\sim320\,\mathrm{km}) / A_\rho(\sim150\,\mathrm{km})$")
        ax.legend(loc="lower right", fontsize=15, framealpha=1, edgecolor="0.7")
        fig.tight_layout()
    return fig


def plot_figure3_left_panel(
    ax,
    growth_summary: Dict[str, Any],
    exobase_alt_km: float = EXOBASE_ALT_KM,
    panel_label: Optional[str] = "(a)",
):
    """Fig 3(a): Centerline and edge relative density-amplitude growth vs altitude."""
    altitudes = growth_summary["altitudes"]
    ax.plot(TIAN_FIG3_GROW, TIAN_FIG3_ALTITUDES, ls="--", color=FIG3_COLORS["1d_cart"],
            linewidth=3, label="1D Cartesian")
    ax.plot(SPHERICAL_1D_GROW, SPHERICAL_1D_ALTITUDES, ls="-", color=FIG3_COLORS["1d_sph"],
            linewidth=3, label="1D Spherical Model")
    ax.plot(growth_summary["center"], altitudes, ls="-", color=FIG3_COLORS["2d_center"],
            linewidth=3.2, label="2D Spherical Center")
    ax.plot(growth_summary["edge"], altitudes, ls="-", color=FIG3_COLORS["2d_edges"],
            linewidth=3.2, label="2D Spherical Edges")

    ax.axhline(exobase_alt_km, color=FIG3_COLORS["exobase"], linestyle=":", linewidth=2.2,
               alpha=0.9, label=f"Exobase (~{int(exobase_alt_km)} km)", zorder=2)

    ax.set_ylabel("Altitude (km)", fontsize=20)
    ax.set_xlabel(r"$n(z)$ Amplitude Peak / $n(\mathrm{150\ km})$ Amplitude Peak", fontsize=18)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(149.0, 330.0)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white",
              edgecolor="#cccccc", fontsize=14)
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")
    return ax


def plot_figure3_right_panel(
    ax,
    temp_s6: Dict[str, Any],
    temp_1d: Dict[str, Any],
    h_km: float = H0_KM,
    t23_slope_K_H: float = T23_SLOPE_K_H,
    exobase_alt_km: float = EXOBASE_ALT_KM,
    panel_label: Optional[str] = "(b)",
):
    """Fig 3(b): Cycle-averaged temperature perturbation <dT> and linear least-squares fits."""
    x_alt = np.asarray(temp_s6["altitudes"], dtype=float)
    pairs = centre_and_edge_columns(temp_s6["azimuthal_indices"])
    y_center = np.ma.filled(
        np.ma.mean(np.ma.vstack([temp_s6["temp_last_cycle"][c] for c in pairs["center"]]), axis=0),
        np.nan,
    )
    y_edge = np.ma.filled(
        np.ma.mean(np.ma.vstack([temp_s6["temp_last_cycle"][c] for c in pairs["edge"]]), axis=0),
        np.nan,
    )
    s6_fits = group_heating_slopes(temp_s6, h_km=h_km)
    fit_center, fit_edge = s6_fits["center"], s6_fits["edge"]

    alt_1d = np.asarray(temp_1d["altitudes"], dtype=float)
    col_1d = int(temp_1d["azimuthal_indices"][0])
    y_1d = np.ma.filled(np.ma.asarray(temp_1d["temp_last_cycle"][col_1d], dtype=float), np.nan)
    fit_1d = heating_slope(alt_1d, y_1d, h_km=h_km)
    x_pred_1d = np.linspace(fit_1d["z_min"], fit_1d["z_max"], 100)
    y_pred_1d = fit_1d["slope_K_km"] * x_pred_1d + fit_1d["intercept_K"]

    # 1D Cartesian reference line (9.4 K/H0), anchored at the mean of S6 center and edge base points
    line_t23 = t23_slope_K_H * (x_alt - x_alt[0]) / h_km + np.mean([y_center[0], y_edge[0]])
    ax.plot(line_t23, x_alt, color=FIG3_COLORS["1d_cart"], linestyle="--", linewidth=3, label="1D Cartesian")

    # 1D Spherical (1D_5p / P26) scatter + linear fit
    ax.scatter(y_1d, alt_1d, color=FIG3_COLORS["1d_sph"], alpha=0.35, edgecolors="k",
               linewidths=0.5, s=40, zorder=3)
    ax.plot(y_pred_1d, x_pred_1d, color=FIG3_COLORS["1d_sph"], linewidth=3.5,
            label="1D Spherical", zorder=4)

    # 2D Spherical Center scatter + linear fit
    ax.scatter(y_center, x_alt, color=FIG3_COLORS["2d_center"], alpha=0.35, edgecolors="k",
               linewidths=0.5, s=45, zorder=3)
    ax.plot(fit_center["slope_K_km"] * x_alt + fit_center["intercept_K"], x_alt,
            color=FIG3_COLORS["2d_center"], linestyle="-", linewidth=3.5,
            label="2D Spherical Center", zorder=4)

    # 2D Spherical Edges scatter + linear fit
    ax.scatter(y_edge, x_alt, color=FIG3_COLORS["2d_edges"], alpha=0.35, edgecolors="k",
               linewidths=0.5, s=45, zorder=3)
    ax.plot(fit_edge["slope_K_km"] * x_alt + fit_edge["intercept_K"], x_alt,
            color=FIG3_COLORS["2d_edges"], linestyle="-", linewidth=3.5,
            label="2D Spherical Edges", zorder=4)

    ax.axhline(exobase_alt_km, color=FIG3_COLORS["exobase"], linestyle=":", linewidth=2.2,
               alpha=0.9, label=f"Exobase (~{int(exobase_alt_km)} km)", zorder=2)

    ax.set_ylabel("Altitude (km)", fontsize=20)
    ax.set_xlabel(r"$\langle T \rangle - T_0$ (K)", fontsize=20)
    ax.set_ylim(149.0, 330.0)

    ax.legend(loc="lower right", frameon=True, framealpha=0.92, facecolor="white",
              edgecolor="#cccccc", fontsize=18)
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")
    return ax


def plot_figure3_manuscript(
    growth_summary: Dict[str, Any],
    temp_s6: Dict[str, Any],
    temp_1d: Dict[str, Any],
    figsize=(18, 8.5),
):
    """Combined two-panel manuscript Figure 3: (a) density-amplitude growth, (b) <dT> and heating slopes."""
    rc = {
        **sns.axes_style("whitegrid"),
        **sns.plotting_context("paper", font_scale=1.8),
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
    with plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        plot_figure3_left_panel(ax1, growth_summary, panel_label="(a)")
        plot_figure3_right_panel(ax2, temp_s6, temp_1d, panel_label="(b)")
        ax2.set_ylabel("")
        fig.tight_layout()
    return fig


def plot_growth_profile_facets(profiles, references: Sequence[Dict[str, Any]] = (), edge_simulation: str = "S6",
                               cmap: str = "YlOrRd", height: float = 4.5):
    """Fig 6: one panel per simulation, a line per forced column coloured by |distance from centre|.

    profiles: long table with 'Abs Distance (km)' (resolution.add_distance_from_centre).
    references: dicts {x, y, label, color} drawn in every panel. The edge average of
    edge_simulation (mean of its first and last forced columns) is added as a reference.
    Returns the Figure.
    """
    from matplotlib.colors import Normalize

    edge = profiles[profiles["Simulation"] == edge_simulation]
    nums = edge["Azi Num"]
    edge_avg = (edge[nums.isin([nums.min(), nums.max()])]
                .groupby(ALTITUDE)[GROWTH].mean().reset_index())
    references = list(references) + [dict(x=edge_avg[GROWTH], y=edge_avg[ALTITUDE], color="#1f77b4",
                                          label=f"Edge Avg {edge_simulation}")]

    norm = Normalize(vmin=0, vmax=profiles["Abs Distance (km)"].max())
    colormap = plt.get_cmap(cmap)
    sims = list(dict.fromkeys(profiles["Simulation"]))
    ncols = 3
    nrows = int(np.ceil(len(sims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * height * 1.1, nrows * height), squeeze=False)
    for ax, sim in zip(axes.flat, sims):
        sub = profiles[profiles["Simulation"] == sim]
        for _, col in sorted(sub.groupby("Azi Num"), key=lambda kv: kv[0]):
            ax.plot(col[GROWTH], col[ALTITUDE], lw=2.5, color=colormap(norm(col["Abs Distance (km)"].iloc[0])))
        for ref in references:
            ax.plot(ref["x"], ref["y"], lw=2.5, ls="--", color=ref["color"], label=ref["label"],
                    zorder=ref.get("zorder", 4))
        ax.set_title(f"Simulation = {sim}")
    for ax in axes.flat[len(sims):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Altitude (km)")
    for ax in axes[-1]:
        ax.set_xlabel(r"$n(z)$ / $n(150 \rm{km})$ Amplitude Peak")
    axes.flat[0].legend(fontsize=9, loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.tight_layout()
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("|Distance from Center| (km)")
    return fig


def plot_peak_diagnostics(series, time, altitudes, peak_indices, wave_end=None, ncols: int = 3,
                          panel_size=(4.2, 1.9), title: str = ""):
    """Visual check of peak detection: one panel per altitude, the perturbation time series with the
    chosen peaks as red dots (last cycle ringed).

    series: (n_time, n_alt) perturbation in the masked time window; time: (n_time,) [s];
    peak_indices: per-altitude index arrays from process_growth_num_amp(..., return_indices=True).
    Returns the Figure.
    """
    n_alt = len(altitudes)
    nrows = int(np.ceil(n_alt / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * panel_size[0], nrows * panel_size[1]),
                             sharex=True, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n_alt:
            ax.set_visible(False)
            continue
        ax.plot(time, series[:, i], lw=0.9, color="0.3")
        idx = np.asarray(peak_indices[i], dtype=int)
        if idx.size:
            ax.plot(time[idx], series[idx, i], "o", color="red", ms=4)
            ax.plot(time[idx[-1]], series[idx[-1], i], "o", mfc="none", mec="red", ms=9)
        else:
            ax.text(0.5, 0.5, "detection failed", transform=ax.transAxes, ha="center", color="red")
        if wave_end is not None:
            ax.axvline(wave_end, color="C0", ls=":", lw=1)
        ax.text(0.01, 0.95, f"{altitudes[i]:.1f} km", transform=ax.transAxes, va="top", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
