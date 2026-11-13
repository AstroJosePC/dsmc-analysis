#!/usr/bin/env python3
"""Compute pre-forcing vertical profiles of lambda(z), c_bar(z), nu(z), D(z), and delta(z) for S6 (Route #2).

    python campaigns/s1_s6_azimuthal/compute_transport_profiles.py

Reads S6 macroscopic cubes over the pre-forcing steady-state window (last 17 outputs before
wave forcing turns on, t = 4907.8-5012.3 s, averaged over all 1024 azimuthal cells) and
overlays the lock-in lateral attenuation length L_att(z) from
outputs/s1_s6_azimuthal/cache/lockin_attenuation_S6_azimean_preforcing.npz.

Writes:
  * outputs/s1_s6_azimuthal/cache/preforcing_transport_S6_azimean_preforcing.npz (mirrored to analysis/)
  * outputs/s1_s6_azimuthal/tables/table_s6_preforcing_transport_azimean_preforcing.csv
  * outputs/s1_s6_azimuthal/figures/figure_s6_preforcing_transport_stokes_azimean_preforcing.png
"""

from __future__ import annotations

import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import numpy as np

from compute_attenuation_cache import cache_path as lockin_cache_path
from dsmclib import attenuation as att
from dsmclib import registry
from dsmclib.paths import mirror_legacy_cache, output_dir
from dsmclib.run_data import DSMCRun


def transport_cache_path(alias: str, background: str = "azimean_preforcing"):
    safe_bg = re.sub(r"[^A-Za-z0-9_-]+", "_", background)
    return output_dir("s1_s6_azimuthal", "cache") / f"preforcing_transport_{alias}_{safe_bg}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="S6", help="Registry alias (default: S6)")
    ap.add_argument("--background", default="azimean_preforcing")
    ap.add_argument("--snapshots", type=int, default=None,
                    help="Number of pre-forcing snapshots before wave_start_time (default: PREFORCING_SNAPSHOTS=17)")
    ap.add_argument("--z-max", type=float, default=550.0,
                    help="Upper altitude limit [km] for the diagnostic figure (default: 550.0)")
    ap.add_argument("--rebuild", action="store_true", help="Recompute even if cache exists")
    args = ap.parse_args()

    alias = args.run
    lck_path = lockin_cache_path(alias, args.background)
    reduction = None
    table_s3_rows = None
    if lck_path.exists():
        with np.load(lck_path) as f:
            reduction = {k: f[k] for k in f.files}
        reduction["alias"] = alias
        table_s3_rows = att.attenuation_table(reduction, (150.0, 230.0, 320.0))

    cpath = transport_cache_path(alias, args.background)
    if cpath.exists() and not args.rebuild:
        print(f"Loading cached transport profiles: {cpath}")
        with np.load(cpath) as f:
            profiles = {k: f[k] for k in f.files}
    else:
        run_info = registry.run(alias)
        run = DSMCRun(run_info.path, alias=alias)
        profiles = att.compute_preforcing_transport_profiles(
            run,
            n_snapshots=args.snapshots,
            reduction=reduction,
        )
        np.savez_compressed(cpath, **profiles)
        mirror_legacy_cache(cpath)
        print(f"Saved {cpath}")
        run.release_macro()

    # Export CSV table of 1D vertical profiles
    table_path = output_dir("s1_s6_azimuthal", "tables") / f"table_{alias.lower()}_preforcing_transport_{args.background}.csv"
    columns = [
        "altitude_km",
        "mean_free_path_m",
        "thermal_speed_ms",
        "nu_m2s",
        "D_m2s",
        "delta_km",
        "scale_height_m",
        "knudsen_number",
        "nu_vhs_m2s",
        "D_vhs_m2s",
        "delta_vhs_km",
        "L_att_km",
        "sigma_L_total_km",
        "ratio_L_att_to_delta",
        "ratio_L_att_to_delta_vhs",
        "density_m3",
        "temp_trans_K",
        "temp_trans_pw_K",
        "thermal_speed_pw_ms",
        "mean_free_path_from_bg_m",
        "nu_from_bg_m2s",
        "D_from_bg_m2s",
        "delta_from_bg_km",
        "valid_cell_fraction",
    ]
    n_alt = int(np.asarray(profiles["altitude_km"]).size)
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in range(n_alt):
            writer.writerow({col: float(np.asarray(profiles[col])[i]) for col in columns})
    print(f"Saved {table_path}")

    # Generate diagnostic figure
    fig = att.plot_preforcing_transport_profiles(profiles, table_s3_rows=table_s3_rows, z_max_km=args.z_max)
    fig_path = output_dir("s1_s6_azimuthal", "figures") / f"figure_{alias.lower()}_preforcing_transport_stokes_{args.background}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    # Print key altitude summary
    z = np.asarray(profiles["altitude_km"], dtype=float)
    print(f"\nPre-forcing window: outputs {int(profiles['preforcing_start_index'])}..{int(profiles['preforcing_end_index'])} "
          f"(n={int(profiles['preforcing_n_snapshots'])}, t={float(profiles['preforcing_time_start_s']):.2f}.."
          f"{float(profiles['preforcing_time_end_s']):.2f} s; forcing starts at index {int(profiles['forcing_start_index'])}, "
          f"t={float(profiles['wave_start_time_s']):.2f} s)")
    print(f"Configured wave period: P = {float(profiles['wave_period_s']):.4f} s (omega = {float(profiles['omega_freq_rad_s']):.5f} rad/s)")
    print(f"Exact exobase crossing (lambda = H, Kn = 1): z = {float(profiles['exobase_altitude_km']):.2f} km\n")

    header = (f"{'Target':>12} {'z_cell [km]':>11} {'lambda [m]':>12} {'c_bar [m/s]':>11} "
              f"{'Kn':>8} {'nu_CE [m2/s]':>13} {'D_CE [m2/s]':>13} {'delta [km]':>10} "
              f"{'delta_VHS [km]':>14} {'L_att [km]':>15}")
    print(header)
    print("-" * len(header))
    key_targets = [
        ("150 km", 150.0),
        ("230 km (exo)", 230.0),
        ("Kn=1 (exact)", float(profiles["exobase_altitude_km"])),
        ("320 km (top)", 320.0),
        ("Grid top", float(z.max())),
    ]
    for label, z_t in key_targets:
        zi = int(np.argmin(np.abs(z - z_t)))
        l_val = float(np.asarray(profiles["L_att_km"])[zi])
        l_err = float(np.asarray(profiles["sigma_L_total_km"])[zi])
        l_str = f"{l_val:6.2f} +- {l_err:4.2f}" if np.isfinite(l_val) else "      --       "
        print(
            f"{label:>12} {z[zi]:11.2f} {float(np.asarray(profiles['mean_free_path_m'])[zi]):12.4e} "
            f"{float(np.asarray(profiles['thermal_speed_ms'])[zi]):11.2f} "
            f"{float(np.asarray(profiles['knudsen_number'])[zi]):8.3f} "
            f"{float(np.asarray(profiles['nu_m2s'])[zi]):13.4e} "
            f"{float(np.asarray(profiles['D_m2s'])[zi]):13.4e} "
            f"{float(np.asarray(profiles['delta_km'])[zi]):10.2f} "
            f"{float(np.asarray(profiles['delta_vhs_km'])[zi]):14.2f} "
            f"{l_str:>15}"
        )


if __name__ == "__main__":
    main()
