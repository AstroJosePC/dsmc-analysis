#!/usr/bin/env python3
"""SI Table S3 (L_att with bootstrap + cycle-to-cycle uncertainty) and Figures S5/S6 from the lock-in caches.

    python campaigns/s1_s6_azimuthal/attenuation_table.py                 # table + Figure S5
    python campaigns/s1_s6_azimuthal/attenuation_table.py --time-windows  # also Figure S6 (loads raw S6)

Reads outputs/s1_s6_azimuthal/cache/lockin_attenuation_<run>_<background>.npz (build them with
compute_attenuation_cache.py). Writes tables/table_s3_lockin_attenuation_<background>.csv and
figures/figure_s5_two_branch_<background>.png, figure_s6_time_windows_<background>.png.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dsmclib import attenuation as att
from dsmclib import registry
from dsmclib.paths import output_dir
from dsmclib.run_data import DSMCRun
from compute_attenuation_cache import cache_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["S2", "S3", "S4", "S5", "S6"])
    ap.add_argument("--background", default="azimean_preforcing")
    ap.add_argument("--altitudes", type=float, nargs="+", default=[150.0, 230.0, 320.0])
    ap.add_argument("--snr-cut", type=float, default=att.DEFAULT_SNR_CUT)
    ap.add_argument("--max-dist", type=float, default=400.0)
    ap.add_argument("--n-boot", type=int, default=att.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=att.DEFAULT_SEED)
    ap.add_argument("--time-windows", action="store_true", help="Figure S6: loads the raw S6 density cube")
    args = ap.parse_args()

    reductions, rows = {}, []
    for alias in args.runs:
        with np.load(cache_path(alias, args.background)) as f:
            red = {key: f[key] for key in f.files}
        red["alias"] = alias
        reductions[alias] = red
        rows += att.attenuation_table(red, args.altitudes, snr_cut=args.snr_cut, max_distance_km=args.max_dist,
                                      n_boot=args.n_boot, seed=args.seed)

    table = output_dir("s1_s6_azimuthal", "tables") / f"table_s3_lockin_attenuation_{args.background}.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {table}")
    print(f"{'run':4} {'z km':>6} {'L':>6} {'boot':>5} {'cyc':>5} {'tot':>5} {'n':>3} {'ext':>3} {'R^2':>6} "
          f"{'sigma_c':>8} {'A_c(0)':>7} {'L(+1)':>6} {'drift%':>6}")
    for r in rows:
        print(f"{r['run']:4} {r['altitude_km']:6.1f} {r['L_att_km']:6.1f} {r['sigma_L_boot_km']:5.1f} "
              f"{r['sigma_L_cycle_km']:5.1f} {r['sigma_L_total_km']:5.1f} {r['n_points']:3d} {r['n_exterior']:3d} "
              f"{r['r_squared']:6.4f} {r['sigma_c']:8.5f} {r['A_edge_interp']:7.4f} "
              f"{r['L_start_plus1_km']:6.1f} {r['start_drift_pct']:6.1f}")

    figures = output_dir("s1_s6_azimuthal", "figures")
    if "S6" in reductions:
        fig = att.plot_two_branch(reductions["S6"], args.altitudes, rows=rows, snr_cut=args.snr_cut,
                                  max_distance_km=args.max_dist)
        path = figures / f"figure_s5_two_branch_{args.background}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    if args.time_windows:
        run = DSMCRun(registry.run("S6").path, alias="S6")
        run.load_parameters()
        run.load_data()
        windows = att.lockin_time_windows(run, reductions["S6"], background_method=args.background)
        run.release_macro()
        for w in windows:
            print(f"S6 {w['altitude_km']:.1f} km, s = {w['s_km']:.1f} km (pairs k = {w['k_pairs']}): "
                  f"|Z| = {abs(w['Z']):.5f} = {abs(w['Z']) / w['sigma_c']:.2f} sigma_c (sigma_c = {w['sigma_c']:.5f})")
        fig = att.plot_lockin_time_windows(windows)
        path = figures / f"figure_s6_time_windows_{args.background}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
