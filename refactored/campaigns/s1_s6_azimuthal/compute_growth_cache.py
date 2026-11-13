"""Batch: density growth for every forced column of S1..S6 -> outputs/s1_s6_azimuthal/cache/.

    python campaigns/s1_s6_azimuthal/compute_growth_cache.py                 # S1..S6, skip cached
    python campaigns/s1_s6_azimuthal/compute_growth_cache.py --runs S4_2025 --recompute
    python campaigns/s1_s6_azimuthal/compute_growth_cache.py --background azimean_preforcing --altitude-cells nearest   # refactored recipe

Reads multi-GB fields over the MyBook mount; run it in the background with a log. The
notebook density_growth.ipynb then only reads the caches.
"""

import argparse

from dsmclib import registry
from dsmclib.background import BACKGROUND_METHODS
from dsmclib.growth import cached_growth
from dsmclib.run_data import ALTITUDE_CELL_RULES

CAMPAIGN = "s1_s6_azimuthal"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=[r.alias for r in registry.campaign(CAMPAIGN)])
    parser.add_argument("--alt-min", type=float, default=150.0, help="base altitude [km] (growth = 1 there)")
    parser.add_argument("--alt-max", type=float, default=320.0)
    parser.add_argument("--static", action="store_true", help="fixed peak-detection settings at every altitude")
    parser.add_argument("--background", default="mad_column0", choices=BACKGROUND_METHODS,
                        help="pre-forcing background (azimean_snapshot = 2024 pipeline; see dsmclib.background)")
    parser.add_argument("--altitude-cells", default="first_above", choices=ALTITUDE_CELL_RULES,
                        help="cells for --alt-min/--alt-max: first above (root) or nearest (2024 / refactored S1-S6)")
    parser.add_argument("--recompute", action="store_true", help="overwrite existing caches")
    args = parser.parse_args()

    for alias in args.runs:
        print(f"===== {alias} =====", flush=True)
        cached_growth(alias, recompute=args.recompute, alt_range=(args.alt_min, args.alt_max),
                      use_dynamic_peak_detection=not args.static, background=args.background,
                      altitude_cells=args.altitude_cells)
