#!/usr/bin/env python3
"""Compute symmetric lock-in attenuation reductions for selected S1-S6 runs."""

from __future__ import annotations

import argparse
import re

import numpy as np

from dsmclib import registry
from dsmclib.attenuation import compute_symmetric_lockin_reduction
from dsmclib.paths import mirror_legacy_cache, output_dir
from dsmclib.run_data import DSMCRun


def cache_path(alias: str, background: str):
    safe_background = re.sub(r"[^A-Za-z0-9_-]+", "_", background)
    return output_dir("s1_s6_azimuthal", "cache") / f"lockin_attenuation_{alias}_{safe_background}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["S2", "S3", "S4", "S5", "S6"])
    parser.add_argument("--background", default="azimean_preforcing",
                        choices=["mad_column0", "azimean_snapshot", "azimean_preforcing"])
    parser.add_argument("--rebuild", action="store_true", help="Recompute even if the cache exists")
    args = parser.parse_args()

    for alias in args.runs:
        path = cache_path(alias, args.background)
        if path.exists() and not args.rebuild:
            print(f"Using cached reduction: {path}")
            continue
        run_info = registry.run(alias)
        run = DSMCRun(run_info.path, alias=alias)
        run.load_parameters()
        run.load_data()
        reduction = compute_symmetric_lockin_reduction(run, background_method=args.background)
        reduction["alias"] = np.asarray(alias)
        np.savez_compressed(path, **reduction)
        mirror_legacy_cache(path)
        print(f"Saved {path}")
        run.release_macro()


if __name__ == "__main__":
    main()
