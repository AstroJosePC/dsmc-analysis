"""Consolidate raw run outputs into one run file (<alias>_run.nc) per run, then render each run's simulation card.

    python campaigns/oh2_escape_rerun/compile_netcdf.py                 # all campaign runs, skip existing
    python campaigns/oh2_escape_rerun/compile_netcdf.py --runs OH2_A --overwrite
    python campaigns/oh2_escape_rerun/compile_netcdf.py --no-card

Thin wrapper around dsmclib.runfile (python -m dsmclib.runfile --campaign oh2_escape_rerun does the same).
Writes <Results>/<alias>_run.nc and <Results>/<alias>_card.{pdf,json} (+ outputs/cards/).
"""

import sys

from dsmclib import runfile

CAMPAIGN = "oh2_escape_rerun"

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--runs" not in argv:
        argv = ["--campaign", CAMPAIGN] + argv
    sys.exit(runfile.main(argv))
