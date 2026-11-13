"""Build the run file (run tier, docs/CONVENTIONS.md) of registered runs with the dsmc_netcdf_webapp
builder (>= 1.2.0), then their cards.

    python -m dsmclib.runfile --campaign h2_ubc_rtop                # every finished run; skip complete existing files
    python -m dsmclib.runfile --runs OH2_A OH2_B_r1 --overwrite     # rebuild (run file and card)
    python -m dsmclib.runfile --campaign oh2_escape_rerun --no-card

Writes <Results>/<alias>_run.nc (the builder only writes inside the run's Results folder) and,
unless --no-card, the simulation card (dsmclib.card) next to it and in outputs/cards/.
Logs: outputs/<campaign>/logs/compile_<alias>.log. Runs are compiled concurrently; each builder
process also parses its timesteps in parallel.
"""

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import registry
from .ncrun import RUN_SUFFIX
from .paths import REFACTORED_ROOT, output_dir

_NC_LOCK = threading.Lock()  # HDF5 in env dsmc is not thread-safe; compile_run runs in a pool
BUILDER = REFACTORED_ROOT.parent.parent / "dsmc_netcdf_webapp" / "dsmc_netcdf_builder.py"


def run_file_path(alias: str) -> Path:
    return registry.run(alias).path / f"{alias}{RUN_SUFFIX}"


def run_finished(alias: str) -> bool:
    """The solver writes COMPLETE.ascii in the run folder (parent of Results) when it exits normally."""
    return (registry.run(alias).path.parent / "COMPLETE.ascii").is_file()


def run_file_complete(path: Path) -> bool:
    """run_complete attribute the builder (>= 1.2.0) wrote: False for a snapshot taken mid-run."""
    import netCDF4

    with _NC_LOCK, netCDF4.Dataset(path) as ds:
        return getattr(ds, "run_complete", "false") == "true"


def compile_run(alias: str, threads: int = 4, overwrite: bool = False, card: bool = True) -> tuple:
    """Build <Results>/<alias>_run.nc; returns (one-line status, built). Raises on builder failure.

    Unfinished runs (no COMPLETE.ascii) are skipped. An existing run file built mid-run
    (run_complete=false) is stale and is rebuilt, with its card, even without overwrite."""
    run = registry.run(alias)
    out = run.path / f"{alias}{RUN_SUFFIX}"
    if not run_finished(alias):
        return f"{alias}: run not finished (no COMPLETE.ascii), skipped", False
    stale = out.exists() and not run_file_complete(out)
    if out.exists() and not overwrite and not stale:
        msg, built = f"{alias}: run file exists, skipped ({out})", False
    else:
        log = output_dir(run.campaign, "logs") / f"compile_{alias}.log"
        cmd = [sys.executable, "-B", str(BUILDER), "--results-dir", str(run.path), "--output-file", str(out),
               "--alias", alias, "--strict", "--threads", str(threads)] + (["--overwrite"] if out.exists() else [])
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"{alias}: builder failed (exit {rc}); see {log}")
        msg = f"{alias}: {'rebuilt stale mid-run' if stale else 'wrote'} {out} ({out.stat().st_size / 1e6:.1f} MB)"
        built = True
    if card:
        from .card import make_card  # local import: matplotlib/tectonic only when cards are wanted

        res = make_card(alias, overwrite=overwrite or built)
        msg += f"; card {res['status']}"
    return msg, built


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--campaign", help="compile every run of this registry campaign")
    sel.add_argument("--runs", nargs="+", help="registry aliases")
    parser.add_argument("--jobs", type=int, default=4, help="runs compiled concurrently")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 4),
                        help="parser threads per builder process")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing run files and cards")
    parser.add_argument("--no-card", action="store_true", help="do not render simulation cards")
    args = parser.parse_args(argv)
    aliases = args.runs or [r.alias for r in registry.campaign(args.campaign)]
    failed, built = [], set()

    def one(alias):
        try:
            msg, did_build = compile_run(alias, args.threads, args.overwrite, card=False)
            if did_build:
                built.add(alias)
            return msg
        except Exception as exc:  # report every run, fail at the end
            failed.append(alias)
            return f"{alias}: FAILED: {exc}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(one, aliases):
            print(msg, flush=True)
    # Cards after all builds: one process renders them (xarray/netCDF in threads is not safe in env dsmc).
    if not args.no_card:
        from .card import make_card

        for alias in aliases:
            if alias in failed or not (alias in built or run_finished(alias)):
                continue
            try:
                res = make_card(alias, overwrite=args.overwrite or alias in built)
                print(f"{alias}: card {res['status']} ({res.get('pdf', '')})", flush=True)
            except Exception as exc:
                failed.append(alias)
                print(f"{alias}: card FAILED: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
