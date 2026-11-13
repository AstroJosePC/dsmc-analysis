"""Runs by name.

    from dsmclib import registry
    registry.run("S4").path          # /Volumes/MyBook/ArchiveSims/Results_20240722_034301.982
    registry.campaign("s1_s6_azimuthal")
    registry.open_run("S6")          # loaded DSMCRun

Backed by refactored/registry/simulation_metadata.db: a copy of the root DB with two added
tables, campaigns and run_aliases (built by registry/build_registry.py). Paths are rebuilt
from results_base on whatever machine this runs on; the DB's stored data_path is ignored.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .paths import REGISTRY_DB, archive_root


@dataclass
class Run:
    alias: str
    campaign: str
    results_base: str
    role: str
    notes: str
    n_azi_cell: Optional[int] = None
    n_rad_cell: Optional[int] = None
    num_periods: Optional[int] = None
    rmin: Optional[float] = None
    rmax: Optional[float] = None
    ubc: Optional[str] = None

    @property
    def path(self) -> Path:
        """Run folder. Relative results_base resolves under the ArchiveSims mount; an absolute
        results_base (runs not yet archived, e.g. ~/MarsAtmos-OH2-rerun/runs/<run>/Results) is used as-is."""
        base = Path(self.results_base)
        return base if base.is_absolute() else archive_root() / base

    def __repr__(self) -> str:
        return (f"Run({self.alias}: {self.results_base}, campaign={self.campaign}, role={self.role}, "
                f"n_azi={self.n_azi_cell}, periods={self.num_periods})")


_QUERY = """
    SELECT a.alias, a.campaign, a.results_base, a.role, a.notes,
           r.n_azi_cell, r.n_rad_cell, r.num_periods, r.rmin, r.rmax, r.UBC
    FROM run_aliases a
    LEFT JOIN simulation_runs r ON r.results_base = a.results_base
"""


def _connect() -> sqlite3.Connection:
    if not REGISTRY_DB.exists():
        raise FileNotFoundError(f"{REGISTRY_DB} missing; run `python registry/build_registry.py` first")
    return sqlite3.connect(f"file:{REGISTRY_DB}?mode=ro", uri=True)


def _rows(where: str = "", args=()) -> List[Run]:
    with _connect() as conn:
        rows = conn.execute(_QUERY + where + " ORDER BY a.campaign, a.sort_order", args).fetchall()
    return [Run(*row) for row in rows]


def run(alias: str) -> Run:
    rows = _rows("WHERE a.alias = ?", (alias,))
    if not rows:
        raise KeyError(f"Unknown run alias {alias!r}; known: {', '.join(r.alias for r in all_runs())}")
    return rows[0]


def campaign(name: str, include_alternates: bool = False) -> List[Run]:
    """Runs of a campaign in their natural order (S1..S6). Alternate re-runs only on request."""
    runs = _rows("WHERE a.campaign = ?", (name,))
    if not runs:
        raise KeyError(f"Unknown campaign {name!r}; known: {', '.join(campaigns())}")
    return [r for r in runs if include_alternates or r.role != "alternate"]


def campaigns() -> List[str]:
    with _connect() as conn:
        return [name for (name,) in conn.execute("SELECT name FROM campaigns ORDER BY name")]


def all_runs() -> List[Run]:
    return _rows()


def open_run(alias: str):
    """DSMCRun for the alias with parameters and fields loaded."""
    from .run_data import DSMCRun

    return DSMCRun(run(alias).path, alias=alias).load()
