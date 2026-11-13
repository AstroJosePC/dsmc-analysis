"""Data-tier constants and the required analysis-file attributes (docs/CONVENTIONS.md).

    from dsmclib import tiers
    path = tiers.analysis_path("oh2_escape_rerun", "OH2_A")            # outputs/<campaign>/analysis/OH2_A_analysis.nc
    ds.attrs.update(tiers.analysis_attrs("OH2_A: O+H2 1-D DSMC analysis", [run.path], "compute_analysis.py"))
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from .ncrun import RUN_SUFFIX  # noqa: F401  (re-exported: <alias>_run.nc)
from .paths import output_dir

CONVENTION_VERSION = "1.0"
ANALYSIS_SUFFIX = "_analysis.nc"


def sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analysis_path(campaign: str, name: str, variant: Optional[str] = None) -> Path:
    """outputs/<campaign>/analysis/<name>[_<variant>]_analysis.nc; name is a run alias or the campaign (cross-run)."""
    stem = f"{name}_{variant}" if variant else name
    return output_dir(campaign, "analysis") / f"{stem}{ANALYSIS_SUFFIX}"


def analysis_attrs(title: str, sources: Iterable[Union[str, Path]], script: str) -> Dict[str, str]:
    """The global attributes every analysis file must carry (CONVENTIONS.md §2.2)."""
    import dsmclib

    sources = [Path(s) for s in sources]
    return {
        "Conventions": "CF-1.13",
        "data_tier": "analysis",
        "convention_version": CONVENTION_VERSION,
        "title": f"{title} (analysis)",
        "source_files": ",".join(str(s) for s in sources),
        "source_sha256": ",".join(sha256(s) for s in sources),
        "dsmclib_version": dsmclib.__version__,
        "history": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {script}",
    }
