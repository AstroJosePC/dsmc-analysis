"""Where things live: the simulation archive and the staging outputs tree."""

from pathlib import Path

# Mount points of the ArchiveSims folder on the machines this code runs on.
# The first one that exists wins.
ARCHIVE_ROOT_CANDIDATES = [
    Path("/Volumes/MyBook/ArchiveSims"),  # Mac Studio (SMB mount of AstroServer)
    Path(r"I:\ArchiveSims"),  # Windows workstation
    Path(r"Z:\ArchiveSims"),  # Windows, network drive letter
    Path("/media/astrojose/AstroBook/ArchiveSims"),  # Linux (paths stored in simulation_metadata.db)
]

REFACTORED_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REFACTORED_ROOT / "outputs"
REGISTRY_DB = REFACTORED_ROOT / "registry" / "simulation_metadata.db"
# Frozen manuscript release (read only): legacy inputs for the regression notebooks and checks
LEGACY_ZENODO = REFACTORED_ROOT.parent / "zenodo_release"


def archive_root() -> Path:
    """Return the first ArchiveSims mount that exists on this machine."""
    for candidate in ARCHIVE_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No ArchiveSims mount found. Tried: " + ", ".join(str(c) for c in ARCHIVE_ROOT_CANDIDATES)
    )


def mirror_legacy_cache(path: Path) -> Path:
    """Copy a legacy-run cache file (outputs/<campaign>/cache/x) to outputs/<campaign>/analysis/x, untagged.

    Legacy runs (no raw ASCII, so no run file) are outside the tier scheme; their caches stay in cache/,
    which the code reads, and are mirrored to analysis/ for redundancy (docs/CONVENTIONS.md §1.5).
    """
    import shutil

    path = Path(path)
    dest = path.parent.parent / "analysis" / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def output_dir(campaign: str, kind: str) -> Path:
    """Return (and create) refactored/outputs/<campaign>/<kind>/, e.g. kind = figures, cache, tables."""
    path = OUTPUT_ROOT / campaign / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
