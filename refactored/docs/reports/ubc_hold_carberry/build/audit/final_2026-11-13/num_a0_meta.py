"""Metadata audit: data_tier, source_sha256 (and that it matches the run file), mtime ordering; raw BallisticHold re-read."""
import glob, hashlib, json, os, re, sys
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored")
REP = ROOT / "docs/reports/ubc_hold_carberry"
files = sorted((ROOT / "outputs/h2_hold/analysis").glob("*_analysis.nc")) + \
    sorted(f for f in (ROOT / "outputs/h2_ubc_rtop/analysis").glob("H2_*_analysis.nc")) + \
    [ROOT / "outputs/h2_ubc_rtop/analysis/h2_ubc_rtop_analysis.nc"]
CHECK_SHA = "--sha" in sys.argv


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


bad = 0
for f in files:
    with xr.open_dataset(f) as ds:
        a = dict(ds.attrs)
    srcs = [s for s in str(a.get("source_files", "")).split(",") if s]
    shas = [s for s in str(a.get("source_sha256", "")).split(",") if s]
    issues = []
    if a.get("data_tier") != "analysis":
        issues.append(f"data_tier={a.get('data_tier')}")
    if not shas:
        issues.append("no source_sha256")
    if len(shas) != len(srcs):
        issues.append(f"{len(srcs)} sources vs {len(shas)} sha")
    for s, h in zip(srcs, shas):
        p = Path(s)
        if not p.exists():
            issues.append(f"missing source {s}")
            continue
        if p.stat().st_mtime > f.stat().st_mtime:
            issues.append(f"source newer: {p.name}")
        if CHECK_SHA and sha(p) != h:
            issues.append(f"sha mismatch {p.name}")
    print(f"{f.name:45s} tier={a.get('data_tier')} nsrc={len(srcs)} {'OK' if not issues else issues}")
    bad += bool(issues)
print("files with issues:", bad, "of", len(files))

st = json.loads((REP / "numbers_status.json").read_text())
from collections import Counter
print("status counts", Counter(v["status"] for v in st["inputs"].values()), "pending_macros", st["pending_macros"])
# analysis files referenced in status match those present
for k, v in st["inputs"].items():
    if v["analysis_file"] and not Path(v["analysis_file"]).exists():
        print("status refs missing", k)
