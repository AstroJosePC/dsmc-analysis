"""Regression: staging dsmclib.attenuation.attenuation_table vs the frozen root Table S3 recipe.

    python checks/regress_lockin_table.py

Reference: root ``lockin_symmetric_rederivation.py`` output ``exports/lockin_symmetric_table.csv``
(Sep 2026, column-0 background, 2024 S4) and the reductions it cached in
``exports/lockin_symmetric_<n>azi.npz``. The staged table is run on those same root caches, so
any difference is in the fit/bootstrap/cycle code, not the reduction. L_att, both uncertainty
terms, n and R^2 must match to 1e-6 relative (the root CSV was written under another scipy;
the cycle-term fits then differ at ~3e-8, i.e. curve_fit tolerance).
Report: outputs/checks/tables/regress_lockin_table.json
"""

import csv
import json
import sys

import numpy as np

sys.dont_write_bytecode = True

from dsmclib.attenuation import attenuation_table  # noqa: E402
from dsmclib.paths import REFACTORED_ROOT, output_dir  # noqa: E402

EXPORTS = REFACTORED_ROOT.parent / "exports"
COLUMNS = ["L_att_km", "sigma_L_boot_km", "sigma_L_cycle_km", "sigma_L_total_km", "n_points", "r_squared"]


def main():
    with open(EXPORTS / "lockin_symmetric_table.csv") as f:
        root_rows = {(r["run"], float(r["target_altitude_km"])): r for r in csv.DictReader(f)}
    report, worst = [], 0.0
    for n_azi in (256, 384, 512, 640, 1024):
        with np.load(EXPORTS / f"lockin_symmetric_{n_azi}azi.npz") as f:
            red = {key: f[key] for key in f.files}
        # The root table never applied a point minimum (S3 rows are nan only through n <= 2).
        for row in attenuation_table(red, min_points=3):
            ref = root_rows[(row["run"], row["target_altitude_km"])]
            for col in COLUMNS:
                a, b = float(row[col]), float(ref[col])
                rel = 0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b) / max(abs(b), 1e-300)
                worst = max(worst, rel)
                report.append({"run": row["run"], "altitude_km": row["target_altitude_km"],
                               "column": col, "staged": a, "root": b, "rel_diff": rel})
    ok = worst < 1e-6
    path = output_dir("checks", "tables") / "regress_lockin_table.json"
    path.write_text(json.dumps({"pass": ok, "worst_rel_diff": worst, "rows": report}, indent=1))
    print(f"{'PASS' if ok else 'FAIL'}: worst relative difference {worst:.3g} over {len(report)} values -> {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
