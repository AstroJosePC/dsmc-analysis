# h2_hold: Kepler hold vs explicit RK4 ballistic tracking for pure-H2 Mars escape

Campaign directory: `~/MarsAtmos-H2Hold-campaign/` (local SSD). Solver: MarsAtmos branch `feature/ubc-hold-explicit`
(worktree `~/MarsAtmos-UBC-explicit`, draft PR #14 stacked on #13), binary at `43064c24` (`DSMC.commit`). Designed 2026-11-11.

## Question
Do the two ballistic upper boundary conditions agree in a collisional column over a long run?

- **UBC 2:** the analytic Kepler hold.
- **UBC 5, `UBC_BALLISTIC_EXPLICIT`:** held particles are integrated with the domain's RK4 above r_top. The escape rule is the same as for UBC 2: escape if the orbit is unbound or its apoapsis is beyond r_Hill.

How do both compare with reflect (UBC 1)? This is the analogue of Carberry Mogan et al. (2021, PSS 205, 105302), Figs. 1–3 and Table 1.

## Runs (registry campaign `h2_hold`)
| Alias | What |
|---|---|
| `Hold_H2_U{1,2,5}_z{2500,10000,40000}_dr250` | Pure H2, 270 K, collisions on (VHS), dr_max 250 km, t_end 2e6 s (dt 1 s), outputs every 2000 s. `Hold_` prefix because the run ids are h2_ubc_rtop aliases. |
| `E7a_kepler`, `E7b_explicit`, `E7c_explicit_dbg`, `E7d_explicit_restart`, `E8_main_kepler`, `E9_reflect` (role `verification`) | Collisionless verification runs of PR #14, z_top 2,500 km nominal; `verification/experiments.tsv`. |

Every run with the new binary writes `Results/Diagnostics/Output<N>_Proc<r>_BallisticHold.ascii` per averaging interval. The columns are: hold entries, hold returns, escapes at the crossing, escapes by orbit or Hill cutoff, held count at the end and mean, and the RK4 return-delay statistics. The hold balance Δheld = in − ret is exact.

## Analysis
```zsh
python -B campaigns/h2_hold/compute_analysis.py
```

It writes these files (docs/CONVENTIONS.md, analysis tier):

- `outputs/h2_hold/analysis/<alias>_analysis.nc`, one per finished run:
  - profiles (`profile_*`, with `_sem`) over the run's window: verification t > 30 ks; production, the steady window of the H2 inventory, else the second half;
  - escape, held and inventory series;
  - the BallisticHold series (`hold_*`, dim `hold_time`), with `hold_invariant_violation`;
  - window escape (`dsmc_escape_*` attrs) and wall time;
  - for E7a, E7b and E9, the cell-averaged collisionless profile.
- `outputs/h2_hold/analysis/h2_hold_analysis.nc`: the collisionless references from `dsmclib.exosphere` for every sweep and campaign top, and the E7–E9 verification metrics (`verification_json`). The references are escape against r_top, held(t), tau90 and the longest return time.
- `outputs/h2_hold/tables/run_inventory.csv` and `verification_summary.csv`.

The run file is traced with `source_files`/`source_sha256`. The BallisticHold files are traced by glob, count and a combined SHA-256 (`hold_source_*` attributes).

The consumer is `docs/reports/ubc_hold_carberry/`.
