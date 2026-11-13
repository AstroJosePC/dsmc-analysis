# h2_ubc_rtop: upper-boundary altitude and condition for an isothermal H2-bearing Mars atmosphere

Campaign directory: `~/MarsAtmos-H2UBC-sweep/` (local SSD; archive to `/Volumes/MyBook/ArchiveSims` when done).
Solver: MarsAtmos branch `fix/ubc-reflect-ballistic` (worktree `~/MarsAtmos-UBC`), commit recorded in `DSMC.commit`
and in every `runs/<id>/run_meta.txt`. Designed 2026-11-09.

## Question
Which **upper-boundary altitude z_top** and **upper boundary condition (UBC)**, and which temporal and spatial
numerical parameters, does a 1-D isothermal (270 K) DSMC model of a hydrogen-bearing Mars upper atmosphere need,
for steady-state density/temperature profiles and the H2 escape rate to be independent of those choices?

**Hypotheses (user):**
- H1. z_top needs to be at least about 30,000 km.
- H2. The UBC must be free escape or ballistic. A reflective top was thought to keep adding particles and heat
  the atmosphere. On `main` that symptom is (at least largely) a code bug: UBC_REFLECT duplicated every
  reflected particle. It is fixed on `fix/ubc-reflect-ballistic`.
- H3. Steady state for H2 needs long averages (tens of ks) and runs of several 1e5 s.

**Theory expectation, to be tested.** In 1-D spherical symmetry, a bound particle returning ballistically reaches
z_top with the same speed and angular momentum and the opposite radial velocity. The fixed reflect UBC is
therefore the zero-delay limit of the ballistic UBC:
- Both should give the same steady state, already at low z_top.
- Free escape also removes bound particles whose apoapsis lies above z_top, and counts them as escape.
- So free escape is the UBC whose profiles and escape rate depend on z_top.

## Prior evidence
O+H2 reruns (Nov 2026, 40,000 km, free escape, 1.3e4 s), from `dsmc_analyzer/refactored/outputs/oh2_escape_rerun`:
- O is steady after about 3 ks.
- The H2 inventory is still rising at 4.7–6.1 %/ks at the end.
- Zero escaping computational particles. The median transit from the exobase to 40,000 km is 17–19 ks, longer than the run.
- The H2 exobase is at about 400–410 km, the O exobase at about 240 km.

## Matrix (coarse pass, 60 runs, 1 seed each)
| Factor | Levels |
|---|---|
| Atmosphere | `H2`: pure H2 (n0 = 2.3e13 m⁻³); `H2O`: H2 + O (n0 = 2.3e13 and 1e16 m⁻³) |
| UBC | 1 reflect (fixed), 2 ballistic (analytic Kepler hold; apoapsis > r_Hill counts as escape), 3 free escape |
| z_top (nominal) | 2,500 / 5,000 / 10,000 / 20,000 / 40,000 km |
| Δr_max (`RadialCellWidth.input`) | 1000 / 250 km |

**Run ids:** `<atm>_U<ubc>_z<z_top>_dr<dr_max>`, e.g. `H2O_U2_z10000_dr250`. `matrix.csv` lists every run with
n_steps, dt, the cell count and the actual top. The solver extends the last cell past the requested top, by up
to one cap width; the analysis must use the actual top.

## Fixed parameters
- **Lower boundary:** 100 km, LBC_ALL_RETURN, uniform 270 K, start empty (`Fill` 0). Waves off.
- **Collisions:** elastic VHS. H2: d_ref = 2.92 Å, ω = 0.67. O: σ_ref = 4.5e-20 m², ω = 0.67.
- **`max_MB_vel_dis` = 8000 m/s.** It must exceed v_esc(100 km) = 4.96 km/s.
  - Injection is truncated at this speed. For H2 that is x = v/v0 = 5.36, so the truncated flux fraction
    (1+x²)e^{−x²} = 3e-11, compared with a Jeans fraction (1+λ0)e^{−λ0} = 1.9e-4 (λ0 = 11.0).
  - `make_configs.py` asserts this.
- **Statistics:** k = `max_inject_per_step` is 10 for H2O and 100 for H2. The weight follows as W = flow·dt/k per
  species: H2 1.43e28 (H2O) or 1.48e28 (H2); O 2.2e30. (F0(H2) = 1.48e30 s⁻¹; an earlier version of this line
  gave that flow as the pure-H2 weight; the run cards confirm W = 1.48e28.)
- **dt:**
  - H2O: 0.0965 s, set by the O MFP-sized 1.54 km bottom cell. It is independent of Δr_max.
    dt/τ_c(H2 at the bottom) = 0.23, and the H2 MFP there (~0.7 km) is below the bottom cell width. Both are
    inherited from the O+H2 reruns; a refinement check is listed below.
  - H2: 1 s, the solver's cap; dt/τ_c = 0.015.
- **Duration:** t_end = 3e5 s for all runs: 300,000 steps for H2, 3,118,360 steps for H2O.
- **Outputs:** about every 1000 s (`avg_steps` 1000 or 10,360), so about 300 outputs per run.
- **Restart:** a restart file is always written at the end, so a run can be extended through the 3rd line of
  `Restart.input`.

**Settling estimate:** the period of a bound orbit with apoapsis at z_top is 9 ks (2,500 km), 14 ks (5,000),
24 ks (10,000), 48 ks (20,000) and 110 ks (40,000). 3e5 s is at least 2.7 such periods.

## Execution
- Task Spooler `~/.dsmc_ts.sock` with 16 slots (the P-cores), np = 1 per run.
- `just queue-all` queues longest first.
- Each job runs `run_config.zsh`: it runs the solver, zstd-compresses `dsmc.log`, then runs `dsmc_builder --strict
  --alias <id> --overwrite` to `runs/<id>/Results/<id>_run.nc` (docs/CONVENTIONS.md). `--overwrite` is deliberate: a
  file already there when the solver exits can only be a mid-run snapshot (2026-11-11: the 16 first-wave H2O runs had
  `_L1.nc` files built ~34 min in, so their end-of-run build failed; script switched to `_run.nc` the same day, jobs
  started before then still write `_L1.nc`).
- `python -m dsmclib.runfile --campaign h2_ubc_rtop` builds run files and cards for finished runs only (needs
  `COMPLETE.ascii`), and rebuilds any existing run file with `run_complete=false` without needing `--overwrite`.
  Re-run it after the sweep to give the jobs that still write `_L1.nc` their `_run.nc` and card.

## Acceptance criteria (later analysis; dsmclib `convergence`, `profiles`, `escape`, `exosphere`)
1. **Steady state.** `convergence.steady_window` on the per-species inventory. Profiles and escape are averaged
   over the trailing steady window: at least 5e4 s, target 2e5 s.
2. **z_top convergence,** per (atmosphere, UBC, Δr_max), between successive z_top values:
   - at and below z_ref = 2,000 km: |Δn/n| < 5 % and |ΔT|/T < 2 %, or within 2σ (batch means);
   - H2 escape rate within 10 % or 2σ.
   - The answer is the lowest z_top for which the next level also passes.
3. **Resolution.** The same metrics between Δr_max = 1000 and 250 km.
4. **Theory.**
   - Above the exobase, compare with `dsmclib.exosphere.profile`: the collisionless Liouville solution, using
     the exobase as the source.
   - Compare escape with Jeans at the DSMC exobase (`escape.jeans_at_exobase`).
   - Compare inject − return − escape with dN/dt (`escape.lower_boundary_balance`).

**Analysis caveats (found in verification):**
- **Temperature.** In 1-D, the solver's `temp_trans` and `temp_kin` subtract the cell's *Cartesian* mean velocity,
  which is about 0 for a spherically symmetric radial flow. Both therefore include the bulk term m·u_r²/(3k).
  - Use T = `temp_trans` − m·`vel_rad`²/(3k).
  - In E4 (free escape, z_top 2,500 km), the raw value is 27 K too hot at the top cell; the corrected value
    matches the analytic T within 0.2 %.
  - The bias grows with the outflow, so it differs between UBCs and between z_top values.
- **Ballistic held reservoir.** Near-escape orbits have times of flight up to about 1e7 s, bounded by r_Hill.
  - The held count (`num_part_ball`) therefore keeps growing slowly through a 3e5 s run. In E3, at 30–60 ks, it
    was 53 % of its asymptotic value.
  - The returning flux from those orbits is small, so in-domain profiles converge much sooner.
  - Compare with `exosphere.held_count(..., t=time)`, not with the t → ∞ value.

## Refinement pass (after the coarse results)
- Bisect z_top around the convergence edge.
- Run Δr_max = 60 km at the selected z_top.
- Run 3 seeds at the selected configuration.
- dt check for H2O: lower dt/τ_c, e.g. via a finer bottom cell or a higher `max_MB_vel_dis`.
- Extend any unconverged runs from their end-of-run restart.

## Verification of the solver fixes
Experiments E0–E6 are in `~/MarsAtmos-UBC/verification/` (`results/summary.md`), and the PR is
`fix/ubc-reflect-ballistic` → `main`.
