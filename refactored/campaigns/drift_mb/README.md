# drift_mb: 1-D drifting-Maxwellian lower-boundary forcing in pure O and O+H₂

Campaign directory: `~/MarsAtmos-DriftMB-sweep/` (local SSD; design in `EXPERIMENT_DESIGN.md` there).
Solver: MarsAtmos `feature/velwave-1d` (`9088aab3`), builder 1.2.1. Runs 2026-11-13.

## Runs (registry campaign `drift_mb`)
| Alias | What |
|---|---|
| `SPIN_O`, `SPIN_OH2` (role `spinup`) | 1.0e5 s spin-ups from empty, no forcing; outputs every ~990 s |
| `<atm>_s<100 s_O>_P<P>` (role `current`) | Restart of the spin-up: VEL_WAVE_LINEAR, s_O ∈ {0.05, 0.10, 0.20}, P ∈ {165, 330, 660} s, 20 forcing + 5 ring-down periods, 16 outputs per period |
| `O_s000_P330`, `OH2_s000_P330` | s = 0 controls (same restart, drift sampler at w′ = 0) |

The registry rows come from `matrix.csv` (`registry/build_registry.py`, `_driftmb_aliases`).

## Pipeline
```zsh
python -B campaigns/drift_mb/compute_analysis.py              # ~1 min: analysis files + tables
MPLBACKEND=Agg python -B campaigns/drift_mb/make_figures.py   # 17 figures (png + pdf)
python -B checks/validate_drift_mb.py                         # 2493 checks -> tables/validation.csv
```

**Outputs** (`outputs/drift_mb/`, analysis tier per `docs/CONVENTIONS.md`):
- `analysis/<alias>_analysis.nc`: one per run.
  - Forced runs and controls carry:
    - the schedule;
    - phase-folded injection VDFs (16 phases + ring-down) with the mixture theory, KS/χ², binned and unbinned moments vs theory;
    - per-output injection moments and mass/momentum/energy rates vs theory, and return rates;
    - escape series, per-period and phase-folded rates;
    - escape-speed VDFs (fit, ring-down, phases, control, spin-up, Jeans-mapped reference);
    - harmonic wave response (u_r, Δn/n₀, ΔT, ΔT_rad, ΔT_tan) with the control noise floor;
    - altitude-phase and altitude-time fields;
    - per-period u_r amplitude and cycle-mean fields;
    - cycle-mean heating vs control and vs spin-up;
    - propagation (phase speed, damping) and ring-down timescales.
  - Spin-ups carry window profiles, exobase, escape and escape VDF.
- `analysis/drift_mb_analysis.nc`: cross-run scalars (run × species).
- `tables/`: `run_summary`, `vdf_moments`, `flux_budget`, `escape_summary`, `wave_response`, `heating`, `ringdown` and `validation` (CSV).
- `figures/`: VDF phase folds, moments vs phase, injection budget, H₂ escape (per period, summary, speed VDF), altitude-time and altitude-phase maps, wave profiles, propagation summary, heating profiles and exobase summary, ring-down.
- `logs/`: logs of the three steps.

## Definitions (also stored as attributes)
- **Restart-interval correction.** A `reload_inputs` restart (`feature/velwave-1d`) re-reads `avg_step` but keeps `avg_time_interval` from the spin-up.
  - So every Rates quantity of a forced run (`*_rate`, `num_part_*`, `*_energy`) and `therm_conduct_*` is divided by the spin-up interval (988–991 s) instead of avg·dt.
  - The rates are therefore too low by a factor of 96, 48 or 24 (P165, P330, P660).
  - `dsmclib.drift.correct_restart_rates` rescales them in memory; the run files stay unmodified.
  - Counts, Macro n/T/u and the diagnostics groups are unaffected.
  - `checks/validate_drift_mb.py` verifies the correction against the drift-diagnostic injection counts, an independent group.
  - **The simulation cards of the 20 forced runs show the stale interval.**
- **Fit window:** forcing periods 10–19 (the last 10 of 20).
- **Ring-down:** periods 20–24.
- **Common window:** 10–20 × P165 of the atmosphere (1.65–3.29 ks after the restart). It lies inside the forcing for every P, so escape and heating are compared at equal forcing duration; the fit window sits at a different elapsed time for each P.
- **Baseline:** the s = 0 control of the same atmosphere, averaged over the same elapsed window (`time_matched`).
  - The P330 control lasts 8.2 ks, so the P660 fit and ring-down windows use the control's second half (`control_second_half`).
  - The controls themselves use the spin-up's last 2e4 s.
  - Every heating and escape number also has a `_vs_spin` cross-check.
- **Harmonic fits** use interval mid-times, with amplitudes divided by sinc(π/16) = 0.9936 (interval averaging).
  - Phases are relative to w′ sin ωt′ (0 = in phase).
  - Cells with fewer than 10 computational particles in the baseline are masked; the figures draw profiles only where there are at least 50.
- **Moment standard errors** are exact asymptotic values: the delta method on E[y⁰…⁸] of the flux-weighted drifting-Maxwellian mixture (`dsmclib.drift.sample_moment_se`, checked against Monte Carlo).
- **Ring-down τ** comes from an exponential fit to the last forcing period plus the 5 ring-down periods. It is reported only when resolved (|a| > 3σ_a and σ_τ < τ); raw fits are kept as `*_fit`.

## Validation (2026-11-13): 2493 checks, all PASS
- **Exact identities:** completeness, provenance (SHA-256), schedule, NaN, restart-factor identities, VDF/escape bookkeeping.
- **Mass balance** < 2e-3.
- **Injection vs theory:** number, momentum and energy within 5σ (ranges 0.9991–1.0005, 0.997–1.002, 0.997–1.003).
- **VDFs:** 510 groups; max KS D/D_crit = 0.92, max moment |z| = 3.65, all χ² p-values above Bonferroni.
- **Controls vs spin-up:** escape and exobase ΔT consistent with 0.
- **Signal vs noise:** forced signal > 5× the noise floor and > 5 SE.
- **Linearity:** u_r(150 km) is linear in s to within 20 %.
- **Threshold changed after seeing the results:** the noise-floor test was first 10× the floor and was changed to 5× floor and 5 SE. The s = 0.05, P165 runs sit at 7.7–9× the floor, which is a clear detection.

## Caveats
- **One replicate per cell.** gfortran 15 seeds from the OS, so the runs are independent and not paired across s.
- **P660 common window.** The common-window ΔT has large SEs at P660 (≈ 2.5 periods in the window, batch means).
- **Phase speed vs theory.** At P660 the measured O phase speed (518–554 m/s) is well below the isothermal estimate c_s/√(1−ω_a²/ω²) = 611 m/s. P165 and P330 agree within a few %. H₂'s phase speed (550–600 m/s) is faster than O's.
- **H₂ escape per period at P660.** The excess peaks around period 10 and then declines while the exospheric heating persists, consistent with depletion of the upper H₂ column. The fit-window escape enhancement therefore depends on the window.
- **P165 above the exobase.** The response is at the noise floor (10 s outputs).
