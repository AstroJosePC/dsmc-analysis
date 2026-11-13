# refactored/ — staging sub-project

A self-contained copy of the analysis code, organised by **simulation campaign**. Notebooks drive the work, backed by a small library (`dsmclib`).
The repository root is **frozen**: manuscript scripts, `zenodo_release/`, the `s6_*` reviewer checks and `manuscript_review/` stay there, untouched, as the reproduction path. Nothing in here writes to the root or into ArchiveSims run folders.

## Setup (once)

```bash
mamba create -n dsmc -c conda-forge python=3.12 numpy scipy pandas matplotlib seaborn jupyter nbconvert pyarrow plotly
conda run -n dsmc pip install -e "<path>/dsmc_analyzer/refactored"
conda run -n dsmc python registry/build_registry.py        # from refactored/: copies the root DB + adds aliases
```

Once it's installed, `import dsmclib` works from any notebook or cwd, with no `sys.path` lines.

## Layout

```
dsmclib/          shared code, one module per step
  paths           ArchiveSims mount (Mac /Volumes/MyBook, I:\, Z:\, /media/astrojose) + outputs/ tree
  registry        runs by name: registry.run("S4").path, registry.campaign("s1_s6_azimuthal")
  params          SimulationParameters from params_data.npz or Info/*.ascii
  macro           fields + surface/escape data from .npy or the macro pickle (ensemble pickle first)
  run_data        DSMCRun: one run, loaded on demand
  background      pre-forcing background (mad_column0 = root; azimean_snapshot = 2024 recipe), perturbations
  peaks           per-cycle peak detection (interactive=True opens the click-to-pick window)
  growth          density-amplitude growth, cached per run in outputs/<campaign>/cache/
  resolution      G_max vs azimuthal cells, Eq. 10 fit + band, Table 4 (staging caches or the frozen 2024 parquet)
  heating         last-cycle <dT> vs constant T0 = 270 K and its gradient in K/H0 with a t interval (Fig 3b, Table 2)
  attenuation     symmetric lock-in amplitude, Table S3 bootstrap + cycle budget, Figures S5/S6 (Fig 4c, Text S5)
  plotting        figures; plotting.use_style() for the manuscript look
  -- species-aware, from the run file <alias>_run.nc (dsmc_netcdf_webapp builder >= 1.2.0) --
  ncrun           NcRun / open_run_nc(alias or path): one run, fields (species, time, radial), species by name
  convergence     domain inventory, boundary rates, steady_window drift test, filling_timescale
  profiles        time-mean profiles (batch-means SEM), isothermal_reference, exobase (Kn = 1), regrid, replicate_stats
  escape          dsmc_escape (+ Poisson limit), jeans_escape / _at_exobase / _at_lower_boundary,
                  transit_expected_count (exobase-to-top travel time), lower_boundary_balance
  wave            1-D forcing response: harmonic fit with drift term, amplitude/phase vs altitude
                  (dn, dT relative; dv_r absolute in m/s), pre-forcing noise floor (null fit at omega)
  -- pipeline and presentation --
  runfile         python -m dsmclib.runfile --campaign X | --runs A B: build <Results>/<alias>_run.nc, then cards
  tiers           CONVENTION_VERSION, analysis_path(), analysis_attrs(): required analysis-file metadata
  card            python -m dsmclib.card ALIAS | --campaign X | --all | --run-file PATH: one-page simulation card
  graphite        Graphite Telemetry style (palette, rcParams, bundled fonts in assets/graphite/) for reports and cards
registry/         build_registry.py + simulation_metadata.db (copy of root DB + campaigns, run_aliases)
campaigns/
  s1_s6_azimuthal/  density_growth.ipynb, compute_growth_cache.py,
                    resolution_fit.ipynb (Fig 7 / Table 4 / Fig 6: staging vs published)
                    peak_diagnostics.ipynb (time series + picked peaks per altitude, both backgrounds)
                    heating_slope.ipynb (S6 centre/edge temperature gradient ± 95% t interval, Table 2)
                    figure3_manuscript.py (combined Fig 3a/3b manuscript figure + Table 2 summary)
                    compute_attenuation_cache.py + attenuation_figure.ipynb (Fig 4c signed-distance lock-in amplitude)
                    attenuation_table.py (Table S3 + SI Figures S5/S6 from the lock-in caches)
  radial_1d/        density_growth.ipynb, compute_growth_cache.py
  oh2_escape_rerun/ compile_netcdf.py (raw ASCII -> <Results>/<alias>_run.nc + card; wraps dsmclib.runfile),
                    compute_analysis.py (-> outputs/.../analysis/<alias>_analysis.nc, oh2_escape_rerun_analysis.nc, tables/*.csv),
                    oh2_escape.ipynb (figures: inventory, profiles, escape, wave response)
  synthetic_methods/  (not ported yet)
checks/           regress_density_growth.py: staging vs frozen root, number for number
                  regress_resolution_fit.py: Fig 7 / Table 4 legacy path vs zenodo_release
                  regress_heating_slope.py: S6 heating profiles + slopes vs root optimized_loader (writes the cache)
                  regress_legacy_2024_growth.py: the 2024 run-folder growth pkls rebuilt from raw data
                  regress_lockin_table.py: staging Table S3 fit + bootstrap + cycle budget vs root lockin_symmetric_table.csv
                  regress_oh2_netcdf_vs_ascii.py: run file vs raw ASCII bit-for-bit + geometry + physics sanity
                  regress_oh2_rerun_vs_2025.py: B reruns vs archived Results_20250216_153021.481 (exact Info, stochastic z)
                  regress_oh2_wave_series.py: OH2_A wave series vs raw ASCII, shot noise, noise floor calibration
                  regress_sim_card.py: card values vs raw Info/Simulation_Parameters, ncrun and the O+H2 report Table 1
outputs/          gitignored: outputs/<campaign>/{analysis,figures,tables} (+ cache/ for legacy runs); outputs/cards/ (card copies + cards.csv)
```

## Conventions

**Data tiers, file naming, required NetCDF metadata and vocabulary are defined in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)** (convention 1.0: raw → run file `<alias>_run.nc` → analysis `outputs/<campaign>/analysis/`; "computational particles" in prose, `part`/`npart` in data names). The bullets below are working practice.

- **Runs by name.** Use `registry.run("S6")`, never a hard-coded `Results_...` path. To add or rename aliases, edit `ALIASES` in `registry/build_registry.py` and re-run it. The notes column holds each run's provenance caveats (the S1–S3 ensembles, the two S4 runs).
- **Notebook or script, case by case.** Exploration and figures go in `.ipynb`. Anything slow (GB-scale loads, sweeps) is an argparse CLI in the campaign folder that writes to `outputs/<campaign>/analysis/` (legacy runs: `cache/`, mirrored to `analysis/`), and notebooks read from there.
- **Porting a root script.** Move reusable physics into `dsmclib` by copying it, leaving the root untouched. Add a comparison to `checks/`, and keep the plotting in the notebook.
- **Check the background before porting.** If the root analysis builds its perturbations from `get_background_profiles` (azimuth column 0 only), flag it, and run the port on the S1–S6 recipe (`background="azimean_preforcing"`) with the root recipe kept for comparison. See *Background choice for the 2D runs* below.
- **Horizontal attenuation.** The Figure 4c port keeps the symmetric lock-in estimator and pair geometry (`jL-k`, `jR+k`) from the current re-derivation. Its cache CLI computes S2–S6 for resolution comparisons, while the manuscript figure plots S4–S6. The CLI defaults to `azimean_preforcing`; use `--background mad_column0` only when reproducing the root-background comparison. The plot labels negative signed distance as “source interior” and the zero-distance boundary as “source edge.”
- Attenuation caches are keyed by run and background. Use `--rebuild` after changing estimator settings (cycle window, distance range, or period detection) so saved reductions are recomputed.

## Runs

| alias | run | notes |
|---|---|---|
| S1–S3 | Results_20240719_* | 128/256/384 azi, 10-member ensemble means |
| S4 | Results_20240722_034301.982 | 512 azi, 2024 run (current manuscript choice) |
| S4_2025 | Results_20250904_113551.097 | alternate ±5° band re-run (`campaign(..., include_alternates=True)`) |
| S5, S6 | Results_20240722_040346.456 / _041906.318 | 640 / 1024 azi |
| 1D_5p, 1D_10p | Results_20250326_133319.916 / _135431.191 | 1D, 5 / 10 periods |
| 1D_10p_jun, 1D_20p_jun | Results_20250605_015133.742 / _042405.584 | 1D, 10 / 20 periods |
| OH2_A | ~/MarsAtmos-OH2-rerun/runs/OH2_A_128cell_20261109_180509/Results | O+H2 1D rerun of Results_20250216_103247.560, 128 cells, flux wave (4 whole periods) |
| OH2_B_r1-r3 | ~/MarsAtmos-OH2-rerun/runs/OH2_B_35cell_r{1,2,3}_*/Results | O+H2 1D reruns of the 2025-02-16 35-cell settings, extended to 130,650 steps, 3 replicates |

The OH2 runs are not archived yet: their `results_base` is an absolute path, which `registry.Run.path`
uses as-is (relative bases still resolve under ArchiveSims). After moving them into ArchiveSims, change
`ALIASES` to the archive folder names and re-run `build_registry.py`. `add_missing_run` counts only the
forcing periods that fit inside the run (OH2_A: 4, not the 10 configured).

## Verification

```bash
MPLBACKEND=Agg python checks/regress_density_growth.py            # ~all runs; log to outputs/checks/
```

It compares staging growth with the root `optimized_loader` (S-runs) and with root `loader.py` + `helper_funcs` (1D). Every forced column must be bit-identical.

```bash
python checks/regress_resolution_fit.py      # needs pyarrow (dsmc env); report in outputs/checks/tables/
```

It reproduces the published Fig 7 / Table 4 from `zenodo_release/` exactly (the legacy path) and checks the staging G_max against the raw caches. The published S1–S6 points come from 2024 growth caches that the current code does not reproduce, so the staging figure differs. `resolution_fit.ipynb` shows both.

### Background choice for the 2D runs

The root loaders build the pre-forcing background from **azimuth column 0 only** (`mad_column0`). That is exact for the 1D runs. For S1–S6, though, each altitude's reference carries several % of particle noise, which is comparable to the ≈0.12 perturbation at 150 km, so the growth profiles come out crooked above ~270 km (S1–S3 worst).

The lost 2024 pipeline behind the published S1–S6 growth used:
- the azimuthal mean of the snapshot 2 outputs before forcing (`azimean_snapshot`),
- the cells nearest 150 and 320 km (`altitude_cells="nearest"`: 148.3–315.4 km),
- 600 s of padding.

`regress_legacy_2024_growth.py` rebuilds its 157 pkls from raw data:

| Run | Result |
|---|---|
| S2, S3 | bit-identical |
| S1 | identical to 1e-14 |
| S6 | 87/88 columns; column 512 was picked by hand in 2024 |

**Refactored S1–S6 recipe** (`dsmclib.resolution.STAGING_RECIPE`; used by the S1–S6 notebooks):
- `background="azimean_preforcing"`: the mean over all azimuth columns and the 17 pre-forcing outputs that every run has (t = 4908–5012 s). Do not extend this to the whole S4–S6 pre-forcing record: it starts at t = 7 s, includes the spin-up, and biases 315 km by about +2.9%.
- `altitude_cells="nearest"`: the growth window is 148.3–315.4 km.

Its caches are `density_growth_<run>_azimean_preforcing_nearest.pkl`. The defaults of `compute_density_growth` stay at the root recipe (`mad_column0`, `first_above`), so `regress_density_growth.py` stays bit-identical to root.

```bash
python campaigns/s1_s6_azimuthal/compute_growth_cache.py --background azimean_preforcing --altitude-cells nearest
MPLBACKEND=Agg python checks/regress_legacy_2024_growth.py
```

## Simulation cards

Every run with a run file gets a one-page **simulation card** (PDF) and its JSON twin: an audit sheet
of the run's configuration, populations and provenance. `dsmclib.runfile` renders it after each run-file build;
`dsmclib.card` (re-)renders it on its own.

```bash
python -m dsmclib.runfile --campaign h2_ubc_rtop     # build missing run files, then their cards
python -m dsmclib.card --all                         # cards for every registered run with a run file
python -m dsmclib.card --run-file /path/Results/X_run.nc  # any run file, registered or not
python -m dsmclib.card --all --overwrite             # re-render after a template change
python checks/regress_sim_card.py                    # card values vs raw Info, ncrun and the report
```

- **Source.** Only the run file (root attributes incl. every `param_*` line, `/macroscopic`,
  `/rates_*`, `/info_surface`) plus the run folder's `run_meta.txt`, `COMPLETE.ascii`, `Input/` and
  `DSMC` binary, which are hashed (SHA-256). No value is typed by hand. A card is a pure function of
  its run file, so cards can be re-rendered after any style change without rebuilding it.
- **Outputs.** `<Results>/<alias>_card.{pdf,json}` next to the run file, copied to `outputs/cards/`,
  which also holds `cards.csv` (one row per run: grid, time, forcing, warnings, commit, run-file hash). JSON keys use `npart_*` for computational-particle counts.
  Existing cards are kept unless `--overwrite`.
- **Content.** Status strip (completion, output count, time axis, N_phys = W·N_part, strict build, run-file convention, parameters absent from the solver,
  forcing end beyond run end), domain ruler with cell edges, grid/time/forcing blocks, species at the
  lower boundary (m, W, n0, T0, c̄, flux, flow, λ0, Jeans), collision pairs, populations
  (computational particles: final/peak, bottom/top cell, injected/returned/escaped) with inventory panels, provenance and hashes.
- **Rendering.** matplotlib + tectonic, Graphite Telemetry (`dsmclib.graphite`), one US Letter page
  (a card that does not fit one page is an error).
- **Runs without a run file** are skipped. The 11 legacy runs (radial_1d, S1–S6) have no raw ASCII left,
  so they have no run file and no card.
- **Snapshots of running runs.** A run file built while the solver is still running is a snapshot; its card
  says "not finished". Rebuild with `python -m dsmclib.runfile --campaign X --overwrite` once the runs end.

## Multi-species runs (oh2_escape_rerun)

`DSMCRun` and the background/peaks/growth/heating/attenuation modules assume one species. Multi-species
1-D runs go through the run file instead:

```bash
python campaigns/oh2_escape_rerun/compile_netcdf.py       # raw ASCII -> <Results>/<alias>_run.nc (4 runs in ~10 s)
python campaigns/oh2_escape_rerun/compute_analysis.py     # -> outputs/oh2_escape_rerun/{analysis,tables}
jupyter nbconvert --execute --inplace campaigns/oh2_escape_rerun/oh2_escape.ipynb
python checks/regress_oh2_netcdf_vs_ascii.py && python checks/regress_oh2_rerun_vs_2025.py
python checks/regress_oh2_wave_series.py                 # wave series: raw ASCII, shot noise, noise floor
```

- **Physical constants** in `constants.py` (`G_NEWTON`, `K_BOLTZMANN`, `AMU`) are the solver's values
  (MOD_Constants.f90), so Jeans rates reproduce `Info/Surface_Cells.ascii` exactly.
- **Archived `surface_data.npz` of multi-species runs is scrambled.** The solver writes the per-species
  Surface/Escape columns field-major (`inject_rate(1:ns)`, then `return_rate(1:ns)`, ...). The script that
  produced the archived `.npz` files split them species-major. For one species the two orders coincide,
  so the manuscript runs are unaffected. For two species, e.g. Results_20250216_153021.481, every
  per-species surface field in the `.npz` is wrong. Use the raw ASCII or the run file instead.
- **1-D temperatures count radial bulk flow as heat.** The solver subtracts the Cartesian mean velocity
  (~0 in a spherical shell). `profiles.temp_trans_corrected(run)` gives `temp_trans_corr = temp_trans - m vel_rad^2/(3k)` per
  output for 1-D runs (analysis tier; the reader no longer adds it). The oh2 analysis attaches it and uses it for the exobase Jeans estimate, profiles and the wave dT.
- **Wave-series scatter in OH2_A is DSMC shot noise.** At 100-150 km a cell holds ~105 O / ~40 H2
  computational particles, so dn/n scatters by ~1/sqrt(N) ~ 0.1 per window; `regress_oh2_wave_series.py` confirms the
  run-file series equal the raw ASCII. The null fit over the 4 pre-forcing periods gives the noise floor. AR(1)
  errors are calibrated for O and H2 dv_r, and ~25-30% low for H2 dn/dT.
- **Escape at a distant top boundary.** Under UBC_FREE_ESCAPE the count is made at the top boundary.
  An escaping particle needs a long time to travel there from the exobase: the median is ~17-19 ks
  for H2 to 40,000 km. `escape.transit_expected_count` gives the count to expect in a finite run, so
  a zero count can be interpreted.
