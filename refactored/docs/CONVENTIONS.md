# Data conventions (refactored/)

**Convention version 1.0**, adopted 2026-11-10. Every NetCDF written under these rules carries
`convention_version = "1.0"`. Change the number when a rule below changes, and say what changed in
*Changelog* at the end.

This file is the single source for these rules. The README and CLAUDE.md point here and do not
restate them.

Scope: the tiers and where files live, the metadata every NetCDF must carry, and vocabulary in data
names. Units and the naming of variables and CSV columns are not standardised yet.

---

## 1. Data tiers

Data moves through three tiers. Each one has a descriptive name; there is no numbered ladder.

| Tier | What it is | Where it lives | Written by |
|---|---|---|---|
| **raw** | The solver's own output, untouched | the run folder: `Results/{Macro,Rates,Info,…}`, `Input/`, `run_meta.txt`, `COMPLETE.ascii`, the `DSMC` binary | the solver and the run scripts |
| **run** | One NetCDF per run holding the raw tier's numbers, consolidated, with metadata | `<Results>/<alias>_run.nc` | the builder, `dsmc_netcdf_webapp` (≥ 1.2.0), through `python -m dsmclib.runfile` |
| **analysis** | Anything computed from run files | `outputs/<campaign>/analysis/` | the campaign's analysis scripts (for example `compute_analysis.py`) |

Figures, CSV tables, reports and simulation cards are **products**, not a tier (§1.4).

### 1.1 raw

- Never modified, moved or renamed by the pipeline. The pipeline only reads it and may add files
  beside it (the run file and the card).
- A run that is still going has a raw tier that is still growing. Anything built from it is a
  snapshot (§1.2).

### 1.2 run

- **No value changes.** Each data variable equals its raw ASCII source bit for bit. Checks:
  `checks/regress_oh2_netcdf_vs_ascii.py` and `checks/regress_oh2_wave_series.py`.
- **No computed fields.** The file is not modified after the builder writes it, and the reader
  (`dsmclib.ncrun`) adds no derived variables on load. A correction of a known solver bias, even a
  closed-form per-cell one such as `temp_trans_corr`, belongs to the analysis tier.
- **Names come from the solver.** Variables keep the solver's names (`num_part`, `num_dens`,
  `vel_rad`, `temp_trans`, …). The vocabulary rule in §3 does not rename them.
- **Snapshots.** A run file built while the solver is still running is a snapshot. Its
  `run_complete` attribute is `"false"` and its card says "not finished". Rebuild it with
  `--overwrite` once the run ends.
- **Rebuilding** uses the same name, so the file is replaced. Rebuild every run file when the
  builder's major or minor version changes.

### 1.3 analysis

- **Format: NetCDF** for anything new.
- **Per-run** products go in `outputs/<campaign>/analysis/<alias>_analysis.nc`.
- **Cross-run** products (summaries, ensembles, fits across runs) go in
  `outputs/<campaign>/analysis/<campaign>_analysis.nc`.
- **Variants are named in the file name**, never implied by a default. Put the variant between the
  alias and the suffix: `<alias>_<variant>_analysis.nc` (for example
  `S4_azimean_preforcing_analysis.nc`). A file without a variant tag means the analysis has only one
  variant.
- Every analysis file names its sources by path and SHA-256 (§2.2), so it can be traced back to the
  exact run files it was computed from.

### 1.4 Products

- **Figures** go in `outputs/<campaign>/figures/`, and **CSV tables** in
  `outputs/<campaign>/tables/`. Reports live in `docs/reports/<name>/` and keep their own
  `figures/`, `tables/` and `build/`.
- A product is regenerated from run and analysis files by code. Values are never typed into a
  product by hand.
- **Simulation cards** (`<Results>/<alias>_card.{pdf,json}`, copied to `outputs/cards/`) are built
  straight from the run file. Their computed values are descriptive only: counts, extrema and
  hashes, not physics.

### 1.5 Legacy runs (outside the scheme)

The radial_1d and S1–S6 runs have no raw ASCII left, so they have no run file and no tier. They
keep the derived files the old loader made (`.npy`, `.pkl`, `.npz`).

- Their derived caches stay in `outputs/<campaign>/cache/`, which is what the code reads.
- Copies are kept in `outputs/<campaign>/analysis/` for redundancy. These copies carry no tier tag
  and keep their original names.
- New legacy-run outputs are written to `cache/` and copied to `analysis/` (`dsmclib.paths.mirror_legacy_cache`, called by the growth, heating and lock-in cache writers).
- `cache/` is used for nothing else.

---

## 2. Required metadata (global attributes)

### 2.1 Run files (written by the builder, ≥ 1.2.0)

| Attribute | Value |
|---|---|
| `data_tier` | `"run"` |
| `convention_version` | `"1.0"` |
| `title` | `"<alias>: DSMC run file (solver output, unmodified)"` |
| `Conventions` | `"CF-1.13"` |
| `builder_version` | builder version, e.g. `"1.2.0"` |
| `run_alias` | the registry alias (builder `--alias`) |
| `source_commit`, `source_branch`, `run_started`, `run_finished` | from `run_meta.txt`; always present, `""` when unknown |
| `run_complete` | `"true"` / `"false"`: whether `COMPLETE.ascii` exists (§1.2, snapshots) |
| `absent_params` | the `Simulation_Parameters` keys the builder types into root attributes (`NUMERIC_ROOT_ATTRS`) that this solver version did not write, comma-separated; `""` if none |
| `history` | ISO-8601 UTC build time |
| `param_*` | every `Simulation_Parameters.ascii` line, verbatim |

**Gaps between solver versions are explicit.** When the solver does not write a parameter (for
example, the wave settings on commit `64f21a1`), the builder leaves out its `param_*` attribute and
lists the parameter in `absent_params`. A missing attribute never has to be guessed at.

### 2.2 Analysis files

| Attribute | Value |
|---|---|
| `data_tier` | `"analysis"` |
| `convention_version` | `"1.0"` |
| `title` | `"<alias or campaign>: <what> (analysis)"` |
| `source_files` | run (or analysis) files used, absolute paths, comma-separated |
| `source_sha256` | SHA-256 of each source file, in the same order |
| `dsmclib_version` | version of the dsmclib package that wrote the file |
| `history` | ISO-8601 UTC build time and the script that wrote it |

Written with `dsmclib.tiers.analysis_attrs(title, sources, script)`; paths with `tiers.analysis_path(campaign, name, variant)`.

Analysis-specific attributes (windows, thresholds, fit settings) are added beside these. Every
setting that changes a number belongs in an attribute.

---

## 3. Vocabulary

- **In prose, captions, labels and docstrings:** "computational particles", or "virtual particles".
  Never "simulators".
- **In data names (variables, attributes, CSV columns, JSON keys):** use the solver's token,
  `part`.
  - Counts of computational particles use `npart`: `dsmc_escape_npart`, `background_npart`,
    `npart_final`.
  - `N_part` pairs with `N_phys` in the domain inventory (`N_phys = W · N_part`).
  - Per-particle quantities and thresholds use `part`: `dsmc_one_part_per_window_s`,
    `wave_min_part`.
- **In typeset math:** `N_\mathrm{cp}` for a count of computational particles.
- **Particle weight** `W`: physical molecules per computational particle.
- **Run-file variables are exempt.** They keep the solver's names (§1.2).

### Renames made at convention 1.0

| Old | New |
|---|---|
| `<alias>_L1.nc` | `<alias>_run.nc` |
| `<alias>_L2.nc`, `oh2_summary_L2.nc` (in `cache/`) | `analysis/<alias>_analysis.nc`, `analysis/oh2_escape_rerun_analysis.nc` |
| `N_sim` | `N_part` |
| `escape_sim_count`, `dsmc_escape_sim_count` | `escape_npart`, `dsmc_escape_npart` |
| `dsmc_escape_sims` | `dsmc_escape_npart` |
| `dsmc_one_sim_per_window_s` | `dsmc_one_part_per_window_s` |
| `expected_sims`, `naive_expected_sims` | `expected_npart`, `naive_expected_npart` |
| `transit_expected_sims` | `transit_expected_npart` |
| `expected_sims_at_jeans_exobase` | `expected_npart_at_jeans_exobase` |
| `background_sims` | `background_npart` |
| `min_sim` / `wave_min_sim` | `min_part` / `wave_min_part` |
| analysis attribute `source_l1_file` | `source_files` + `source_sha256` |
| card JSON `ncp_*`, `populations.ncp` | `npart_*`, `populations.npart` |
| `dsmclib.l1`, `L1_SUFFIX` | `dsmclib.runfile`, `RUN_SUFFIX` |
| `temp_trans_corr` added by `ncrun` on read | computed in the analysis tier (`dsmclib.profiles.temp_trans_corrected`) |

The old `*_L1.nc` files are kept beside the new run files. Nothing in the code reads them any more.

---

## Changelog

- **1.0** (2026-11-10): first written version. Descriptive tiers (raw / run / analysis) replace
  L0/L1/L2. Builder 1.2.0 writes the tier, title and `absent_params` attributes. The `part` token
  is used in data names. Corrections move from read time to the analysis tier.
