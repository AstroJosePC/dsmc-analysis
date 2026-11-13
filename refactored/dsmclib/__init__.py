"""Shared code for the DSMC exosphere gravity-wave campaigns (staging copy; root scripts are untouched).

Modules, roughly in pipeline order:
    paths       archive mount + outputs tree
    registry    runs by name (S1..S6, 1D runs), backed by registry/simulation_metadata.db
    params      SimulationParameters from params_data.npz or Info/*.ascii
    macro       fields and surface/escape data from .npy files or the macro pickle
    run_data    DSMCRun: one run, loaded on demand
    background  pre-forcing background and relative perturbations
    peaks       per-cycle peak detection
    growth      density-amplitude growth with altitude
    plotting    figures (call plotting.use_style() for the manuscript look)
    resolution  G_max vs azimuthal cells, Eq. 10 fit, Table 4 (staging caches or the frozen parquet)
    constants   Mars radius, padding, reference growth curves, solver physical constants

Species-aware modules for multi-species 1-D runs read from the run file (<alias>_run.nc, dsmc_netcdf_builder >= 1.2.0; tiers in docs/CONVENTIONS.md):
    ncrun       NcRun / open_run_nc: one run from its <alias>_run.nc, species labelled by name
    convergence domain inventory, boundary rates, steady-state drift test, filling timescale
    profiles    time-mean profiles with batch-means errors, isothermal reference, exobase, replicate stats
    escape      DSMC escape count + Poisson limit, Jeans escape (exobase, lower boundary), mass balance
    wave        1-D forcing response: harmonic fit with drift term, amplitude/phase vs altitude
    drift       drifting-Maxwellian boundary theory, output/phase schedule, reload_inputs restart-rate correction

Pipeline and presentation:
    runfile     build <alias>_run.nc for registered runs (dsmc_netcdf_webapp builder), then their cards
    card        one-page simulation card (PDF + JSON) per run from its run file: python -m dsmclib.card
    graphite    Graphite Telemetry style (palette, rcParams, bundled fonts) shared by reports and cards
"""

__version__ = "0.5.0"

from .run_data import DSMCRun

__all__ = ["DSMCRun"]
