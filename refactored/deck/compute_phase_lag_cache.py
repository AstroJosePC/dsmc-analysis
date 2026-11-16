"""Cache the S6 per-column phase lags behind lambda_h (slide 6, card 2 sparkline).

The published retrieval lives in the frozen root scripts and never saved its points. This runs
that production code unchanged (compare_horizontal_phase_lag_profiles.process_horizontal_at_altitude
on both sides of the forced patch at the 230 km reference altitude, then the combined ODR fit of
s6_reviewer_facts.odr_fit_full) and stores the points and the fit.

The root code uses its own loader and the column-0 background ("mad_column0"); that is the
construction behind the quoted 450.4 km, so it is kept here on purpose.

Run from refactored/ (loads the full S6 cube, a few minutes):
    conda run -n dsmc python deck/compute_phase_lag_cache.py
Writes outputs/s1_s6_azimuthal/analysis/phase_lag_S6_mad_column0.npz
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import dsmclib
from dsmclib import registry
from dsmclib.paths import output_dir

LEGACY_ROOT = Path(dsmclib.__file__).resolve().parents[2]   # frozen root scripts sit beside refactored/
sys.path.insert(0, str(LEGACY_ROOT))

import compare_horizontal_phase_lag_profiles as chp  # noqa: E402
from helper_funcs import detect_wave_period  # noqa: E402
from optimized_loader import OptimizedDSMCDataProcessor  # noqa: E402
from s6_reviewer_facts import odr_fit_full  # noqa: E402

ALIAS = "S6"


def main():
    processor = OptimizedDSMCDataProcessor(str(registry.run(ALIAS).path))
    processor.load_parameters()
    processor.load_data()

    osc_cols = processor.compute_osc_cols()
    j_left, j_right = int(osc_cols[0]), int(osc_cols[-1])
    time = processor.params.time
    wave_start = processor.params.start_step_wave * processor.params.timestep
    wave_end = processor.params.final_step_wave * processor.params.timestep
    bg = processor.get_background_profiles()

    r_idx = int(np.argmin(np.abs(processor.params.radial_cells - chp.REFERENCE_ALTITUDE_KM)))
    centre = int((j_left + j_right) / 2)
    in_wave = (time >= wave_start) & (time <= wave_end)
    series = processor.density_data[np.where(in_wave)[0], r_idx, centre]
    pert = (series - bg["density"][r_idx]) / bg["density"][r_idx] * 100
    period, _, _, _ = detect_wave_period(time[in_wave], pert)
    d_phi_km = chp.compute_horizontal_resolution(processor.params, chp.REFERENCE_ALTITUDE_KM)

    side = {d: chp.process_horizontal_at_altitude(processor, chp.REFERENCE_ALTITUDE_KM, period, bg, time,
                                                  wave_start, wave_end, direction=d)
            for d in ("left", "right")}
    L, R = side["left"], side["right"]
    distance = np.concatenate([L["distances_km"], R["distances_km"][1:]])
    lag = np.concatenate([L["phase_lags"], R["phase_lags"][1:]])
    lag_err = np.concatenate([L["uncertainties"], R["uncertainties"][1:]])
    distance_err = np.full(distance.size, d_phi_km)
    distance_err[0] = 0.0
    is_left = np.arange(distance.size) < len(L["distances_km"])

    fit = odr_fit_full(distance, lag, lag_err, distance_err)
    print(f"lambda_h = {fit['lam']:.2f} +/- {fit['lam_err']:.2f} km, n = {fit['n']} "
          f"(excluded {fit['n_excluded']}), reduced chi2 = {fit['chi2']:.3f}")

    path = output_dir("s1_s6_azimuthal", "analysis") / f"phase_lag_{ALIAS}_mad_column0.npz"
    np.savez(path, distance_km=distance, lag_periods=lag, lag_err_periods=lag_err,
             distance_err_km=distance_err, is_left=is_left,
             used=(lag > 0.001) & (lag_err > 0) & (distance_err > 0),
             slope_per_km=fit["slope"], intercept_periods=fit["intercept"],
             lambda_h_km=fit["lam"], lambda_h_err_km=fit["lam_err"], n=fit["n"],
             reduced_chi2=fit["chi2"], altitude_km=float(processor.params.radial_cells[r_idx]),
             period_s=float(period), d_phi_km=float(d_phi_km), alias=ALIAS, background="mad_column0")
    print(path)


if __name__ == "__main__":
    main()
