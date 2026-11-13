import sys, numpy as np, xarray as xr
R = "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored/docs/reports/ubc_hold_carberry/scripts"
sys.path.insert(0, R)
H = "/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored/outputs/h2_hold/analysis/"
d = xr.open_dataset(H + "E8_main_kepler_analysis.nc")
print([v for v in d.data_vars if v.startswith('hold')])
i = d.hold_n_hold_in.values
print('E8 in per interval', i)
print('sum', i.sum(), 'mean rate', i.sum() / float(d.hold_time[-1]))
import analysis as an
A = an.Analysis() if hasattr(an, 'Analysis') else None
print([n for n in dir(an) if not n.startswith('_')])
