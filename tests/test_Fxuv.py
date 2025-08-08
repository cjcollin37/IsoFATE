from IsoFATE.isofate.isofunks import Fxuv
from IsoFATE.isofate.constants import s2yr
import numpy as np

t=1e6, F0=1e2, t0 = 1e6, t_sat = 5e8, beta = -1.23, step_fn = False, F_final = 0, t_pms = 1e7, pms_factor = 1e2
time = t*s2yr
assert 0 < time < t_pms
F_pms0 = F0*pms_factor
s = (np.log10(F0) - np.log10(F_pms0)) / (np.log10(t_pms) - np.log10(t0))
assert Fxuv(t,F0) == F_pms0*(time/t0)**s

t=1e6, F0=1e2, t0 = 1e6, t_sat = 5e8, beta = -1.23, step_fn = False, F_final = 0, t_pms = 1e7, pms_factor = 1e2
time = t*s2yr
assert time < t_sat
assert Fxuv(t,F0) == F0

t=1e6, F0=1e2, t0 = 1e6, t_sat = 0, beta = -1.23, step_fn = False, F_final = 0, t_pms = 1e7, pms_factor = 1e2
time = t*s2yr
assert time >= t_sat
assert step_fn == False
assert Fxuv(t,F0) == F0*(time/t_sat)**beta

time=1e6, F0=1e2, t0 = 1e6, t_sat = 5e8, beta = -1.23, step_fn = True, F_final = 0, t_pms = 1e7, pms_factor = 1e2
time = t*s2yr
assert step_fn == True
assert Fxuv(t,F0) == F_final
