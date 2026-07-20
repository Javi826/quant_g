from scipy.stats import norm
import numpy as np

EULER_GAMMA = 0.5772156649
n_eff  = 7.9088
var_sr = 0.000767

z_n  = norm.ppf(1 - 1/n_eff)
z_ne = norm.ppf(1 - 1/(n_eff*np.e))
sr0  = np.sqrt(var_sr) * ((1-EULER_GAMMA)*z_n + EULER_GAMMA*z_ne)
print(sr0)