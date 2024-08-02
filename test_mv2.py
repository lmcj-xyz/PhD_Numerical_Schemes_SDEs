import dsdes as ds
from numpy.random import default_rng
import matplotlib.pyplot as plt
import numpy as np
import sys


rng = default_rng()
time_steps = 2**8
epsilon = 10e-6
beta = 1/4
hurst = 1 - beta
y0 = 1
sample_paths = 10**4
time_start = 0
time_end = 1
dt = (time_end - time_start)/(time_steps - 1)
time_grid = np.linspace(time_start + dt, time_end, time_steps)
noise = rng.normal(
        loc=0.0, scale=np.sqrt(dt),
        size=(time_steps, sample_paths)
        )
# Parameters to create fBm
points_x = 2**12  # According to the lower bound in the paper
half_support = 10
eta = 1/((hurst-1/2)**2 + 2 - hurst)
lower_bound = 2*half_support*time_steps**(eta/2)
eqn = 66
if (points_x <= lower_bound):
    msg = 'You need to define your fBm on at least %.2f \
            points as per equation (%d) in the paper.' % (lower_bound, eqn)
    raise ValueError(msg)
    sys.exit(1)
delta_x = half_support/(points_x-1)
grid_x = np.linspace(start=-half_support, stop=half_support, num=points_x)
grid_x0 = np.linspace(start=0, stop=2*half_support, num=points_x)
fbm_array = ds.fbm(hurst, points_x, half_support)
bridge_array = ds.bridge(fbm_array, grid_x0)
# Variance of heat kernel
var_heat_kernel = ds.heat_kernel_var(time_steps, hurst)
# Integral between grid points
integral_grid = ds.integral_between_grid_points(var_heat_kernel, grid_x, half_support)
# Drift creation
drift_array = ds.create_drift_array(bridge_array, integral_grid)


# PDE solution

y = ds.solve_mv(y0, drift_array, noise, time_start, time_end, time_steps, sample_paths, grid_x, half_support, 10, 2**8)

plt.plot(y[:, 0:5])
plt.show()
