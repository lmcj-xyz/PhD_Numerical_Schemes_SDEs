from numpy.random import default_rng
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds


rng = default_rng()
time_steps = 2**10
epsilon = 10e-6
beta = 3/8
hurst = 1 - beta
sample_paths = 10**4
y0 = rng.normal(size=sample_paths)
time_start = 0
time_end = 1
dt = (time_end - time_start)/(time_steps - 1)
time_grid = np.linspace(time_start + dt, time_end, time_steps)
# Parameters to create fBm
points_x = 2**9  # According to the lower bound in the paper
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
gaussian_fbm = rng.standard_normal(size=points_x)
drift_array = ds.drift(gaussian_fbm, hurst, points_x, half_support, time_steps)[0],
noise = rng.normal(
        loc=0.0, scale=np.sqrt(dt),
        size=(time_steps, sample_paths)
        )


# SDE solution
def nonl(x):
    return np.sin(x)


y = ds.solve_mv(y0, drift_array, noise, time_start, time_end, time_steps,
                sample_paths, grid_x, half_support, points_x, time_steps, nonl)

#plt.plot(grid_x, drift_array, label=r"$b^N$")
plt.hist(y[0][0, :], bins=100, density=True, label="SDE terminal density")
plt.plot(grid_x, y[1][0, :], label="PDE density t = 0")
plt.plot(grid_x, y[1][int(time_steps/2), :], label="PDE density t = 1/2")
plt.plot(grid_x, y[1][-1, :], label="PDE density t = 1")
plt.legend()
plt.show()
