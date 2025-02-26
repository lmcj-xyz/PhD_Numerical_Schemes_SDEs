import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dsdes import fbm, bridge, heat_kernel_var, integral_between_grid_points

theseed = 1894832884316783
rng = np.random.default_rng(seed=theseed)

half_support = 5
points = 100
gaussian = rng.standard_normal(points)
hurst = 0.75
time_steps = 2**10

grid = np.linspace(-half_support, half_support, points)
grid0 = np.linspace(0, 2*half_support, points)
fbm_array = fbm(gaussian, hurst, points, half_support)
fbb_array = bridge(fbm_array, grid0)
hk = heat_kernel_var(time_steps, hurst)
ig = integral_between_grid_points(hk, grid, half_support)

plt.plot(grid, ig)
plt.show()
