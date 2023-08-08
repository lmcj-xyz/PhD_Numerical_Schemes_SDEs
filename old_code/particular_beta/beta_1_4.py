import euler as e
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dist_coeff import *

import time

st = time.process_time()

# Time steps
M = 10**3
# Instance of distributional coefficient
#dist = Distribution(hurst=0.75, limit=5, points=10**2)

# n(m) = m^(8/3)

# Distributional drift
beta = 0.25
h = 1 - beta
l = 5
def_points_bn = 10**2
def bn (t, x, m):
    dist = Distribution(hurst=h, limit=l, points=def_points_bn, time_steps=m)
    return dist.func(t, x, m)

# Constant diffusion
def sigma (t, x, m):
    return 1

# Euler approximation
y = e.Euler(
        drift = bn,
        diffusion = sigma,
        time_steps = M,
        paths = 10,
        y0 = 1
        )

# Solution
#y.plot_solution(paths_plot=3, save_plot=False)

# Rate of convergence
error, rate = y.rate(real_solution = y.solve(), approximations = 2, 
        show_plot = True, save_plot = False)
print("error array", error)
print("rate =", rate)

et = time.process_time()
print("time: ", et-st)
