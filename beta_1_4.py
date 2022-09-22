import euler as e
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dist_coeff import *

# Time steps
M = 10**3
# Instance of distributional coefficient
#dist = Distribution(hurst=0.75, limit=5, points=10**2)

# n(m) = m^(8/3)

# Distributional drift
def bn (t, x, m):
    dist = Distribution(hurst=0.75, limit=5, points=10**2, time_steps=m)
    return dist.func(t, x, m)

# Constant diffusion
def sigma (t, x, m):
    return 1

# Euler approximation
y = e.Euler(
        drift = bn,
        diffusion = sigma,
        time_steps = M,
        paths = 100,
        y0 = 1
        )

# Rate of convergence
error, rate = y.rate(real_solution = y.solve(), approximations = 2, 
        show_plot = True, save_plot = False)
print("error array", error)
print("rate =", rate)
