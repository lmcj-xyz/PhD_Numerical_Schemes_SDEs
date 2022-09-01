import euler as e
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Time steps
M = 10**5

# n(m) = m^2

# Distributional drift
def bn (t, x, m):
    n = m**2
    return norm.cdf(x, loc=0, scale=1/n)

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
error, rate = y.rate(real_solution = y.solve(), approximations = 3, 
        show_plot = True, save_plot = True)
print("error array", error)
print("rate =", rate)
