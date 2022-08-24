import Euler as e
import numpy as np
import matplotlib.pyplot as plt

# Time steps
m = 10**6

# Distributional drift
def bn (t, x):
    return 0

# Constant diffusion
def sigma (t, x):
    return 0

# Euler approximation
y = e.Euler(
        drift = bn,
        diffusion = sigma,
        time_steps = m,
        paths = 10,
        y0 = 0
        )

# Rate of convergence
print(y.rate())
