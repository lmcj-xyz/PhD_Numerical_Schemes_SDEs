import euler as e
import numpy as np
import matplotlib.pyplot as plt

# Time steps
M = 10**5

# Distributional drift
def bn (t, x, m):
    #return np.sqrt(M**2/2*np.pi)*np.trapz(y=np.exp(-(m**2*x**2)/2), x=x)
    return 3

# Constant diffusion
def sigma (t, x, m):
    return 1*x

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
