import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def exponential_decay(t, y):
    return -0.5*y


sol = solve_ivp(fun=exponential_decay,
                t_span=[0, 10],
                y0=[2, 4, 8])

plt.plot(sol.t, sol.y.T)
plt.show()
