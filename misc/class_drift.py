import numpy as np
from numpy.random import default_rng

rng = default_rng()


class DriftFunction:
    def __init__(self, drift_array, grid):
        self.drift_array = drift_array
        self.grid = grid
        self.points = len(grid)
        self.delta_half = grid[-1]/(self.points - 1)
        self.grid_list = [(i - self.delta_half <= x)*(x < i + self.delta_half) for i in grid]
        self.drift_list = [drift_array[i] for i in range(self.points)]

    def eval(self, x):
        return np.piecewise(x, self.grid_list, self.drift_list)


def euler_scheme(drift, time_min=0, time_max=1, time_steps=100):
    y = np.zeros(time_steps)
    std = (time_max - time_min)/(time_steps - 1)
    noise = rng.normal(loc=0.0, scale=std, size=time_steps)
    for i in range(time_steps):
        y[i+1] = y[i] + drift.eval(y[i]) + noise[i]
    return y


x = np.linspace(-5, 5, 100)
da = np.sin(x)
drift_test = DriftFunction(x, da)

euler_scheme(drift=drift_test)
