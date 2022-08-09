# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:06:58 2022

@author: mmlmcj

@title: Euler scheme
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()

class Euler:
    """
    
    This will give us the solution to an SDE that satisfies the usual
    assumptions, refer to Kloeden & Platen, Numerical Solutions for Stochastic
    Differential Equations

    """
    def __init__ (self, drift, diffusion, time_steps = None, time_start = None, time_end = None, y0 = None, paths = None):
        """

        Parameters
        ----------
        drift:
            Drift coefficient of the equation, satisfying usual conditions.
        diffusion:
            Diffusion coefficient of the equation, satisfying usual conditions.
        time_steps: int
            The amount of time steps we use to compute the solution, 
            which defines how fine is the grid.
        time_start: float
            The starting time of the scheme.
        time_end: float
            The ending time of the scheme.
        y0: float
            Initial condition for the equation.
        paths: int
            Amount of paths to comput the solution.

        Returns
        -------
        z: float
            A random variable to use in the Euler scheme as the Brownian motion.
        dt:
            The step size in time.
        time_grid: float
            Time grid to compute the solution.
        y: float
            Placeholder for the solution of the SDE.

        """
        self.drift = drift
        self.diffusion = diffusion
        self.time_steps = time_steps if time_steps is not None else 100
        self.time_start = time_start if time_start is not None else 0
        self.time_end = time_end if time_end is not None else 1
        self.y0 = y0 if y0 is not None else 0
        self.paths = paths if paths is not None else 100

        """

        Creation of the time grid

        """
        self.time_grid = np.linspace(
                start = self.time_start, 
                stop = self.time_end,
                num = self.time_steps
                )

        """

        Creation of normal random variable

        """
        self.dt = self.time_grid[1]
        self.z = rng.normal(
                loc = 0.0,
                scale = np.sqrt(self.dt),
                size = (self.paths, self.time_steps)
                )

        """

        Creation of the placeholder for the solution.

        """
        self.y = np.zeros(shape = (self.paths, self.time_steps))
        self.y[:, 0] = self.y0

    """

    Solve the SDE

    """
    def solve (self):
        for i in range(self.time_steps - 1):
            self.y[:, i+1] = self.y[:, i] \
                    + self.drift(
                            self.y[:, i], 
                            self.time_grid[i]
                            )*self.dt \
                    + self.diffusion(
                            self.y[:, i],
                            self.time_grid[i]
                            )*self.z[:, i+1]
        return self.y

    """

    Plot the solutions

    Here you can select the number of paths to plot with paths_plot

    Remember that this method is not very useful in any other situation
    besides testing

    """
    def plot_solution (self, paths_plot):
        plt.figure()
        plt.plot(self.y[range(paths_plot)].T)
        plt.show()


"""

Testing area

"""
def mu(x, t):
    return 1*x

def sigma(x, t):
    return x

y = Euler(drift = mu, diffusion = sigma, time_steps = 100, paths = 10)#, paths_plot = 3)
print("time steps:\n", y.time_steps)
#print("time grid:", y.time_grid)
#print("dt:\n", y.dt)
print("random variable:\n", y.z)
#print("solution y placeholder:\n", y.y)
#print("drift:\n", y.drift(y.y, y.time_grid))
print("solution:\n", y.solve())
y.plot_solution(paths_plot = 3)
