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
from datetime import datetime

class Euler:
    """
    Class used to solve a Stochastic Differential Equation (SDE) using the
    Euler-Maruyama method.

    This will give us the solution to an SDE that satisfies the usual
    assumptions, refer to Kloeden & Platen, Numerical Solutions for Stochastic
    Differential Equations
    """

    def __init__ (
            self, 
            drift, 
            diffusion, 
            time_steps = None,
            time_start = None,
            time_end = None,
            y0 = None, 
            paths = None
            ):
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

        # Initialize arguments
        self.drift = drift
        self.diffusion = diffusion
        self.time_steps = time_steps if time_steps \
                is not None else 10**2
        self.time_start = time_start if time_start \
                is not None else 0
        self.time_end = time_end if time_end \
                is not None else 1
        self.y0 = y0 if y0 \
                is not None else 1
        self.paths = paths if paths \
                is not None else 100

        # Time step
        self.dt = (self.time_end - self.time_start) / self.time_steps

        # Creation of normal random variable
        ## Since the time grid would start in t_1, and the random variable
        ## in z_1 we have a bijection between the time grid and each path
        ## of the rv which cannot achieved if we start the time grid in t_0.
        ## Well it can be achieved, but is not what we need given the recursive
        ## nature of the Brownian motion when having coarse grids
        self.z = rng.normal(
                loc = 0.0,
                scale = np.sqrt(self.dt),
                size = (self.paths, self.time_steps)
                )
        # Creation of the time grid
        ## We create the time grid starting from the first non-zero term,
        ## effectively we do not need to use the first time (typically 0)
        ## in any moment
        ## Also this is the finest grid, in the solve method we can use this
        ## or a coarser one
        self.time_grid = self.generate_time_grid()

    #def coarse_z (self, self.z):

    #def coarse_time_grid (self, self.time_grid):

    #def rate (self):

    def generate_time_grid (
            self, 
            time_start_grid = None,
            time_end_grid = None,
            dt_grid = None,
            time_steps_grid = None
            ):
        time_start_grid = time_start_grid if time_start_grid \
                is not None else self.time_start
        time_end_grid = time_end_grid if time_end_grid \
                is not None else self.time_end
        time_steps_grid = time_steps_grid if time_steps_grid \
                is not None else self.time_steps
        dt_grid = dt_grid if dt_grid \
                is not None else self.dt

        time_grid_generated = np.linspace(
                start = time_start_grid + dt_grid, 
                stop = time_end_grid,
                num = time_steps_grid
                )
        return time_grid_generated

    def solve (
            self,
            time_steps_solve = None,
            #time_grid_solve = None,
            dt_solve = None,
            z_solve = None
            ):
        time_steps_solve = time_steps_solve if time_steps_solve \
                is not None else self.time_steps
        time_grid_solve = self.generate_time_grid(time_steps_solve) if time_steps_solve \
                is not None else self.time_grid
        dt_solve = dt_solve if dt_solve \
                is not None else self.dt
        z_solve = z_solve if z_solve \
                is not None else self.z

        """ Solve the SDE """
        # Creation of the placeholder for the solution.
        self.y = np.zeros(shape = (self.paths, time_steps_solve))
        # And adding intial condition.
        self.y[:, 0] = self.y0

        for i in range(time_steps_solve - 1):
            self.y[:, i+1] = self.y[:, i] \
                    + self.drift(
                            self.y[:, i], 
                            time_grid_solve[i]
                            )*dt_solve \
                    + self.diffusion(
                            self.y[:, i],
                            time_grid_solve[i]
                            )*z_solve[:, i+1]
        return self.y

    def plot_solution (self, paths_plot, save_plot = False):
        """
        Plot the solutions

        Here you can select the number of paths to plot with paths_plot

        Remember that this method is not very useful in any other situation
        besides testing, its just a sanity check
        """
        solution = plt.figure()
        plt.plot(self.solve()[range(paths_plot)].T)
        plt.show()
        if save_plot == True:
            # For organization reasons the figures are saved into the
            # ./figures_solution/ directory with a time stamp
            solution.savefig(
                    fname = 
                    'figures_solution/'
                    +
                    datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
                    +
                    '_solution'
                    )
