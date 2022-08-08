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
    
    This will give us the solution to an SDE that satisfies the usual assumptions, refer to Kloeden & Platen, Numerical Solutions for Stochastic Differential Equations

    """
    def __init__ (self, time_steps, time_start, time_end, drift, diffusion, y0):
        """

        Parameters
        ----------
        time_steps: int
            The amount of time steps we use to compute the solution, which defines how fine is the grid.
        time_start: float
            The starting time of the scheme.
        time_end: float
            The ending time of the scheme.
        drift:
            Drift coefficient of the equation, satisfying usual conditions.
        diffusion:
            Diffusion coefficient of the equation, satisfying usual conditions.
        y0: 
            Initial condition for the equation.

        """
