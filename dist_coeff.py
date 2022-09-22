# -*- coding: utf-8 -*-
"""
Created on Fri Sept  9 14:38:38 2022

@author: Luis Mario Chaparro Jáquez

@title: Approximation of distributional coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from datetime import datetime
from scipy.stats import norm
from scipy.integrate import quad_vec, quad

#from numba import jit

class Distribution:
    """
    Generates a distributional coefficient by computing the generalised
    derivative of a fractional Brownian motion (fBm).

    Parameters
    ----------
    hurst: float
        Hurst coefficient for the fBm
    limit: float
        Upper limit of the interval to compute the fBm, to create an interval
        of the form [-limit, limit]
    points: int
        Amount of points that will be in the interval generated, and in the fBm
        itself

    Returns
    -------

    """
    def __init__(self, hurst, limit, points, time_steps):
        self.hurst = hurst
        self.limit = limit
        self.points = points
        self.time_steps = time_steps
        #self.time_steps2 = time_steps/10

        self.t_heat = np.sqrt(1/(self.time_steps**(8/3)))
        #self.t_heat2 = np.sqrt((1/(self.time_steps2**(8/3))))

        self.grid = np.linspace(
                start = -limit,
                stop = limit,
                num = points
                )
        self.length_grid = self.grid.shape[0]

        self.fbm_path = self.fbm()

        #self.df = self.normal_differences(0.1)
        self.df = self.normal_differences(self.t_heat)
        
        ##self.dist, self.dist1 = np.sum(np.multiply(self.fbm_path, self.df), axis=1)
        self.dist_array = np.sum(np.multiply(self.fbm_path, self.df.T), axis=1)
        ##self.dist_array2 = np.sum(np.multiply(self.fbm_path, self.df2.T), axis=1)

        # Tests
        self.con = self.constant()
        self.conconv = np.sum(np.multiply(self.con, self.df.T), axis=1)
        #self.conconv2 = np.sum(np.multiply(self.con, self.df2.T), axis=1)

        self.zer = self.zeros()
        self.zerconv = np.sum(np.multiply(self.zer, self.df.T), axis=1)
        #self.zerconv2 = np.sum(np.multiply(self.zer, self.df2.T), axis=1)

        self.lin = self.linear()
        self.linconv = np.sum(np.multiply(self.lin, self.df.T), axis=1)
        #self.linconv2 = np.sum(np.multiply(self.lin, self.df2.T), axis=1)
    
    def zeros(self):
        zeros_arr = np.zeros_like(self.grid)
        return zeros_arr

    def constant(self):
        constant_arr = np.ones_like(self.grid)
        return constant_arr
        
    def linear(self):
        linear_arr = self.grid
        return linear_arr

    def fbm(self):
        x_grid, y_grid = np.meshgrid(
                self.grid, 
                self.grid, 
                sparse=False,
                indexing='ij'
                )
        covariance = 0.5*(
                np.abs(x_grid)**(2*self.hurst) +
                np.abs(y_grid)**(2*self.hurst) - 
                np.abs(x_grid - y_grid)**(2*self.hurst)
                )
        g = rng.standard_normal(size=self.points)
        cholesky = np.linalg.cholesky(a=covariance)
        fbm_arr = np.matmul(cholesky, g)
        return fbm_arr

    # This is creating the array to perform the convolution of
    # f*p(x) where x is the same as thea argument x received by the function
    #@jit(nopython=True)
    def normal_differences(self, t_var):
        diff_norm = np.zeros(shape=(self.length_grid, self.length_grid))
        delta = self.limit/self.length_grid
        const = -1/t_var**2

        # Array of functions
        p = lambda u: const*(self.grid - u)*norm.pdf(self.grid, loc=u, scale=t_var)
        for j in range(self.length_grid):
            jj = self.grid[j]
            diff_norm[j, :] = quad_vec(p, jj - delta, jj + delta)[0]

        ## Horrible nested loops
        #for i in range(self.length_grid):
        #    ii = self.grid[i]
        #    p = lambda u: const*(ii - u)*norm.pdf(ii, loc=u, scale=t_var)
        #    for j in range(self.length_grid):
        #        jj = self.grid[j]
        #        diff_norm[j, i] = quad(p, jj - delta, jj + delta)[0]
        
        # Meshgrid attempt
        #xi, xj = np.meshgrid(self.grid, self.grid, sparse=False, indexing='ij')
        #diff_norm = (xj + delta)*norm.cdf(xj - delta, loc=xi, scale=t_var) \
        #        - (xj - delta)*norm.cdf(xj - delta, loc=xi, scale=t_var) \
        #        - 2*delta*norm.cdf(xj, loc=xi, scale=t_var)
        ##diff_norm = (xj + delta)*norm.cdf(xi, loc=xj - delta, scale=t_var) \
        ##          - (xj - delta)*norm.cdf(xi, loc=xj - delta, scale=t_var) \
        ##               - 2*delta*norm.cdf(xi, loc=xj, scale=t_var)
        #diff_norm = const*diff_norm

        return diff_norm#, diff_norm1

    def func(self, t, x, m):
        #var_heat = np.sqrt(1/(m**(8/3)))
        var_heat = self.t_heat
        #df = self.normal_differences(t_var=var_heat)
        df = self.df
        #dist_a = np.sum(np.multiply(self.fbm_path, df.T), axis=1)
        dist_a = self.dist_array
        #dist_a_1 = np.sum(np.multiply(self.fbm_path, df_1.T), axis=1)
        delta = self.limit/self.length_grid
        return np.piecewise(
                x, 
                [(k - delta <= x)*(x < k + delta) for k in self.grid],
                [k for k in dist_a]
                )
