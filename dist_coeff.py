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
from scipy.integrate import quad

class distribution:
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

        self.grid = np.linspace(
                start = -limit,
                stop = limit,
                num = points
                )

        self.fbm_path = self.fbm()

        self.df = self.normal_difference()
        
        #self.dist, self.dist1 = np.sum(np.multiply(self.fbm_path, self.df), axis=1)
        self.dist = np.sum(np.multiply(self.fbm_path, self.df.T), axis=1)

        # Tests
        self.con = self.constant()
        self.conconv = np.sum(np.multiply(self.con, self.df.T), axis=1)

        self.zer = self.zeros()
        self.zerconv = np.sum(np.multiply(self.zer, self.df.T), axis=1)

        self.lin = self.linear()
        self.linconv = np.sum(np.multiply(self.lin, self.df.T), axis=1)
    
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
    def normal_difference(self):
        length_grid = self.grid.shape[0]
        diff_norm = np.zeros(shape=(length_grid, length_grid))
        #print("length of grid: ", length_grid)
        delta = self.limit/length_grid
        for i in range(length_grid):
            for j in range(length_grid):
                p = lambda u: \
                        (-self.time_steps**(8/3))* \
                        (self.grid[i] - u)* \
                        norm.pdf( 
                                self.grid[i], 
                                loc=u,
                                scale=1/(self.time_steps**(8/3)) 
                                )
                diff_norm[j, i] = quad(
                        p,
                        self.grid[j] - self.grid[i] - delta,
                        #self.grid[j] - dx,
                        self.grid[j] - self.grid[i] + delta
                        #self.grid[j] + dx
                        )[0]
        #xi, xj = np.meshgrid(self.grid, self.grid, sparse=False, indexing='ij')
        #diff_norm_1 = quad(
        #                lambda w: 
        #                w*norm.pdf(
        #                    w, 
        #                    loc=xi,
        #                    scale=1/(self.time_steps**(8/3))
        #                    ),
        #                xj - dx,
        #                xj + dx
        #                )[0]

        return diff_norm#, diff_norm_1
       
# Tests
x = distribution(hurst=0.75, limit=1, points=10**1, time_steps=10**(1))
#print(x.grid)
## Covariance matrix
#cov = x.fbm()
#print(cov)
#plt.imshow(cov)
#plt.colorbar()
#plt.show()
## fBm
#frac = x.fbm_path
#plt.figure()
#plt.plot(x.grid, frac)
#plt.show()
#print("grid: ", x.grid)
#print("value of b: ", x.normal_difference(x=1, t=1, m=10))
#print(x.normal_difference_m)
#plt.figure()
#plt.plot(x.grid, x.fbm_path)
#plt.show()
#x.dist

#print(x.fbm_path)
#print(x.dist)
#print(x.conconv)
#print(x.zerconv)
#print(x.linconv)

plt.figure()
plt.plot(x.grid, x.fbm_path, label="fBm")
plt.plot(x.grid, x.dist, label="dist")
plt.plot(x.grid, x.conconv, label="constant")
#plt.plot(x.grid, x.zerconv, label="zeros")
plt.plot(x.grid, x.linconv, label="linear")
plt.legend()
plt.show()

#print(x.dist)
#print(x.df)
#plt.figure()
#plt.plot(x.grid, x.df)
#plt.show()

#plt.figure()
#plt.plot(x.df)
#plt.show()

#print(x.df.size)
#print(x.df.shape)
