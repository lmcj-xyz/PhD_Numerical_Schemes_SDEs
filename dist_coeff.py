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
        self.dist = np.sum(np.multiply(self.fbm_path, self.df), axis=1)
    
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
        fbm_array = np.matmul(cholesky, g)
        return fbm_array

    # This is creating the array to perform the convolution of
    # f*p(x) where x is the same as thea argument x received by the function
    def normal_difference(self):
        length_grid = np.shape(self.grid)[0]
        diff_norm = np.zeros(shape=(length_grid, length_grid))
        #print("length of grid: ", length_grid)
        #dx = 2*self.limit/length_grid
        dx = self.limit/length_grid
        for i in range(length_grid):
            for j in range(length_grid):
                diff_norm[j, i] = quad(
                        lambda w: 
                        w*norm.pdf(
                            w, 
                            loc=self.grid[i],
                            scale=1/(self.time_steps**(8/3))
                            ),
                        self.grid[j] - dx,
                        self.grid[j] + dx
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
x = distribution(hurst = 0.75, limit = 4, points = 10**2, time_steps=10**1)
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
plt.figure()
plt.plot(x.grid, x.fbm_path)
plt.plot(x.grid, x.dist)
#plt.plot(x.grid, x.dist1)
plt.show()
#print(x.dist)
#print(x.df)
#plt.figure()
#plt.plot(x.grid, x.df)
#plt.show()
