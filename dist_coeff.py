# September 9th 14:38:38
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
    def __init__(self, hurst, limit, points):
        self.hurst = hurst
        self.limit = limit
        self.points = points

        self.grid = np.linspace(
                start = -limit,
                stop = limit,
                num = points
                )

        self.fbm_path = self.fbm()
    
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

    def b(self, x, t, m):
        diff_norm = np.zeros_like(self.grid)
        length_grid = np.shape(self.grid)[0]
        dx = 2*self.limit/length_grid
        for i in range(length_grid):
            diff_norm[i] = norm(i + dx, loc=i, scale=1/(m**(8/3))) \
                    - norm(i - dx, loc=i, scale=1/(m**(8/3)))
        b_array = diff_norm
        return b_array
        
       
# Tests
x = distribution(hurst = 0.75, limit = 4, points = 10)
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
print(x.b(x=2, t=1, m=10))
