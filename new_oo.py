#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:50:37 2023

@author: tari
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm
from scipy.integrate import quad_vec
rng = np.random.default_rng()
plt.rcParams['figure.dpi'] = 500
#%% fBm class
class FractionalBrownianMotion:
    def __init__(self, 
                 hurst: float, 
                 points: int):
        
        self.hurst = hurst
        self.points = points
        self.gaussian = rng.standard_normal(size=self.points)
        self.grid = np.linspace(start=1/self.points, stop=1, num=self.points)
        self.fbm = self.path()
        
    def path(self) -> np.ndarray:
        x, y = np.meshgrid(self.grid, self.grid, sparse=False, indexing='ij')
        covariance = 0.5*(
            np.abs(x)**(2*self.hurst) +
            np.abs(y)**(2*self.hurst) - 
            np.abs(x - y)**(2*self.hurst)
            )
        cholesky = np.linalg.cholesky(a=covariance)
        fbm_arr = np.matmul(cholesky, self.gaussian)
        return fbm_arr
    
    def plot(self):
        plt.figure()
        plt.plot(self.grid, self.fbm)
        plt.title("Path of fBm with %d points" % (self.points+1))
        plt.show()
        
#%% Brownian motion class
class BrownianMotion:
    def __init__(self, 
                 time_steps: int, 
                 initial_time: float, 
                 final_time: float, 
                 paths: int):
        
        self.time_steps = time_steps
        self.dt = (final_time - initial_time)/(time_steps-1)
        self.paths = paths
        self.bm = rng.normal(loc=0.0,
                             scale=m.sqrt(self.dt),
                             size=(self.time_steps, self.paths)
                             )
        
    def lower_resolution(self, new_time_steps: int) -> np.ndarray:
        coarse_bm = np.zeros(shape=(new_time_steps, self.paths))
        original_bm = self.bm.copy()
        quotient = int(self.time_steps/new_time_steps)
        if quotient == 1:
            coarse_bm = original_bm
            print("\nThe number of time steps \
                  provided are the same as the \
                      maximum amount of time steps.\
                          \nThe output is the original Brownian motion!\n")
            return coarse_bm
        elif quotient > 1:
            temp = original_bm.reshape(new_time_steps, quotient, self.paths)
            coarse_bm = np.sum(temp, axis=1)
            print("\nThe output is the corresponding \
                  Brownian motion now with %d time \
                      steps instead of the maximum amount of \
                          %d.\n" % (new_time_steps, self.time_steps))
            return coarse_bm
        else:
            raise ValueError("Impossible to lower the \
                             resolution of the Brownian \
                                 motion if the new time \
                                     steps are more than \
                                         the maximum time steps.\
                                             \nTry a smaller number!")

#%% Distributional drift class
class DistributionalDrift:
    def __init__(self,
                 fbm: np.ndarray,
                 hurst: float,
                 time_steps: int,
                 points: int,
                 grid: np.ndarray,
                 half_support: float):
        self.fbm = fbm
        self.hurst = hurst
        self.time_steps = time_steps
        self.eta = 1/(2*(self.hurst-1/2)**2 + 2 - self.hurst)
        self.heat_parameter = 1/(self.time_steps**(self.eta))
        self.points = points
        self.grid = grid
        self.half_support = half_support
        self.drift_array = np.convolve(self.fbm, self.kernel, 'same')
        self.delta = self.half_support/self.points
        self.kernel = self.kernel()
        
    def kernel(self):
        kernel = np.zeros_like(self.grid)
        constant = -1/self.heat_parameter
        p = lambda u: constant*(self.grid + u)*norm.pdf(
            self.grid+u,
            loc=0,
            scale=m.sqrt(self.heat_parameter)
            )
        kernel = quad_vec(p, -self.delta, self.delta)[0]
        return kernel
    
    def drift(self, x):
        drift = np.piecewise(
            x, 
            [(i - self.delta <= x)*(x < i + self.delta) for i in self.grid], 
            [self.drift_array[i] for i in range(self.points)]
            )
        return drift
#%% Euler-Maruyama class
class EulerMaruyama:
    def __init__(self, 
                 time_steps: int, 
                 time_start: float,
                 time_end: float,
                 initial_condition: float, 
                 brownian_motion: np.ndarray,
                 ):
        
        self.time_steps = time_steps
        self.y0 = initial_condition
        self.bm = brownian_motion
#%%
hurst = 0.75
points = 2**9
time_steps = 2**14
half_support = 10
grid = np.linspace(-half_support, half_support, points)

fbm_object = FractionalBrownianMotion(hurst=hurst, points=points)
fbm_object.plot()
fbm_path = fbm_object.fbm.copy()

brownian_motion = brownian_motion_object = BrownianMotion(time_steps=2**4, initial_time=0, final_time=1, paths=3)
brownian_motion.lower_resolution(new_time_steps=2**4)
brownian_motion.bm
type(brownian_motion.lower_resolution(new_time_steps=2**3)) # See type
np.shape(brownian_motion.lower_resolution(new_time_steps=2**8)) # Raise an error

drift = DistributionalDrift(fbm_path, hurst, time_steps, points, grid, half_support)
