#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:50:37 2023

@author: tari
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm
from scipy.integrate import quad_vec
rng = np.random.default_rng()
plt.rcParams['figure.dpi'] = 500


class FractionalBrownianMotion:
    def __init__(self,
                 hurst: float,
                 points: int):
        self.hurst = hurst
        self.points = points
        self.gaussian = rng.standard_normal(size=self.points)
        self.grid = np.linspace(start=1/self.points, stop=1, num=self.points)
        self.fbm = self.path()
        self.fbb = self.bridge()

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

    def bridge(self) -> np.ndarray:
        return self.fbm - (self.fbm[-1]/self.grid[-1])*self.grid

    def plot_fbm(self):
        plt.plot(self.grid, self.fbm)
        plt.title('Path of fBm with %d points' % (self.points+1))

    def plot_fbb(self):
        plt.plot(self.grid, self.fbb)
        plt.title('Path of "fB bridge" with %d points' % (self.points+1))


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
        self.delta = self.half_support/self.points
        self.kernel_array = self.kernel()
        self.drift_array = np.convolve(self.fbm, self.kernel_array, 'same')

    def kernel(self):
        kernel = np.zeros_like(self.grid)
        constant = -1/self.heat_parameter
        kernel = quad_vec(lambda u: constant*(self.grid + u)*norm.pdf(
            self.grid+u,
            loc=0,
            scale=m.sqrt(self.heat_parameter)
            ), -self.delta, self.delta)[0]
        return kernel

    def drift(self, x):
        drift = np.piecewise(
            x,
            [(i - self.delta <= x)*(x < i + self.delta) for i in self.grid],
            [self.drift_array[i] for i in range(self.points)]
            )
        return drift


class EulerMaruyama:
    def __init__(self,
                 time_steps: int,
                 time_start: float,
                 time_end: float,
                 initial_condition: float,
                 brownian_motion: np.ndarray,
                 drift
                 ):
        self.time_steps = time_steps
        self.y0 = initial_condition
        self.bm = brownian_motion
        self.time_steps = time_steps
        self.time_start = time_start
        self.time_end = time_end
        self.sample_paths = self.bm.shape[1]
        self.drift = drift
        self.y = self.solve()

    def solve(self):
        y = np.zeros(shape=(self.time_steps+1, self.sample_paths))
        dt = (self.time_end - self.time_start)/(self.time_steps-1)
        y[0, :] = self.y0
        for i in range(time_steps):
            y[i+1, :] = y[i, :] + self.drift(x=y[i, :])*dt + self.bm[i, :]
        return y


# Testing
hurst = 0.75
points = 2**9
time_steps = 2**5
half_support = 10
grid = np.linspace(-half_support, half_support, points)
grid2x5 = np.linspace(-half_support, half_support, 2**5)
t0 = 0
t1 = 1
sp = 3
y0 = 1

fbm_object = FractionalBrownianMotion(hurst=hurst, points=points)
plt.figure()
fbm_object.plot_fbm()
fbm_object.plot_fbb()
plt.show()
fbm_path = fbm_object.fbm.copy()

brownian_motion = BrownianMotion(
        time_steps=time_steps, initial_time=t0, final_time=t1, paths=sp)
brownian_motion.lower_resolution(new_time_steps=2**4)
brownian_motion.bm
type(brownian_motion.lower_resolution(new_time_steps=2**3))  # See type
np.shape(brownian_motion.lower_resolution(new_time_steps=2**8))  # Raise except

drift = DistributionalDrift(
        fbm_path, hurst, time_steps, points, grid, half_support)
drift_array = drift.drift_array
drift_function = drift.drift(grid)
drift_function_2x5 = drift.drift(grid2x5)

solution1 = EulerMaruyama(
        time_steps=time_steps, time_start=t0, time_end=t1,
        initial_condition=y0,
        brownian_motion=brownian_motion.bm,
        drift=drift.drift)
solution2 = EulerMaruyama(
        time_steps=2**4, time_start=t0, time_end=t1,
        initial_condition=y0,
        brownian_motion=brownian_motion.lower_resolution(2**4),
        drift=drift.drift)
paths_solution1 = solution1.y
paths_solution2 = solution2.y
