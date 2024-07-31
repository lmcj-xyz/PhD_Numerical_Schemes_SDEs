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


class FractionalBrownianMotion:
    def __init__(self, hurst: float, points: int):
        self.hurst = hurst
        self.points = points
        self.gaussian = rng.standard_normal(size=self.points)
        self.grid = np.linspace(start=1/self.points, stop=1, num=self.points)
        self.fbm = self.path()
        self.fbb = self.bridge()

    def __str__(self):
        return f'FractionalBrownianMotion(hurst = {self.hurst})'

    def path(self) -> np.ndarray:
        x, y = np.meshgrid(self.grid, self.grid, sparse=False, indexing='ij')
        covariance = 0.5*(
            np.abs(x)**(2*self.hurst) +
            np.abs(y)**(2*self.hurst) -
            np.abs(x - y)**(2*self.hurst)
            )
        cholesky = np.linalg.cholesky(covariance)
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

    def lower_resolution(self, new_time_steps: int, verbose=False) -> np.ndarray:
        coarse_bm = np.zeros(shape=(new_time_steps, self.paths))
        original_bm = self.bm.copy()
        quotient = int(self.time_steps/new_time_steps)
        if quotient == 1:
            coarse_bm = original_bm
            if verbose:
                print("The number of time steps \
                    provided are the same as the \
                    maximum amount of time steps.\
                    \nThe output is the original Brownian motion!\n")
            return coarse_bm
        elif quotient > 1:
            temp = original_bm.reshape(new_time_steps, quotient, self.paths)
            coarse_bm = np.sum(temp, axis=1)
            if verbose:
                print(f"The output is the corresponding \
                    Brownian motion now with {new_time_steps} \
                    time steps instead of the maximum amount of time steps \
                    {self.time_steps}.\n")
            return coarse_bm
        else:
            raise ValueError(
                    f"Impossible to lower the resolution of the Brownian \
                    motion if new_time_steps > time_steps \
                    \nTry a choosing new_time_steps less than \
                    {self.time_steps}!")


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

    def eval(self, x):
        drift = np.piecewise(
            x,
            [(i - self.delta <= x)*(x < i + self.delta) for i in self.grid],
            [self.drift_array[i] for i in range(self.points)]
            )
        return drift


class DistributionalSDE:
    def __init__(self,
                 initial_condition: float,
                 time_start: float,
                 time_end: float,
                 brownian_motion: BrownianMotion,
                 drift: DistributionalDrift
                 ):
        self.y0 = initial_condition
        self.time_start = time_start
        self.time_end = time_end
        self.brownian_motion = brownian_motion
        self.sample_paths = self.brownian_motion.bm.shape[1]
        self.drift = drift.eval

    def real_solution(self, time_steps: int):
        return self.solve(time_steps)

    def approx(self, time_steps: list[int]):
        approx_list = []
        for t in time_steps:
            approx_list.append(self.solve(t))
        return approx_list

    def solve(self, time_steps):
        y = np.zeros(shape=(time_steps+1, self.sample_paths))
        dt = (self.time_end - self.time_start)/(time_steps-1)
        y[0, :] = self.y0
        bm = self.brownian_motion.lower_resolution(time_steps)
        for i in range(time_steps):
            y[i+1, :] = y[i, :] + self.drift(x=y[i, :])*dt + bm[i, :]
        return y

    def plot(self, trajectories):
        pass


class StrongError:
    def __init__(self,
                 real_solution: np.ndarray,
                 approximation: list[np.ndarray]):
        self.real_solution = real_solution
        self.approximation = approximation
        self.keys = tuple(f'approx{i+1}' for i, _ in enumerate(self.approximation))

    def calculate(self):
        error = dict.fromkeys(self.keys)
        for i, k in enumerate(error):
            error[k] = np.mean(
                    np.abs(self.real_solution[-1] - self.approximation[i][-1])
                    )
        return error

    def log(self):
        error = self.calculate()
        log_error = dict.fromkeys(error.keys())
        for k, v in error.items():
            log_error[k] = np.log(error[k])
        return log_error

    def plot(self):
        pass


def main():
    return 0


if __name__ == "__main__":
    main()
