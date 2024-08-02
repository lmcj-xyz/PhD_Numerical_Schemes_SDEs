#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:09:33 2023

@author: mmlmcj
"""
# %% libraries
import numpy as np
import math as m
from scipy.integrate import quad_vec
from numpy.random import default_rng
from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    PDEBase, \
    ScipySolver, \
    Controller
from scipy.stats import norm
from scipy.interpolate import interpn
import numpy as np
import matplotlib.pyplot as plt

rng = default_rng()


# %% fbm func
# Fractional Brownian motion (fBm) creation function
def fbm(hurst: float, points: int, half_support: float) -> np.ndarray:
    fbm_grid = np.linspace(
            start=1/points,
            stop=2*half_support,
            # stop=1,
            num=points
            )
    xv, yv = np.meshgrid(
            fbm_grid,
            fbm_grid,
            sparse=False,
            indexing='ij'
            )
    covariance = 0.5*(
            np.abs(xv)**(2*hurst) +
            np.abs(yv)**(2*hurst) -
            np.abs(xv - yv)**(2*hurst)
            )
    g = rng.standard_normal(size=points)
    cholesky = np.linalg.cholesky(covariance)
    fbm_array = np.matmul(cholesky, g)
    return fbm_array


# %% bridge func
def bridge(f: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bridge_array = f - (f[-1]/grid[-1])*grid
    return bridge_array


# %% heat parameter func
# Heat kernel parameter creation based on time steps of the Euler scheme
def heat_kernel_var(time_steps: int, hurst: float) -> float:
    eta = 1/((hurst-1/2)**2 + 2 - hurst)
    variance = 1/(time_steps**eta)
    return variance


# %% integral between grid points func
def integral_between_grid_points(heat_kernel_var: float,
                                 grid_x: np.ndarray,
                                 half_support: float) -> np.ndarray:
    points = len(grid_x)
    heat_kernel_std = m.sqrt(heat_kernel_var)
    integral = np.zeros_like(grid_x)
    delta_half = half_support/(points)
    integral, error = quad_vec(lambda z:
                               ((grid_x - z)/heat_kernel_var)*norm.pdf(
                                   grid_x - z,
                                   loc=0,
                                   scale=heat_kernel_std),
                               a=-delta_half, b=delta_half)
    return integral


# %% Drift array
def create_drift_array(rough_func: np.ndarray,
                       integral_on_grid: np.ndarray) -> np.ndarray:
    return -np.convolve(rough_func, integral_on_grid, 'same')


# %% coarse noise func
# Coarse noise
def coarse_noise(z: np.ndarray,
                 time_steps: int,
                 sample_paths: int) -> np.ndarray:
    z_coarse = np.zeros(shape=(time_steps, sample_paths))
    q = int(np.shape(z)[0] / time_steps)
    if q == 1:
        z_coarse = z
    else:
        temp = z.reshape(time_steps, q, sample_paths)
        z_coarse = np.sum(temp, axis=1)
    return z_coarse


# %% sde solver func terminal time using np.interp
def solve(y0: float,
          drift_array: np.ndarray,
          z: np.ndarray,
          time_start: float, time_end: float, time_steps: int,
          sample_paths: int,
          grid: np.ndarray,) -> np.ndarray:
    y = np.zeros(shape=(1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[0, :] = y[0, :] \
                + np.interp(x=y[0, :], xp=grid, fp=drift_array)*dt \
                + z_coarse[i, :]
    return y


# Euler scheme solver for the distributional drift
# This function keeps the entire array corresponding to the solutions
# Use with care because it will eat up your memory very quick
def solve_keep_paths(y0: float,
                     drift_array: np.ndarray,
                     z: np.ndarray,
                     time_start: float, time_end: float, time_steps: int,
                     sample_paths: int,
                     grid: np.ndarray,) -> np.ndarray:
    y = np.zeros(shape=(time_steps+1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[i+1, :] = y[i, :] \
                + np.interp(x=y[i, :], xp=grid, fp=drift_array)*dt \
                + z_coarse[i, :]
    return y


# Euler scheme solver for a generic McKean-Vlasov SDE
class FokkerPlanckPDE(PDEBase):
    def __init__(self, drift, nonlinear, bc="dirichlet"):
        self.drift = drift
        self.nonlinear = nonlinear
        self.bc = bc

    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        v = state
        drift2 = v * self.nonlinear(v) * self.drift(v)
        v_x = drift2.gradient(bc=self.bc)[0]
        v_xx = v.laplace(bc=self.bc)
        v_t = 0.5 * v_xx - v_x
        return v_t


def solve_fp(drift_a, grid_a, limx=1, nonlinear_f=lambda x: np.sin(x), ts=0, te=1, xpoints=10, tpoints=2**8):
    xn = xpoints
    #x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), xn)
    x = np.linspace(-limx, limx, xn)
    ic = norm.pdf(x)
    grid_bounds = (-limx, limx)
    grid = CartesianGrid(bounds=[grid_bounds], shape=xn, periodic=False)
    state = ScalarField(grid=grid, data=ic)
    storage = MemoryStorage()

    def drift_f(x: np.ndarray, drift_array=drift_a, grid=grid_a):
        return np.interp(x=x.data, xp=grid, fp=drift_array)

    eq = FokkerPlanckPDE(drift=drift_f, nonlinear=nonlinear_f)
    solver = ScipySolver(pde=eq)
    time_steps = tpoints
    dt = 1/time_steps
    time_range = (ts, te)
    cont = Controller(solver=solver, t_range=time_range,
                      tracker=storage.tracker(dt))
    soln = cont.run(state)
    return storage


def solve_mv(y0: float,
             drift_array: np.ndarray,
             z: np.ndarray,
             time_start: float, time_end: float, time_steps: int,
             sample_paths: int,
             grid: np.ndarray,
             half_support,
             xpde, tpde) -> np.ndarray:
    y = np.zeros(shape=(time_steps+1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    rho = solve_fp(drift_a=drift_array, grid_a=grid, limx=half_support, xpoints=xpde, tpoints=tpde)
    rho_usable = np.array(rho.data)
    tsde = np.linspace(time_start, time_end, tpde+1)
    xsde = np.linspace(-half_support, half_support, xpde)
    ti = 0
    for i in range(time_steps):
        ti += dt
        tti = np.repeat(ti, y[i+1, :].shape[0])
        y[i+1, :] = y[i, :] + \
            interpn((tsde, xsde), rho_usable, list(zip(tti, y[i, :])), 'cubic', False, 1) * \
            np.interp(x=y[i, :], xp=grid, fp=drift_array)*dt + \
            z_coarse[i, :]
    return y


#####################
# Below are the functions no longer used
#####################
# %% sde solver func terminal time
# Euler scheme solver for the distributional drift
# This function only keeps the terminal time of the solution
def solves(
        y0,
        drift_array,
        z,
        time_start, time_end, time_steps,
        sample_paths,
        grid
        ):
    y = np.zeros(shape=(1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[0, :] = y[0, :] \
                + create_drift_function(
                        x=y[0, :],
                        drift_array=drift_array,
                        grid=grid
                        )*dt \
                + z_coarse[i, :]
    return y


# %% generic sde solver func
# Euler scheme solver for a generic SDE
def gen_solve(
        y0,
        drift,
        diffusion,
        z,
        time_start, time_end, time_steps,
        sample_paths
        ):
    y = np.zeros(shape=(time_steps+1, sample_paths))
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[i+1, :] = y[i, :] \
                + drift(t=i, x=y[i, :])*dt \
                + diffusion(t=i, x=y[i, :])*z[i, :]
    return y


# %%
# Create solutions/approximations
# This will be tested in approximate_tests.py
# To test each one of the functions above separately use manual_tests.py
def approximate(
        fbm,
        time_steps,
        hurst,
        time_start, time_end,
        noise,
        sample_paths,
        y0,
        grid_x, points_x, delta_x,
        half_support):
    df = integral_between_grid_points(
        np.sqrt(heat_kernel_var(time_steps, hurst)),
        # 15/11, # for testing with a fixed heat kernel parameter
        points_x,
        grid_x,
        half_support
        )
    drift_array = np.convolve(fbm, df, 'same')
    dt = (time_end - time_start)/(time_steps-1)
    time_grid = np.linspace(time_start + dt, time_end, time_steps)
    time_grid0 = np.insert(time_grid, 0, 0)
    z = coarse_noise(noise, time_steps, sample_paths)
    solution = solve(
        y0,
        drift_array,
        z,
        time_start, time_end, time_steps,
        sample_paths,
        grid_x, points_x, delta_x
        )
    return solution, time_grid, time_grid0, drift_array


# fBm generator giving the the N(0,1) RV
def fbm_alt(hurst, gaussian, half_support):
    points = len(gaussian)
    fbm_grid = np.linspace(
            start=1/points,
            stop=2*half_support,
            # stop=1,
            num=points
            )
    xv, yv = np.meshgrid(
            fbm_grid,
            fbm_grid,
            sparse=False,
            indexing='ij'
            )
    covariance = 0.5*(
            np.abs(xv)**(2*hurst) +
            np.abs(yv)**(2*hurst) -
            np.abs(xv - yv)**(2*hurst)
            )
    cholesky = np.linalg.cholesky(covariance)
    fbm_array = np.matmul(cholesky, gaussian)
    return fbm_array


# %% drift func
# Drift coefficient as a piecewise function created out of an array
def create_drift_function(x, drift_array, grid):
    points = len(grid)
    delta_half = grid[-1]/(points-1) # Half support divided by the points
    return np.piecewise(
        x,
        [(i - delta_half <= x)*(x < i + delta_half) for i in grid],
        [drift_array[i] for i in range(points)]  # piecewise constant
        )
