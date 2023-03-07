# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:15:08 2023

@author: mmlmcj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm
import time

from dist_sde_no_oop import *

#%%
##########
# Usage of the functions above to solve multiple SDEs
# and compute convergence rate of approximations
##########

# QOL parameters
plt.rcParams['figure.dpi'] = 500

# Variables to modify for the scheme
epsilon = 10e-6
beta = 1/2
hurst = 1 - beta
time_steps_max = 2**10
time_steps_approx1 = 2**4
time_steps_approx2 = 2**5
time_steps_approx3 = 2**6
time_steps_approx4 = 2**7
time_steps_approx5 = 2**8

# Variables to create fBm
points_x = 2**8
half_support = 3
delta_x = half_support/(points_x-1)
x_grid = np.linspace(
    start = -half_support,
    stop = half_support, 
    num = points_x
    )

# Create an array of fBm
fbm_array = fbm(hurst, points_x, half_support)

# Parameter for Euler scheme
y0 = 1
sample_paths = 10**2
time_start = 0
time_end = 1

# Parameters for real solution
delta_t = (time_end - time_start)/(time_steps_max-1)
z = rng.normal(
    loc=0.0,
    scale=np.sqrt(delta_t),
    size=(time_steps_max, sample_paths)
    )

# Computation of approximations with function approximate
real_solution, t_real, t0_real = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_max, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

approx1, t_a1, t0_a1 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx1, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

approx2, t_a2, t0_a2 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx2, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

approx3, t_a3, t0_a3 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx3, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

approx4, t_a4, t0_a4 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx4, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

approx5, t_a5, t0_a5 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx5, 
    hurst=hurst, 
    time_start=time_start, 
    time_end=time_end, 
    noise=z, 
    sample_paths=sample_paths, 
    y0=y0, 
    x_grid=x_grid, 
    points_x=points_x, 
    delta_x=delta_x,
    half_support=half_support)

#%%
# Computation of errors at terminal time
pathwise_error = np.zeros(shape=(5, sample_paths))
pathwise_error[0, :] = np.abs(real_solution[-1, :] - approx1[-1, :])
pathwise_error[1, :] = np.abs(real_solution[-1, :] - approx2[-1, :])
pathwise_error[2, :] = np.abs(real_solution[-1, :] - approx3[-1, :])
pathwise_error[3, :] = np.abs(real_solution[-1, :] - approx4[-1, :])
pathwise_error[4, :] = np.abs(real_solution[-1, :] - approx5[-1, :])

strong_error = np.mean(pathwise_error, axis=1)

rate_fig = plt.figure('rate_fig')
plt.title("strong error")
plt.semilogy(strong_error, marker='o')
plt.show()

#%%
# Errors between consecutive approximations
pw_error_consecutive = np.zeros(shape=(4, sample_paths))
pw_error_consecutive[0, :] = np.abs(approx2[-1, :] - approx1[-1, :])
pw_error_consecutive[1, :] = np.abs(approx3[-1, :] - approx2[-1, :])
pw_error_consecutive[2, :] = np.abs(approx4[-1, :] - approx3[-1, :])
pw_error_consecutive[3, :] = np.abs(approx5[-1, :] - approx4[-1, :])

consecutive_strong_error = np.mean(pw_error_consecutive, axis=1)

consecutive_error_fig = plt.figure('consecutive_error_fig')
plt.title("error between consecutive approximations")
plt.semilogy(consecutive_strong_error, marker='o')
plt.show()


