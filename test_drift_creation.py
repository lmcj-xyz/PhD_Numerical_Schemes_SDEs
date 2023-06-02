#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:49:52 2023

@author: lmcj
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm, linregress
from scipy.stats import linregress
import time
import math as m

from dsdes import approximate, bridge, coarse_noise, drift_func, fbm, \
    gen_solve, heat_kernel_parameter, mv_solve, integral_between_grid_points, solve, solves

# QOL parameters
plt.rcParams['figure.dpi'] = 500

# Variables to modify for the scheme
epsilon = 10e-6
beta = 7/16
hurst = 1 - beta
time_steps_max = 2**15
time_steps_approx1 = 2**10
time_steps_approx2 = 2**11
time_steps_approx3 = 2**12
time_steps_approx4 = 2**13
time_steps_approx5 = 2**14
time_steps_list = [time_steps_max, 
                   time_steps_approx1, 
                   time_steps_approx2, 
                   time_steps_approx3, 
                   time_steps_approx4, 
                   time_steps_approx5]
# Variables to create fBm
points_x = 2**8
half_support = 10
delta_x = half_support/(points_x-1)
grid_x = np.linspace(
    start = -half_support, stop = half_support, num = points_x
    )
# For the Brownian bridge
grid_x0 = np.linspace(
    start = 0, stop =2*half_support, num = points_x
    )
fbm_array = fbm(hurst, points_x, half_support)
bridge_array = bridge(fbm_array, grid_x0)
smooth_array = np.sin(grid_x)

var_heat_kernel_real = heat_kernel_parameter(time_steps_max, hurst)
var_heat_kernel_approx1 = heat_kernel_parameter(time_steps_approx1, hurst)
var_heat_kernel_approx2 = heat_kernel_parameter(time_steps_approx2, hurst)
var_heat_kernel_approx3 = heat_kernel_parameter(time_steps_approx3, hurst)
var_heat_kernel_approx4 = heat_kernel_parameter(time_steps_approx4, hurst)
var_heat_kernel_approx5 = heat_kernel_parameter(time_steps_approx5, hurst)

df_array_real = integral_between_grid_points(
    var_heat_kernel_real,
    points_x, grid_x, half_support)
df_array1 = integral_between_grid_points(
    var_heat_kernel_approx1, 
    points_x, grid_x, half_support)
df_array2 = integral_between_grid_points(
    var_heat_kernel_approx2,
    points_x, grid_x, half_support)
df_array3 = integral_between_grid_points(
    var_heat_kernel_approx3,
    points_x, grid_x, half_support)
df_array4 = integral_between_grid_points(
    var_heat_kernel_approx4,
    points_x, grid_x, half_support)
df_array5 = integral_between_grid_points(
    var_heat_kernel_approx5,
    points_x, grid_x, half_support)

drift_array_real = np.convolve(smooth_array, df_array_real, 'same')
drift_array1 = np.convolve(smooth_array, df_array1, 'same')
drift_array2 = np.convolve(smooth_array, df_array2, 'same')
drift_array3 = np.convolve(smooth_array, df_array3, 'same')
drift_array4 = np.convolve(smooth_array, df_array4, 'same')
drift_array5 = np.convolve(smooth_array, df_array5, 'same')

manually_computed_sin = np.exp(
    -heat_kernel_parameter(time_steps_max, hurst)/2)*np.cos(grid_x)
manually_computed_cos = np.exp(
    -heat_kernel_parameter(time_steps_max, hurst)/2)*np.sin(grid_x)
limy = 2
drift_fig = plt.figure('drift')
plt.plot(grid_x, drift_array_real, label="drift real solution")
plt.plot(grid_x, manually_computed_sin, label="drift for sin instead of fbm")
#plt.plot(grid_x, manually_computed_cos, label="drift for cos instead of fbm")
plt.plot(grid_x, drift_array1, label="drift approximation 1")
plt.plot(grid_x, drift_array2, label="drift approximation 2")
plt.plot(grid_x, drift_array3, label="drift approximation 3")
plt.plot(grid_x, drift_array4, label="drift approximation 4")
plt.plot(grid_x, drift_array5, label="drift approximation 5")
plt.ylim([-limy, limy])
plt.legend()
plt.show()