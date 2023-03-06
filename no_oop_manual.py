# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:13:37 2023

@author: mmlmcj
"""

#%%
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
x_grid = np.linspace(start = -half_support, stop = half_support, num = points_x)

# Create an array of fBm
fbm_array = fbm(hurst, points_x, half_support)

#%%
##### OPTIONAL #####
# Plot fBm
fbm_fig = plt.figure('fbm')
plt.plot(fbm_array)
plt.show()

#%%
# Create a dF
df_array_real = normal_differences(np.sqrt(heat_param(time_steps_max, hurst)), points_x, x_grid, half_support)
df_array1 = normal_differences(np.sqrt(heat_param(time_steps_approx1, hurst)), points_x, x_grid, half_support)
df_array2 = normal_differences(np.sqrt(heat_param(time_steps_approx2, hurst)), points_x, x_grid, half_support)
df_array3 = normal_differences(np.sqrt(heat_param(time_steps_approx3, hurst)), points_x, x_grid, half_support)
df_array4 = normal_differences(np.sqrt(heat_param(time_steps_approx4, hurst)), points_x, x_grid, half_support)
df_array5 = normal_differences(np.sqrt(heat_param(time_steps_approx5, hurst)), points_x, x_grid, half_support)

#%%
##### OPTIONAL #####
# Plot dF
df_fig = plt.figure('df')
plt.plot(df_array_real, label="df real solution")
plt.plot(df_array1, label="df approximation 1")
plt.plot(df_array2, label="df approximation 2")
plt.plot(df_array3, label="df approximation 3")
plt.legend()
plt.show()

#%%
# Create drift by convolution
drift_array_real = np.convolve(fbm_array, df_array_real, 'same')
drift_array1 = np.convolve(fbm_array, df_array1, 'same')
drift_array2 = np.convolve(fbm_array, df_array2, 'same')
drift_array3 = np.convolve(fbm_array, df_array3, 'same')
drift_array4 = np.convolve(fbm_array, df_array4, 'same')
drift_array5 = np.convolve(fbm_array, df_array5, 'same')


#%%
##### OPTIONAL #####
# Plot drift
drift_fig = plt.figure('drift')
plt.plot(drift_array_real, label="drift real solution")
plt.plot(drift_array1, label="drift approximation 1")
plt.plot(drift_array2, label="drift approximation 2")
plt.plot(drift_array3, label="drift approximation 3")
plt.legend()
plt.show()

#%%
# Distance between points in grid for x
delta_x = half_support/(points_x-1)

# Evaluate and plot some drift functions
support = np.linspace(-half_support, half_support, points_x)

eval_real = drift_func(support, drift_array_real, x_grid, points_x, delta_x)
eval1 = drift_func(support, drift_array1, x_grid, points_x, delta_x)
eval2 = drift_func(support, drift_array2, x_grid, points_x, delta_x)
eval3 = drift_func(support, drift_array3, x_grid, points_x, delta_x)
eval4 = drift_func(support, drift_array4, x_grid, points_x, delta_x)
eval5 = drift_func(support, drift_array5, x_grid, points_x, delta_x)

#%%
##### OPTIONAL #####
drift_func_fig = plt.figure('driftfunc')
plt.plot(support, eval_real, label="real solution drift function")
plt.plot(support, eval1, label="approximation 1 drift function")
plt.plot(support, eval2, label="approximation 2 drift function")
plt.plot(support, eval3, label="approximation 3 drift function")
plt.grid()
plt.legend()
plt.show()

#%%
# Convolute deterministic functions with dF and plot for testing purposes
# Create a deterministic function for testing
square = x_grid**2
cube = x_grid**3
sine = np.sin(x_grid)
# Create dF, you can change ts to see different parameters of the heat kernel
ts = 2**10
df_array_det = normal_differences(np.sqrt(heat_param(ts, hurst)), points_x, x_grid, half_support)
# Convolute
square_conv = np.convolve(square, df_array_det, 'same')
cube_conv = np.convolve(cube, df_array_det, 'same')
sine_conv = np.convolve(sine, df_array_det, 'same')
# Plot; comment out whatever
det_fig = plt.figure('det_fig')
plt.plot(support, square_conv, label="derivative of square: linear")
plt.plot(support, support, label="linear function")
plt.plot(support, cube_conv, label="derivative of cube: square")
plt.plot(support, support**2, label="square function")
plt.plot(support, sine_conv, label="derivative of sin: cos")
plt.plot(support, np.cos(support), label="cos function")
plt.legend()
plt.ylim([-5, 5])
plt.show()

#%%
# Parameter for Euler scheme
y0 = 1
sample_paths = 10**1
time_start = 0
time_end = 1

#%%
# Parameters for real solution
dt_real = (time_end - time_start)/(time_steps_max-1)
time_grid_real = np.linspace(time_start + dt_real, time_end, time_steps_max)
time_grid_real0 = np.insert(time_grid_real, 0, 0)
z_real = rng.normal(loc=0.0, scale=np.sqrt(dt_real), size=(time_steps_max, sample_paths))

# Parameters for approximation 1
dt_approx1 = (time_end - time_start)/(time_steps_approx1 - 1)
time_grid_approx1 = np.linspace(time_start + dt_approx1, time_end, time_steps_approx1)
time_grid_approx10 = np.insert(time_grid_approx1, 0, 0)
z_approx1 = coarse_noise(z_real, time_steps_approx1, sample_paths)

# Parameters for approximation 2
dt_approx2 = (time_end - time_start)/(time_steps_approx2 - 1)
time_grid_approx2 = np.linspace(time_start + dt_approx2, time_end, time_steps_approx2)
time_grid_approx20 = np.insert(time_grid_approx2, 0, 0)
z_approx2 = coarse_noise(z_real, time_steps_approx2, sample_paths)

# Parameters for approximation 3
dt_approx3 = (time_end - time_start)/(time_steps_approx3 - 1)
time_grid_approx3 = np.linspace(time_start + dt_approx3, time_end, time_steps_approx3)
time_grid_approx30 = np.insert(time_grid_approx3, 0, 0)
z_approx3 = coarse_noise(z_real, time_steps_approx3, sample_paths)

# Parameters for approximation 4
dt_approx4 = (time_end - time_start)/(time_steps_approx4 - 1)
time_grid_approx4 = np.linspace(time_start + dt_approx4, time_end, time_steps_approx4)
time_grid_approx40 = np.insert(time_grid_approx4, 0, 0)
z_approx4 = coarse_noise(z_real, time_steps_approx4, sample_paths)

# Parameters for approximation 5
dt_approx5 = (time_end - time_start)/(time_steps_approx5 - 1)
time_grid_approx5 = np.linspace(time_start + dt_approx5, time_end, time_steps_approx5)
time_grid_approx50 = np.insert(time_grid_approx5, 0, 0)
z_approx5 = coarse_noise(z_real, time_steps_approx5, sample_paths)

#%%
##### OPTIONAL #####
# Visualize coarse noises
coarse_fig = plt.figure()
plt.plot(time_grid_approx4, z_approx4[:,0], label="approximation")
plt.plot(time_grid_real, z_real[:,0], label="real solution")
plt.legend()
plt.show()

#%%%
# Solve an SDE
st = time.process_time()
real_solution = solve(y0, drift_array1, z_real, time_start, time_end, time_steps_max, sample_paths, x_grid, points_x, delta_x)
approx1 = solve(y0, drift_array1, z_approx1, time_start, time_end, time_steps_approx1, sample_paths, x_grid, points_x, delta_x)
approx2 = solve(y0, drift_array2, z_approx2, time_start, time_end, time_steps_approx2, sample_paths, x_grid, points_x, delta_x)
approx3 = solve(y0, drift_array3, z_approx3, time_start, time_end, time_steps_approx3, sample_paths, x_grid, points_x, delta_x)
approx4 = solve(y0, drift_array4, z_approx4, time_start, time_end, time_steps_approx4, sample_paths, x_grid, points_x, delta_x)
approx5 = solve(y0, drift_array5, z_approx5, time_start, time_end, time_steps_approx5, sample_paths, x_grid, points_x, delta_x)
et = time.process_time()
rt = et - st

#%% plot SDE
##### OPTIONAL #####
# Plot solution to SDE
emreal_fig = plt.figure('emreal_fig')
plt.plot(time_grid_real0, real_solution[:, 0:3])
plt.title("real solution")
plt.show()

em1_fig = plt.figure('em2_fig')
plt.plot(time_grid_approx10, approx1[:, 0:3])
plt.title("approximation 1")
plt.show()

em2_fig = plt.figure('em2_fig')
plt.plot(time_grid_approx20, approx2[:, 0:3])
plt.title("approximation 2")
plt.show()

em3_fig = plt.figure('em3_fig')
plt.plot(time_grid_approx30, approx3[:, 0:3])
plt.title("approximation 3")
plt.show()

em4_fig = plt.figure('em4_fig')
plt.plot(time_grid_approx40, approx4[:, 0:3])
plt.title("approximation 4")
plt.show()

em5_fig = plt.figure('em5_fig')
plt.plot(time_grid_approx50, approx5[:, 0:3])
plt.title("approximation 5")
plt.show()

#%%
##### OPTIONAL #####
# Comparing single sample paths of different approximations
emssp_fig = plt.figure('emssp_fig')
plt.plot(time_grid_real0, real_solution[:, 0], label="real solution")
plt.plot(time_grid_approx10, approx1[:, 0], label="approximation 1")
plt.plot(time_grid_approx20, approx2[:, 0], label="approximation 2")
plt.plot(time_grid_approx30, approx3[:, 0], label="approximation 3")
plt.title("single sample path comparison")
plt.legend()
plt.show()

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
plt.semilogy(strong_error, marker='o')
plt.show()

































