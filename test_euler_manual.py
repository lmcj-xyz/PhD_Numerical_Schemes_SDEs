# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:13:37 2023

@author: mmlmcj
"""

#%% delete variables
# To delete variables at the start in case we need to reclaim need memory
#from IPython import get_ipython
#ipython = get_ipython()
#ipython.magic("%reset")

#%% packages and parameters
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm, linregress
from scipy.stats import linregress
import time
import math as m

from dsdes import approximate, bridge, coarse_noise, create_drift_function, fbm, \
    gen_solve, heat_kernel_var, mv_solve, integral_between_grid_points, solve, solves

# QOL parameters
plt.rcParams['figure.dpi'] = 500

#%%
##########
# Usage of the functions above to solve multiple SDEs
# and compute convergence rate of approximations
##########

#%% some parameters
# Variables to modify for the scheme
epsilon = 10e-6
beta = 4/16
hurst = 1 - beta
time_steps_max = 2**15
time_steps_approx1 = 2**10
time_steps_approx2 = 2**11
time_steps_approx3 = 2**12
time_steps_approx4 = 2**13
time_steps_approx5 = 2**14

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

# Create an array of fBm
fbm_array = fbm(hurst, points_x, half_support)

# fBm "bridge"
bridge_array = bridge(fbm_array, grid_x0)

# sine array
sine_array = np.sin(grid_x)

# Euler scheme
y0 = 1
sample_paths = 10**4
time_start = 0
time_end = 1

#%% ##### OPTIONAL #####
# Plot fBm
fbm_fig = plt.figure('fbm')
plt.plot(grid_x, fbm_array, label='fbm')
plt.plot(grid_x, bridge_array, label='brownian bridge')
plt.grid(linestyle='--', axis='y', linewidth=0.5)
plt.legend()
plt.show()

#%% Create a dF
####### USE PYTHON map() for this and many other chunks
# We are supposed to use the square root of the heat kernel parameter
# The function integral_between_grid_points computes it, but here we give it the
# square root of the parameter from the start, which means it will compute the
# square root of the square root of the parameter
# this gives us reasonable convergence rates but IT IS INCORRECT
# we need to see how to use the real parameter
var_heat_kernel_real = heat_kernel_var(time_steps_max, hurst)
var_heat_kernel_approx1 = heat_kernel_var(time_steps_approx1, hurst)
var_heat_kernel_approx2 = heat_kernel_var(time_steps_approx2, hurst)
var_heat_kernel_approx3 = heat_kernel_var(time_steps_approx3, hurst)
var_heat_kernel_approx4 = heat_kernel_var(time_steps_approx4, hurst)
var_heat_kernel_approx5 = heat_kernel_var(time_steps_approx5, hurst)

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

#%% ##### OPTIONAL #####
# Plot dF
df_fig = plt.figure('df')
plt.plot(grid_x, df_array_real, label="df real solution")
plt.plot(grid_x, df_array1, label="df approximation 1")
plt.plot(grid_x, df_array2, label="df approximation 2")
plt.plot(grid_x, df_array3, label="df approximation 3")
plt.plot(grid_x, df_array4, label="df approximation 4")
plt.plot(grid_x, df_array5, label="df approximation 5")
plt.legend()
plt.show()

#%% Create drift by convolution with a smooth function
#drift_array_real = np.convolve(sine_array, df_array_real, 'same')
#drift_array1 = np.convolve(sine_array, df_array1, 'same')
#drift_array2 = np.convolve(sine_array, df_array2, 'same')
#drift_array3 = np.convolve(sine_array, df_array3, 'same')
#drift_array4 = np.convolve(sine_array, df_array4, 'same')
#drift_array5 = np.convolve(sine_array, df_array5, 'same')

#%% Create drift by convolution with the bridge
drift_array_real = np.convolve(bridge_array, df_array_real, 'same')
drift_array1 = np.convolve(bridge_array, df_array1, 'same')
drift_array2 = np.convolve(bridge_array, df_array2, 'same')
drift_array3 = np.convolve(bridge_array, df_array3, 'same')
drift_array4 = np.convolve(bridge_array, df_array4, 'same')
drift_array5 = np.convolve(bridge_array, df_array5, 'same')

#%% ##### OPTIONAL ##### Create drift by convolution with the fBm
#drift_array_real = np.convolve(fbm_array, df_array_real, 'same')
#drift_array1 = np.convolve(fbm_array, df_array1, 'same')
#drift_array2 = np.convolve(fbm_array, df_array2, 'same')
#drift_array3 = np.convolve(fbm_array, df_array3, 'same')
#drift_array4 = np.convolve(fbm_array, df_array4, 'same')
#drift_array5 = np.convolve(fbm_array, df_array5, 'same')

#%% ##### OPTIONAL ###### Plot drift
manually_computed = np.exp(-heat_kernel_var(time_steps_max, hurst)/2)*np.cos(grid_x)
limy = 2
drift_fig = plt.figure('drift')
plt.plot(grid_x, drift_array_real, label="drift real solution")
plt.plot(grid_x, manually_computed, label="drift for sine instead of fbm")
plt.plot(grid_x, drift_array1, label="drift approximation 1")
#plt.plot(grid_x, drift_array2, label="drift approximation 2")
#plt.plot(grid_x, drift_array3, label="drift approximation 3")
#plt.plot(grid_x, drift_array4, label="drift approximation 4")
#plt.plot(grid_x, drift_array5, label="drift approximation 5")
plt.ylim([-limy, limy])
plt.legend()
plt.show()

#%% Distance between points in grid for x
delta_x = half_support/(points_x-1)

# Evaluate and plot some drift functions
support = np.linspace(-half_support, half_support, points_x)

#%% ########OPTIONAL##########
eval_real = create_drift_function(support, drift_array_real, grid_x, points_x, delta_x)
eval1 = create_drift_function(support, drift_array1, grid_x, points_x, delta_x)
eval2 = create_drift_function(support, drift_array2, grid_x, points_x, delta_x)
eval3 = create_drift_function(support, drift_array3, grid_x, points_x, delta_x)
eval4 = create_drift_function(support, drift_array4, grid_x, points_x, delta_x)
eval5 = create_drift_function(support, drift_array5, grid_x, points_x, delta_x)

#%% ##### OPTIONAL #####
drift_func_fig = plt.figure('driftfunc')
plt.plot(support, bridge_array, label="fbm")
plt.plot(support, eval_real, label="real solution drift function")
#plt.plot(support, eval1, label="approximation 1 drift function")
#plt.plot(support, eval2, label="approximation 2 drift function")
#plt.plot(support, eval3, label="approximation 3 drift function")
plt.ylim([-2, 2])
plt.grid()
plt.legend()
plt.show()

#%% #### OPTIONAL ####
# Convolute deterministic functions with dF and plot for testing purposes
# Create a deterministic function for testing
square = grid_x**2
cube = grid_x**3
sine = np.sin(grid_x)
# Create dF, you can change ts to see different parameters of the heat kernel
ts = 2**10
points_x_test = 2**8
grid_x_test = np.linspace(
    start = -half_support, stop = half_support, num = points_x
    )
df_array_det = integral_between_grid_points(
    np.sqrt(heat_kernel_var(time_steps_approx1, hurst)), 
    #1/10**4,
    points_x_test, grid_x_test, half_support
    )
# Convolute
square_conv = np.convolve(square, df_array_det, 'same')
cube_conv = np.convolve(cube, df_array_det, 'same')
sine_conv = np.convolve(sine, df_array_det, 'same')
# Plot; comment out whatever
det_fig = plt.figure('det_fig')
plt.plot(support, square_conv, label="derivative of square: linear")
plt.plot(support, 2*support, label="linear function")
plt.plot(support, cube_conv, label="derivative of cube: square")
plt.plot(support, 3*support**2, label="square function")
plt.plot(support, sine_conv, label="derivative of sin: cos")
plt.plot(support, np.cos(support), label="cos function")
plt.legend()
#plt.ylim([-5, 5])
plt.show()

#%% Parameters for solution/approximations
# Parameters for real solution
dt_real = (time_end - time_start)/(time_steps_max-1)
time_grid_real = np.linspace(time_start + dt_real, time_end, time_steps_max)
time_grid_real0 = np.insert(time_grid_real, 0, 0)
z_real = rng.normal(loc=0.0, scale=np.sqrt(dt_real), 
                    size=(time_steps_max, sample_paths))

# Parameters for approximation 1
dt_approx1 = (time_end - time_start)/(time_steps_approx1 - 1)
time_grid_approx1 = np.linspace(
    time_start + dt_approx1, time_end, time_steps_approx1
    )
time_grid_approx10 = np.insert(time_grid_approx1, 0, 0)
#z_approx1 = coarse_noise(z_real, time_steps_approx1, sample_paths)

# Parameters for approximation 2
dt_approx2 = (time_end - time_start)/(time_steps_approx2 - 1)
time_grid_approx2 = np.linspace(
    time_start + dt_approx2, time_end, time_steps_approx2
    )
time_grid_approx20 = np.insert(time_grid_approx2, 0, 0)
#z_approx2 = coarse_noise(z_real, time_steps_approx2, sample_paths)

# Parameters for approximation 3
dt_approx3 = (time_end - time_start)/(time_steps_approx3 - 1)
time_grid_approx3 = np.linspace(
    time_start + dt_approx3, time_end, time_steps_approx3
    )
time_grid_approx30 = np.insert(time_grid_approx3, 0, 0)
#z_approx3 = coarse_noise(z_real, time_steps_approx3, sample_paths)

# Parameters for approximation 4
dt_approx4 = (time_end - time_start)/(time_steps_approx4 - 1)
time_grid_approx4 = np.linspace(
    time_start + dt_approx4, time_end, time_steps_approx4
    )
time_grid_approx40 = np.insert(time_grid_approx4, 0, 0)
#z_approx4 = coarse_noise(z_real, time_steps_approx4, sample_paths)

# Parameters for approximation 5
dt_approx5 = (time_end - time_start)/(time_steps_approx5 - 1)
time_grid_approx5 = np.linspace(
    time_start + dt_approx5, time_end, time_steps_approx5
    )
time_grid_approx50 = np.insert(time_grid_approx5, 0, 0)
#z_approx5 = coarse_noise(z_real, time_steps_approx5, sample_paths)

#%% ##### OPTIONAL #####
# Visualize coarse noises
#coarse_fig = plt.figure()
#plt.plot(time_grid_approx4, z_approx4[:,0], label="approximation")
#plt.plot(time_grid_real, z_real[:,0], label="real solution")
#plt.legend()
#plt.show()

#%%% Solve an SDE by Euler scheme
st = time.process_time()
real_solution = solves(
    y0,
    drift_array_real,
    z_real,
    time_start,
    time_end,
    time_steps_max,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
approx1 = solves(
    y0, 
    drift_array1,
    z_real,
    time_start,
    time_end,
    time_steps_approx1,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
approx2 = solves(
    y0,
    drift_array2,
    z_real,
    time_start,
    time_end,
    time_steps_approx2,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
approx3 = solves(
    y0,
    drift_array3,
    z_real,
    time_start,
    time_end,
    time_steps_approx3,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
approx4 = solves(
    y0,
    drift_array4,
    z_real,
    time_start,
    time_end,
    time_steps_approx4,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
approx5 = solves(
    y0,
    drift_array5,
    z_real,
    time_start,
    time_end,
    time_steps_approx5,
    sample_paths,
    grid_x,
    points_x,
    delta_x
    )
et = time.process_time()
rt = et - st

#%% plot SDE ##### OPTIONAL ##### this doesn't make sense when using "solves"
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

#%% ##### OPTIONAL #####
# Comparing single sample paths of different approximations
emssp_fig = plt.figure('emssp_fig')
plt.plot(time_grid_real0, real_solution[:, 0], label="real solution")
plt.plot(time_grid_approx10, approx1[:, 0], label="approximation 1")
plt.plot(time_grid_approx20, approx2[:, 0], label="approximation 2")
plt.plot(time_grid_approx30, approx3[:, 0], label="approximation 3")
plt.title("single sample path comparison")
plt.legend()
plt.show()

#%% Computation of strong errors at terminal time
pathwise_error = np.zeros(shape=(5, sample_paths))
pathwise_error[0, :] = np.abs(real_solution[-1, :] - approx1[-1, :])
pathwise_error[1, :] = np.abs(real_solution[-1, :] - approx2[-1, :])
pathwise_error[2, :] = np.abs(real_solution[-1, :] - approx3[-1, :])
pathwise_error[3, :] = np.abs(real_solution[-1, :] - approx4[-1, :])
pathwise_error[4, :] = np.abs(real_solution[-1, :] - approx5[-1, :])

strong_error = np.mean(pathwise_error, axis=1)
# error bars
a = 5
seci = np.zeros(shape=a)
for i in range(a):
    seci[i] = 1.96*np.sqrt(
        np.sum(
            (
                pathwise_error[i, :] - strong_error[i]
            )**2/(sample_paths-1))/sample_paths
        )

#%% weak error
# Computation of weak errors at terminal time
pw_weak_error = np.zeros(shape=(5, sample_paths))
pw_weak_error[0, :] = real_solution[-1, :] - approx1[-1, :]
pw_weak_error[1, :] = real_solution[-1, :] - approx2[-1, :]
pw_weak_error[2, :] = real_solution[-1, :] - approx3[-1, :]
pw_weak_error[3, :] = real_solution[-1, :] - approx4[-1, :]
pw_weak_error[4, :] = real_solution[-1, :] - approx5[-1, :]
#pw_weak_error[5, :] = real_solution[-1, :] - approx6[-1, :]
#pw_weak_error[6, :] = real_solution[-1, :] - approx7[-1, :]

weak_error = np.abs(np.mean(pw_weak_error, axis=1))
# error bars
a = 5
weci = np.zeros(shape=a)
for i in range(a):
    weci[i] = 1.96*np.sqrt(
        np.sum(
            (pw_weak_error[i, :] - weak_error[i])**2/(sample_paths-1)
            )/sample_paths
        )

#%% consecutive strong error
# Errors between consecutive approximations
pw_consecutive_error = np.zeros(shape=(4, sample_paths))
pw_consecutive_error[0, :] = np.abs(approx2[-1, :] - approx1[-1, :])
pw_consecutive_error[1, :] = np.abs(approx3[-1, :] - approx2[-1, :])
pw_consecutive_error[2, :] = np.abs(approx4[-1, :] - approx3[-1, :])
pw_consecutive_error[3, :] = np.abs(approx5[-1, :] - approx4[-1, :])
#pw_consecutive_error[4, :] = np.abs(approx6[-1, :] - approx5[-1, :])
#pw_consecutive_error[5, :] = np.abs(approx7[-1, :] - approx6[-1, :])

consecutive_strong_error = np.mean(pw_consecutive_error, axis=1)
# error bars
b = 4
cseci = np.zeros(shape=b)
for i in range(b):
    cseci[i] = 1.96*np.sqrt(
        np.sum(
            (
                pw_consecutive_error[i, :] - consecutive_strong_error[i]
            )**2/(sample_paths-1)
            )/sample_paths
        )

#%% consecutive weak error
# Bias between consecutive approximations
pw_consecutive_bias = np.zeros(shape=(4, sample_paths))
pw_consecutive_bias[0, :] = approx2[-1, :] - approx1[-1, :]
pw_consecutive_bias[1, :] = approx3[-1, :] - approx2[-1, :]
pw_consecutive_bias[2, :] = approx4[-1, :] - approx3[-1, :]
pw_consecutive_bias[3, :] = approx5[-1, :] - approx4[-1, :]
#pw_consecutive_bias[4, :] = approx6[-1, :] - approx5[-1, :]
#pw_consecutive_bias[5, :] = approx7[-1, :] - approx6[-1, :]

consecutive_bias = np.mean(pw_consecutive_bias, axis=1)
consecutive_weak_error = np.abs(consecutive_bias)
# error bars
b = 4
cweci = np.zeros(shape=b)
for i in range(b):
    cweci[i] = 1.96*np.sqrt(
        np.sum(
            (
                pw_consecutive_bias[i, :] - consecutive_weak_error[i]
            )**2/(sample_paths-1)
            )/sample_paths
        )

#%% rate of convergence
deltas = [(time_end - time_start)/time_steps_approx1,
          (time_end - time_start)/time_steps_approx2,
          (time_end - time_start)/time_steps_approx3,
          (time_end - time_start)/time_steps_approx4,
          (time_end - time_start)/time_steps_approx5
          #,
          #(time_end - time_start)/time_steps_approx6,
          #(time_end - time_start)/time_steps_approx7
          ] 

log_strong_error = np.log10(strong_error)
log_weak_error = np.log10(weak_error)
log_deltas = np.log10(deltas)

reg_strong = linregress(log_deltas, log_strong_error)
rate_strong = reg_strong.slope
intersection_strong = reg_strong.intercept
print(rate_strong)

reg_weak = linregress(log_deltas, log_weak_error)
rate_weak = reg_weak.slope
intersection_weak = reg_weak.intercept
print(rate_weak)

#%% Several plots ##### OPTIONAL #####

#%% all errors plot
both_error_fig = plt.figure('both_error_fig')
plt.title(
    "errors for beta=%.5f \n strong error rate = %f \n weak error rate = %f" 
    % (beta, rate_strong, rate_weak)
    )
#plt.errorbar([0, 1, 2, 3, 4], 
#        strong_error, 
#        yerr=seci, 
#        marker='',
#        linewidth=1,
#        label='strong error')
#plt.errorbar([0, 1, 2, 3, 4], 
#        weak_error,
#        yerr=weci,
#        marker='',
#        linewidth=1,
#        label='weak error')
#plt.errorbar([0.5, 1.5, 2.5, 3.5],
#        consecutive_strong_error,
#        yerr=cseci,
#        marker='',
#        linewidth=1,
#        label='error between consecutive approximations')
#plt.errorbar([0.5, 1.5, 2.5, 3.5],
#        consecutive_weak_error,
#        yerr=cweci,
#        marker='',
#        linewidth=1,
#        label='bias between consecutive approximations')
plt.plot([0, 1, 2, 3, 4], 
         strong_error,
         marker='o',
         label='strong error')
plt.plot([0, 1, 2, 3, 4],
         weak_error,
         marker='o',
         label='weak error')
plt.plot([0.5, 1.5, 2.5, 3.5],
         consecutive_strong_error,
         marker='o',
         label='error between consecutive approximations')
plt.plot([0.5, 1.5, 2.5, 3.5],
         consecutive_weak_error,
         marker='o',
         label='bias between consecutive approximations')
plt.yscale('log')
plt.legend()
plt.show()

#%% strong error plot
rate_fig = plt.figure('rate_fig')
plt.title("strong error")
plt.semilogy(strong_error, marker='o')
plt.show()

#%% consecutive error plot
consecutive_error_fig = plt.figure('consecutive_error_fig')
plt.title("error between consecutive approximations")
plt.semilogy(strong_error, marker='o')
plt.show()

#%% one sample path
#path = 9
#plt.figure('paths')
#plt.plot(t0_real, real_solution[:,path])
#plt.plot(t0_a1,         approx1[:,path])
#plt.plot(t0_a2,         approx2[:,path])
#plt.plot(t0_a3,         approx3[:,path])
#plt.plot(t0_a4,         approx4[:,path])
#plt.plot(t0_a5,         approx5[:,path])
#plt.show()

#%% histograms
fig, ax = plt.subplots()
ax.hist(real_solution[1,:], bins=30)
ax.hist(approx1[1,:], bins=30)
plt.show()


#%% ##### OPTIONAL #####
# Plot of strong error
rate_fig = plt.figure('rate_fig')
plt.semilogy(strong_error, marker='o')
plt.show()

#%% ##### OPTIONAL #####
# Printing errors and CIs
print(strong_error)
print(seci)
print(weak_error)
print(weci)
print(consecutive_strong_error)
print(cseci)
print(consecutive_bias)
print(cweci)































