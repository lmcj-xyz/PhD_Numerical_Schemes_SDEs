# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:15:08 2023

@author: mmlmcj
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm, linregress
import time

from dist_sde_no_oop import *

# QOL parameters
plt.rcParams['figure.dpi'] = 500


#%% euler scheme
##########
# Usage of the functions above to solve multiple SDEs
# and compute convergence rate of approximations
##########

# Variables to modify for the scheme
epsilon = 10e-6
beta = epsilon
hurst = 1 - beta
time_steps_max = 2**12
time_steps_approx1 = 2**4
time_steps_approx2 = 2**5
time_steps_approx3 = 2**6
time_steps_approx4 = 2**7
time_steps_approx5 = 2**8

# Variables to create fBm
points_x = 2**10
half_support = 10
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
sample_paths = 10**4
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
real_solution, t_real, t0_real, drift_real = approximate(
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

approx1, t_a1, t0_a1, drift1 = approximate(
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

approx2, t_a2, t0_a2, drift2 = approximate(
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

approx3, t_a3, t0_a3, drift3 = approximate(
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

approx4, t_a4, t0_a4, drift4 = approximate(
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

approx5, t_a5, t0_a5, drift5 = approximate(
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

#%% strong error
# Computation of errors at terminal time
pathwise_error = np.zeros(shape=(5, sample_paths))
pathwise_error[0, :] = np.abs(real_solution[-1, :] - approx1[-1, :])
pathwise_error[1, :] = np.abs(real_solution[-1, :] - approx2[-1, :])
pathwise_error[2, :] = np.abs(real_solution[-1, :] - approx3[-1, :])
pathwise_error[3, :] = np.abs(real_solution[-1, :] - approx4[-1, :])
pathwise_error[4, :] = np.abs(real_solution[-1, :] - approx5[-1, :])

strong_error = np.mean(pathwise_error, axis=1)


#%% consecutive strong error
# Errors between consecutive approximations
pw_error_consecutive = np.zeros(shape=(4, sample_paths))
pw_error_consecutive[0, :] = np.abs(approx2[-1, :] - approx1[-1, :])
pw_error_consecutive[1, :] = np.abs(approx3[-1, :] - approx2[-1, :])
pw_error_consecutive[2, :] = np.abs(approx4[-1, :] - approx3[-1, :])
pw_error_consecutive[3, :] = np.abs(approx5[-1, :] - approx4[-1, :])

consecutive_strong_error = np.mean(pw_error_consecutive, axis=1)

#%% consecutive weak error
# Errors between consecutive approximations
pw_error_consecutive_bias = np.zeros(shape=(4, sample_paths))
pw_error_consecutive_bias[0, :] = approx2[-1, :] - approx1[-1, :]
pw_error_consecutive_bias[1, :] = approx3[-1, :] - approx2[-1, :]
pw_error_consecutive_bias[2, :] = approx4[-1, :] - approx3[-1, :]
pw_error_consecutive_bias[3, :] = approx5[-1, :] - approx4[-1, :]

consecutive_weak_error = np.mean(pw_error_consecutive_bias, axis=1)

#%% rate of convergence
deltas = [(time_end - time_start)/time_steps_approx1,
          (time_end - time_start)/time_steps_approx2,
          (time_end - time_start)/time_steps_approx3,
          (time_end - time_start)/time_steps_approx4,
          (time_end - time_start)/time_steps_approx5] 

log_strong_error = np.log10(strong_error)
log_deltas = np.log10(deltas)

reg = linregress(log_deltas, log_strong_error)
rate = reg.slope
intersection = reg.intercept
print(rate)

#%%
# Several plots

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

#%% both errors plot
both_error_fig = plt.figure('both_error_fig')
plt.title("errors for beta=%.5f \n rate = %f" % (beta, rate))
plt.semilogy(strong_error, marker='o', label='strong error')
plt.semilogy([0.5, 1.5, 2.5, 3.5],consecutive_strong_error, marker='o', label='error between consecutive approximations')
plt.semilogy([0.5, 1.5, 2.5, 3.5],np.abs(consecutive_weak_error), marker='o', label='bias between consecutive approximations')
plt.legend()
plt.show()


#%% one sample path
path = 9
plt.figure('paths')
plt.plot(t0_real, real_solution[:,path])
plt.plot(t0_a1,         approx1[:,path])
plt.plot(t0_a2,         approx2[:,path])
plt.plot(t0_a3,         approx3[:,path])
plt.plot(t0_a4,         approx4[:,path])
plt.plot(t0_a5,         approx5[:,path])
plt.show()














































