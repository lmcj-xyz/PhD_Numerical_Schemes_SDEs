#-*- coding: utf-8 -*-
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
beta = 7/16
hurst = 1 - beta
time_steps_max = 2**12
time_steps_approx1 = 2**4
time_steps_approx2 = 2**5
time_steps_approx3 = 2**6
time_steps_approx4 = 2**7
time_steps_approx5 = 2**8
time_steps_approx6 = 2**9
time_steps_approx7 = 2**10

#%% fbm
# Variables to create fBm
points_x = 2**12
half_support = 10
delta_x = half_support/(points_x-1)
x_grid = np.linspace(
    start = -half_support,
    stop = half_support, 
    num = points_x
    )

# Create an array of fBm
fbm_array = fbm(hurst, points_x, half_support)

#%% using stored fbm
fbm_array = np.load("fbm1.npy")

#%%
# Parameter for Euler scheme
y0 = 3
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

#%% real solution
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

#%% approximations

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

#%% extra approximations
approx6, t_a6, t0_a6, drift6 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx6, 
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

approx7, t_a7, t0_a7, drift7 = approximate(
    fbm=fbm_array, 
    time_steps=time_steps_approx7, 
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
# Computation of strong errors at terminal time
pw_strong_error = np.zeros(shape=(5, sample_paths))
pw_strong_error[0, :] = np.abs(real_solution[-1, :] - approx1[-1, :])
pw_strong_error[1, :] = np.abs(real_solution[-1, :] - approx2[-1, :])
pw_strong_error[2, :] = np.abs(real_solution[-1, :] - approx3[-1, :])
pw_strong_error[3, :] = np.abs(real_solution[-1, :] - approx4[-1, :])
pw_strong_error[4, :] = np.abs(real_solution[-1, :] - approx5[-1, :])
#pw_strong_error[5, :] = np.abs(real_solution[-1, :] - approx6[-1, :])
#pw_strong_error[6, :] = np.abs(real_solution[-1, :] - approx7[-1, :])

strong_error = np.mean(pw_strong_error, axis=1)

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

#%% Several plots

#%% all errors plot
both_error_fig = plt.figure('both_error_fig')
plt.title("errors for beta=%.5f \n strong error rate = %f \n weak error rate = %f" % (beta, rate_strong, rate_weak))
plt.semilogy(strong_error, marker='o', label='strong error')
plt.semilogy(weak_error, marker='o', label='weak error')
plt.semilogy([0.5, 1.5, 2.5, 3.5], consecutive_strong_error, marker='o', label='error between consecutive approximations')
plt.semilogy([0.5, 1.5, 2.5, 3.5], consecutive_weak_error, marker='o', label='bias between consecutive approximations')
#plt.semilogy([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], consecutive_strong_error, marker='o', label='error between consecutive approximations')
#plt.semilogy([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], consecutive_weak_error, marker='o', label='bias between consecutive approximations')
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
path = 9
plt.figure('paths')
plt.plot(t0_real, real_solution[:,path])
plt.plot(t0_a1,         approx1[:,path])
plt.plot(t0_a2,         approx2[:,path])
plt.plot(t0_a3,         approx3[:,path])
plt.plot(t0_a4,         approx4[:,path])
plt.plot(t0_a5,         approx5[:,path])
plt.show()

#%% histograms
fig, ax = plt.subplots()
ax.hist(real_solution[1,:], bins=30)
ax.hist(approx1[1,:], bins=30)
plt.show()















































