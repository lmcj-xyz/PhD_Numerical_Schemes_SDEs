#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:58:44 2023

@author: lmcj
"""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import pickle
import time

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds


start_time = time.time()

rng = default_rng()

plt.rcParams['figure.dpi'] = 200

# Parameters for Euler scheme
keys = ('real', 'approx1', 'approx2', 'approx3', 'approx4', 'approx5')

time_steps_tuple = (2**11, 2**5, 2**6, 2**7, 2**8, 2**9)
time_steps = dict(zip(keys, time_steps_tuple))

error_keys = ('e1', 'e2', 'e3', 'e4', 'e5')

epsilon = 10e-6
beta = 1/2 - epsilon
hurst = 1 - beta
y0 = 1
sample_paths = 10**4
time_start = 0
time_end = 1
dt_tuple = tuple(
        map(
            lambda t: (time_end - time_start)/(t - 1), time_steps.values()
            )
        )
dt = dict(zip(keys, dt_tuple))
time_grid_tuple = tuple(
        map(
            lambda dt, t: np.linspace(time_start + dt, time_end, t),
            dt.values(),
            time_steps.values()
            )
        )
time_grid = dict(zip(keys, time_grid_tuple))

noise = rng.normal(
        loc=0.0, scale=np.sqrt(dt['real']),
        size=(time_steps['real'], sample_paths)
        )

# Parameters to create fBm
points_x = 2**12  # According to the lower bound in the paper
half_support = 10

eta = 1/((hurst-1/2)**2 + 2 - hurst)
lower_bound = 2*half_support*time_steps['real']**(eta/2)
eqn = 66
if (points_x <= lower_bound):
    msg = 'You need to define your fBm on at least %.2f \
            points as per equation (%d) in the paper.' % (lower_bound, eqn)
    raise ValueError(msg)
    sys.exit(1)

delta_x = half_support/(points_x-1)
grid_x = np.linspace(start=-half_support, stop=half_support, num=points_x)
grid_x0 = np.linspace(start=0, stop=2*half_support, num=points_x)
fbm_array = ds.fbm(hurst, points_x, half_support)
bridge_array = ds.bridge(fbm_array, grid_x0)

# Variance of heat kernel
var_tuple = tuple(map(lambda t: ds.heat_kernel_var(t, hurst), time_steps.values()))
var_heat_kernel = dict(zip(keys, var_tuple))

# Integral between grid points
integral_tuple = tuple(
        map(
            lambda t: ds.integral_between_grid_points(t, grid_x, half_support),
            var_heat_kernel.values()
            )
        )
integral_grid = dict(zip(keys, integral_tuple))

# Drift creation
drift_tuple = tuple(
        map(
            lambda i: ds.create_drift_array(bridge_array, i),
            integral_grid.values()
            )
        )
drift_array = dict(zip(keys, drift_tuple))

solution_tuple = tuple(
        map(
            lambda d, t: ds.solve_mv(
                y0, d, noise,
                time_start, time_end, t,
                sample_paths, grid_x, half_support, 2**8, 10,
                lambda x: 5*np.sin(0.1*x - 10)
                ),
            drift_array.values(),
            time_steps.values(),
            )
        )
solution = dict(zip(keys, solution_tuple))
#%%
strong_error = dict.fromkeys(error_keys)
strong_error['e1'] = np.abs(solution['real'] - solution['approx1'])
strong_error['e2'] = np.abs(solution['real'] - solution['approx2'])
strong_error['e3'] = np.abs(solution['real'] - solution['approx3'])
strong_error['e4'] = np.abs(solution['real'] - solution['approx4'])
strong_error['e5'] = np.abs(solution['real'] - solution['approx5'])

plot_error = [np.mean(value) for key, value in strong_error.items()]
plot_dt = [value for key, value in dt.items() if key not in 'real']

log_strong_error = np.log10(plot_error)
log_dt = np.log10(plot_dt)

reg_strong = linregress(log_dt, log_strong_error)
rate_strong = reg_strong.slope
intersection_strong = reg_strong.intercept

plot_dict = {
        'beta': beta,
        'rate': rate_strong,
        'dt': plot_dt,
        'error': plot_error,
        }

plot = True
if plot:
    fig, ax = plt.subplots()
    ax.set_title(
        r'Rate of convergence r = %f for $\beta$=%f' % (rate_strong, beta)
        )
    ax.plot(plot_dt,
            plot_error,
            marker='o',
            label='Strong error')
    ax.grid(which='both')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\log_{10}(\Delta t)$')
    ax.set_ylabel(r'$\log_{10}(\epsilon)$')
    ax.legend()
    plt.show()

end_time = time.time()
running_time = end_time - start_time
print(running_time)

#%%
#saving = input('Do you want to save to files the plot and its corresponding dictionary? (yes/no): ')
saving = False
date_string = time.strftime("%Y-%m-%d-%H-%M")
if saving:
    plot_string = date_string + '-rate.pdf'
    dict_string = date_string + '-dict_plot.pkl'
    fig.savefig(plot_string, dpi=200)
    with open(dict_string, 'wb') as fp:
        pickle.dump(plot_dict, fp)
        print('Files saved succesfully')
