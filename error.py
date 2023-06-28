#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:58:44 2023

@author: lmcj
"""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import dsdes as ds
from scipy.stats import linregress

rng = default_rng()

plt.rcParams['figure.dpi'] = 200

for i in range(10):
    # Parameters for Euler scheme
    keys = ('real', 'approx1', 'approx2', 'approx3', 'approx4', 'approx5')
    time_steps_tuple = (2**15, 2**8, 2**9, 2**10, 2**11, 2**12)
    time_steps = dict(zip(keys, time_steps_tuple))

    error_keys = ('e1', 'e2', 'e3', 'e4', 'e5')

    epsilon = 10e-6
    beta = 7/16
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
                lambda d, t: ds.solves2(
                    y0, d, noise, time_start, time_end, t, sample_paths, grid_x
                    ),
                drift_array.values(),
                time_steps.values(),
                )
            )
    solution = dict(zip(keys, solution_tuple))

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

    both_error_fig = plt.figure('strong_error_fig')
    plt.title(
        r"Rate of convergence = %f for $\beta$=%.2f" % (rate_strong, beta)
        )
    plt.plot(plot_dt,
             plot_error,
             marker='o',
             label='Strong error')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    # plt.show()
    plt.savefig('beta04375-'+str(i)+'.png')
    plt.clf()
    print('rate = ', rate_strong)
    file = open("rates.txt", "a")
    file.write(str(rate_strong))
    file.write("\n")
    file.close()
