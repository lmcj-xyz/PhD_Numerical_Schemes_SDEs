# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:44:23 2023

@author: mmlmcj
"""

#%% Packages
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
seed = 123456
rng = default_rng(seed)

#%% Functions
def mu_func(x):
    return 2.5*x

def sigma_func(x):
    return 0.5*x

def coarse_noise(noise, time_steps):
    sample_paths = np.shape(noise)[1]
    coarse_noise = np.zeros(shape = (time_steps, sample_paths))
    q = int(np.shape(noise)[0] / time_steps)
    if q == 1:
        coarse_noise = noise
    else:
        temp = noise.reshape(time_steps, q, sample_paths)
        coarse_noise = np.sum(temp, axis=1)
    return coarse_noise

def euler(
        drift, 
        diffusion, 
        noise,
        y0,
        sample_paths, 
        time_start, time_end, time_steps
        ):
    t = np.linspace(time_start, time_end, time_steps)
    t0 = np.insert(t, 0, 0)
    dt = (time_end - time_start)/(time_steps-1)
    z = coarse_noise(noise, time_steps)
    y = np.zeros(shape=(time_steps+1, sample_paths))
    y[0,:] = y0
    for ti in range(time_steps):
        y[ti+1, :] = y[ti, :] \
            + drift(y[ti, :])*dt \
                + diffusion(y[ti, :])*z[ti, :]
    return t0, dt, y


#%% Solutions
y0 = 10
sp = 10**5
ts = 0
te = 1
t_steps = 2**10
delta_min = (te - ts)/(t_steps-1)


z = rng.normal(
    loc=0.0,
    scale=np.sqrt(delta_min),
    size=(t_steps, sp)
    )

# Approximations parameters
tsp1 = 2**5
tsp2 = 2**6
tsp3 = 2**7
tsp4 = 2**8

# Solutions
time, dt, solution = euler(mu_func, sigma_func, z, y0, sp, ts, te, t_steps)
ta1, dt1, approx1 = euler(mu_func, sigma_func, z, y0, sp, ts, te, tsp1)
ta2, dt2, approx2 = euler(mu_func, sigma_func, z, y0, sp, ts, te, tsp2)
ta3, dt3, approx3 = euler(mu_func, sigma_func, z, y0, sp, ts, te, tsp3)
ta4, dt4, approx4 = euler(mu_func, sigma_func, z, y0, sp, ts, te, tsp4)

#%% Plots of a single path
path = 0
plt.figure('Single sample path')
plt.plot(time,  solution[:, path], label="Real solution")
plt.plot(ta1,    approx1[:, path], label="Approximation 1")
plt.plot(ta2,    approx2[:, path], label="Approximation 2")
plt.plot(ta3,    approx3[:, path], label="Approximation 3")
plt.plot(ta4,    approx4[:, path], label="Approximation 4")
plt.legend()
plt.show()

#%% Rate of convergence
pathwise_error = np.zeros(shape=(4, sp))
pathwise_error[0,:] = np.abs(solution[-1, :] - approx1[-1, :])
pathwise_error[1,:] = np.abs(solution[-1, :] - approx2[-1, :])
pathwise_error[2,:] = np.abs(solution[-1, :] - approx3[-1, :])
pathwise_error[3,:] = np.abs(solution[-1, :] - approx4[-1, :])

strong_error = np.zeros(4)
strong_error[0] = np.mean(pathwise_error[0, :])
strong_error[1] = np.mean(pathwise_error[1, :])
strong_error[2] = np.mean(pathwise_error[2, :])
strong_error[3] = np.mean(pathwise_error[3, :])

### Linear regression to compute rate of convergence
reg = np.ones(4)
x_axis = np.array([dt1, dt2, dt3, dt4])
A = np.vstack([x_axis, reg]).T
y_reg = np.log2(strong_error[:, np.newaxis])
rate, intersection = np.linalg.lstsq(A, y_reg, rcond=None)[0]

plt.figure('Rate of convergence')
plt.semilogy(x_axis, strong_error)
plt.show()






















