# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:09:33 2023

@author: mmlmcj
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm

#%%
# QOL parameters
plt.rcParams['figure.dpi'] = 500

#%%
# Variables to modify for the scheme
hurst = 0.75
time_steps_max = 2**10
time_steps_approx1 = 2**8
time_steps_approx2 = 2**6
time_steps_approx3 = 2**9

# Variables to create fBm
points_x = 2**4
half_support = 3
x_grid = np.linspace(
        start = -half_support,
        stop = half_support,
        num = points_x
        )

#%% 
# Creation of fBm
def fbm(hurst, points_x, half_support):
    fbm_grid = np.linspace(
            start = 1/points_x,
            stop = 2*half_support,
            #stop = 1,
            num = points_x
            )
    x_grid, y_grid = np.meshgrid(
            fbm_grid, 
            fbm_grid, 
            sparse=False,
            indexing='ij'
            )
    covariance = 0.5*(
            np.abs(x_grid)**(2*hurst) +
            np.abs(y_grid)**(2*hurst) - 
            np.abs(x_grid - y_grid)**(2*hurst)
            )
    g = rng.standard_normal(size=points_x)
    cholesky = np.linalg.cholesky(a=covariance)
    fbm_arr = np.matmul(cholesky, g)
    fbm_arr = np.concatenate([np.zeros(1),fbm_arr])
    return fbm_arr
#%%
# Create an array of fBm
fbm_array = fbm(hurst, points_x, half_support)

#%%
# Plot fBm
fbm_fig = plt.figure('fbm')
plt.plot(fbm_array)
plt.show()

#%%
# Function for heat parameter
def heat_param(time_steps, hurst):
    eta = 1/(2*(hurst-1/2)**2 + 2 - hurst)
    param = np.sqrt(1/(time_steps**(eta)))
    return param

#%%
# Create function for convolution
def normal_differences(sqrt_heat_parameter):
    diff_norm = np.zeros(shape=points_x)
    delta = half_support/points_x
    const = -1/sqrt_heat_parameter**2

    p = lambda u: const*(x_grid + u)*norm.pdf(x_grid+u, loc=0, scale=sqrt_heat_parameter)
    diff_norm = quad_vec(p, -delta, delta)[0]

    return diff_norm

#%%
# Create a dF
df_array_real = normal_differences(np.sqrt(heat_param(time_steps_max, hurst)))
df_array1 = normal_differences(np.sqrt(heat_param(time_steps_approx1, hurst)))
df_array2 = normal_differences(np.sqrt(heat_param(time_steps_approx2, hurst)))
df_array3 = normal_differences(np.sqrt(heat_param(time_steps_approx3, hurst)))

#%% Plot dF
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

#%%
# Plot drift
drift_fig = plt.figure('drift')
plt.plot(drift_array_real, label="drift real solution")
plt.plot(drift_array1, label="drift approximation 1")
plt.plot(drift_array2, label="drift approximation 2")
plt.plot(drift_array3, label="drift approximation 3")
plt.legend()
plt.show()

#%%
# Define a piecewise function out of the array
delta_x = half_support/(points_x-1)
def drift_func(x, dd):
    return np.piecewise(
        x, 
        [(i - delta_x <= x)*(x < i + delta_x) for i in x_grid], 
        [dd[i] for i in range(points_x)]
        )

#%%
# Evaluate and plot some drift functions
x2 = np.linspace(-half_support, half_support, 2**8)
eval_real = drift_func(x2, drift_array_real)
eval1 = drift_func(x2, drift_array1)
eval2 = drift_func(x2, drift_array2)
eval3 = drift_func(x2, drift_array3)
drift_func_fig = plt.figure('driftfunc')
plt.plot(x2, eval_real, label="real solution drift function")
plt.plot(x2, eval1, label="approximation 1 drift function")
plt.plot(x2, eval2, label="approximation 2 drift function")
plt.plot(x2, eval3, label="approximation 3 drift function")
plt.grid()
plt.legend()
plt.show()

#%%
# Euler scheme
y0 = 1
sample_paths = 10**1
time_start = 0
time_end = 1
# Parameters for real solution
dt_real = (time_end - time_start)/(time_steps_max-1)
time_grid_real = np.linspace(time_start + dt_real, time_end, time_steps_max)
time_grid_real0 = np.insert(time_grid_real, 0, 0)
z_real = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt_real),
        size=(time_steps_max, sample_paths)
        )

# Make a coarser Z
def coarse_noise(z, time_steps):
    z_coarse = np.zeros(shape = (time_steps, sample_paths))
    q = int(np.shape(z)[0] / time_steps)
    if q == 1:
        z_coarse = z
    else:
        temp = z.reshape(
                time_steps, 
                q,
                sample_paths
                )
        z_coarse = np.sum(temp, axis=1)
    return z_coarse

# Parameters for approximation 1
dt_approx1 = (time_end - time_start)/(time_steps_approx1-1)
time_grid_approx1 = np.linspace(time_start + dt_approx1, time_end, time_steps_approx1)
time_grid_approx10 = np.insert(time_grid_approx1, 0, 0)
z_approx1 = coarse_noise(z_real, time_steps_approx1)

# Parameters for approximation 2
dt_approx2 = (time_end - time_start)/(time_steps_approx2-1)
time_grid_approx2 = np.linspace(time_start + dt_approx2, time_end, time_steps_approx2)
time_grid_approx20 = np.insert(time_grid_approx2, 0, 0)
z_approx2 = coarse_noise(z_real, time_steps_approx2)

# Parameters for approximation 3
dt_approx3 = (time_end - time_start)/(time_steps_approx3-1)
time_grid_approx3 = np.linspace(time_start + dt_approx3, time_end, time_steps_approx3)
time_grid_approx30 = np.insert(time_grid_approx3, 0, 0)
z_approx3 = coarse_noise(z_real, time_steps_approx3)


# Euler scheme function
def solve(y0, drift_array, z, time_steps, sample_paths):
    y = np.zeros(shape=(time_steps+1, sample_paths))
    y[0, :] = y0
    for i in range(time_steps):
        y[i+1, :] = y[i, :] \
                + drift_func(
                        x = y[i, :], 
                        dd = drift_array
                        )*dt_real \
                    + z[i, :]
    return y

#%%
# Solve an SDE
real_solution = solve(y0, drift_array1, z_real, time_steps_max, sample_paths)
approx1 = solve(y0, drift_array1, z_approx1, time_steps_approx1, sample_paths)
approx2 = solve(y0, drift_array2, z_approx2, time_steps_approx2, sample_paths)
approx3 = solve(y0, drift_array3, z_approx3, time_steps_approx3, sample_paths)

#%%
# Plot solution to SDE
emreal_fig = plt.figure('emreal_fig')
plt.plot(time_grid_real0, real_solution)
plt.title("real solution")
plt.show()

em1_fig = plt.figure('em2_fig')
plt.plot(time_grid_approx10, approx1)
plt.title("approximation 1")
plt.show()


em2_fig = plt.figure('em2_fig')
plt.plot(time_grid_approx20, approx2)
plt.title("approximation 2")
plt.show()

em3_fig = plt.figure('em3_fig')
plt.plot(time_grid_approx30, approx3)
plt.title("approximation 3")
plt.show()

#%%
# Comparing single sample paths of different approximations
emssp_fig = plt.figure('emssp_fig')
plt.plot(time_grid_real0, real_solution[:, 0], label="real solution")
plt.plot(time_grid_approx10, approx1[:, 0], label="approximation 1")
plt.plot(time_grid_approx20, approx2[:, 0], label="approximation 2")
plt.plot(time_grid_approx30, approx3[:, 0], label="approximation 3")
plt.title("single sample path comparison")
plt.legend()
plt.show()
































