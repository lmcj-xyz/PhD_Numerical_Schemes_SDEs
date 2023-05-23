# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:09:33 2023

@author: mmlmcj
"""
#%% libraries
import numpy as np
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm
import math as m

#%% fbm func
# Fractional Brownian motion (fBm) creation function
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
    #g_bridge = g - (fbm_grid/half_support)*g[-1]
    cholesky = np.linalg.cholesky(a=covariance)
    #fbm_arr = np.matmul(cholesky, g_bridge)
    fbm_arr = np.matmul(cholesky, g)
    #fbm_arr = np.concatenate([np.zeros(1),fbm_arr])
    return fbm_arr

#%% bridge func
def bridge(f, grid):
    return f - (f[-1]/grid[-1])*grid

#%% heat parameter func
# Heat kernel parameter creation based on time steps of the Euler scheme
def heat_param(time_steps, hurst):
    #eta = 1/(2*(hurst-1/2)**2 + 2 - hurst) # Parameter that was being used
    eta = 1/((hurst-1/2)**2 + 2 - hurst) # Parameter from paper?
    #eta = 1/((hurst-1/2)**2 + 2) # Some testing parameter
    
    #param = np.sqrt(1/(time_steps**(eta))) # Incorrect parameter
    #param = 1/(time_steps**0.001) # Using a different parameter
    param = 1/(time_steps**(eta)) # Parameter according to the theory
    return param
    #return 0.5

#%% normal differences func
# Creation of the drift by convoluting fBm with the 
# derivative of the heat kernel
def normal_differences(heat_parameter, points_x, x_grid, half_support):
    sqrt_heat_parameter = m.sqrt(heat_parameter)
    diff_norm = np.zeros(shape=points_x)
    delta = half_support/(points_x)
    const = -1/heat_parameter

    p = lambda u: const*(x_grid - u)*norm.pdf(x_grid - u,
                                              loc=0, 
                                              scale=sqrt_heat_parameter)
    # 'quad_vec' gives you the the result of the integral and the error
    # this is why we select the element '0' from that computation
    diff_norm = quad_vec(p, -delta, delta)[0]

    #return p, diff_norm
    return diff_norm

#%% drift func
# Drift coefficient as a piecewise function created out of an array
def drift_func(x, drift_array, grid, points, delta):
    return np.piecewise(
        x, 
        [(i - delta <= x)*(x < i + delta) for i in grid], 
        [drift_array[i] for i in range(points)]
        )


#%% coarse noise func
# Coarse noise
def coarse_noise(z, time_steps, sample_paths):
    z_coarse = np.zeros(shape = (time_steps, sample_paths))
    q = int(np.shape(z)[0] / time_steps)
    if q == 1:
        z_coarse = z
    else:
        temp = z.reshape(time_steps, q, sample_paths)
        z_coarse = np.sum(temp, axis=1)
    return z_coarse


#%% sde solver func
# Euler scheme solver for the distributional drift
# This function keeps the entire array corresponding to the solutions
def solve(
        y0, 
        drift_array, 
        z, 
        time_start, 
        time_end, 
        time_steps, 
        sample_paths,
        grid,
        points,
        delta
        ):
    y = np.zeros(shape=(time_steps+1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[i+1, :] = y[i, :] \
                + drift_func(
                        x=y[i, :], 
                        drift_array=drift_array,
                        grid=grid,
                        points=points,
                        delta=delta
                        )*dt \
                    + z_coarse[i, :]
    return y

#%% sde solver func terminal time
# Euler scheme solver for the distributional drift
# This function only keeps the terminal time of the solution
def solves(
        y0, 
        drift_array, 
        z, 
        time_start, 
        time_end, 
        time_steps, 
        sample_paths,
        grid,
        points,
        delta
        ):
    y = np.zeros(shape=(1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[0, :] = y[0, :] \
                + drift_func(
                        x=y[0, :], 
                        drift_array=drift_array,
                        grid=grid,
                        points=points,
                        delta=delta
                        )*dt \
                    + z_coarse[i, :]
    return y

#%% generic sde solver func
# Euler scheme solver for a generic SDE
def gen_solve(
        y0, 
        drift,
        diffusion,
        z, 
        time_start, 
        time_end, 
        time_steps, 
        sample_paths
        ):
    y = np.zeros(shape=(time_steps+1, sample_paths))
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        y[i+1, :] = y[i, :] \
                + drift(t = i, x=y[i, :])*dt \
                    + diffusion(t = i, x=y[i, :])*z[i, :]
    return y

#%% mckean-vlasov sde solver func
# Euler scheme solver for a generic McKean-Vlasov SDE
def mv_solve(
        y0, 
        drift,
        diffusion,
        z, 
        time_start, 
        time_end, 
        time_steps, 
        sample_paths
        ):
    y = np.zeros(shape=(time_steps+1, sample_paths))
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    for i in range(time_steps):
        nu = 1# use the KDE from scikit learn
        y[i+1, :] = y[i, :] \
                + drift(t = i, x=y[i, :], law=nu(y[i,:]))*dt \
                    + diffusion(t = i, x=y[i, :], law=nu(y[i,:]))*z[i, :]
    return y

#%%
# Create solutions/approximations
# This will be tested in approximate_tests.py
# To test each one of the functions above separately use manual_tests.py
def approximate(
        fbm, 
        time_steps, 
        hurst, 
        time_start, 
        time_end,
        noise,
        sample_paths,
        y0,
        x_grid,
        points_x,
        delta_x,
        half_support):
    df = normal_differences(
        np.sqrt(heat_param(time_steps, hurst)), 
        #15/11, # for testing with a fixed heat kernel parameter
        points_x, 
        x_grid, 
        half_support
        )
    drift_array = np.convolve(fbm, df, 'same')
    dt = (time_end - time_start)/(time_steps-1)
    time_grid = np.linspace(time_start + dt, time_end, time_steps)
    time_grid0 = np.insert(time_grid, 0, 0)
    z = coarse_noise(noise, time_steps, sample_paths)
    solution = solve(
        y0, 
        drift_array, 
        z, 
        time_start, 
        time_end, 
        time_steps, 
        sample_paths, 
        x_grid, 
        points_x, 
        delta_x
        )
    return solution, time_grid, time_grid0, drift_array