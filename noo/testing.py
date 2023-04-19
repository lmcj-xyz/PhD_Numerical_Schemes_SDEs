# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:39:45 2023

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

from dsdes import *
#%%
fbm_trajectory = fbm(0.55, 2**8, 5)
drift = np.convolve(fbm_trajectory, df, 'same')
drift1 = np.convolve(fbm_trajectory, df1, 'same')

plt.figure()
plt.plot(x, fbm_trajectory, label='fbm')
plt.plot(x, drift1, label='derivative 1')
plt.plot(x, drift, label='derivative 2')
plt.legend()
plt.grid()
plt.show()

#%%
# The next tests are to run only if approximate_tests.py has
# been used at least once, so that we can use the variables
# defined there
#%%
i = 2*m.sqrt(1/(time_steps_approx2**(12/11)))
density = ((x_grid < i) & (x_grid > -i)).sum()
#%% max and min in sample paths
max_soln_perc = (real_solution.max(axis=0) > 6).sum()
min_soln_perc = (real_solution.min(axis=0) < 0).sum()
#%% plot fbm and der
drift_array_real_total = np.convolve(fbm_array, df_array_real)
plt.figure()
plt.plot(fbm_array)
plt.plot(drift_array_real)
#plt.plot(drift_array1)
#plt.plot(drift_array3)
#plt.ylim([-5, 10])
plt.show()

#%%
fbm = np.load("fbm_bad_result.npy")

#%%
x = 4

st1 = time.process_time()
np.sqrt(x)
et1 = time.process_time()
print("numpy time", et1-st1)

st2 = time.process_time()
math.sqrt(x)
et2 = time.process_time()
print("math time", et1-st1)

#%%
rng = default_rng(2222)
def fbm1(hurst, points_x, half_support):
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
    #fbm_arr = np.concatenate([np.zeros(1),fbm_arr])
    return fbm_arr
def fbm2(hurst, points_x, half_support):
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
    #fbm_arr = np.concatenate([np.zeros(1),fbm_arr])
    return fbm_arr
#%%

points = 2**8
t = 1/1000
hk = norm.pdf(x, 0, t)
df = normal_differences(t, points, np.linspace(-1, 1, points), 1)
plt.figure()
plt.plot(x, hk)
plt.title("Densidad gaussiana t=1/50")
plt.show()

#%%
const = np.ones(points)
x = np.linspace(-1, 1, points)
x2 = 3*x**2
x3 = x**3
cos = np.cos(x)
sin = np.sin(x)
f = fbm(0.65, points, 1)
d = np.convolve(f, df, mode='same')

convx = np.convolve(x, df, mode='same')
convx3 = np.convolve(x3, df, mode='same')
convsin = np.convolve(sin, df, mode='same')

plt.figure()
plt.plot(x, const, label="f'(x)=1")
plt.plot(x, convx, label="Convolución")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, x2, label="g'(x)=2x³")
plt.plot(x, convx3, label="Convolución")
plt.legend()
plt.show()

plt.figure()
plt.plot(x, cos, label="h'(x)=cos(x)")
plt.plot(x, convsin, label="Convolución")
plt.legend()
plt.show()

#%%
plt.figure()
plt.plot(x, f, label='B^H(x)')
plt.plot(x, d, label='dB^H(x)')
plt.ylim([-2, 5])
plt.show()

#%%
t1 = np.linspace(0, 1, 2**5)
t2 = np.linspace(0, 1, 2**3)

x = rng.normal(size=2**5, loc=1, scale=1/2**5)
y = np.sum(x.reshape(2**3, 2**2), axis=1)*1/2**2

plt.figure()
plt.plot(t1, x)
plt.plot(t2, y)
plt.grid()
plt.show()