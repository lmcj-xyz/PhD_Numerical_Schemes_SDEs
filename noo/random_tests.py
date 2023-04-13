# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:44:16 2023

@author: mmlmcj
"""

#%%
import numpy as np
import math
import time
from numpy.random import default_rng
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













