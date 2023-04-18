#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:04:51 2023

@author: lmcj
"""

#%% packages and parameters
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm, linregress
import time

from dist_sdes import *

# QOL parameters
plt.rcParams['figure.dpi'] = 500

#%%
a = 3
b = 0.5

def mu(t, x):
    return a*x

def sigma(t, x):
    return b*x

ic = 1
t0 = 0
t1 = 1
ts = 2**5
tg = np.linspace(t0, t1, ts+1)
#tg = np.insert(tg, 0, 0)
dt = (t1 - t0)/(ts - 1)
sp = 10**3
bm = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(ts, sp))

soln = np.zeros(shape=(ts+1, sp))
soln[0, :] = ic
bmcs = np.cumsum(bm, axis=0)
for i in range(ts):
    soln[i+1, :] = ic*np.exp((a - 0.5*b**2)*tg[i+1] + b*bmcs[i])
    
approx = gen_solve(
    y0 = ic, 
    drift = mu, 
    diffusion = sigma, 
    z = bm, 
    time_start = t0,
    time_end = t1,
    time_steps = ts,
    sample_paths = sp
    )

#%%
plt.figure()
plt.plot(soln[:, 3])
plt.plot(approx[:, 3])
plt.show()