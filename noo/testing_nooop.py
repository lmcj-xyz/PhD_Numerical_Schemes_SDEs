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

from dist_sde_no_oop import *
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
# The next tests are to run only if no_oop_approximate.py has
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
