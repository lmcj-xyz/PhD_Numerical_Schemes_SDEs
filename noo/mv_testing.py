#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:00:40 2023

@author: lmcj
"""
#%%
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy.random import default_rng
rng = default_rng()
#%%
a1 = approx1[0, :]
a1r = a1[:, np.newaxis] # Adding axis
a1rr = a1.reshape(-1, 1)
kde_scott = KernelDensity(kernel="gaussian", bandwidth="scott").fit(a1r)
kde_silverman = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(a1r)
kde_silverman_tophat = KernelDensity(kernel="tophat", bandwidth="silverman").fit(a1r)
x = np.linspace(-3, 6, 1000)[:, np.newaxis]
x_kde1 = kde_scott.score_samples(x)
x_kde2 = kde_silverman.score_samples(x)
x_kde3 = kde_silverman_tophat.score_samples(x)
#%%
plt.figure()
plt.hist(a1, weights=np.ones_like(a1)/len(a1))
plt.plot(x, np.exp(x_kde1))
#plt.plot(x, np.exp(x_kde2))
#plt.fill(x, np.exp(x_kde2))
plt.show()
#%%
np.exp(kde_scott.score(3))
#%% euler scheme
y0 = 2
sample_paths = int(10e2)
time_steps = 2**7
time_end = 1
time_start = 0
dt = (time_end - time_start)/(time_steps-1)
z = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(time_steps, sample_paths))
#y = np.full(shape=(time_steps+1, sample_paths), fill_value=y0)
y = np.zeros(shape=(time_steps+1, sample_paths))
y[0, :] = y0
for i in range(time_steps):
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(y[i,:].reshape(-1, 1))
    y[i+1, :] = y[i, :] + 5*np.exp(kde.score_samples(y[i,:].reshape(-1, 1)))*dt + 5*z[i, :]
    #y[i+1, :] = y[i, :] - 5*dt + 0.5*z[i, :]

