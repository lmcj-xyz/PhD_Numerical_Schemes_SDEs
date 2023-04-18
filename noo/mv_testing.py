#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:00:40 2023

@author: lmcj
"""
#%%
from sklearn.neighbors import KernelDensity
#%%
a1 = approx[1, :]
a1r = a1[:, np.newaxis]
kde_scott = KernelDensity(kernel="gaussian", bandwidth="scott").fit(a1r)
kde_silverman = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(a1r)
a2 = approx[2, :]
a2r = a2[:, np.newaxis]
x = np.linspace(0.75, 1.5, 100)[:, np.newaxis]
x_kde1 = kde_scott.score_samples(x)
x_kde2 = kde_silverman.score_samples(x)
#%%
plt.figure()
plt.hist(a1, weights=np.ones_like(a1)/len(a1))
plt.plot(x, np.exp(x_kde1))
plt.plot(x, np.exp(x_kde2))
plt.show()