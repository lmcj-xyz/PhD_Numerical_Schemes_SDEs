#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:10:33 2023

@author: lmcj
"""
import dsdes
import matplotlib.pyplot as plt
from numpy.random import default_rng

plt.rcParams['figure.dpi'] = 200
cc = ['#377eb8', '#ff7f00', '#4daf4a',
      '#f781bf', '#a65628', '#984ea3',
      '#999999', '#e41a1c', '#dede00']
rng = default_rng()
H1 = 0.65
H2 = 0.85
pointsX = 2**10
L = 5
gaussian = rng.standard_normal(size=pointsX)
fbm1 = dsdes.fbm_alt(hurst=H1, gaussian=gaussian, half_support=L)
fbm2 = dsdes.fbm_alt(hurst=H2, gaussian=gaussian, half_support=L)

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
axs[0].plot(fbm1, label='H=0.65', color=cc[4])
axs[1].plot(fbm2, label='H=0.85', color=cc[5])
fig.suptitle('FBM with different values of H')
fig.legend()
fig.show()
