# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:39:45 2023

@author: mmlmcj
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from scipy.integrate import quad_vec
from scipy.stats import norm
import time

from dist_sde_no_oop import *

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