# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:28:41 2022

@author: mmlmcj
"""

#%%
# Numeric
import numpy as np
import matplotlib.pyplot as plt
beta_n = np.linspace(0, 0.25, 100)
rate_n = -0.5 + 0.25*((beta_n + 1)/((0.5 - beta_n)**2 + beta_n + 1))
plt.figure()
plt.plot(beta_n, rate_n)
plt.xlabel(r"$\hat{\beta}$")
plt.ylabel("Rate")
plt.grid()
plt.show()

#%%
# Symbolic
from sympy import Symbol
from sympy.plotting import plot
beta_s = Symbol('beta')
rate_s = -(1/2) + (1/4)*((beta_s + 1)/(((1/2) - beta_s)**2 + beta_s + 1))
plot(rate_s, xlim=[0, 1/4], ylim=[-0.305, -0.26])
