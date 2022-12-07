# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:28:41 2022

@author: mmlmcj
"""
#%%
# Graphical parameters
pdpi = 500
#%%
##### Numeric
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# beta
beta_n = np.linspace(0, 0.25, 100)
#%%
# eta
eta_n = 1/(2*(0.5 - beta_n)**2 + beta_n + 1)
plt.figure(dpi=pdpi)
plt.plot(beta_n, eta_n, color="orange")
plt.title(r"Parameter $\eta$ with respect to $\beta$ from $p_{f_{m}} = m^{-\eta}$")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\eta$")
plt.grid()
plt.savefig("eta.png")
plt.show()
#%%
# Rate of convergence
rate_n = eta_n*(0.5 - beta_n)**2
plt.figure(dpi=pdpi)
plt.plot(beta_n, rate_n, color="purple")
plt.title(r"Rate of convergence of EM scheme as a function of $\beta$")
plt.xlabel(r"$\beta$")
plt.ylabel("Rate")
plt.grid()
plt.savefig("theoretical_rate.png")
plt.show()
#%%
# Empirical rate of convergence
def f_rate_discrete(x):
    r = (1/(2*(0.5 - x)**2 + x + 1)*(0.5 - x)**2)
    return r
beta_discrete = np.array([0, 1/64, 1/32, 1/16, 1/8, 3/16, 1/4])
rate_discrete = f_rate_discrete(beta_discrete)
data = pd.read_csv("runs_process.csv")
plt.figure(dpi=pdpi)
plt.title("Empirical rate of convergence EM scheme")
plt.boxplot(data, notch=True, 
         labels=[
             r"$\epsilon$", 
             r"$\frac{1}{64}$",
             r"$\frac{1}{32}$",
             r"$\frac{1}{16}$",
             r"$\frac{2}{16}$",
             r"$\frac{3}{16}$",
             r"$\frac{1}{4}-\epsilon$"
             ]
         )
plt.xlabel(r"$\beta$")
plt.ylabel("Rate of convergence")
plt.plot(range(1, 8), rate_discrete, marker="x", label="Theoretical rate")
plt.legend()
plt.savefig("empirical_rate_bp.png")
plt.show()
#%%
# Symbolic
from sympy import Symbol
from sympy.plotting import plot
#%%
beta_s = Symbol('beta')
rate_s = -(1/2) + (1/4)*((beta_s + 1)/(((1/2) - beta_s)**2 + beta_s + 1))
plot(rate_s, xlim=[0, 1/4], ylim=[-0.305, -0.26])
