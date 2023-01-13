# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:40:52 2022

@author: mmlmcj
"""
#%%
from sympy import symbols, exp, sqrt, pi, limit, oo, plot, sign

#%%
z = symbols('z')

expr = z * exp(-z**2/2) / sqrt(4*pi)

limit_expr = limit(expr, z, oo)

print(limit_expr)

plot(expr, xlim=[-10, 10])

#%%
from sympy import Abs

#%%
x = symbols('x')
epsilon = 0.00001
beta1 = epsilon
beta2 = 1/128
beta3 = 1/64
beta4 = 1/32
beta5 = 1/16
beta6 = 1/8
beta7 = 1/4 - epsilon

#mfunct1 = - Abs(x)**(1 - beta1)
#mfunct2 = - Abs(x)**(1 - beta2)

mfunct1 = sign(1-x) * Abs(1 - x)**(1 - beta1)
mfunct2 = sign(1-x) * Abs(1 - x)**(1 - beta2)

plot(mfunct1, xlim=[-10, 10])
plot(mfunct2, xlim=[-10, 10])

