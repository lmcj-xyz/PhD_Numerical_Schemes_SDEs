# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:40:52 2022

@author: mmlmcj
"""

from sympy import symbols, exp, sqrt, pi, limit, oo, plot

z = symbols('z')

expr = z * exp(-z**2/2) / sqrt(4*pi)

limit_expr = limit(expr, z, oo)

print(limit_expr)

plot(expr, xlim=[-10, 10])