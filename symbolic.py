#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:59:26 2023

@author: lmcj
"""
from sympy import exp, sin, pi, sqrt, Integral, Derivative, plot, oo, simplify
from sympy import symbols

x, y, z, t = symbols('x y z t', real=True)

heat_kernel = (1/sqrt(2*pi*t))*exp(-(x-y)**2/(2*t))
der_sine = Derivative.diff(sin(y), y)
der_heat_kernel = Derivative.diff((1/sqrt(2*pi*t))*exp(-(y)**2/(2*t)), y)

convolution_hk_sine = Integral.integrate(heat_kernel*der_sine, (y, -oo, oo))

t_hk = 8/11
convolution_for_t_hk = convolution_hk_sine.subs(t, t_hk)

plot(convolution_for_t_hk)
