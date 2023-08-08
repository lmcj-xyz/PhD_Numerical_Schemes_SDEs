#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 21:59:26 2023

@author: lmcj
"""
#%% importing
from sympy import exp, sin, cos, pi, sqrt, Integral, Derivative, plot, oo, simplify
from sympy import symbols

x, y, z, t = symbols('x y z t', real=True)

#%% computing
heat_kernel = (1/sqrt(2*pi*t))*exp(-(x-y)**2/(2*t))
der_sin = Derivative.diff(sin(y), y)
der_cos = Derivative.diff(cos(y), y)
der_lin = Derivative.diff(y, y)
der_heat_kernel = Derivative.diff(heat_kernel, x)

convolution_hk_sin = Integral.integrate(heat_kernel*der_sin, (y, -oo, oo))
convolution_hk_cos = Integral.integrate(heat_kernel*der_cos, (y, -oo, oo))
convolution_hk_lin = Integral.integrate(heat_kernel*der_lin, (y, -oo, oo))

conv1 = Integral.integrate(heat_kernel*der_sin, (y, -oo, oo))

t_hk = 4/5
convolution_for_t_hk_sin = convolution_hk_sin.subs(t, t_hk)
convolution_for_t_hk_cos = convolution_hk_cos.subs(t, t_hk)
convolution_for_t_hk_lin = convolution_hk_lin.subs(t, t_hk)

plot(convolution_for_t_hk_sin)
plot(convolution_for_t_hk_cos)
plot(convolution_for_t_hk_lin)
