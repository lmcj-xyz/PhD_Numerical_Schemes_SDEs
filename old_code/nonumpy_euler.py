# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:24:21 2023

@author: mmlmcj
"""
import random
from math import *

def polar_marsaglia():
    w = 0
    while((w >= 1) or (w <= 0)):
        v1 = 2*random.uniform(0, 1)
        v2 = 2*random.uniform(0, 1)
        w = v1**2 + v2**2
    sq = sqrt(-2*log(w)/w)
    n1 = v1*sq
    n2 = v2*sq
    return n1, n2

t1 = 1
t0 = 0
time_steps = 2**3
time_stepsy = time_steps/2
delta = (t1 - t0)/time_steps
deltay = 2*delta

deltas = 4
paths = 10

for i in range(deltas):
    deltay = deltay/2
    sqdelta = sqrt(deltay)
    time_stepsy = 2*time_stepsy
    for j in range(paths):
        for k in range(time_stepsy):
            
        



