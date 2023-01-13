# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:22:31 2023

@author: LM
"""

import matplotlib.pyplot as plt
from numpy import genfromtxt

data = genfromtxt('rates_comp.csv', delimiter=',', names=True)

plt.plot(data[0,], data[1,])

plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')

plt.show()