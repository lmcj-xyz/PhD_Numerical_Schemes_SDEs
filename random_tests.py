# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:44:16 2023

@author: mmlmcj
"""


import numpy as np
import math
import time

x = 4

st1 = time.process_time()
np.sqrt(x)
et1 = time.process_time()
print("numpy time", et1-st1)

st2 = time.process_time()
math.sqrt(x)
et2 = time.process_time()
print("math time", et1-st1)