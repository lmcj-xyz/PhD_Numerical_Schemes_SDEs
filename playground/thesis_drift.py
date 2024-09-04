import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

rng = np.random.default_rng(seed=1392917848)

hurst = 0.76
time_steps = 2**12

points = 10**3
half_support = 10
gaussian = rng.standard_normal(points)
bn, bH, bB, x = ds.drift(gaussian, hurst, points, half_support, time_steps)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x, bH, linewidth='1', color='orange', label=r'$B^H$')
ax.plot(x, bB, linewidth='1', color='red', label=r'$B^H_b$')
ax.plot(x, bn, color='blue', label=r'$b^N$')
ax.grid(linestyle=':', color='green')
ax.legend()
plt.show()
