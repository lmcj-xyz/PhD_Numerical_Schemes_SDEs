import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

rng = np.random.default_rng(seed=8392917)

time_steps = 2**12
dt = 1/time_steps
sample_paths = 10**4
bm1 = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(time_steps, sample_paths))
bm2 = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(time_steps, sample_paths))
bm3 = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(time_steps, sample_paths))

hurst = 0.9
points = 10**3
half_support = 10
gaussian = rng.standard_normal(points)
bn, bH, bB, x = ds.drift(gaussian, hurst, points, half_support, time_steps)

plt.plot(x, bn)
plt.plot(x, bH)
plt.plot(x, bB)
plt.axhline(y=0, color='k')
plt.show()
