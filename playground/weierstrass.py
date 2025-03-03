import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

#rng = np.random.default_rng()
#gaussian = rng.standard_normal(2**12)
#f1 = ds.fbm(gaussian, 0.9, 2**12, 10)
#f2 = ds.fbm(gaussian, 0.8, 2**12, 10)
#f3 = ds.fbm(gaussian, 0.7, 2**12, 10)
#
#x1, w1 = ds.weierstrass(0.9, 10)
#x2, w2 = ds.weierstrass(0.8, 10)
#x3, w3 = ds.weierstrass(0.7, 10)
#grid0 = np.linspace(0, 20, 2**12)
#b1 = ds.bridge(f1, grid0)
#b2 = ds.bridge(f2, grid0)
#b3 = ds.bridge(f3, grid0)

wd1, w1, x1, var1 = ds.wdrift(alpha=0.9, b=12, points=2**12, half_support=10, time_steps= 2**20)
wd2, w2, x2, var2 = ds.wdrift(alpha=0.6, b=12, points=2**12, half_support=10, time_steps= 2**30)

plt.plot(x1, wd1, label='alpha 0.9')
plt.plot(x2, wd2, label='alpha 0.6')
plt.plot(x1, w1, label='alpha 0.9')
plt.plot(x2, w2, label='alpha 0.6')
# plt.plot(x3, w3, label='alpha 0.7')
# plt.plot(x1, b1, label='hurst 0.9')
# plt.plot(x2, b2, label='hurst 0.8')
# plt.plot(x3, b3, label='hurst 0.7')
plt.legend()
plt.show()


# x = np.linspace(-1, 1, 50)
# y1 = np.heaviside(x - 0.2, 0.5)
# y2 = np.heaviside(-x + 0.2, 0.5)
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.show()
