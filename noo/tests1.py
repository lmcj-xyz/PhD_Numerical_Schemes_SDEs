import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()

t1 = np.linspace(0, 1, 2**5)
t2 = np.linspace(0, 1, 2**3)

x = rng.normal(size=2**5, loc=1, scale=1/2**5)
y = np.sum(x.reshape(2**3, 2**2), axis=1)*1/2**2

plt.figure()
plt.plot(t1, x)
plt.plot(t2, y)
plt.grid()
plt.show()
