import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 1, 0])
y = np.arange(5)

con = np.convolve(x, y, 'same')

plt.figure()
plt.plot(con)
plt.show()
