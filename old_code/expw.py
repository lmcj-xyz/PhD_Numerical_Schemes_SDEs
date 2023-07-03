from numpy import piecewise, linspace
import matplotlib.pyplot as plt

# Ilegal
conditions = [x > 3, x <= 3]
functions = [1, 0]

x = linspace(0, 5, 100)
f = piecewise(x, conditions, functions)

plt.figure()
plt.plot(f)
plt.show()

# Legal
plt.figure()
plt.plot(piecewise(x, conditions, functions))
plt.show()
