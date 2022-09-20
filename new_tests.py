import numpy as np

x = np.linspace(-1, 1, 10)
f = lambda w: x - w

print(f(1))
print(f(0))
print(f(-1))
