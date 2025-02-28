import numpy as np
import matplotlib.pyplot as plt

def weierstrass(alpha,
                b,
                terms: int = 50,
                points: int = 2**12,
                half_support: float = 10) -> tuple[np.ndarray, np.ndarray]:
    #assert a > 0 and a < 1
    #assert a*b > 1 + 3*np.pi/2
    grid = np.linspace(start=-half_support, stop=half_support, num=points)
    w = np.zeros(points)
    for k in range(terms):
        w += b**(-k*alpha) * np.cos(b**k * np.pi * grid)
    return grid, w

#x, w = weierstrass(0.9, 10)
#plt.plot(x, w)
#plt.show()

x = np.linspace(-1, 1, 50)
y1 = np.heaviside(x - 0.2, 0.5)
y2 = np.heaviside(-x + 0.2, 0.5)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()
