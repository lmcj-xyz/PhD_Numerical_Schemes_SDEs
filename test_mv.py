import numpy as np
import dsdes as ds
from numpy.random import default_rng
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

rng = default_rng()
time_steps = 2**8
grid = np.linspace(-10, 10, 10**3)
linear = grid
time_start = 0
time_end = 1
dt = (time_end - time_start)/time_steps
sample_paths = 10**2
noise = rng.normal(loc=0.0, scale=np.sqrt(dt),
                   size=(time_steps, sample_paths))
y0 = 1
soln, law = ds.mv_solve(y0=y0,
                        drift_array=linear,
                        z=noise,
                        time_start=time_start,
                        time_end=time_end,
                        time_steps=time_steps,
                        sample_paths=sample_paths,
                        grid=grid)

det_y = norm.cdf(1)*np.exp(np.linspace(time_start, time_end, time_steps))
sqrt_t = np.sqrt(np.linspace(time_start, time_end, time_steps))
fig, ax = plt.subplots()
ax.plot(soln[:, 0:3])
ax.legend()
plt.show()

sg = np.linspace(0, 2.5, 1000)
fig, ax = plt.subplots()
for i in range(50):
    ax.plot(sg, law[i].evaluate(sg), label='ap')
fig.legend()
plt.show()


# 3d plot
def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


ax = plt.figure().add_subplot(projection='3d')

x = np.linspace(0., 2.5, 1000)
t = range(20)
verts = [polygon_under_graph(x, law[ti].evaluate(x)) for ti in t]
facecolors = plt.colormaps['viridis_r'](np.linspace(0., 1, len(verts))) # I think linspace here is not the best

poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
ax.add_collection3d(poly, zs=t, zdir='y')

ax.set(xlim=(0, 2.5), ylim=(0, 20), zlim=(0, 20),
       xlabel='x', ylabel='t', zlabel='prob')

plt.show()
