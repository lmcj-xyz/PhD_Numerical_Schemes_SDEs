import numpy as np
import dsdes as ds
from numpy.random import default_rng
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm

rng = default_rng()
time_steps = 2**10
grid = np.linspace(-10, 10, 10**3)
linear = grid
fbm = ds.fbm(hurst=0.8, points=10**3, half_support=10)
time_start = 0
time_end = 1
dt = (time_end - time_start)/time_steps
sample_paths = 10**4
noise = rng.normal(loc=0.0, scale=np.sqrt(dt),
                   size=(time_steps, sample_paths))
y0 = 1
soln, law = ds.mv_solve(y0=y0,
                        drift_array=fbm,
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

sg = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
for i in range(50):
    ax.plot(sg, law[i].evaluate(sg), label='ap')
fig.legend()
plt.show()


# 3d plot
def polygon_under_graph(x, y):
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


ax = plt.figure().add_subplot(projection='3d')

x = np.linspace(0., 10, 1000)
t = range(20)
verts = [polygon_under_graph(x, law[ti].evaluate(x)) for ti in t]

# I think linspace here is not the best
facecolors = plt.colormaps['viridis_r'](np.linspace(0., 1, len(verts)))

poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
ax.add_collection3d(poly, zs=t, zdir='y')

ax.set(xlim=(0, 10), ylim=(0, 20), zlim=(0, 20),
       xlabel='x', ylabel='t', zlabel='prob')

plt.show()

#plt.style.use('_mpl-gallery')
#
## Make data
#X = np.linspace(0, 5, 1000)
#T = range(20)
#Z = [law[ti].evaluate(X) for ti in T]
#X, T = np.meshgrid(X, T)
#
## Plot the surface
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#ax.plot_surface(X, T, Z, vmin=None, cmap=cm.Blues)
#
#ax.set(xticklabels=[],
#       yticklabels=[],
#       zticklabels=[])
#
#plt.show()
