import numpy as np
import dsdes as ds
from numpy.random import default_rng
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt

rng = default_rng()
time_steps = 2**8
grid = np.linspace(-10, 10, 10**3)
linear = grid
time_start = 0
time_end = 1
dt = (time_end - time_start)/time_steps
sample_paths = 10**2
noise = rng.normal(
        loc=0.0, scale=np.sqrt(dt),
        size=(time_steps, sample_paths))
soln = ds.mv_solve(y0=1, drift_array=linear, z=noise,
                   time_start=time_start, time_end=time_end,
                   time_steps=time_steps, sample_paths=sample_paths, grid=grid)

det_soln = norm.cdf(1)*np.exp(np.linspace(time_start, time_end, time_steps))
sqrt_t = np.sqrt(np.linspace(time_start, time_end, time_steps))
fig, ax = plt.subplots()
ax.plot(soln[:, 0:3])
ax.plot(det_soln, label='Soln ODE')
ax.plot(det_soln - sqrt_t, linestyle='dotted')
ax.plot(det_soln + sqrt_t, linestyle='dotted')
ax.legend()
plt.show()

sg = np.linspace(0.82, 1, 1000)
plt.plot(sg, gaussian_kde(soln[1, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[2, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[3, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[4, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[5, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[6, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[7, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[8, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[9, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[10, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[11, :]).evaluate(sg))
plt.plot(sg, gaussian_kde(soln[12, :]).evaluate(sg))
plt.show()
