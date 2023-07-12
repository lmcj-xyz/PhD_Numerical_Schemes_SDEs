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
sample_paths = 10**1
noise = rng.normal(loc=0.0, scale=np.sqrt(dt),
                   size=(time_steps, sample_paths))
iloc = 1
y0 = rng.normal(loc=iloc, scale=np.sqrt(0.001),
                size=sample_paths)
# soln = ds.mv_solve(y0=1,
#                    drift_array=linear,
#                    z=noise,
#                    time_start=time_start,
#                    time_end=time_end,
#                    time_steps=time_steps,
#                    sample_paths=sample_paths,
#                    grid=grid)

# def mv_solve(y0: float,
#              drift_array: np.ndarray,
#              z: np.ndarray,
#              time_start: float, time_end: float, time_steps: int,
#              sample_paths: int,
#              grid: np.ndarray,) -> np.ndarray:
y = np.zeros(shape=(time_steps+1, sample_paths))
z_coarse = ds.coarse_noise(noise, time_steps, sample_paths)
dt = (time_end - time_start)/(time_steps-1)
nu0 = y0
y[0, :] = y0*nu0
for i in range(time_steps):
    kde = gaussian_kde(dataset=y[i, :], bw_method='scott')
    y[i+1, :] = y[i, :] + \
        kde.evaluate(y[i, :]) * \
        np.interp(x=y[i, :], xp=grid, fp=linear)*dt + \
        z_coarse[i, :]

det_y = norm.cdf(1)*np.exp(np.linspace(time_start, time_end, time_steps))
sqrt_t = np.sqrt(np.linspace(time_start, time_end, time_steps))
fig, ax = plt.subplots()
ax.plot(y[:, 0:3])
#ax.plot(det_y[0:10], label='y ODE')
#ax.plot(det_y[0:10] - sqrt_t[0:10], linestyle='dotted')
#ax.plot(det_y[0:10] + sqrt_t[0:10], linestyle='dotted')
ax.legend()
plt.show()

sg = np.linspace(0, 2.5, 1000)
fig, ax = plt.subplots()
ax.plot(sg, gaussian_kde(y[0, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[1, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[2, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[3, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[4, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[5, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[6, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[7, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[8, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[9, :]).evaluate(sg), label='ap')
ax.plot(sg, gaussian_kde(y[10, :]).evaluate(sg), label='ap')
fig.legend()
plt.show()
