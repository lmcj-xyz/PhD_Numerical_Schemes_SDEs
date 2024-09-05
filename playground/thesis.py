import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

# Graphical parameters
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)

# Graph colors
lgreen = (0, 0.31, 0.18)
lred = (0.77, 0.07, 0.19)
lwhite = (0.96, 0.95, 0.89)

# Scheme parameters
rng = np.random.default_rng(seed=1392917848)

hurst = 0.76
time_steps = 2**10
dt = 1/time_steps

points = 10**3
half_support = 10

sample_paths = 10**4
y0 = rng.normal(size=sample_paths)
time_start = 0
time_end = 1

# Drift
gaussian1 = rng.standard_normal(points)
bn1, ibn1, bH1, bB1, x1 = ds.drift(gaussian1, hurst, points, half_support, time_steps)

# Law
bm1 = rng.normal(loc=0.0, scale=np.sqrt(dt),
                 size=(time_steps, sample_paths))

soln1 = ds.solve(y0, bn1, bm1, time_start, time_end, time_steps, sample_paths, x1)
law1 = ds.solve_fp(bn1, x1, half_support, lambda x: x**0, time_start, time_end, points, time_steps)
mvsoln1 = ds.solve_mv(y0, bn1, bm1, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, lambda x: np.sin(x))


def plot_drift(drift, bridge, grid):
    fig, ax = plt.subplots()
    ax.plot(grid, bridge,  linewidth='1', color=lgreen, label=r'$B^H_b$')
    ax.plot(grid, drift,  linewidth='1', color=lred, label=r'$b^N$')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lwhite)
    frame.set_edgecolor(lwhite)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    plt.show()


def plot_law(soln, law, grid, title):
    fig, ax = plt.subplots()
    ax.hist(soln[0, :], bins=250, density=True, color=lred, label='Empirical density')
    ax.plot(grid, np.array(law.data)[-1, :], color=lgreen, label='Fokker-Planck solution')
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    plot_drift(bn1, bB1, x1)
    plot_law(soln1, law1, x1, 'SDE densities')
    plot_law(mvsoln1[0], mvsoln1[1], x1, 'MVSDE densities')
