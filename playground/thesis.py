import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import floor, log, pi, exp

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

# Graphical parameters
params = {
   'axes.labelsize': 8,
   'font.size': 11,
   'legend.fontsize': 11,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': True,
   'figure.figsize': [10, 8],
   'figure.constrained_layout.use': True
   }
rcParams.update(params)

# Graph colors
lgreen = '#00502f'
lgreent = '#00502f88'
lred = '#910000'
lbrightred = '#c70000'
lcream = '#f6eee5'
lcoral = '#ff4a36'
lcoralt = '#ff4a3688'
lblack = '#212121'

# Scheme parameters
#theseed = 1392917848  # better rates
theseed = 4392327879
#theseed = 1334917848 # the one Jan and Elena didn't like
rng = np.random.default_rng(seed=theseed)

# Parameters to control
hurst = 0.6
time_steps = 2**11
extra_steps = 2**4
dt = 1/time_steps

points = 10**4
half_support = 10

sample_paths = 1 * 10**4
y0 = rng.normal(size=sample_paths)
time_start = 0
time_end = 1

## Nonlinear functions
def nonlinear1(x):
    return np.sin(x)

def nonlinear2(x):
    return np.cos(x)

def nonlinear3(x):
    return np.sin(10*x)

def nonlinear4(x):
    return np.cos(10*x)

def nonlinear5(x):
    return 1/(1 + np.exp(-100*x))

def nonlinear6(x):
    return 1/(1 + np.exp(100*x))

## Brownian motion driver
bm = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(time_steps, sample_paths))

## Gaussian for fBm and random drift
gaussian = rng.standard_normal(points)

# drifts and generators
rand_drift1, ibn1, bH1, brownian_bridge1, x1, var1 = ds.drift(gaussian, hurst, points, half_support, time_steps)
rand_drift2, ibn2, bH2, brownian_bridge2, x2, var2 = ds.drift(gaussian, hurst, points, half_support, time_steps*extra_steps)
#weier_drift1, weier1, x3, var3 = ds.wdrift(alpha=hurst, points=points, half_support=half_support, time_steps=time_steps)
#weier_drift2, weier2, x4, var4 = ds.wdrift(alpha=hurst, points=points, half_support=half_support, time_steps=time_steps*extra_steps)

# McKean
## Law and solution for equivalent of linear SDE?
#soln1 = ds.solve(y0, bn1, bm1, time_start, time_end, time_steps, sample_paths, x1)
#law1 = ds.solve_fp(bn1, x1, half_support, lambda x: x**0, time_start, time_end, points, time_steps)

##################
# Nonlinear function change
##################
## Laws McKean
print('Obtaining MV law for random drift')
rand_mvlaw1 = ds.solve_fp(rand_drift1, x1, half_support, nonlinear1, time_start, time_end, points, time_steps)
print('Obtaining MV law for random drift')
rand_mvlaw2 = ds.solve_fp(rand_drift2, x2, half_support, nonlinear1, time_start, time_end, points, time_steps*extra_steps)
# print('Obtaining MV law for deterministic drift')
# weier_mvlaw1 = ds.solve_fp(weier_drift1, x3, half_support, nonlinear1, time_start, time_end, points, time_steps)
# print('Obtaining MV law for deterministic drift')
# weier_mvlaw2 = ds.solve_fp(weier_drift2, x4, half_support, nonlinear1, time_start, time_end, points, time_steps*extra_steps)
#
# ## Solutions McKean
print('Obtaining MV solution random drift')
rand_mvsoln1 = ds.solve_mv(y0, rand_drift1, bm, rand_mvlaw1, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear1)
print('Obtaining MV solution random drift')
rand_mvsoln2 = ds.solve_mv(y0, rand_drift2, bm, rand_mvlaw2, time_start, time_end, time_steps*extra_steps, sample_paths, x2, half_support, points, time_steps, nonlinear1)
# print('Obtaining MV solution deterministic drift')
# weier_mvsoln1 = ds.solve_mv(y0, weier_drift2, bm, weier_mvlaw2, time_start, time_end, time_steps, sample_paths, x2, half_support, points, time_steps, nonlinear1)
# print('Obtaining MV solution deterministic drift')
# weier_mvsoln2 = ds.solve_mv(y0, weier_drift2, bm, weier_mvlaw2, time_start, time_end, time_steps*extra_steps, sample_paths, x2, half_support, points, time_steps, nonlinear1)

##################
# Plot functions
##################
def plot_generators(gen1, gen2, grid):
    fig, ax = plt.subplots()
    ax.plot(grid, gen1,  linewidth='1', color=lgreent)
    ax.plot(grid, gen2,  linewidth='1', color=lred)
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    #fig.savefig('sde_drift.png')
    #fig.savefig('sde_drift.pdf')
    #fig.savefig('sde_drift.eps')
    plt.show()

def plot_drift(drift1, drift2, bridge, grid):
    fig, ax = plt.subplots()
    ax.plot(grid, drift2,  linewidth='1', color=lgreent, label=r'$b^N$ for $N=$' + str(floor(1/var2)))
    ax.plot(grid, drift1,  linewidth='1', color=lred, label=r'$b^N$ for $N=$' + str(floor(1/var1)))
    ax.plot(grid, bridge,  linewidth='1', color=lgreen, label=r'$B^H_b$')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    #fig.savefig('sde_drift.png')
    #fig.savefig('sde_drift.pdf')
    #fig.savefig('sde_drift.eps')
    plt.show()


def plot_mckean_drift(drift1, drift2, law1, law2, nl, grid):
    fig, axs = plt.subplots(3, 1)
    axs[0].set_title(r'Function $b^N$')
    axs[0].plot(grid, drift2,  linewidth='1', label=r'$b^N$ for $N=$' + str(floor(1/var2)), color=lcoral, alpha=0.7)
    axs[0].plot(grid, drift1,  linewidth='1', label=r'$b^N$ for $N=$' + str(floor(1/var1)), color=lgreen)
    axs[1].set_title(r'Law density as obtained by solving FPE for $b^N$')
    axs[1].plot(grid, nl(law2.data[-1]),  linewidth='1', label=r'$F(\rho^N)$ for $N=$' + str(floor(1/var2)), color=lcoral, alpha=0.7)
    axs[1].plot(grid, nl(law1.data[-1]),  linewidth='1', label=r'$F(\rho^N)$ for $N=$' + str(floor(1/var1)), color=lgreen)
    axs[2].set_title(r'McKean equation drift')
    axs[2].plot(grid, nl(law2.data[-1])*drift2,  linewidth='1', label=r'$F(\rho^N)b^N$ for $N=$' + str(floor(1/var2)), color=lcoral, alpha=0.7)
    axs[2].plot(grid, nl(law1.data[-1])*drift1,  linewidth='1', label=r'$F(\rho^N)b^N$ for $N=$' + str(floor(1/var1)), color=lgreen)
    for ax in axs:
        legend = ax.legend(framealpha=1, loc='upper right')
        frame = legend.get_frame()
        frame.set_facecolor(lcream)
        frame.set_edgecolor(lcream)
        ax.grid(linestyle='--', linewidth='0.5', color='gray')
    fig.suptitle(r'Action of $F(v^N)$ on $b^N$', fontsize=14)
    #fig.savefig('mckean_drift.png')
    #fig.savefig('mckean_drift.pdf')
    #fig.savefig('mckean_drift.eps')
    plt.show()


def plot_law(soln, law, grid):
    fig, ax = plt.subplots()
    ax.hist(soln[0, :], bins=100, density=True, color=lred, alpha=0.85, label='Empirical density')
    ax.plot(grid, np.array(law.data)[-1, :], color=lgreen, label='FP PDE solution')
    ax.set_title('SDE densities')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    #fig.savefig('sde_law.png')
    #fig.savefig('sde_law.pdf')
    #fig.savefig('sde_law.eps')
    plt.show()


def plot_mckean_law(soln1, soln2, grid, t1, t2):
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(t1)
    axs[0].hist(soln1[0][0, :], bins=100, density=True, color=lred, alpha=0.85, label='Empirical density')
    axs[0].plot(grid, np.array(soln1[1].data)[-1, :], color=lgreen, label='FP PDE solution')
    axs[1].set_title(t2)
    axs[1].hist(soln2[0][0, :], bins=100, density=True, color=lred, alpha=0.85, label='Empirical density')
    axs[1].plot(grid, np.array(soln2[1].data)[-1, :], color=lgreen, label='FP PDE solution')
    fig.suptitle('MVSDE densities', fontsize=14)
    for ax in axs:
        legend = ax.legend(framealpha=1)
        frame = legend.get_frame()
        frame.set_facecolor(lcream)
        frame.set_edgecolor(lcream)
        ax.grid(linestyle='--', linewidth='0.5', color='gray')
    #fig.savefig('mckean_law.png')
    #fig.savefig('mckean_law.pdf')
    #fig.savefig('mckean_law.eps')
    plt.show()


def plot_wdrift(drift, weier, grid):
    fig, ax = plt.subplots()
    ax.plot(grid, drift,  linewidth='1', color=lred, label=r'$b^N$ for $N=$' + str(floor(1/var3)))
    ax.plot(grid, weier,  linewidth='1', color=lgreen, label=r'$W_\alpha$')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    #fig.savefig('sde_drift.png')
    #fig.savefig('sde_drift.pdf')
    #fig.savefig('sde_drift.eps')
    plt.show()

def plot_integral():
    hs = 5
    grid = np.linspace(-hs, hs, 10000)
    
    #plt.rcParams['figure.dpi'] = 200
    
    ts1 = 2**3
    ts2 = 2**5
    
    hk1 = ds.heat_kernel_var(time_steps=ts1, hurst=0.8)
    hk2 = ds.heat_kernel_var(time_steps=ts2, hurst=0.8)
    
    i1 = ds.integral_between_grid_points(heat_kernel_var=hk1,
                                         grid_x=grid, half_support=hs)
    i2 = ds.integral_between_grid_points(heat_kernel_var=hk2,
                                         grid_x=grid, half_support=hs)
    
    ticks1 = np.arange(-5, 6, 1)
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.suptitle(r'$\mathcal{I}^{N}(y)$ for different $1/N$')
    axs[0].set_title(r'$1/N = %.6f$' % (hk1))
    axs[0].plot(grid, i1, color=lred)
    axs[0].set_xticks(ticks1)
    axs[0].grid()
    axs[1].set_title(r'$1/N = %.6f$' % (hk2))
    axs[1].plot(grid, i2, color=lred)
    axs[1].set_xticks(ticks1)
    axs[1].grid()
    plt.show()
    #fig.savefig('integral_grid.png')
    #fig.savefig('integral_grid.pdf')
    #fig.savefig('integral_grid.eps')

if __name__ == "__main__":
    #plot_drift(rand_drift1, rand_drift2, brownian_bridge1, x1)
    #plot_drift(weier_drift1, weier_drift2, weier1, x1)
    plot_mckean_drift(rand_drift1, rand_drift2, rand_mvlaw1, rand_mvlaw2, nonlinear1, x1)
    #plot_law(soln1, law1, x1)
    #plot_mckean_law(mvsoln5, mvsoln6, x1, r'$F(x) = sin(x)$', r'$F(x) = sin(10x)$')
    #plot_wdrift(bn3, w1, x3)
    #plot_generators(bB1, w1, x3)
    #plot_integral()
