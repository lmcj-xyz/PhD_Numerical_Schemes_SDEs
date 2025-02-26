import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import floor

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

hurst = 0.99
time_steps = 2**11
dt = 1/time_steps

points = 4 * 10**3
half_support = 10

sample_paths = 1 * 10**4
y0 = rng.normal(size=sample_paths)
time_start = 0
time_end = 1

# Drift
gaussian1 = rng.standard_normal(points)
bn1, ibn1, bH1, bB1, x1, var1 = ds.drift(gaussian1, hurst, points, half_support, time_steps)
print('N_1 = ' + str(1/var1))
bn2, ibn2, bH2, bB2, x2, var2 = ds.drift(gaussian1, hurst, points, half_support, time_steps*2**6)
print('N_2 = ' + str(1/var2))
bn3, w1, x3, var3  = ds.wdrift(a=1/2, b=12, points=points, half_support=half_support, time_steps=2**8)
print("W drift created")

## Law
#bm1 = rng.normal(loc=0.0, scale=np.sqrt(dt),
#                 size=(time_steps, sample_paths))
#
#soln1 = ds.solve(y0, bn1, bm1, time_start, time_end, time_steps, sample_paths, x1)
#law1 = ds.solve_fp(bn1, x1, half_support, lambda x: x**0, time_start, time_end, points, time_steps)
#
#
#def sinusoid(A, omega, phi, x):
#    return A*np.sin(omega*x + phi)
#
#
#def nonlinear1(x):
#    return sinusoid(1, 1, 0, x)
#
#
#def nonlinear2(x):
#    return sinusoid(5, 1, 0, x)
#
#
#mvlaw1 = ds.solve_fp(bn1, x1, half_support, nonlinear1, time_start, time_end, points, time_steps)
#mvlaw2 = ds.solve_fp(bn2, x2, half_support, nonlinear1, time_start, time_end, points, time_steps)
#mvlaw3 = ds.solve_fp(bn2, x2, half_support, nonlinear2, time_start, time_end, points, time_steps)
#
#mvsoln1 = ds.solve_mv(y0, bn1, bm1, mvlaw1, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear1)
#mvsoln3 = ds.solve_mv(y0, bn1, bm1, mvlaw3, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear2)


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
    fig.savefig('sde_drift.png')
    fig.savefig('sde_drift.pdf')
    fig.savefig('sde_drift.eps')
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
    fig.savefig('mckean_drift.png')
    fig.savefig('mckean_drift.pdf')
    fig.savefig('mckean_drift.eps')
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
    fig.savefig('sde_law.png')
    fig.savefig('sde_law.pdf')
    fig.savefig('sde_law.eps')
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
    fig.savefig('mckean_law.png')
    fig.savefig('mckean_law.pdf')
    fig.savefig('mckean_law.eps')
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
    fig.savefig('integral_grid.png')
    fig.savefig('integral_grid.pdf')
    fig.savefig('integral_grid.eps')
    #plt.savefig(fname='integral_grid.pdf', dpi=200, format='pdf')
    #plt.clf()

if __name__ == "__main__":
    #plot_drift(bn1, bn2, bB1, x1)
    #plot_mckean_drift(bn1, bn2, mvlaw1, mvlaw2, nonlinear1, x1)
    #plot_law(soln1, law1, x1)
    #plot_mckean_law(mvsoln1, mvsoln3, x1, r'$F(x) = sin(x)$', r'$F(x) = sin(10x)$')
    #plot_wdrift(bn3, w1, x3)
    plot_integral()
