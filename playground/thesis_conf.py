import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import floor
from scipy.integrate import cumulative_trapezoid
from scipy.stats import ks_1samp

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
   'figure.figsize': [10, 4],
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

hurst = 0.75
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
bn2, ibn2, bH2, bB2, x2, var2 = ds.drift(gaussian1, hurst, points, half_support, 2**9)
print('N_2 = ' + str(1/var2))
bn3, ibn3, bH3, bB3, x3, var3 = ds.drift(gaussian1, hurst, points, half_support, 2**7)
print('N_3 = ' + str(1/var3))

# Law
bm1 = rng.normal(loc=0.0, scale=np.sqrt(dt),
                 size=(time_steps, sample_paths))

#soln1 = ds.solve(y0, bn1, bm1, time_start, time_end, time_steps, sample_paths, x1)
#law1 = ds.solve_fp(bn1, x1, half_support, lambda x: x**0, time_start, time_end, points, time_steps)


def sinusoid(A, omega, phi, x):
    return A*np.sin(omega*x + phi)


def nonlinear1(x):
    return sinusoid(1, 1, 0, x)


def nonlinear2(x):
    return sinusoid(5, 1, 0, x)

def nonlinear3(x):
    return 1/(1 + np.exp(-100*(x - 0.2)))

def nonlinear4(x):
    return 1/(1 + np.exp(100*(x - 0.2)))

def nonlinear5(x):
    return np.cos(x)


#mvlaw1 = ds.solve_fp(bn1, x1, half_support, nonlinear1, time_start, time_end, points, time_steps) # sin
#mvlaw2 = ds.solve_fp(bn1, x1, half_support, nonlinear2, time_start, time_end, points, time_steps) # 5sin
mvlaw3_1 = ds.solve_fp(bn1, x1, half_support, nonlinear5, time_start, time_end, points, time_steps) # cos
mvlaw3_2 = ds.solve_fp(bn1, x1, half_support, nonlinear5, time_start, time_end, points, 2**9) # cos
mvlaw3_3 = ds.solve_fp(bn1, x1, half_support, nonlinear5, time_start, time_end, points, 2**7) # cos
#mvlaw4 = ds.solve_fp(bn1, x1, half_support, nonlinear4, time_start, time_end, points, time_steps) # exp
#mvlaw5 = ds.solve_fp(bn1, x1, half_support, nonlinear3, time_start, time_end, points, time_steps) # negexp
#mvlaw5_2 = ds.solve_fp(bn1, x1, half_support, nonlinear3, time_start, time_end, points, time_steps*2**3) # negexp

#for conference
#mvlaw2 = ds.solve_fp(bn2, x2, half_support, nonlinear1, time_start, time_end, points, time_steps)
#mvlaw3 = ds.solve_fp(bn2, x2, half_support, nonlinear2, time_start, time_end, points, time_steps)

#mvlaw4 = ds.solve_fp(bn2, x2, half_support, nonlinear5, time_start, time_end, points, time_steps)

#mvsoln1 = ds.solve_mv(y0, bn1, bm1, mvlaw1, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear1)
#mvsoln2 = ds.solve_mv(y0, bn1, bm1, mvlaw2, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear2)
mvsoln3_1 = ds.solve_mv(y0, bn1, bm1, mvlaw3_1, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear5)
mvsoln3_2 = ds.solve_mv(y0, bn1, bm1, mvlaw3_2, time_start, time_end, 2**9, sample_paths, x1, half_support, points, 2**9, nonlinear5)
mvsoln3_3 = ds.solve_mv(y0, bn1, bm1, mvlaw3_3, time_start, time_end, 2**7, sample_paths, x1, half_support, points, 2**7, nonlinear5)
#mvsoln4 = ds.solve_mv(y0, bn1, bm1, mvlaw4, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear4)
#mvsoln5 = ds.solve_mv(y0, bn1, bm1, mvlaw5, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear3)
#mvsoln5_2 = ds.solve_mv(y0, bn1, bm1, mvlaw5_2, time_start, time_end, time_steps*2**3, sample_paths, x1, half_support, points, time_steps*2**3, nonlinear3)

#for conference
#mvsoln3 = ds.solve_mv(y0, bn1, bm1, mvlaw4, time_start, time_end, time_steps, sample_paths, x1, half_support, points, time_steps, nonlinear5)


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

def plot_mckean_law_single(soln1, grid, file):
    fig, ax = plt.subplots(1, 1)
    #axs[0].set_title(title1)
    ax.hist(soln1[0][0, :], bins=100, density=True, color=lred, alpha=0.85, label='Empirical density')
    ax.plot(grid, np.array(soln1[1].data)[-1, :], color=lgreen, label='FP PDE solution')
    #fig.suptitle('MVSDE densities', fontsize=14)
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    fig.savefig(f'mckean_law_{file}.png')
    fig.savefig(f'mckean_law_{file}.pdf')
    fig.savefig(f'mckean_law_{file}.eps')
    plt.show()

def ks(solution, x):
    data = solution[0][0, :]
    pdf = solution[1][-1, :]
    cdf  = cumulative_trapezoid(pdf, x, initial=0)
    return ks_1samp(data, lambda vals: np.interp(vals, x1, cdf))

if __name__ == "__main__":
    #plot_drift(bn1, bn2, bB1, x1)
    #plot_mckean_drift(bn1, bn2, mvlaw5, mvlaw5_2, nonlinear3, x1)
    #plot_law(soln1, law1, x1)

    #plot_mckean_law(mvsoln1, mvsoln3, x1, r'$F(x) = sin(x)$', r'$F(x) = \cos(x)$')
    #plot_mckean_law(mvsoln1, mvsoln3, x1, r'$F(x) = sin(x)$', r'$F(x) = 1/(1 + \exp(100*(x - 0.2)))$')
    #plot_mckean_law(mvsoln1, mvsoln3, x1, r'$F(x) = sin(x)$', r'$F(x) = 1/(1 + \exp(-100*(x - 0.2)))$')
    #plot_mckean_law(mvsoln1, mvsoln3, x1, r'$F(x) = sin(x)$', r'$F(x) = \sin(5x)$')

    #plot_mckean_law_single(mvsoln1, x1, 'h_099_sin')
    #plot_mckean_law_single(mvsoln2, x1, 'h_099_5sin')
    #plot_mckean_law_single(mvsoln3, x1, 'h_099_cos')
    #plot_mckean_law_single(mvsoln4, x1, 'h_099_exp')
    #plot_mckean_law_single(mvsoln5, x1, 'h_099_negexp')

    plot_mckean_law_single(mvsoln3_1, x1, 'h_075_cos_ts_211')
    print(ks(mvsoln3_1, x1))
    plot_mckean_law_single(mvsoln3_2, x1, 'h_075_cos_ts_209')
    print(ks(mvsoln3_2, x1))
    plot_mckean_law_single(mvsoln3_3, x1, 'h_075_cos_ts_207')
    print(ks(mvsoln3_3, x1))
