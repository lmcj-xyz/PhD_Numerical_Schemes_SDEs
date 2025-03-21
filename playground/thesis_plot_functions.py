import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import floor, log, pi, exp

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dsdes as ds

# Graphical parameters
params = {'axes.labelsize': 8,
          'font.size': 11,
          'legend.fontsize': 11,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10, 'text.usetex': True,
          'figure.figsize': [10, 8],
          'figure.constrained_layout.use': True}
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

##################
# Plot functions
##################
def plot_generators(gen1, gen2, grid, save=False):
    """
    Plots the generators of the drift, either Weierstrass or fBm
    """
    fig, ax = plt.subplots()
    ax.plot(grid, gen1,  linewidth='1', color=lgreent)
    ax.plot(grid, gen2,  linewidth='1', color=lred)
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    if save:
        fig.savefig(f'sde_gen_{save}.png')
        fig.savefig(f'sde_gen_{save}.pdf')
        fig.savefig(f'sde_gen_{save}.eps')
    plt.show()

def plot_drift(drift1, drift2, bridge, grid, save=False, name=''):
    """
    Plot drifts (drift1 and drift2) generated with different variances
    and compares them with their generator (bridge)

    Optionally saves the figure if save=True
    and appends an identifier to the file name using 'name'
    """
    fig, ax = plt.subplots()
    ax.plot(grid, drift2,  linewidth='1', color=lgreent, label=r'$b^N$ for $N=$' + str(floor(1/var2)))
    ax.plot(grid, drift1,  linewidth='1', color=lred, label=r'$b^N$ for $N=$' + str(floor(1/var1)))
    ax.plot(grid, bridge,  linewidth='1', color=lgreen, label=r'$B^H_b$')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    if save:
        fig.savefig(f'sde_drift_{name}.png')
        fig.savefig(f'sde_drift_{name}.pdf')
        fig.savefig(f'sde_drift_{name}.eps')
    plt.show()


def plot_mckean_drift(drift1, drift2, law1, law2, nl, grid, save=False, name=''):
    """
    Plots the three row figure for the McKean drift

    Plots:
    - the drifts drift1 and drift2
    - its laws law1 and law2
    - and finally the product

    The nonlinear function nl is applied to the given laws to plot
    both in the second and third row

    Optionally saves the figure if save=True
    and appends an identifier to the file name using 'name'
    """
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
    if save:
        fig.savefig(f'mckean_drift_{name}.png')
        fig.savefig(f'mckean_drift_{name}.pdf')
        fig.savefig(f'mckean_drift_{name}.eps')
    plt.show()


def plot_law(soln, law, grid, save=False, name=''):
    """
    Plots the empirical density of the SDE at terminal time and compares it
    with the obtained by solving the Fokker-Planck PDE

    Optionally saves the figure if save=True
    and appends an identifier to the file name using 'name'
    """
    fig, ax = plt.subplots()
    ax.hist(soln[0, :], bins=100, density=True, color=lred, alpha=0.85, label='Empirical density')
    ax.plot(grid, np.array(law.data)[-1, :], color=lgreen, label='FP PDE solution')
    ax.set_title('SDE densities')
    legend = ax.legend(framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor(lcream)
    frame.set_edgecolor(lcream)
    ax.grid(linestyle='--', linewidth='0.5', color='gray')
    if save:
        fig.savefig(f'sde_law_{name}.png')
        fig.savefig(f'sde_law_{name}.pdf')
        fig.savefig(f'sde_law_{name}.eps')
    plt.show()


def plot_mckean_law(soln1, soln2, grid, t1, t2, save=False, name=''):
    """
    Plots the empirical density of the McKean SDE at terminal time and compares it
    with the obtained by solving the Fokker-Planck PDE

    ¡¡¡¡¡¡¡¡¡This is needed because the solution of a McKean SDE and the solver of a linear
    SDE have different return values!!!!!!!!!!!!!!!

    Optionally saves the figure if save=True
    and appends an identifier to the file name using 'name'
    """
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
    if save:
        fig.savefig(f'mckean_law_{name}.png')
        fig.savefig(f'mckean_law_{name}.pdf')
        fig.savefig(f'mckean_law_{name}.eps')
    plt.show()


def plot_wdrift(drift, weier, grid, save=False, name=''):
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

def plot_integral(save=False, name=''):
    """
    Plots the heat kernel generated integrals that are used in the convolution
    to create the drift

    Optionally saves the figure if save=True
    and appends an identifier to the file name using 'name'
    """
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
    if save:
        fig.savefig(f'integral_grid_{name}.png')
        fig.savefig(f'integral_grid_{name}.pdf')
        fig.savefig(f'integral_grid_{name}.eps')

