import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
import pickle
import time

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

lgreen = (0, 0.31, 0.18)
lred = (0.77, 0.07, 0.19)
lwhite = (0.96, 0.95, 0.89)

# Code
#theseed = 1392917848  # better rates
theseed = 1334917848
rng = np.random.default_rng(seed=theseed)

# Parameters for Euler scheme
keys = ('real', 'approx1', 'approx2', 'approx3', 'approx4', 'approx5')

time_steps_tuple = (2**13, 2**6, 2**7, 2**8, 2**9, 2**10)
time_steps = dict(zip(keys, time_steps_tuple))

error_keys = ('e1', 'e2', 'e3', 'e4', 'e5')

epsilon = 10e-6
beta = epsilon
hurst = 1 - beta
sample_paths = 10**4
y0 = rng.normal(size=sample_paths)
time_start = 0
time_end = 1
# Parameters to create fBm
points_x = 2**10  # According to the lower bound in the paper
half_support = 10

eta = 1/((hurst-1/2)**2 + 2 - hurst)
lower_bound = 2*half_support*time_steps['real']**(eta/2)
eqn = 66
if (points_x <= lower_bound):
    msg = 'You need to define your fBm on at least %.2f \
            points as per equation (%d) in the paper.' % (lower_bound, eqn)
    raise ValueError(msg)
    #sys.exit(1)

delta_x = half_support/(points_x-1)
grid_x = np.linspace(start=-half_support, stop=half_support, num=points_x)
#grid_x0 = np.linspace(start=0, stop=2*half_support, num=points_x)
#gaussian_fbm = rng.standard_normal(size=points_x)

# todo: check how to use the beta/alpha properly
drift_tuple = tuple(map(lambda t: ds.wdrift(hurst, 12, points_x, half_support, t)[0],
                        time_steps.values()))
drift_array = dict(zip(keys, drift_tuple))
plt.plot(drift_array['real'])
plt.show()

dt_tuple = tuple(
        map(lambda t: (time_end - time_start)/(t - 1), time_steps.values())
        )
dt = dict(zip(keys, dt_tuple))
time_grid_tuple = tuple(map(
    lambda dt, t: np.linspace(time_start + dt, time_end, t),
    dt.values(), time_steps.values()
    ))
time_grid = dict(zip(keys, time_grid_tuple))

loopint = 40
for i in range(loopint):
    # loop here?
    noise = rng.normal(loc=0.0, scale=np.sqrt(dt['real']),
                       size=(time_steps['real'], sample_paths))

    solution_tuple = tuple(map(
        lambda d, t: ds.solve(
            1, d, noise, time_start, time_end, t, sample_paths, grid_x
            ),
        drift_array.values(),
        time_steps.values(),
        ))
    solution = dict(zip(keys, solution_tuple))

    strong_error = dict.fromkeys(error_keys)
    strong_error['e1'] = np.abs(solution['real'] - solution['approx1'])
    strong_error['e2'] = np.abs(solution['real'] - solution['approx2'])
    strong_error['e3'] = np.abs(solution['real'] - solution['approx3'])
    strong_error['e4'] = np.abs(solution['real'] - solution['approx4'])
    strong_error['e5'] = np.abs(solution['real'] - solution['approx5'])

    plot_error = [np.mean(value) for key, value in strong_error.items()]
    plot_dt = [value for key, value in dt.items() if key not in 'real']

    log_strong_error = np.log10(plot_error)
    log_dt = np.log10(plot_dt)

    reg_strong = linregress(log_dt, log_strong_error)
    rate_strong = reg_strong.slope
    intersection_strong = reg_strong.intercept

    plot_dict = {
            'beta': beta,
            'rate': rate_strong,
            'dt': plot_dt,
            'error': plot_error,
            }

    plot = False
    if plot:
        fig, ax = plt.subplots()
        ax.set_title(
                r'Rate of convergence r = %f for $\beta$=%f' % (rate_strong, beta)
                )
        ax.plot(plot_dt,
                plot_error,
                marker='o',
                label='Strong error',
                color=lgreen)
        ax.grid(which='both', linestyle='--')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\log_{10}(\Delta t)$')
        ax.set_ylabel(r'$\log_{10}(\epsilon)$')
        ax.legend()
        plt.show()

    #saving = input('Do you want to save to files the plot and its corresponding dictionary? (yes/no): ')
    saving = False
    if saving:
        date_string = time.strftime("%Y-%m-%d-%H-%M")
        fig.savefig(date_string + 'rate.pdf', dpi=200)
        with open(date_string + 'dict_plot.pkl', 'wb') as fp:
            pickle.dump(plot_dict, fp)
            print('Files saved succesfully')

    if loopint == 40:
        print('rate = ', rate_strong)
        file = open("rates_linear.txt", "a")
        file.write(str(rate_strong))
        file.write("\n")
        file.close()
