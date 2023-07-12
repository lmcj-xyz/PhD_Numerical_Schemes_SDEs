import matplotlib.pyplot as plt
import dsdes as ds
import numpy as np
import pandas as pd
import pickle
from numpy.random import default_rng

# Integral between grid points plots
hs = 5
grid = np.linspace(-hs, hs, 10000)

plt.rcParams['figure.dpi'] = 200

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
fig.suptitle(r'$\mathcal{I}^{N}(y_{k})$ for different $1/N$')
axs[0].set_title(r'$1/N = %.6f$' % (hk1))
axs[0].plot(grid, i1)
axs[0].set_xticks(ticks1)
axs[0].grid()
axs[1].set_title(r'$1/N = %.6f$' % (hk2))
axs[1].plot(grid, i2)
axs[1].set_xticks(ticks1)
axs[1].grid()
plt.show()
#plt.savefig(fname='integral_grid.pdf', dpi=200, format='pdf')
#plt.clf()


# Computation of lower bound for points of definition
def eta(beta):
    return 1/(1 + beta + (1/2 - beta)**2)


def points_drift(m, L, beta):
    etap = eta(beta)
    return int(2*L*m**(etap/2))


mA = 2**12
lA = 5
betaA = [10e-6, 1/8, 1/6, 1/4, 1/2]
etaA = [eta(b) for b in betaA]
hurstA = [1 - b for b in betaA]
n_1A = [ds.heat_kernel_var(time_steps=mA, hurst=h) for h in hurstA]
aA = [points_drift(mA, lA, b) for b in betaA]

# Plot of different convergence rates
df = pd.read_csv('rates.csv')
plt.figure('rates')
plt.plot(df['Beta'], df['Average'], marker='o',
         label='Average of empirical rate')
plt.fill_between(df['Beta'],
                 df['Average'] - df['CI'],
                 df['Average'] + df['CI'],
                 alpha=0.15,
                 label='Confidence intervals at 95%')
plt.plot(df['Beta'], df['Conjecture'], linestyle='dotted', marker='.',
         label=r'Rate $1/2 - \beta/2$')
plt.plot(df['Beta'], df['Theoretical'], marker='o',
         label='Theoretical rate')
plt.xlabel(r'$\beta$')
plt.ylabel('Rate of convergence')
plt.legend()
plt.grid()
plt.show()

with open('dict_plot-beta_00.pkl', 'rb') as fp00:
    beta00 = pickle.load(fp00)
with open('dict_plot-beta_18.pkl', 'rb') as fp18:
    beta18 = pickle.load(fp18)
with open('dict_plot-beta_14.pkl', 'rb') as fp14:
    beta14 = pickle.load(fp14)
with open('dict_plot-beta_12.pkl', 'rb') as fp12:
    beta12 = pickle.load(fp12)

beta00C = [e * 1000 for e in beta00['error']]
beta18C = [e * 8 for e in beta18['error']]
beta14C = [e * 3.6 for e in beta14['error']]
beta12C = [e for e in beta12['error']]

fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle(r'Empirical convergence rate for different values of $\beta$')
axs.plot(beta12['dt'],
         beta12C,
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta12['rate'], beta12['beta']))
axs.plot(beta14['dt'],
         beta14C,
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta14['rate'], beta14['beta']))
axs.plot(beta18['dt'],
         beta18C,
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta18['rate'], beta18['beta']))
axs.plot(beta00['dt'],
         beta00C,
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta00['rate'], beta00['beta']))
axs.grid(which='both')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylabel(r'$\log_{10}(\epsilon)$')
axs.set_xlabel(r'$\log_{10}(\Delta t)$')
axs.legend()
plt.show()

time_start = 0
time_end = 1
time_steps = 2**8
dt = (time_end - time_start)/time_steps
half_support = 10
points_x = 2**8
hurst = 0.7
delta_x = half_support/(points_x-1)
grid_x = np.linspace(start=-half_support, stop=half_support, num=points_x)
grid_x0 = np.linspace(start=0, stop=2*half_support, num=points_x)
fbm_array = ds.fbm(hurst, points_x, half_support)
bridge_array = ds.bridge(fbm_array, grid_x0)
var = ds.heat_kernel_var(time_steps, hurst)
y0 = 1
integral = ds.integral_between_grid_points(var, grid_x, half_support)
drift_array = ds.create_drift_array(bridge_array, integral)
sample_paths = 3
rng = default_rng()
z = rng.normal(
        loc=0.0, scale=np.sqrt(dt),
        size=(time_steps, sample_paths)
        )
paths = ds.solve_keep_paths(
        y0,
        drift_array,
        z,
        time_start, time_end, time_steps,
        sample_paths,
        grid_x
        )

fig, ax = plt.subplots()
ax.plot(paths[:, 0])
plt.show()
