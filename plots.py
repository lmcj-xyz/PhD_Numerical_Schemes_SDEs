import matplotlib.pyplot as plt
from dsdes import heat_kernel_var, integral_between_grid_points
import numpy as np
import pandas as pd
import pickle

# Integral between grid points plots
hs = 5
grid = np.linspace(-hs, hs, 10000)

plt.rcParams['figure.dpi'] = 200

ts1 = 2**3
ts2 = 2**5

hk1 = heat_kernel_var(time_steps=ts1, hurst=0.8)
hk2 = heat_kernel_var(time_steps=ts2, hurst=0.8)

i1 = integral_between_grid_points(heat_kernel_var=hk1,
                                  grid_x=grid, half_support=hs)
i2 = integral_between_grid_points(heat_kernel_var=hk2,
                                  grid_x=grid, half_support=hs)

ticks1 = np.arange(-5, 6, 1)
fig, axs = plt.subplots(1, 2, sharey=True)
fig.suptitle(r'$\mathcal{I}(y_{k})$ for different $f_{m}$')
axs[0].set_title(r'$m = %d$, $f_m = %.6f$' % (ts1, hk1))
axs[0].plot(grid, i1)
axs[0].set_xticks(ticks1)
axs[0].grid()
axs[1].set_title(r'$m = %d$, $f_m = %.6f$' % (ts2, hk2))
axs[1].plot(grid, i2)
axs[1].set_xticks(ticks1)
axs[1].grid()
plt.show()
#plt.savefig(fname='integral_grid.pdf', dpi=200, format='pdf')
#plt.clf()


# Computation of lower bound for points of definition
def eta(hurst):
    return 1/((hurst-1/2)**2 + 2 - hurst)


def points_drift(m, M, beta):
    hurst = 1 - beta
    etap = eta(hurst)
    return int(2*M*m**(etap/2))


m_A = 2**12
M_A = 10
beta_A = [0, 1/6, 1/8, 1/4, 1/2]
hurst_A = [1 - b for b in beta_A]
eta_A = [eta(h) for h in hurst_A]
A_A = [points_drift(m_A, M_A, b) for b in beta_A]

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

fig, axs = plt.subplots(1, 1, sharey=True)
fig.suptitle(r'Convergence rate visualised for different values of $\beta$')
axs.plot(beta12['dt'],
         beta12['error'],
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta12['rate'], beta12['beta']))
axs.plot(beta14['dt'],
         beta14['error'],
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta14['rate'], beta14['beta']))
axs.plot(beta18['dt'],
         beta18['error'],
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta18['rate'], beta18['beta']))
axs.plot(beta00['dt'],
         beta00['error'],
         marker='o',
         label=r'rate = %f for $\beta$=%f' % (beta00['rate'], beta00['beta']))
axs.grid(which='both')
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylabel(r'$\log_{10}(\epsilon)$')
axs.set_xlabel(r'$\log_{10}(\Delta t)$')
axs.legend()
plt.show()
