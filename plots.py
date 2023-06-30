import matplotlib.pyplot as plt
from dsdes import heat_kernel_var, integral_between_grid_points
import numpy as np
import pandas

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
df = pandas.read_csv('rates.csv')
dfT = df.T
