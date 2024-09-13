import matplotlib.pyplot as plt
import numpy as np

linear_rates = np.genfromtxt('rates_linear.csv', delimiter=',')#, names=True)
linear_drift = np.genfromtxt('rates_linear_drift.csv', delimiter=',')#, names=True)
mckean_drift = np.genfromtxt('rates_mckean_drift.csv', delimiter=',')#, names=True)
mckean_rates = np.genfromtxt('rates_mckean.csv', delimiter=',')#, names=True)

epsilon = 10e-6
pos = [epsilon, 1/8, 3/8, 1/2 - epsilon]

linear_data = [linear_rates[:, i] for i in range(4)]
linear_ci = 1.96 * np.std(linear_data, axis=1) / np.sqrt(np.shape(linear_data)[1])
linear_ddata = [linear_drift[:, i] for i in range(4)]
linear_cci = 1.96 * np.std(linear_ddata, axis=1) / np.sqrt(np.shape(linear_data)[1])
mckean_data = [mckean_rates[:, i] for i in range(4)]
mckean_ci = 1.96 * np.std(mckean_data, axis=1) / np.sqrt(np.shape(linear_data)[1])
mckean_ddata = [mckean_drift[:, i] for i in range(4)]
mckean_cci = 1.96 * np.std(mckean_ddata, axis=1) / np.sqrt(np.shape(linear_data)[1])

ax1 = plt.subplot(221)
ax1.violinplot(linear_data, positions=pos,
               points=40, widths=0.1,
               showmeans=True, quantiles=[[0.25, 0.75] for i in range(4)])
ax1.set_title('Linear SDE, same drift, 40 batches of BM with 10,000 sample paths')
ax1.set_ylabel('Rate of convergence')

ax2 = plt.subplot(222, sharey=ax1)
ax2.violinplot(linear_ddata, positions=pos,
               points=40, widths=0.1,
               showmeans=True, quantiles=[[0.25, 0.75] for i in range(4)])
ax2.set_title('Linear SDE, 40 drifts, same bathc of BM with 10,000 sample paths')

ax3 = plt.subplot(223, sharey=ax1)
ax3.violinplot(mckean_data, positions=pos,
               points=40, widths=0.1,
               showmeans=True, quantiles=[[0.25, 0.75] for i in range(4)])
ax3.set_title('McKean SDE, same drift, 40 batches of BM with 10,000 sample paths')
ax3.set_ylabel('Rate of convergence')
ax3.set_xlabel(r'$\beta$')

ax4 = plt.subplot(224, sharey=ax1)
ax4.violinplot(mckean_ddata, positions=pos,
               points=40, widths=0.1,
               showmeans=True, quantiles=[[0.25, 0.75] for i in range(4)])
ax4.set_title('Mckean SDE, 40 drifts, same bathc of BM with 10,000 sample paths')
ax4.set_xlabel(r'$\beta$')

ax1.plot(pos, [0.16, 0.1, 0.01, 0])
ax1.grid()
ax2.plot(pos, [0.16, 0.1, 0.01, 0])
ax2.grid()
ax3.plot(pos, [0.16, 0.1, 0.01, 0])
ax3.grid()
ax4.plot(pos, [0.16, 0.1, 0.01, 0])
ax4.grid()
plt.show()

fig, axs = plt.subplots(ncols=2, nrows=2)
for j in range(4):
    axs[0, 0].hist(linear_data[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
    axs[0, 0].legend()
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_title('Linear SDE, same drift, 40 batches of BM with 10,000 sample paths')
for j in range(4):
    axs[0, 1].hist(linear_ddata[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
    axs[0, 1].legend()
    axs[0, 1].set_title('Linear SDE, 40 drifts, same bathc of BM with 10,000 sample paths')
for j in range(4):
    axs[1, 0].hist(mckean_data[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Rate of convergence')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('McKean SDE, same drift, 40 batches of BM with 10,000 sample paths')
for j in range(4):
    axs[1, 1].hist(mckean_ddata[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('Rate of convergence')
    axs[1, 1].set_title('Mckean SDE, 40 drifts, same bathc of BM with 10,000 sample paths')
plt.show()
