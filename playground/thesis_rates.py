import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Graphical parameters
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [10, 8],
   'figure.constrained_layout.use': True
   }
rcParams.update(params)

# Graph colors
lgreen = '#00502f'
lred = '#910000'
lbrightred = '#c70000'
lcream = '#f6eee5'
lcoral = '#ff4a36'
lblack = '#212121'

linear_rates = np.genfromtxt('files_archive/rates_linear.csv', delimiter=',')#, names=True)
linear_drift = np.genfromtxt('files_archive/rates_linear_drift.csv', delimiter=',')#, names=True)
mckean_drift = np.genfromtxt('files_archive/rates_mckean_drift.csv', delimiter=',')#, names=True)
mckean_rates = np.genfromtxt('files_archive/rates_mckean.csv', delimiter=',')#, names=True)

epsilon = 10e-6
pos = [epsilon, 1/8, 1/4, 3/8, 1/2 - epsilon]

linear_data = [linear_rates[:, i] for i in range(5)]
linear_ci = 1.96 * np.std(linear_data, axis=1) / np.sqrt(np.shape(linear_data)[1])
linear_ddata = [linear_drift[:, i] for i in range(5)]
linear_cci = 1.96 * np.std(linear_ddata, axis=1) / np.sqrt(np.shape(linear_data)[1])
mckean_data = [mckean_rates[:, i] for i in range(5)]
mckean_ci = 1.96 * np.std(mckean_data, axis=1) / np.sqrt(np.shape(linear_data)[1])
mckean_ddata = [mckean_drift[:, i] for i in range(5)]
mckean_cci = 1.96 * np.std(mckean_ddata, axis=1) / np.sqrt(np.shape(linear_data)[1])

# Violin plot
fig = plt.figure()
ax1 = plt.subplot(221)
ax1.set_title('Linear SDE')

ax2 = plt.subplot(223, sharey=ax1)

ax3 = plt.subplot(222, sharey=ax1)
ax3.set_title('McKean SDE')
ax3.yaxis.set_label_position('right')
ax3.set_ylabel('Fixed drift')

ax4 = plt.subplot(224, sharey=ax1)
ax4.yaxis.set_label_position('right')
ax4.set_ylabel('Different drifts')

trate_title = 'Theoretical rate'
krate_title = r'$1/2 - \beta/2$'
for ax, data, ci in zip([ax1, ax2, ax3, ax4],
                        [linear_data, linear_ddata, mckean_data, mckean_ddata],
                        [linear_ci, linear_cci, mckean_ci, mckean_cci]):
    parts = ax.violinplot(data, positions=pos, points=20, widths=0.1, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_alpha(0.8)
    parts['cmaxes'].set_color(lblack)
    parts['cmins'].set_color(lblack)
    parts['cbars'].set_color(lblack)
    parts['cmeans'].set_color(lblack)
    ax.set_xticks(pos)
    ax.plot(pos, [0.16, 0.1, 0.045, 0.01, 0],
            linestyle='--', marker='.', color=lred,
            label=trate_title)
    ax.plot(pos, [1/2 - b/2 for b in pos],
            linestyle='dotted', marker='.', color=lgreen,
            label=krate_title)
    ax.grid(linestyle='dashed')
    expect = np.mean(data, axis=1)
    ax.plot(pos, expect, label='Empirical rate', color=lbrightred, linestyle='dashdot', marker='_')
    ax.fill_between(pos, expect + ci, expect - ci, alpha=0.15, label='95% CI', color=lcoral)

handles, labels = ax4.get_legend_handles_labels()
fig.legend(handles, labels)
fig.supxlabel(r'$\beta$')
fig.supylabel('Rate of convergence')
fig.suptitle('Rates of convergence', fontsize=14)
#fig.savefig('rates_violins.pdf', format='pdf')
#fig.savefig('rates_violins.png', format='png')
#fig.savefig('rates_violins.eps', format='eps')
plt.show()

# Linear rate
fig, ax = plt.subplots()
ax.plot(pos, [0.16, 0.1, 0.045, 0.01, 0],
        linestyle='--', marker='.', color=lred,
        label=trate_title)
ax.plot(pos, [1/2 - b/2 for b in pos],
        linestyle='dotted', marker='.', color=lgreen,
        label=krate_title)
ax.grid(linestyle='dashed')
expect = np.mean(linear_data, axis=1)
ax.plot(pos, expect, label='Empirical rate', color=lbrightred, linestyle='dashdot', marker='_')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Rate of convergence')
ax.set_xticks(pos)
ax.fill_between(pos, expect + linear_ci, expect - linear_ci, alpha=0.15, label='95% CI', color=lcoral)
ax.set_title('Linear SDE\nSingle (arbitrary) drift', fontsize=14)
fig.legend()
#fig.savefig('rates_sde_sd.pdf', format='pdf')
#fig.savefig('rates_sde_sd.png', format='png')
#fig.savefig('rates_sde_sd.eps', format='eps')
plt.show()

# Mckean rate
fig, ax = plt.subplots()
ax.plot(pos, [0.16, 0.1, 0.045, 0.01, 0],
        linestyle='--', marker='.', color=lred,
        label=trate_title)
#ax.plot(pos, [1/2 - b/2 for b in pos],
#        linestyle='dotted', marker='.', color=lgreen,
#        label=krate_title)
ax.grid(linestyle='dashed')
expect = np.mean(mckean_data, axis=1)
ax.plot(pos, expect, label='Empirical rate', color=lbrightred, linestyle='dashdot', marker='_')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Rate of convergence')
ax.set_xticks(pos)
ax.fill_between(pos, expect + linear_ci, expect - linear_ci, alpha=0.15, label='95% CI', color=lcoral)
ax.set_title('McKean SDE\nSingle (arbitrary) drift', fontsize=14)
fig.legend()
fig.savefig('rates_mckean_sd.pdf', format='pdf')
fig.savefig('rates_mckean_sd.png', format='png')
fig.savefig('rates_mckean_sd.eps', format='eps')
plt.show()

# Linear rate multiple drifts
fig, ax = plt.subplots()
ax.plot(pos, [0.16, 0.1, 0.045, 0.01, 0],
        linestyle='--', marker='.', color=lred,
        label=trate_title)
ax.plot(pos, [1/2 - b/2 for b in pos],
        linestyle='dotted', marker='.', color=lgreen,
        label=krate_title)
ax.grid(linestyle='dashed')
expect = np.mean(linear_ddata, axis=1)
ax.plot(pos, expect, label='Empirical rate', color=lbrightred, linestyle='dashdot', marker='_')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Rate of convergence')
ax.set_xticks(pos)
ax.fill_between(pos, expect + linear_cci, expect - linear_cci, alpha=0.15, label='95% CI', color=lcoral)
ax.set_title('Linear SDE\nDifferent drifts', fontsize=14)
fig.legend()
#fig.savefig('rates_sde_dd.pdf', format='pdf')
#fig.savefig('rates_sde_dd.png', format='png')
#fig.savefig('rates_sde_dd.eps', format='eps')
plt.show()

# Mckean rate different drifts
fig, ax = plt.subplots()
ax.plot(pos, [0.16, 0.1, 0.045, 0.01, 0],
        linestyle='--', marker='.', color=lred,
        label=trate_title)
#ax.plot(pos, [1/2 - b/2 for b in pos],
#        linestyle='dotted', marker='.', color=lgreen,
#        label=krate_title)
ax.grid(linestyle='dashed')
expect = np.mean(mckean_ddata, axis=1)
ax.plot(pos, expect, label='Empirical rate', color=lbrightred, linestyle='dashdot', marker='_')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Rate of convergence')
ax.set_xticks(pos)
ax.fill_between(pos, expect + linear_cci, expect - linear_cci, alpha=0.15, label='95% CI', color=lcoral)
ax.set_title('McKean SDE\nDifferent drifts', fontsize=14)
fig.legend()
fig.savefig('rates_mckean_dd.pdf', format='pdf')
fig.savefig('rates_mckean_dd.png', format='png')
fig.savefig('rates_mckean_dd.eps', format='eps')
plt.show()

#fig, axs = plt.subplots(ncols=2, nrows=2)
#for j in range(5):
#    axs[0, 0].hist(linear_data[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
#    axs[0, 0].legend()
#    axs[0, 0].set_ylabel('Frequency')
#    axs[0, 0].set_title('Linear SDE, same drift, 40 batches of BM with 10,000 sample paths')
#for j in range(5):
#    axs[0, 1].hist(linear_ddata[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
#    axs[0, 1].legend()
#    axs[0, 1].set_title('Linear SDE, 40 drifts, same bathc of BM with 10,000 sample paths')
#for j in range(5):
#    axs[1, 0].hist(mckean_data[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
#    axs[1, 0].legend()
#    axs[1, 0].set_xlabel('Rate of convergence')
#    axs[1, 0].set_ylabel('Frequency')
#    axs[1, 0].set_title('McKean SDE, same drift, 40 batches of BM with 10,000 sample paths')
#for j in range(5):
#    axs[1, 1].hist(mckean_ddata[j], histtype='step', fill=False, label='beta=' + str(pos[j]))
#    axs[1, 1].legend()
#    axs[1, 1].set_xlabel('Rate of convergence')
#    axs[1, 1].set_title('Mckean SDE, 40 drifts, same bathc of BM with 10,000 sample paths')
#plt.show()
