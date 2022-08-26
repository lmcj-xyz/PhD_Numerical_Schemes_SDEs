import euler as e
import numpy as np
import matplotlib.pyplot as plt

A = 0.1
B = 0.3
TIME_STEPS = 10**5

def mu(x, t, m):
    return A*x

def sigma(x, t, m):
    return B*x

# Option to print only 3 decimal places
np.set_printoptions(precision = 3)

y = e.Euler(
        drift = mu,
        diffusion = sigma,
        time_steps = TIME_STEPS,
        #time_end = 10,
        paths = 1000,
        y0 = 1
        )
#, paths_plot = 3)

#print("time steps:\n", y.time_steps)
#print("time grid:", y.time_grid)
#print("dt:\n", y.dt)
#print("random variable:\n", y.z)
#plt.figure()
#plt.plot(y.z[0, :].T)
#plt.show()
#print(y.coarse_z(time_steps_z = 10**1))
#plt.figure()
#plt.plot(y.coarse_z(time_steps_z = 10**2)[0, :].T)
#plt.show()
#print(np.shape(y.z)[1])
#print("solution y placeholder:\n", y.y)
#print("drift:\n", y.drift(y.y, y.time_grid))
#print("solution:\n", y.solve())
#y.plot_solution(paths_plot = 5, save_plot = True)
#y.plot_solution(paths_plot = 3, save_plot = False)
#plt.figure()
#plt.plot(y.solve(time_steps_solve = 10**2)[0:3, :].T)
#plt.title("coarse euler")
#plt.show()
#plt.figure()
#plt.plot(y.solve(time_steps_solve = 10**1)[0:3, :].T)
#plt.title("coarse euler")
#plt.show()

# Creation of an explicit GBM to compare
########## This is wrong, check
#gbm = np.zeros_like(y.y)
#print(np.shape(gbm))
#gbm[0, :] = y.y0
#
#for i in range(y.time_steps - 1):
#    gbm[i+1, :] = gbm[0, :]*np.exp(
#            ( A - B**2 / 2)*y.time_grid[i] + B*y.z[i+1, :]
#            #( A - B**2 / 2)*y.dt + B*y.z[i+1, :]
#            )

#gbm*y.y0

#print(gbm)
#plt.figure()
#plt.plot(gbm[:, 0:3])
#plt.plot(gbm.T)
#plt.title("gbm")
#plt.show()

#print(np.shape(gbm[:, ::10**1]))
#print(np.shape(gbm[:, ::10**2]))

#dif = gbm - y.solve()
#print(dif)
#
#print(np.amax(abs(dif), axis=1))

print(y.rate(y.solve(), 4))
