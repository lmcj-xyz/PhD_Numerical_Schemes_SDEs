import euler as e
import numpy as np
import matplotlib.pyplot as plt

def mu(x, t):
    return x

def sigma(x, t):
    return 0.05*x

# Option to print only 3 decimal places
np.set_printoptions(precision = 3)

y = e.Euler(
        drift = mu,
        diffusion = sigma,
        time_steps = 10**1,
        #time_end = 10,
        paths = 10
        )
#, paths_plot = 3)

#print("time steps:\n", y.time_steps)
print("time grid:", y.time_grid)
#print("dt:\n", y.dt)
#print("random variable:\n", y.z)
#print("solution y placeholder:\n", y.y)
#print("drift:\n", y.drift(y.y, y.time_grid))
#print("solution:\n", y.solve())
#y.plot_solution(paths_plot = 5, save_plot = True)
y.plot_solution(paths_plot = 5, save_plot = False)

# Creation of an explicit GBM to compare
########## This is wrong, check
gbm = np.zeros(shape = (y.paths, y.time_steps))
gbm[:, 0] = 1
#print(gbm)

for i in range(y.time_steps - 1):
    gbm[:, i+1] = gbm[:, 0] * np.exp(
                ( 
                    #mu(gbm[:, i], y.time_grid[i])
                    1
                    - 
                    #sigma(gbm[:, i], y.time_grid[i])**2 / 2
                    0.05**2 / 2
                )*y.time_grid[i]
            #+ sigma(gbm[:, i], y.time_grid[i])*y.z[:, i+1]
            + 0.05*y.z[:, i+1]
            )

#gbm*y.y0

#print(gbm)
plt.figure()
plt.plot(gbm[0:4, :].T)
plt.show()

#dif = gbm - y.solve()
#print(dif)
#
#print(np.amax(abs(dif), axis=1))
