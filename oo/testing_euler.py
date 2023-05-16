import euler as e
import numpy as np
import matplotlib.pyplot as plt

A = 1.5
B = 1.0
TIME_STEPS = 2**6

def mu(x, t, m):
    return A*x

def sigma(x, t, m):
    return B*x

# Option to print only 3 decimal places
#np.set_printoptions(precision = 3)

y = e.Euler(
        drift = mu,
        diffusion = sigma,
        time_steps = TIME_STEPS,
        #time_end = 10,
        paths = 100,
        y0 = 1.0, 
        batches=20,
        )
#, paths_plot = 3)

#print("%f empirical variance %f = " % (TIME_STEPS, np.mean(np.var(y.z, axis=0))))
#print("%f theoretical variance %f = " % (TIME_STEPS, 1/TIME_STEPS))
#
#test_steps = 2**7
#print("%f steps empirical variance = %f" %(test_steps, np.mean(np.var(y.coarse_z(test_steps), axis=0))))
##print("%f steps empirical expectation = %f" %(test_steps, np.mean(np.mean(y.coarse_z(test_steps), axis=0))))
#print("%f steps theoretical variance = %f" %(test_steps, 1/test_steps))
#x1 = np.linspace(0, 1, TIME_STEPS)
#x2 = np.linspace(0, 1, test_steps)
###print("length x1 = ", len(x1))
###print("length x2 = ", len(x2))
#plt.figure()
#plt.plot(x1, y.z[:, 1, 1])
#plt.plot(x2, y.coarse_z(test_steps)[:, 1, 1])
#plt.show()

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
#gbm = np.zeros_like(y.z)
##print(np.shape(gbm))
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

error, rate = y.rate(real_solution = y.solve(), approximations = 4,
        show_plot = True, save_plot = False)
print("error array\n", error)
print("rate =", rate)
#print(logerror)
#print(logx)
#y.plot_rate(y.solve(), 4)