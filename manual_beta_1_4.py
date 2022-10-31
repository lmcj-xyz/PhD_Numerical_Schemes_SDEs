# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:06:58 2022

@author: Luis Mario Chaparro Jáquez

@title: Euler scheme
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from datetime import datetime
from scipy.integrate import quad_vec
from scipy.stats import norm

import time

class Distribution:

    def __init__(self, hurst, limit, points, time_steps, approximations):
        self.hurst = hurst
        self.limit = limit
        self.points = points
        self.approximations = approximations

        self.grid = np.linspace(
                start = -limit,
                stop = limit,
                num = points
                )
        self.fbm_grid = np.linspace(
                start = 1/self.points,
                stop = 2*limit,
                #stop = 1,
                num = points
                )
        
        self.length_grid = self.grid.shape[0]

        self.fbm_path = self.fbm()

        # To compute the parameter t of the heat kernel
        self.pow_time_steps = int(np.log2(time_steps))
        
        #self.time_steps_array = [2**(self.pow_time_steps-k) for k in range(approximations+1)]
        # Get the number of time steps for the approximations I need
        # Starting from 2^2 = 4 time steps, and going up to 2^approximations+1
        # For instance if approximations = 5, we would have time steps from
        # 2^2 = 4 to 2^6 = 64
        self.time_steps_array = [2**(k+1) for k in range(1, approximations+1)]
        #print(self.time_steps_array)

        # For the time steps of the approximations we compute the parameter
        # t of the heat kernel
        self.t_pow = 0
        self.t_heat = [np.sqrt(1/(k**(self.t_pow))) for k in self.time_steps_array]
        # The following line helps to test for the same function all the time
        #self.t_heat = [np.sqrt(1/(16**self.t_pow)) for k in self.time_steps_array]
        #print("t's approx", self.t_heat)
        
        # The same but for the real solution
        self.t_real_solution = np.sqrt(1/(time_steps**(self.t_pow)))
        # The following line helps to test for the same function all the time
        #self.t_real_solution = np.sqrt(1/(16**self.t_pow))
        #print("t real", self.t_real_solution)

        self.df = [self.normal_differences(k) for k in self.t_heat]
        self.df_real_solution = self.normal_differences(self.t_real_solution)
        #self.df_dummy = [self.normal_differences(10^(-k)) for k in range(1, 10)]

        #for i in self.df:
        #    plt.figure()
        #    plt.title("df approx")
        #    plt.plot(i)
        #    plt.show()
        #plt.figure()
        #plt.title("df real")
        #plt.plot(self.df_real_solution)
        #plt.show()
        #for i in self.df_dummy:
        #    plt.figure()
        #    plt.title("df dummy")
        #    plt.plot(i)
        #    plt.show()
        
        # This stores the distributional array for each amount of
        # time_steps, i.e. t parameters for the kernel
        self.dist_array = [np.convolve(self.fbm_path, k, 'same') for k in self.df]
        self.dist_array_real_solution = np.convolve(self.fbm_path, self.df_real_solution, 'same')

        #for k in range(approximations):
        #    #print(k)
        #    plt.figure()
        #    plt.title("approx")
        #    plt.plot(np.array(self.dist_array[k][:]))
        #    plt.ylim(-5, 5)
        #    plt.show()
        #plt.figure()
        #plt.title("real soln")
        #plt.plot(self.dist_array_real_solution)
        #plt.ylim(-5, 5)
        #plt.show()

        #plt.figure()
        #plt.title("dist array")
        #plt.plot(np.array(self.dist_array).T)
        #plt.show()
        #print("number dist array = ", len(self.dist_array))
        #for i in self.dist_array:
        #    print("dist array shape = ", np.shape(i))
        #    plt.figure()
        #    plt.title("dist array")
        #    plt.plot(i)
        #    plt.ylim(-5, 5)
        #    plt.show()
        #print("dist array shape = ", np.shape(self.dist_array_real_solution))

        # This is a function to create functions and avoid the function
        # to be defined once only and being repeated
        def create_function(dd):
            def f (t, x, m):
                #var_heat = self.t_heat[k]
                # Limit is half the length of the interval [-limit, limit]
                # therefore, if delta x is the separation between x
                # limit/length_grid = (delta x) / 2 is half of it
                delta = self.limit/self.length_grid
                #print([(i - delta <= x)*(x < i + delta) for i in self.grid])
                # this function must be piecewise linear, not constant
                return np.piecewise(
                        x, 
                        [(i - delta <= x)*(x < i + delta) for i in self.grid],
                        #d
                        [dd[i] for i in range(len(self.grid))]
                        )
            return f
            

        self.func_list = []
        for k in range(approximations):
            #print("grid = ", np.shape(self.grid), "dist = ", np.shape(self.dist_array[k]))
            #plt.figure()
            #plt.plot(self.dist_array[k])
            #plt.ylim(-100, 100)
            #plt.show()
            d = self.dist_array[k].tolist()
            f = create_function(d)
            #def f (t, x, m):
            #    #var_heat = self.t_heat[k]
            #    # Limit is half the length of the interval [-limit, limit]
            #    # therefore, if delta x is the separation between x
            #    # limit/length_grid = (delta x) / 2 is half of it
            #    delta = self.limit/self.length_grid
            #    #print([(i - delta <= x)*(x < i + delta) for i in self.grid])
            #    # this function must be piecewise linear, not constant
            #    return np.piecewise(
            #            x, 
            #            [(i - delta <= x)*(x < i + delta) for i in self.grid],
            #            #d
            #            [self.dist_array[k][i] for i in range(len(self.grid))]
            #            )
            self.func_list.append(f)

        #for k in range(approximations):
        #    print(self.func_list[k](t=1, x=1, m=1))

        #print("grid = ", np.shape(self.grid), "dist = ", np.shape(self.dist_array_real_solution))
        #print("functions = ", len(self.func_list))

        def f1 (t, x, m):
            #var_heat = self.t_real_solution
            delta = self.limit/self.length_grid
            # this function must be piecewise linear, not constant
            return np.piecewise(
                    x,
                    [(i - delta <= x)*(x < i + delta) for i in self.grid],
                    [self.dist_array_real_solution[i] for i in range(len(self.grid))]
                    )
        #self.func_real_solution = f1

        # The last element in this list is the drift for the real solution
        # Careful with that, remember to use this for the approximarion with
        # i in range(approximations) and take i as index
        # and for the real solution take the index -1
        self.func_list.append(f1)

        #print(type(self.func_real_solution))

        ##### Test to see the functions created
        #x_test = np.linspace(-1, 1, 50)
        #for i in range(approximations+1):
        #    plt.figure()
        #    plt.title("dist function approx 2^%d time steps" % (i+2))
        #    plt.plot(self.func_list[i](t=1, x=x_test, m=1))
        #    plt.show()

        #plt.figure()
        #plt.title("dist function real")
        #plt.plot(self.func_list[-1](t=1, x=x_test, m=self.t_real_solution))
        #plt.show()

############################### METHODS


    def fbm(self):
        x_grid, y_grid = np.meshgrid(
                self.fbm_grid, 
                self.fbm_grid, 
                sparse=False,
                indexing='ij'
                )
        covariance = 0.5*(
                np.abs(x_grid)**(2*self.hurst) +
                np.abs(y_grid)**(2*self.hurst) - 
                np.abs(x_grid - y_grid)**(2*self.hurst)
                )
        g = rng.standard_normal(size=self.points)
        cholesky = np.linalg.cholesky(a=covariance)
        fbm_arr = np.matmul(cholesky, g)
        fbm_arr = np.concatenate([np.zeros(1),fbm_arr])
        return fbm_arr

    def normal_differences(self, t_var):
        diff_norm = np.zeros(shape=self.length_grid)
        delta = self.limit/self.length_grid
        const = -1/t_var**2

        p = lambda u: const*(self.grid + u)*norm.pdf(self.grid+u, loc=0, scale=t_var)
        diff_norm = quad_vec(p, -delta, delta)[0]
        #p = lambda u: const*(self.grid - u)*norm.pdf(self.grid, loc=u, scale=t_var)
        #diff_norm = quad_vec(p, -delta, delta)[0]

        return diff_norm

    #def func(self, t, x, m):
    #    var_heat = self.t_heat
    #    #df = self.df
    #    #dist_a = self.dist_array
    #    delta = self.limit/self.length_grid
    #    return np.piecewise(
    #            x, 
    #            [(k - delta <= x)*(x < k + delta) for k in self.grid],
    #            [k for k in dist_a]
    #            )

class Euler:

    def __init__ (
            self, 
            h, l, bp,
            #drift, 
            #diffusion, 
            time_steps = None,
            time_start = None,
            time_end = None,
            y0 = None, 
            paths = None,
            approximations = None,
            batches = None
            ):

        #self.drift = drift
        #self.diffusion = diffusion
        self.time_steps = time_steps if time_steps \
                is not None else 2**6
        self.time_start = time_start if time_start \
                is not None else 0
        self.time_end = time_end if time_end \
                is not None else 1
        self.y0 = y0 if y0 \
                is not None else 1
        self.paths = paths if paths \
                is not None else 100
        self.batches = batches if batches \
                is not None else 20 # This parameter is not currently used
        self.approximations = approximations if approximations \
                is not None else 5

        # Parameters needed for the generation of the distributional coefficient
        self.h = h # Hurst parameter
        self.l = l # Uper limit of support, the function will be supported on [-l, l]
        self.bp = bp # Points of definition for the coefficient

        # Generates a dt for each amount of time steps
        # Used in the method solve
        self.dt = self.generate_dt()

        # Random variable Representing the differences of Brownian motion for each time step
        self.z = rng.normal(
                loc=0.0,
                scale=np.sqrt(self.dt),
                #size=(self.time_steps, self.paths, self.batches)
                size=(self.time_steps, self.paths)
                )
        
        # From the index 0 to approximations-1
        # we have the drift for the approximations
        # the last index, or -1 is the drift of the real solution
        self.dist = Distribution(hurst=self.h, limit=self.l, points=self.bp, time_steps=self.time_steps, approximations=self.approximations)
        self.drift_list = self.dist.func_list
        self.array_list = self.dist.dist_array
        self.array_real = self.dist.dist_array_real_solution
        ############# TESTS ##################
        self.y_lim = [-10, 10]
        for i in self.array_list:
            plt.figure()
            plt.title("array approx dist")
        ### The parameters t and m do not matter, they are kept to work with the generic scheme
            plt.plot(np.linspace(-self.l, self.l, np.shape(i)[0]),i)
            plt.ylim(self.y_lim)
            plt.show()
            
        plt.figure()
        plt.title("array real dist")
        plt.plot(np.linspace(-self.l, self.l, np.shape(self.array_real)[0]), self.array_real)
        plt.ylim(self.y_lim)
        plt.show()
        # Print the length of the list of function
        #print("drift list elements = ", len(self.drift_list))

        # Plot the different distributional coefficient (depending on step size)
        self.x = np.linspace(-3, 3, 50)
        #
        for i in self.drift_list:
            plt.figure()
            plt.title("function")
        ### The parameters t and m do not matter, they are kept to work with the generic scheme
            plt.plot(self.x, i(x = self.x, t = 3, m = 3))
            plt.ylim(self.y_lim)
            plt.show()
        #^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^^^

    def generate_dt (self, time_steps_dt = None):
        time_steps_dt = time_steps_dt if time_steps_dt \
                is not None else self.time_steps
        time_start_dt = self.time_start
        time_end_dt = self.time_end

        dt_generated = (time_end_dt - time_start_dt) / time_steps_dt
        return dt_generated
    
    def generate_time_grid (self, time_steps_grid = None):
        time_steps_grid = time_steps_grid if time_steps_grid \
                is not None else self.time_steps
        time_start_grid = self.time_start
        time_end_grid = self.time_end

        dt_grid = self.generate_dt(time_steps_dt = time_steps_grid)

        time_grid_generated = np.linspace(
                start = time_start_grid + dt_grid, # We don't want the time t=0 since that is given by the initial condition
                stop = time_end_grid,
                num = time_steps_grid
                )
        return time_grid_generated
    
    def coarse_z (self, time_steps_z = None):
        time_steps_z = time_steps_z if time_steps_z \
                is not None else self.time_steps
        z_orig = self.z
        #dt_z = self.generate_dt(time_steps_dt=time_steps_z)
        #z_coarse = np.zeros(shape = (time_steps_z, self.paths, self.batches))
        
        ### We want the coarser Z to have the time steps of the coarser approximations
        z_coarse = np.zeros(shape = (time_steps_z, self.paths))

        ### Quotient between the original amount of time steps
        ### and the time steps for a coarser approximation
        q_z = int(np.shape(z_orig)[0] / time_steps_z)
        
        ### Subsampling or change of resolution for the RV
        if q_z == 1:
            z_coarse = z_orig
        else:
            ### This is a NumPy way of subsampling the RV
            ### It creates a temporary array by reshaping the original RV with shape
            ### (time_steps, paths)
            ### into an array with shape
            ### (time_steps, q_z, paths)
            ### Then it will sum over the axis of the new dimension
            ### to get a new RV z_coarse of shape
            ### (time_steps_z, paths)
            ### Where time_steps_z is the time steps for the approximation
            temp = z_orig.reshape(
                    time_steps_z, 
                    q_z,
                    self.paths#, self.batches
                    )
            z_coarse = np.sum(temp, axis=1)
            ### In the method solve (below), line 374 you can uncomment a series
            ### of print statements where you can see the variance of the change
            ### for different time steps
            ### It naturally deteriorates if we have a small number of time steps

        return z_coarse
    
    def solve (self, drift = None, time_steps_solve = None):
        time_steps_solve = time_steps_solve if time_steps_solve \
                is not None else self.time_steps
        time_grid_solve = self.generate_time_grid(time_steps_solve) if time_steps_solve \
                is not None else self.time_grid
        dt_solve = self.generate_dt(time_steps_dt=time_steps_solve) if time_steps_solve \
                is not None else self.dt
        z_solve = self.coarse_z(time_steps_z=time_steps_solve) if time_steps_solve \
                is not None else self.z
        drift = drift if drift  \
                is not None else self.drift_list[-1]

        ############# TESTS ##################
        ### Here you can see how the variance changes when we change the "resolution"
        ### of the RV that represents the differences of BM
        '''
        print("thoretical variance original Z = ", 1/self.time_steps)
        print(" empirical variance original Z = ", np.mean(np.var(self.z, axis=0)))
        print("     thoretical variance new Z = ", 1/time_steps_solve)
        print("      empirical variance new Z = ", np.mean(np.var(z_solve, axis=0)))
        '''
        #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#
        
        ### We require ONE element more than the time steps in order to use the 
        ### all the elements in the RV representing differences of BM
        #y = np.zeros(shape=(time_steps_solve+1, self.paths, self.batches))
        y = np.zeros(shape=(time_steps_solve+1, self.paths))
        #y[0, :, :] = self.y0
        y[0, :] = self.y0 # Initial condition
        
        #for i in range(time_steps_solve - 1):
        for i in range(time_steps_solve):
            #y[i+1, :, :] = y[i, :, :] \
            y[i+1, :] = y[i, :] \
                    + drift(
                            t = time_grid_solve[i],
                            #x = y[i, :, :], 
                            x = y[i, :], 
                            m = time_steps_solve
                            )*dt_solve \
                    + z_solve[i, :]
                    #+ z_solve[i+1, :]
                    #+ z_solve[i+1, :, :]
        return y

    ### Modify this function if you want to use a different drift
    ### to use it comment and uncomment the tests starting on LINE 433 to avoid the Euler scheme
    ### running for the distributional coeffiecient and only run it for b
    def b(self, t, x, m):
        return 1.5*x

    def rate (self, show_plot=False, save_plot=False):
        error = np.zeros(shape=(self.approximations, self.paths))
        #error = np.zeros(shape=(self.approximations))
        x_axis = np.zeros(self.approximations)
        #m = self.time_steps
        #dist = Distribution(hurst=self.h, limit=self.l, points=self.bp, time_steps=self.time_steps, self.approximations=approximations)
        
        ############# TESTS #################
        ##### Plots for the function used in each step
        ### The function will have support in [-l, l]
        ### the larger domain from x_test is to illustrate the support
        #x_test = np.linspace(-15, 15, 50)
        #plt.figure()
        #plt.title("dist function real")
        #plt.plot(self.drift_list[-1](t=1, x=x_test, m=1))
        ##plt.plot(self.b(t=1, x=x_test, m=1))
        #plt.show()
        #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#
        
        ############# TESTS #################
        # Uncomment as appropriate
        # IMPORTANT: Also uncomment and comment in the TESTS starting in LINE 468 as appropriate
        
        # Euler scheme with Distributional coefficient
        real_solution = self.solve(time_steps_solve=self.time_steps, drift=self.drift_list[-1])
        
        # Euler scheme for known SDE given by function b in line 410
        #real_solution = self.solve(time_steps_solve=self.time_steps, drift=self.b)
        #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#

        #length_solution = int(np.log10(np.shape(real_solution)[0]))
        length_solution = int(np.log2(np.shape(real_solution)[0]))
        for i in range(self.approximations):
            #m = (10**(length_solution-i-1))
            #m = (2**(length_solution-i-1))
            # 2^(i+2) because we want the approximations starting with
            # 2^2 time steps
            m = 2**(i+4)
            delta = (self.time_end - self.time_start)/m
            #############
            #print("m = ", m)
            #print("i = ", i)
            #print("func length = ", len(dist.func_list))
            #############
            #print("delta t = ", delta)

            ############# TESTS #################
            ### If you uncomment as appropriate you will see the different drifts that are used
            ### Used to see that we are actually using the appropriate drift for each approximation
            #plt.figure()
            #plt.title("dist function approx 2^%d" % (i+2))
            #plt.plot(self.drift_list[i](t=1, x=x_test, m=1))
            #plt.show()
            #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#
            
            ############# TESTS #################
            # IMPORTANT: Also uncomment as appropriate the tests starting in LINE 432
            # Approximation with distributional coefficient
            soln = self.solve(time_steps_solve = m, drift=self.drift_list[i])
            
            # Approximation for known SDE with drift given by method b
            #soln = self.solve(time_steps_solve = m, drift=self.b)
            #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#

            ##### Not necesary #####
            
            ### We do not need this because we only want the solution at terminal time
            ### It is needed if we want the maximum or supremum of the paths
            
            #real_solution_coarse = real_solution[::10**(i+1), :, :]
            #real_solution_coarse = real_solution[::2**(i+1), :, :]
            #real_solution_coarse = real_solution[::2**(length_solution-i-1-1), :, :]

            # To get the coarse real solution we divide the original length
            # by the new desired length,
            # then we use that to  select that amount of elements
            #real_solution_coarse = real_solution[::int(2**length_solution/m), :, :]
            #real_solution_coarse = real_solution[::int(2**length_solution/m), :]
            #^^^^ Not necesary ^^^^#
            
            ############# TESTS #################
            ### This is the plot of a single path of the solution to the SDE
            ### Compared with the approximation with fewer time steps
            ### Just for illustrative purposes
            
            #print("shape real soln = ", np.shape(real_solution_coarse)[0])
            #print("shape appr soln = ", np.shape(soln)[0])
            plt.figure()
            plt.title("comparison")
            #plt.plot(real_solution_coarse[:,1])
            plt.plot(np.linspace(0,1,self.time_steps+1),real_solution[:,1])
            plt.plot(np.linspace(0,1,m+1),soln[:,1])
            #plt.plot(np.linspace(0,1,self.time_steps),real_solution[:,1])
            #plt.plot(np.linspace(0,1,m),soln[:,1])
            plt.show()
            #^^^^^^^^^^^^ TESTS ^^^^^^^^^^^^^^^^^#

            ### Not needed
            #error[i] = np.amax(
            #                np.mean(
            #                    np.abs(
            #                        np.subtract(real_solution_coarse, soln)
            #                        ),
            #                    axis = 1
            #                    ),
            #                axis = 0
            #                )

            #error[i, :] = np.mean(
            #                    np.abs(
            #                        np.subtract(real_solution_coarse, soln)
            #                        ),
            #                    axis = 1
            #                    )[-1,:]
            #^^^ Not needed ^^^^^^^^^^#
            
            ### Computation of the error for the approximation i for all paths
            ### of the approximation and all paths of the "real solution"
            error[i, :] = np.abs(real_solution[-1, :] - soln[-1, :])
            x_axis[i] = delta

        #print(error)
        #error_ic = np.zeros(shape=(2, self.approximations))
        error_ic = np.zeros(shape=(self.approximations))
        for i in range(self.approximations):
            error_mean = np.mean(error[i, :])
            #error_mean_log = np.log2(error_mean)
            error_var = np.sum((error[i,:] - error_mean)**2/self.paths)
            #error_var = np.sum((np.log2(error[i,:]) - error_mean)**2/self.paths)
            #error_var = np.var(error[i, :])
            error_sqrt = np.sqrt(error_var/self.paths)
            error_ic[i] = 1.96*error_sqrt

        error_meanv = np.mean(error, axis=1)
    
        print("\t\tThe first pair below is for the minimum amount of time steps\n\t\twhile the last is for the maximum")
        print("upper limit IC: ", np.log2(error_meanv) + error_ic)
        print("lower limit IC: ", np.log2(error_meanv) - error_ic)

        ### Linear regression to compute rate of convergence
        reg = np.ones(self.approximations)
        A = np.vstack([np.log2(x_axis), reg]).T
        y_reg = np.log2(error_meanv[:, np.newaxis])
        rate, intersection = np.linalg.lstsq(A, y_reg, rcond=None)[0]

        """
        rate_plot = plt.figure()
        plt.loglog(x_axis, error, label="Error", marker="o")
        plt.title(
                label="Rate = "
                +str(rate)
                +"\nProxy of solution: 10^"+str(length_solution)+" time steps"
                )
        plt.xlabel("Step size")
        plt.ylabel("log(error)")
        plt.legend()
        """
        rate_plot = plt.figure()
        plt.errorbar(
                x=np.log2(x_axis),
                y=np.log2(error_meanv),
                yerr=[np.log2(error_meanv) - np.log2(error_meanv - error_ic), np.log2(error_meanv + error_ic) - np.log2(error_meanv)],
                label="Error",
                ecolor="red"
                )
        plt.grid()
        #plt.plot(np.log(x_axis), intersection+np.log(x_axis)*rate)
        plt.title(
                label="Rate = "
                +str(rate)
                +"\nProxy of solution: 2^"+str(length_solution)+" time steps"
                )
        plt.xlabel("log2(step size)")
        plt.ylabel("log2(error)")
        plt.legend()

        if show_plot == True:
            plt.show()

        if save_plot == True:
            rate_plot.savefig(
                    fname = 
                    'figures_rate/'
                    +
                    datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
                    +
                    '_rate'
                    )

        return error, error_ic, error_mean, rate#, np.log10(error), np.log10(x_axis)

    # no modificado para batches
    def plot_solution (self, paths_plot, save_plot = False):
        solution = plt.figure()
        plt.plot(self.solve()[:, range(paths_plot)])
        plt.title("Euler scheme")
        plt.show()
        if save_plot == True:
            solution.savefig(
                    fname = 
                    'figures_solution/'
                    +
                    datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
                    +
                    '_solution'
                    )

## Euler scheme for distributional coefficient in C^1/4

st = time.process_time()

# Time steps
#### Careful with the time steps, remember that you only define
#### the array of the dist coeff for so many points,
#### if you have a very small variance then effectivelly you will have integration
#### between points that are not defined
M = 2**12
# Instance of distributional coefficient
#dist = Distribution(hurst=0.75, limit=5, points=10**2)

# n(m) = m^(8/3)

# Distributional drift
beta = 0.25
h = 1 - beta
l = 3
#def_points_bn = M*int(np.ceil(M**(1/3)*2*l))
def_points_bn = 2**8

# Euler approximation
y = Euler(
        h = h,
        l = l,
        bp = def_points_bn,
        #drift = bn,
        #diffusion = sigma,
        time_steps = M,
        paths = 1000,
        batches = 50,
        approximations = 5,
        y0 = 1
        )

# Solution
#y.plot_solution(paths_plot=3, save_plot=False)

# Rate of convergence
#error, rate = y.rate(show_plot = True, save_plot = False)
error, ic, error_mean, rate = y.rate(show_plot = True, save_plot = False)
#print("error array = \n", error)
#print("IC = \n", ic)
#print("shape IC = \n", np.shape(ic))
#print("error array mean = \n", error_mean)
print("rate =", rate)
#print("error shape = ", np.shape(error))

################################################################################
et = time.process_time()
print("time: ", et-st)
