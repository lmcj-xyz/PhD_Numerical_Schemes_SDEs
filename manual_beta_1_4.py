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
        self.t_heat = [np.sqrt(1/(k**(8/3))) for k in self.time_steps_array]
        #print("t's approx", self.t_heat)
        # The same but for the real solution
        self.t_real_solution = np.sqrt(1/(time_steps**(8/3)))
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
        #plt.plot(self.func_real_solution(t=1, x=x_test, m=self.t_real_solution))
        #plt.show()

############################### METHODS


    def fbm(self):
        x_grid, y_grid = np.meshgrid(
                self.grid, 
                self.grid, 
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
        return fbm_arr

    def normal_differences(self, t_var):
        diff_norm = np.zeros(shape=self.length_grid)
        delta = self.limit/self.length_grid
        const = -1/t_var**2

        p = lambda u: const*(self.grid - u)*norm.pdf(self.grid, loc=u, scale=t_var)
        diff_norm = quad_vec(p, -delta, delta)[0]

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
                is not None else 20
        self.approximations = approximations if approximations \
                is not None else 5

        self.h = h
        self.l = l
        self.bp = bp

        self.dt = self.generate_dt()

        self.z = rng.normal(
                loc=0.0,
                scale=np.sqrt(self.dt),
                size=(self.time_steps, self.paths, self.batches)
                )
        self.time_grid = self.generate_time_grid()

        # From the index 0 to approximations-1
        # we have the drift for the approximations
        # the index approximations, or -1 is the drift of the real solution
        self.drift_list = Distribution(hurst=self.h, limit=self.l, points=self.bp, time_steps=self.time_steps, approximations=self.approximations).func_list
        #print("drift list elements = ", len(self.drift_list))

        #self.x = np.linspace(-2, 2, 100)
        #for i in self.drift_list:
        #    plt.figure()
        #    plt.plot(i( x = self.x, t = 3, m = 3))
        #    plt.show()

    def generate_dt (self, time_steps_dt = None):
        time_steps_dt = time_steps_dt if time_steps_dt \
                is not None else self.time_steps
        time_start_dt = self.time_start
        time_end_dt = self.time_end

        dt_generated = (time_end_dt - time_start_dt) / time_steps_dt
        return dt_generated
    
    def coarse_z (self, time_steps_z = None):
        time_steps_z = time_steps_z if time_steps_z \
                is not None else self.time_steps
        z_orig = self.z
        dt_z = self.dt
        z_coarse = np.zeros(shape = (time_steps_z, self.paths, self.batches))

        n_z = int(np.shape(z_orig)[0] / time_steps_z)
        
        if n_z == 1:
            z_coarse = z_orig
        else:
            temp = z_orig.reshape(
                    time_steps_z, 
                    np.shape(z_orig)[0]//time_steps_z,
                    self.paths, 
                    self.batches
                    )
            z_coarse = np.sum(temp, axis=1)

        return z_coarse

    def generate_time_grid (self, time_steps_grid = None):
        time_steps_grid = time_steps_grid if time_steps_grid \
                is not None else self.time_steps
        time_start_grid = self.time_start
        time_end_grid = self.time_end

        dt_grid = self.generate_dt( 
                time_steps_dt = time_steps_grid
                )

        time_grid_generated = np.linspace(
                start = time_start_grid + dt_grid, 
                stop = time_end_grid,
                num = time_steps_grid
                )
        return time_grid_generated

    def solve (self, drift = None, time_steps_solve = None):
        time_steps_solve = time_steps_solve if time_steps_solve \
                is not None else self.time_steps
        time_grid_solve = self.generate_time_grid(time_steps_solve) if time_steps_solve \
                is not None else self.time_grid
        dt_solve = self.generate_dt(time_steps_dt = time_steps_solve) if time_steps_solve \
                is not None else self.dt
        z_solve = self.coarse_z(time_steps_z = time_steps_solve) if time_steps_solve \
                is not None else self.z
        drift = drift if drift  \
                is not None else self.drift_list[-1]

        y = np.zeros(shape=(time_steps_solve, self.paths, self.batches))
        y[0, :, :] = self.y0
        #self.y[0, :] = self.y0
        
        for i in range(time_steps_solve - 1):
            #self.y[i+1, :] = self.y[i, :] \
            y[i+1, :, :] = y[i, :, :] \
                    + drift(
                            t = time_grid_solve[i],
                            x = y[i, :, :], 
                            #x = self.y[i, :], 
                            m = time_steps_solve
                            )*dt_solve \
                    + z_solve[i+1, :, :]
                    #+ z_solve[i+1, :]
        return y

    def b(self, t, x, m):
        return 5*x

    def rate (self, show_plot=False, save_plot=False):
        error = np.zeros(shape=(self.approximations, self.batches))
        x_axis = np.zeros(self.approximations)
        #m = self.time_steps
        #dist = Distribution(hurst=self.h, limit=self.l, points=self.bp, time_steps=self.time_steps, self.approximations=approximations)
        
        ##### Tests for the function used in each step
        #x_test = np.linspace(-15, 15, 50)
        #plt.figure()
        #plt.title("dist function real")
        #plt.plot(self.drift_list[-1](t=1, x=x_test, m=1))
        #plt.show()

        #real_solution = self.solve(time_steps_solve=self.time_steps, drift=self.drift_list[-1])
        # Test with known SDE
        real_solution = self.solve(time_steps_solve=self.time_steps, drift=self.b)

        #length_solution = int(np.log10(np.shape(real_solution)[0]))
        length_solution = int(np.log2(np.shape(real_solution)[0]))
        for i in range(self.approximations):
            #m = (10**(length_solution-i-1))
            #m = (2**(length_solution-i-1))
            # 2^(i+2) because we want the approximations starting with
            # 2^2 time steps
            m = 2**(i+2)
            #############
            print("m = ", m)
            #print("i = ", i)
            #print("func length = ", len(dist.func_list))
            #############
            delta = (self.time_end - self.time_start)/m
            print("delta t = ", delta)

            #plt.figure()
            #plt.title("dist function approx 2^%d" % (i+2))
            #plt.plot(self.drift_list[i](t=1, x=x_test, m=1))
            #plt.show()

            #soln = self.solve(time_steps_solve = m, drift=self.drift_list[i])
            soln = self.solve(time_steps_solve = m, drift=self.b)

            #real_solution_coarse = real_solution[::10**(i+1), :, :]
            #real_solution_coarse = real_solution[::2**(i+1), :, :]
            #real_solution_coarse = real_solution[::2**(length_solution-i-1-1), :, :]

            # To get the coarse real solution we divide the original length
            # by the new desired length,
            # then we use that to  select that amount of elements
            real_solution_coarse = real_solution[::int(2**length_solution/m), :, :]
            
            #print("shape real soln = ", np.shape(real_solution_coarse)[0])
            #print("shape appr soln = ", np.shape(soln)[0])
            plt.figure()
            plt.title("comparison")
            plt.plot(real_solution_coarse[:,1,1])
            plt.plot(soln[:,1,1])
            plt.show()

            #error[i, :] = np.amax(
            #                np.mean(
            #                    np.abs(
            #                        np.subtract(real_solution_coarse, soln)
            #                        ),
            #                    axis = 1
            #                    ),
            #                axis = 0
            #                )

            error[i, :] = np.mean(
                                np.abs(
                                    np.subtract(real_solution_coarse, soln)
                                    ),
                                axis = 1
                                )[-1,:]
            x_axis[i] = delta

        #error_ic = np.zeros(shape=(2, self.approximations))
        error_ic = np.zeros(shape=(self.approximations))
        for i in range(self.approximations):
            error_var = np.var(error[i, :])
            error_sqrt = np.sqrt(error_var/self.batches)
            error_m = np.mean(error[i, :])
            #error_ic[0, i] = error_m - 1.96*error_sqrt
            #error_ic[1, i] = error_m + 1.96*error_sqrt
            #error_ic[0, i] = - 1.96*error_sqrt
            #error_ic[1, i] = + 1.96*error_sqrt
            error_ic[i] = 1.96*error_sqrt

        error_mean = np.mean(error, axis=1)

        reg = np.ones(self.approximations)
        A = np.vstack([np.log2(x_axis), reg]).T
        y_reg = np.log2(error_mean[:, np.newaxis])
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
                #x=x_axis,
                #x=np.log10(x_axis),
                x=np.log2(x_axis),
                #y=error_mean,
                #y=np.log10(error_mean),
                y=np.log2(error_mean),
                #yerr=np.log10(error_ic),
                yerr=error_ic,
                label="Error",
                #marker=".",
                ecolor="red"
                )
        #plt.plot(np.log(x_axis), intersection+np.log(x_axis)*rate)
        plt.title(
                label="Rate = "
                +str(rate)
                +"\nProxy of solution: 2^"+str(length_solution)+" time steps"
                )
        plt.xlabel("Step size")
        plt.ylabel("log(error)")
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
M = 2**8
# Instance of distributional coefficient
#dist = Distribution(hurst=0.75, limit=5, points=10**2)

# n(m) = m^(8/3)

# Distributional drift
beta = 0.25
h = 1 - beta
l = 10
def_points_bn = 10**2

# Euler approximation
y = Euler(
        h = h,
        l = l,
        bp = def_points_bn,
        #drift = bn,
        #diffusion = sigma,
        time_steps = M,
        paths = 100,
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
