# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:34:58 2023

@author: Luis Mario Chaparro Jáquez

@title: Euler scheme for distributional drifts SDEs
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
                num = points
                )
        
        self.length_grid = self.grid.shape[0]

        self.fbm_path = self.fbm()
        self.pow_time_steps = int(np.log2(time_steps))
        self.time_steps_array = [2**(k+1) for k in range(1, approximations+1)]

        self.t_pow = 1/(2*(self.hurst-1/2)**2 + 2 - self.hurst)
        self.t_heat = [np.sqrt(1/(k**(self.t_pow))) for k in self.time_steps_array]
        
        self.t_real_solution = np.sqrt(1/(time_steps**(self.t_pow)))
        
        self.df = [self.normal_differences(k) for k in self.t_heat]
        self.df_real_solution = self.normal_differences(self.t_real_solution)
        
        self.dist_array = [np.convolve(self.fbm_path, k, 'same') for k in self.df]
        self.dist_array_real_solution = np.convolve(self.fbm_path, self.df_real_solution, 'same')

        def create_function(dd):
            def f (t, x, m):
                delta = self.limit/self.length_grid
                return np.piecewise(
                        x, 
                        [(i - delta <= x)*(x < i + delta) for i in self.grid],
                        [dd[i] for i in range(len(self.grid))]
                        )
            return f
            

        self.func_list = []
        for k in range(approximations):
            d = self.dist_array[k].tolist()
            f = create_function(d)
            self.func_list.append(f)

        def f1 (t, x, m):
            delta = self.limit/self.length_grid
            return np.piecewise(
                    x,
                    [(i - delta <= x)*(x < i + delta) for i in self.grid],
                    [self.dist_array_real_solution[i] for i in range(len(self.grid))]
                    )
        self.func_list.append(f1)

        
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
        
        return diff_norm

    
class Euler:

    def __init__ (
            self, 
            h, l, bp,
            time_steps = None,
            time_start = None,
            time_end = None,
            y0 = None, 
            paths = None,
            approximations = None,
            batches = None
            ):

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

        self.h = h # Hurst parameter
        self.l = l # Uper limit of support, the function will be supported on [-l, l]
        self.bp = bp # Points of definition for the coefficient

        self.dt = self.generate_dt()

        self.z = rng.normal(
                loc=0.0,
                scale=np.sqrt(self.dt),
                size=(self.time_steps, self.paths)
                )
        
        self.dist = Distribution(hurst=self.h, limit=self.l, points=self.bp, time_steps=self.time_steps, approximations=self.approximations)
        self.drift_list = self.dist.func_list
        self.array_list = self.dist.dist_array
        self.array_real = self.dist.dist_array_real_solution
        self.y_lim = [-10, 10]
        
        self.x = np.linspace(-3, 3, 50)

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
        z_coarse = np.zeros(shape = (time_steps_z, self.paths))

        q_z = int(np.shape(z_orig)[0] / time_steps_z)
        
        if q_z == 1:
            z_coarse = z_orig
        else:
            temp = z_orig.reshape(
                    time_steps_z, 
                    q_z,
                    self.paths#, self.batches
                    )
            z_coarse = np.sum(temp, axis=1)
            
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
        
        y = np.zeros(shape=(time_steps_solve+1, self.paths))
        y[0, :] = self.y0 # Initial condition
        
        for i in range(time_steps_solve):
            y[i+1, :] = y[i, :] \
                    + drift(
                            t = time_grid_solve[i],
                            x = y[i, :], 
                            m = time_steps_solve
                            )*dt_solve \
                    + z_solve[i, :]
        return y

    def b(self, t, x, m):
        return 1.5*x

    def rate (self, show_plot=False, save_plot=False):
        error = np.zeros(shape=(self.approximations, self.paths))
        x_axis = np.zeros(self.approximations)
        real_solution = self.solve(time_steps_solve=self.time_steps, drift=self.drift_list[-1])
        
        terminal_time = np.zeros(shape=(self.approximations, self.paths))
        
        length_solution = int(np.log2(np.shape(real_solution)[0]))
        for i in range(self.approximations):
            m = 2**(i+4)
            delta = (self.time_end - self.time_start)/m
        
            soln = self.solve(time_steps_solve = m, drift=self.drift_list[i])
            
            error[i, :] = np.abs(real_solution[-1, :] - soln[-1, :])
            terminal_time[i, :] = soln[-1, :]
            x_axis[i] = delta

        error_ic = np.zeros(shape=(self.approximations))
        error_inter_approx = np.zeros(shape=(self.approximations))
    
        for i in range(self.approximations):
            error_mean = np.mean(error[i, :])
            error_var = np.sum((error[i,:] - error_mean)**2/self.paths)
            error_sqrt = np.sqrt(error_var/self.paths)
            error_ic[i] = 1.96*error_sqrt
            if(i == self.approximations-1):
                error_inter_approx[i] = np.mean(np.abs(real_solution[-1, :] - terminal_time[i]))
            else:
                error_inter_approx[i] = np.mean(np.abs(terminal_time[i+1] - terminal_time[i]))

        error_meanv = np.mean(error, axis=1)
        
        reg = np.ones(self.approximations)
        A = np.vstack([np.log2(x_axis), reg]).T
        y_reg = np.log2(error_meanv[:, np.newaxis])
        rate, intersection = np.linalg.lstsq(A, y_reg, rcond=None)[0]

        rate_plot = plt.figure(dpi=500)
        plt.errorbar(
                x=np.log2(x_axis),
                y=np.log2(error_meanv),
                yerr=[np.log2(error_meanv) - np.log2(error_meanv - error_ic), np.log2(error_meanv + error_ic) - np.log2(error_meanv)],
                label="Error",
                ecolor="red"
                )
        plt.grid()
        plt.title(
                label="Rate = "
                +str(rate)
                +"\nProxy of solution: 2^"+str(length_solution)+" time steps"
                )
        plt.xlabel("\log_2(step size)")
        plt.ylabel("\log_2(error)")
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

        return error, error_ic, error_mean, rate, error_inter_approx

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

st = time.process_time()
M = 2**14
e = 0.00001
b1 = e
b14 = 1/256
b13 = 1/128
b12 = 1/64
b11 = 1/32
b2 = 1/16
b3 = 2/16
b4 = 3/16
b5 = 4/16
b51 = 5/16
b52 = 6/16
b53 = 7/16
b6 = 1/2 - e

beta = b6
h = 1 - beta
l = 3
def_points_bn = 2**8
y = Euler(
    h = h,
    l = l,
    bp = def_points_bn,
    #drift = bn,
    #diffusion = sigma,
    time_steps = M,
    paths = 10000,
    batches = 50,
    approximations = 10,
    y0 = 1
    )
error, ic, error_mean, rate, inter_error = y.rate(show_plot = False, save_plot = False)
print("rate =", rate)
print("rate =", inter_error)

et = time.process_time()
print("time: ", et-st)
