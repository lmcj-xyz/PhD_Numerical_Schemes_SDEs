import dsdes as ds
from numpy.random import default_rng
import numpy as np
import sys
from scipy.stats import norm
import pde


rng = default_rng()
time_steps = 10
epsilon = 10e-6
beta = 1/4
hurst = 1 - beta
y0 = 1
sample_paths = 10**4
time_start = 0
time_end = 1
dt = (time_end - time_start)/(time_steps - 1)
time_grid = np.linspace(time_start + dt, time_end, time_steps)
noise = rng.normal(
        loc=0.0, scale=np.sqrt(dt),
        size=(time_steps, sample_paths)
        )
# Parameters to create fBm
points_x = 2**12  # According to the lower bound in the paper
half_support = 10
eta = 1/((hurst-1/2)**2 + 2 - hurst)
lower_bound = 2*half_support*time_steps**(eta/2)
eqn = 66
if (points_x <= lower_bound):
    msg = 'You need to define your fBm on at least %.2f \
            points as per equation (%d) in the paper.' % (lower_bound, eqn)
    raise ValueError(msg)
    sys.exit(1)
delta_x = half_support/(points_x-1)
grid_x = np.linspace(start=-half_support, stop=half_support, num=points_x)
grid_x0 = np.linspace(start=0, stop=2*half_support, num=points_x)
fbm_array = ds.fbm(hurst, points_x, half_support)
bridge_array = ds.bridge(fbm_array, grid_x0)
# Variance of heat kernel
var_heat_kernel = ds.heat_kernel_var(time_steps, hurst)
# Integral between grid points
integral_grid = ds.integral_between_grid_points(var_heat_kernel, grid_x, half_support)
# Drift creation
drift_array = ds.create_drift_array(bridge_array, integral_grid),

#ds.solve_fp(drift_array, grid_x, half_support, lambda x: np.cos(x), ts=0, te=1)
xn = 2**3
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), xn)
ic = norm.pdf(x)
grid_bounds = (-half_support, half_support)
grid = pde.CartesianGrid(bounds=[grid_bounds], shape=xn, periodic=False)
state = pde.ScalarField(grid=grid, data=ic)
storage = pde.MemoryStorage()


def drift_f(x: np.ndarray, drift_array=drift_array, grid=grid_x):
    return np.interp(x=x.data, xp=grid, fp=drift_array)


eq = ds.FokkerPlanckPDE(drift=drift_f, nonlinear=lambda x: np.sin(x))
solver = pde.ScipySolver(pde=eq)
time_steps = 10
dt = 1/time_steps
time_range = (time_start, time_end)
cont = pde.Controller(solver=solver, t_range=time_range, tracker=storage.tracker(dt))
soln = cont.run(state)


#if __name__ == "__main__":
    #main()
