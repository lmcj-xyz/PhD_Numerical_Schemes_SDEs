from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    PDEBase, \
    ScipySolver, \
    Controller
from scipy.stats import norm
from scipy.interpolate import interpn
import numpy as np
import matplotlib.pyplot as plt


class FokkerPlanckPDE(PDEBase):
    def __init__(self, drift, nonlinear, bc={"value": norm.pdf(norm.ppf(0.01))}):
        self.drift = drift
        self.nonlinear = nonlinear
        self.bc = bc

    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        v = state
        drift2 = v * self.nonlinear(v) * self.drift(v)
        v_x = drift2.gradient(bc=self.bc)[0]
        v_xx = v.laplace(bc=self.bc)
        v_t = 0.5 * v_xx - v_x
        return v_t


xn = 2**5
x0 = norm.ppf(0.01)
x1 = norm.ppf(0.99)
x = np.linspace(x0, x1, xn)
ic = norm.pdf(x)

grid_bounds = (x0, x1)
grid = CartesianGrid(bounds=[grid_bounds], shape=xn, periodic=False)
state = ScalarField(grid=grid, data=ic)

storage = MemoryStorage()


def b_drift(x0):
    return np.interp(x0.data, x, x)


def f_nonlinear(x):
    return np.log(x)


eq = FokkerPlanckPDE(drift=b_drift, nonlinear=f_nonlinear)
solver = ScipySolver(pde=eq)

time_steps = 100
dt = 1/time_steps
t0 = 0
t1 = 1
time_range = (t0, t1)
cont = Controller(solver=solver, t_range=time_range,
                  tracker=storage.tracker(dt))

soln = cont.run(state)

tplot = np.linspace(t0, t1, np.shape(storage.data)[0])
xplot = np.linspace(x0, x1, np.shape(storage.data)[1])
Tplot, Xplot = np.meshgrid(tplot, xplot)
yplot = np.array(storage.data).transpose()
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(Tplot, Xplot, yplot)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_zlabel(r"$\rho(t, x)$")
ax.set_title("Solution to PDE")
plt.show()

finer_time_steps = 1000
finer_space_steps = 2**9
finer_x = np.linspace(x0, x1, finer_space_steps)
finer_t = np.linspace(t0, t1, finer_time_steps)
finer_grid = [finer_t, finer_x]
ft, fx = np.meshgrid(*finer_grid)
yfiner = interpn((tplot, xplot), yplot.transpose(), np.meshgrid(*finer_grid), 'cubic')

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(ft, fx, yfiner)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_zlabel(r"$\rho(t, x)$")
ax.set_title("Interpolation")
plt.show()
