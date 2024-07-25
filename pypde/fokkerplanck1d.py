from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    plot_kymograph, \
    PDEBase, \
    ScipySolver, \
    ExplicitSolver, \
    CrankNicolsonSolver, \
    Controller, \
    movie
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class FokkerPlanckPDE(PDEBase):
    def __init__(self, drift, nonlinear, bc="dirichlet"):
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
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), xn)
ic = norm.pdf(x)

grid = CartesianGrid(bounds=[(-1, 1)], shape=xn, periodic=False)
state = ScalarField(grid=grid, data=ic)
state1 = ScalarField(grid=grid, data=ic)
state2 = ScalarField(grid=grid, data=ic)

rhs1 = (0.5 * state.laplace("dirichlet") - ((state * state**2 * 1).gradient("dirichlet"))[0]).data
rhs2 = (0.5 * state.laplace("dirichlet") - ((state * np.tan(state) * 1).gradient("dirichlet"))[0]).data
rhs3 = (0.5 * state.laplace("dirichlet") - ((state * np.log(state) * 1).gradient("dirichlet"))[0]).data
rhs4 = (0.5 * state.laplace("dirichlet") - ((state * np.sin(state) * 1).gradient("dirichlet"))[0]).data
plt.figure()
plt.plot(rhs1, label=r"$b = 1, F = x^2$")
plt.plot(rhs2, label=r"$b = 1, F = \tan{x}$")
plt.plot(rhs3, label=r"$b = 1, F = \log{x}$")
plt.plot(rhs4, label=r"$b = 1, F = \sin{x}$")
plt.ylim([-1.5, 1.5])
plt.legend()
plt.show()

#storage1 = MemoryStorage()
#storage2 = MemoryStorage()
#eq1 = FokkerPlanckPDE(drift=lambda x: x, nonlinear=lambda x: x**2)
#eq2 = FokkerPlanckPDE(drift=lambda x: 1, nonlinear=lambda x: x**2)
#solverSP1 = ScipySolver(pde=eq1)
#solverSP2 = ScipySolver(pde=eq2)
#cont1 = Controller(solver=solverSP1, t_range=(0, 1),
#                  tracker=storage1.tracker(0.01))
#cont2 = Controller(solver=solverSP2, t_range=(0, 1),
#                  tracker=storage2.tracker(0.01))
#soln = cont1.run(state1)
#soln = cont2.run(state2)
#
##movie(storage, filename="fp.mp4")
##plot_kymograph(storage, filename="fp.jpg")
#
#
#tplot = np.linspace(0, 1, np.shape(storage.data)[0])
#xplot = np.linspace(-1, 1, np.shape(storage.data)[1])
#tplot, xplot = np.meshgrid(tplot, xplot)
#yplot = np.array(storage.data).transpose()
#fig = plt.figure()
#ax = fig.add_subplot(projection="3d")
#ax.plot_surface(tplot, xplot, yplot)
#ax.set_xlabel(r"$t$")
#ax.set_ylabel(r"$x$")
#ax.set_zlabel(r"$\rho(t, x)$")
#plt.show()
#
