from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    plot_kymograph, \
    PDEBase, \
    ScipySolver, \
    Controller, \
    movie
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class FokkerPlanckPDE(PDEBase):
    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        dx = state.gradient("auto_periodic_dirichlet")[0]
        dxx = state.laplace("auto_periodic_dirichlet")
        return dxx/2 - dx


xn = 32
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), xn)
ic = norm.pdf(x)

grid = CartesianGrid([[-1, 1]], xn, periodic=False)
state = ScalarField(grid, data=ic)

storage = MemoryStorage()
eq = FokkerPlanckPDE()
solver = ScipySolver(eq)
cont = Controller(solver, t_range=(0, 1),
                  tracker=["plot", storage.tracker(0.05)])
soln = cont.run(state)

plot_kymograph(storage)
