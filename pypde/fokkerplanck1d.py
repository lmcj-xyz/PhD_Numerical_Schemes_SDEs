from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    plot_kymograph, \
    PDEBase
from scipy.stats import norm
import numpy as np


class FokkerPlanckPDE(PDEBase):
    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        dx = state.gradient("auto_periodic_neumann")[0]
        dxx = state.laplace("auto_periodic_neumann")
        return dxx / 2 - dx


x_grid = 32
grid = CartesianGrid([[-1, 1]], [x_grid], periodic=False)
state = ScalarField.from_expression(grid, "sin(x)")

storage = MemoryStorage()
eq = FokkerPlanckPDE()
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), x_grid)
ic = norm.pdf(x)
eq.solve(state, t_range=[0, 10],
         method="scipy", y0=ic, tracker=storage.tracker(0.1))

plot_kymograph(storage)
