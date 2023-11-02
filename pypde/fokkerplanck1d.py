from pde import CartesianGrid, \
    ScalarField, \
    MemoryStorage, \
    plot_kymograph, \
    PDEBase, \
    ScipySolver, \
    Controller
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


class FokkerPlanckPDE(PDEBase):
    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        dx = state.gradient("auto_periodic_neumann")[0]
        dxx = state.laplace("auto_periodic_neumann")
        return dxx / 2 - dx


x_grid = 32
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), x_grid)
ic = norm.pdf(x)

grid = CartesianGrid([[-1, 1]], [x_grid], periodic=False)
state = ScalarField.random_normal(grid)

storage = MemoryStorage()
eq = FokkerPlanckPDE()
solver = ScipySolver(eq)
cont = Controller(solver, t_range=5, tracker=storage.tracker(0.1))
soln = cont.run(state)
eq.solve(state, t_range=10,
         method="scipy",
         tracker=storage.tracker(0.1))

plot_kymograph(storage)
