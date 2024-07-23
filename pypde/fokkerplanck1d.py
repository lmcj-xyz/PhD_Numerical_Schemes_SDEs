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
    def __init__(self,
                 drift=lambda x: np.log(x),
                 nonlinear=lambda x: np.sin(x),
                 bc="auto_periodic_dirichlet"):
        self.drift = drift
        self.nonlinear = nonlinear
        self.bc = bc

    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        v = state
        drift2 = state * self.nonlinear(state) * self.drift(state)
        v_x = drift2.gradient(self.bc)[0]
        v_xx = v.laplace(bc=self.bc)
        v_t = 0.5 * v_xx - v_x
        return v_t


def mydrift(x: float):
    return x + x**2


xn = 2**5
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), xn)
ic = norm.pdf(x)

grid = CartesianGrid([[-1, 1]], xn, periodic=False)
state = ScalarField(grid, data=ic)

storage = MemoryStorage()
eq = FokkerPlanckPDE(drift=lambda x: 1, nonlinear=lambda x: x**2)
solver = ScipySolver(eq)
cont = Controller(solver, t_range=(0, 1),
                  tracker=["progress", storage.tracker(0.01)])
soln = cont.run(state)

movie(storage, filename="fp.mp4")
plot_kymograph(storage, filename="fp.jpg")
