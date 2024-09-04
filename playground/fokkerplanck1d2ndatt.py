from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
from sympy import sin


def fun(x):
    return sin(x) * x


eq = PDE({"v": f"0.5 * laplace(v) - d_dx({fun}() * 5)"})
grid = CartesianGrid([[-1, 1]], [32], periodic=True)
state = ScalarField.from_expression(grid, "10 * sin(x)")

storage = MemoryStorage()
eq.solve(state, t_range=0.5, solver="scipy", tracker=storage.tracker(0.01))

plot_kymograph(storage)
