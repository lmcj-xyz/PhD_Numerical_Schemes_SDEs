from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph

eq = PDE({"v": "0.5 * laplace(v) - d_dx(v * sin(v) * 5)"})
grid = CartesianGrid([[-1, 1]], [32], periodic=True)
state = ScalarField.from_expression(grid, "10 * sin(x)")

storage = MemoryStorage()
eq.solve(state, t_range=0.5, solver="scipy", tracker=storage.tracker(0.01))

plot_kymograph(storage)
