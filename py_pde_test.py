from pde import CartesianGrid, ScalarField, PDE, MemoryStorage

grid = CartesianGrid([[0, 10], [0, 5]], [20, 10], periodic=[True, False])
field = ScalarField(grid)
eq = PDE({"u": "-gradient_squared(u) / 2 - laplace(u - laplace(u))"})
storage = MemoryStorage()
result = eq.solve(field, t_range=10, dt=1e-2, tracker=["progress", storage.tracker(1)])
result.plot()
