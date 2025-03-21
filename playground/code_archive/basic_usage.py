from pde import CartesianGrid, ScalarField, PDE, MemoryStorage

grid = CartesianGrid([[0, 10], [0, 5]], [20, 10], periodic=[True, False])
field = ScalarField.random_uniform(grid)
eq = PDE({'u': '-gradient_squared(u) / 2 - laplace(u - laplace(u))'})
storage = MemoryStorage()
result = eq.solve(field, t_range=10, dt=1e-3,
                  tracker=['progress', storage.tracker(1)])
result.plot()

for time, field in storage.items():
    print(f't={time}, field={field.magnitude}')
