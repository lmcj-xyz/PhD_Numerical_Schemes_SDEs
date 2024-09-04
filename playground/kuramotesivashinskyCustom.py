from pde import PDEBase, ScalarField, UnitGrid


class KuramotoSivashinskyPDE(PDEBase):
    def evolution_rate(self, state, t=0):
        state_lap = state.laplace(bc="auto_periodic_neumann")
        state_lap2 = state_lap.laplace(bc="auto_periodic_neumann")
        state_grad = state.gradient(bc="auto_periodic_neumann")
        return -state_grad.to_scalar("squared_sum") / 2 - state_lap - state_lap2


grid = UnitGrid([32, 32])
state = ScalarField.random_uniform(grid)

eq = KuramotoSivashinskyPDE()
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
