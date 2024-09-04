import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import norm
from scipy.interpolate import interpn
from pde import CartesianGrid, ScalarField, MemoryStorage, \
        PDEBase, ScipySolver, Controller


# Fractional Brownian motion (fBm) creation function
def fbm(gaussian: np.ndarray,
        hurst: float,
        points: int = 2**12,
        half_support: float = 10) -> np.ndarray:
    assert gaussian.shape[0] == points
    fbm_grid = np.linspace(start=1/points, stop=2*half_support, num=points)

    xv, yv = np.meshgrid(fbm_grid, fbm_grid, sparse=False, indexing='ij')

    covariance = 0.5*(np.abs(xv)**(2*hurst) +
                      np.abs(yv)**(2*hurst) -
                      np.abs(xv - yv)**(2*hurst))

    cholesky = np.linalg.cholesky(covariance)
    fbm_array = np.matmul(cholesky, gaussian)
    return fbm_array


def bridge(f: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bridge_array = f - (f[-1]/grid[-1])*grid
    return bridge_array


# Heat kernel parameter creation based on time steps of the Euler scheme
def heat_kernel_var(time_steps: int, hurst: float) -> float:
    eta = 1/((hurst-1/2)**2 + 2 - hurst)
    variance = 1/(time_steps**eta)
    return variance


def integral_between_grid_points(heat_kernel_var: float,
                                 grid_x: np.ndarray,
                                 half_support: float) -> np.ndarray:
    points = len(grid_x)
    heat_kernel_std = math.sqrt(heat_kernel_var)
    integral = np.zeros_like(grid_x)
    delta_half = half_support/(points)
    integral, error = quad_vec(lambda z:
                               ((grid_x - z)/heat_kernel_var)*norm.pdf(
                                   grid_x - z,
                                   loc=0,
                                   scale=heat_kernel_std),
                               a=-delta_half, b=delta_half)
    return integral


def create_drift_array(rough_func: np.ndarray,
                       integral_on_grid: np.ndarray) -> np.ndarray:
    return -np.convolve(rough_func, integral_on_grid, 'same')


def drift(gaussian: np.ndarray, hurst: float, points: int = 2**12, half_support: float = 10,
          time_steps: int = 2**5):
    grid = np.linspace(-half_support, half_support, points)
    grid0 = np.linspace(0, 2*half_support, points)
    fbm_array = fbm(gaussian, hurst, points, half_support)
    fbb_array = bridge(fbm_array, grid0)
    hk = heat_kernel_var(time_steps, hurst)
    ig = integral_between_grid_points(hk, grid, half_support)
    drift_array = create_drift_array(fbb_array, ig)
    return drift_array, fbm_array, fbb_array, grid


# Coarse noise
def coarse_noise(z: np.ndarray,
                 time_steps: int,
                 sample_paths: int) -> np.ndarray:
    z_coarse = np.zeros(shape=(time_steps, sample_paths))
    q = int(np.shape(z)[0] / time_steps)
    if q == 1:
        z_coarse = z
    else:
        temp = z.reshape(time_steps, q, sample_paths)
        z_coarse = np.sum(temp, axis=1)
    return z_coarse


def solve(y0: float,
          drift_array: np.ndarray,
          z: np.ndarray,
          time_start: float, time_end: float, time_steps: int,
          sample_paths: int,
          grid: np.ndarray,
          keep_paths: bool = False) -> np.ndarray:
    y = np.zeros(shape=(1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    if keep_paths:
        y = np.zeros(shape=(time_steps+1, sample_paths))
        y[0, :] = y0
        for i in range(time_steps):
            y[i+1, :] = y[i, :] \
                    + np.interp(x=y[i, :], xp=grid, fp=drift_array)*dt \
                    + z_coarse[i, :]
        return y
    y[0, :] = y0
    for i in range(time_steps):
        y[0, :] = y[0, :] \
                + np.interp(x=y[0, :], xp=grid, fp=drift_array)*dt \
                + z_coarse[i, :]
    return y


# Euler scheme solver for a generic McKean-Vlasov SDE
class FokkerPlanckPDE(PDEBase):
    def __init__(self, drift, nonlinear, bc="dirichlet"):
        self.drift = drift
        self.nonlinear = nonlinear
        self.bc = bc

    def evolution_rate(self, state, t=0):
        assert state.grid.dim == 1
        v = state
        x = np.linspace(state.grid.axes_bounds[0][0],
                        state.grid.axes_bounds[0][1],
                        state.grid.shape[0])
        div = v * self.nonlinear(v) * self.drift(x)
        v_x = div.gradient(bc=self.bc)[0]
        v_xx = v.laplace(bc=self.bc)
        v_t = 0.5 * v_xx - v_x
        return v_t


def solve_fp(drift_a, grid_a, limx=1, nonlinear_f=lambda x: np.sin(x),
             ts=0, te=1, xpoints=10, tpoints=2**8):
    x = np.linspace(-limx, limx, xpoints)
    ic = norm.pdf(x)
    grid = CartesianGrid(bounds=[(-limx, limx)], shape=xpoints, periodic=False)
    state = ScalarField(grid=grid, data=ic)

    def drift_f(x: np.ndarray, drift_array=drift_a, grid=grid_a):
        return np.interp(x=x.data, xp=grid, fp=drift_array)

    eq = FokkerPlanckPDE(drift=drift_f, nonlinear=nonlinear_f)
    solver = ScipySolver(pde=eq)
    dt = 1/tpoints
    time_range = (ts, te)
    storage = MemoryStorage()
    cont = Controller(solver=solver, t_range=time_range,
                      tracker=storage.tracker(dt))
    soln = cont.run(state)
    return storage


def solve_mv(y0: np.ndarray,
             drift_array: np.ndarray,
             z: np.ndarray,
             time_start: float, time_end: float, time_steps: int,
             sample_paths: int,
             grid: np.ndarray,
             half_support,
             xpde, tpde, nl) -> np.ndarray:
    y = np.zeros(shape=(1, sample_paths))
    z_coarse = coarse_noise(z, time_steps, sample_paths)
    dt = (time_end - time_start)/(time_steps-1)
    y[0, :] = y0
    rho = solve_fp(drift_a=drift_array, grid_a=grid, limx=half_support,
                   nonlinear_f=nl, ts=time_start, te=time_end,
                   xpoints=xpde, tpoints=tpde)
    rho_usable = np.array(rho.data)
    tsde = np.linspace(time_start, time_end, tpde+1)
    xsde = np.linspace(-half_support, half_support, xpde)
    ti = 0
    for i in range(time_steps):
        ti += dt
        tti = np.repeat(ti, sample_paths)
        y[0, :] = y[0, :] + \
            interpn((tsde, xsde), nl(rho_usable),
                    np.array(list(zip(tti, y[0, :]))),
                    'linear', False, 1) * \
            np.interp(x=y[0, :], xp=grid, fp=drift_array)*dt + \
            z_coarse[i, :]
    return y, rho_usable
