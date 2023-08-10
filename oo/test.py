import numpy as np
import matplotlib.pyplot as plt
import dsdes

hurst = 0.75
points = 2**9
time_steps = 2**5
half_support = 10
grid = np.linspace(-half_support, half_support, points)
grid2x5 = np.linspace(-half_support, half_support, 2**5)
t0 = 0
t1 = 1
sp = 3
y0 = 1

fbm_object = dsdes.FractionalBrownianMotion(hurst=hurst, points=points)
plt.figure()
fbm_object.plot_fbm()
fbm_object.plot_fbb()
plt.show()
fbm_path = fbm_object.fbm.copy()

brownian_motion = dsdes.BrownianMotion(
        time_steps=time_steps, initial_time=t0, final_time=t1, paths=sp)
brownian_motion.lower_resolution(new_time_steps=2**4)
brownian_motion.bm
type(brownian_motion.lower_resolution(new_time_steps=2**3))  # See type
np.shape(brownian_motion.lower_resolution(new_time_steps=2**8))  # Raise except

drift = dsdes.DistributionalDrift(
        fbm_path, hurst, time_steps, points, grid, half_support)
drift_array = drift.drift_array
drift_function = drift.drift(grid)
drift_function_2x5 = drift.drift(grid2x5)

solution1 = dsdes.EulerMaruyama(
        time_steps=time_steps, time_start=t0, time_end=t1,
        initial_condition=y0,
        brownian_motion=brownian_motion.bm,
        drift=drift.drift)
solution2 = dsdes.EulerMaruyama(
        time_steps=2**4, time_start=t0, time_end=t1,
        initial_condition=y0,
        brownian_motion=brownian_motion.lower_resolution(2**4),
        drift=drift.drift)
paths_solution1 = solution1.y
paths_solution2 = solution2.y
