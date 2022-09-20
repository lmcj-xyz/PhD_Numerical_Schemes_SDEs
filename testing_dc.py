import dist_coeff as dc
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
# Tests
x = dc.distribution(hurst=0.75, limit=5, points=10**4)#, time_steps=10**(1))

xx = np.linspace(-x.limit, x.limit, 10**5)
yy1 = x.func(t=0, x=xx, m=10**0)
yy2 = x.func(t=0, x=xx, m=10**1)
yy3 = x.func(t=0, x=xx, m=10**2)
yy4 = x.func(t=0, x=xx, m=10**3)

#print(df1 - df1_1)
#print(df2 - df2_1)

end_time = time.time()
print(end_time - start_time)

plt.figure()
#plt.plot(x.grid, x.fbm_path, label="fBm")
#plt.plot(x.grid, x.dist_array, label="dist")
plt.plot(xx, yy1, label="dist func $m=10^0$")
plt.plot(xx, yy2, label="dist func $m=10^1$")
plt.plot(xx, yy3, label="dist func $m=10^2$")
plt.plot(xx, yy4, label="dist func $m=10^3$")
#plt.plot(x.grid, x.dist_array2, label="dist2")
#plt.plot(x.grid, x.conconv, label="constant")
#plt.plot(x.grid, x.linconv, label="linear")
plt.legend()
plt.grid()
plt.ylim(-5, 5)
plt.show()

#plt.figure()
#plt.plot(xx, yy1_1, label="dist func")
#plt.plot(xx, yy2_1, label="dist func")
#plt.legend()
#plt.grid()
#plt.show()

#print("delta", x.limit/x.grid.shape[0])
#print(x.grid)
## Covariance matrix
#cov = x.fbm()
#print(cov)
#plt.imshow(cov)
#plt.colorbar()
#plt.show()
## fBm
#frac = x.fbm_path
#plt.figure()
#plt.plot(x.grid, frac)
#plt.show()
#print("grid: ", x.grid)
#print("value of b: ", x.normal_differences(x=1, t=1, m=10))
#print(x.normal_difference_m)
#plt.figure()
#plt.plot(x.grid, x.fbm_path)
#plt.show()
#x.dist

#print(x.fbm_path)
#print(x.dist_array)
#print(x.conconv)
#print(x.zerconv)
#print(x.linconv)

#print(x.func(-1.0))
#print(x.func(-0.75))
#print(x.func(-0.5))
#print(x.func(0.0))
#print(x.func(0.5))
#print(x.func(0.75))
#print(x.func(1.0))

#plt.figure()
#plt.plot(x.grid, x.fbm_path, label="fBm")
#plt.plot(x.grid, x.dist2, label="dist")
#plt.plot(x.grid, x.conconv2, label="constant")
##plt.plot(x.grid, x.zerconv, label="zeros")
#plt.plot(x.grid, x.linconv2, label="linear")
#plt.legend()
#plt.ylim(bottom)
#plt.show()

#print(x.dist)
#print(x.df)
#plt.figure()
#plt.plot(x.grid, x.df)
#plt.show()

#plt.figure()
#plt.plot(x.df)
#plt.show()

#print(x.df.size)
#print(x.df.shape)

