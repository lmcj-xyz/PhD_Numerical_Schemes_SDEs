# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:10:03 2023

@author: mmlmcj
@title: 2-dimensional SDEs (GBM)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x0 = np.array([10, 10])

t_start = 0
t_end = 1
time_steps = 100
dt = (t_end - t_start)/time_steps
t = np.linspace(start=t_start+dt, stop=t_end, num=time_steps)
t = np.concatenate((np.array([0]), t), axis=0)

d = 2
paths = 100000

rng = np.random.default_rng()
z = rng.normal(
    loc=0.0,
    scale=np.sqrt(dt), 
    size=(time_steps, paths, d)
    )

mu = 1
sigma = 0.5
drift = lambda x: mu*x
diffusion = lambda x: sigma*x

x = np.zeros(shape=(time_steps+1, paths, d))
x[0,:,:] = x0

for i in range(time_steps):
    x[i+1,:,:] = x[i,:,:] + drift(x[i,:,:])*dt + diffusion(x[i,:,:])*z[i,:,:]

#%%
plt.figure(1)
plt.plot(t, x[:,0,0])
plt.plot(t, x[:,0,1])
plt.show()