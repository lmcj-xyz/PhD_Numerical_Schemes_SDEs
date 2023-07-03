---
TO DO:
- [ ] Add type hints to all the files used
- [ ] Remove unnecessary old files
- [ ] Add reference to the paper once we have the preprint.
---

# Numerical schemes for SDEs

Some numerical schemes for SDEs with irregular coefficients.

## Brief theory

We are concerned with the equation

$$
dX_t = b(t, X_t) dt + W_t,
$$

where $W$ is a Brownian motion, and $b$ is a distribution living in the Hölder-Zygmund space $C_T \mathcal C^{-\beta}(\mathbb R)$ for some $\beta \in (0, 1/2)$.
For the numerical methods we consider the drift to be time homogeneous, i.e: $\hat b \in \mathcal C^{-\beta}(\mathbb R)$.

For further details look at the paper at...

## Usage

There are three main files here:
- `dsdes.py` where all the functions are defined
- `error.py` where we can test the functions in the file above
- `plots.py` which creates the plots of the paper

If you want to test the numerical methods you could just run the file `error.py`.
The output you will get is the plot of the convergence rate for the Euler-Maruyama scheme.

Using this file I would recommend to just change the parameter `beta` to whichever parameter you want to explore, recall that in theory $\beta \in (0, 1/2)$, so any results you can get with $\beta$ out of that range are potentially nonsense.

You can also modify the dictionary `time_steps` just have the following in mind:
- The point here is that we need to use the Euler-Maruyama method to approximate the real solution of the SDE because there is not an closed form solution to this SDE, so the first `key` of the dictionary is `real`, this corresponds to the amount of time steps we will use to compute our proxy of the real solutions.
- The following `keys` are `approx1`, `approx2`, etc. This is for the amount of approximations that you want to compute. So if you want an extra approximation you must modify three tuples, namely `keys`, `

If you want to try out
