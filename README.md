# Numerical schemes for SDEs

Some numerical schemes for SDEs with irregular coefficients.

Work by

- [Luis Mario Chaparro Jáquez](https://lmcj.xyz), University of Leeds (owner of this repository)
- [Elena Issoglio](https://sites.google.com/view/elenaissoglio), Università degli Studi di Torino
- [Jan Palczewksi](https://www1.maths.leeds.ac.uk/~jp/), University of Leeds

## TO DO:

- [x] Add type hints to `*.py`
- [x] Stop tracking unnecessary old files
- [ ] Add reference to the paper once we have the preprint

## Brief theory

We are concerned with the equation

$$
dX_t = b(t, X_t) dt + W_t,
$$

where $W$ is a Brownian motion, and $b$ is a distribution living in the Hölder-Zygmund space $C_T \mathcal C^{-\beta}(\mathbb R)$ for some $\beta \in (0, 1/2)$.
For the numerical methods we consider the drift to be time homogeneous, i.e: $\hat b \in \mathcal C^{-\beta}(\mathbb R)$.

Since the drift coefficient is a distribution the numerical approximation is not trivial and has to be addressed by finding functions which converge to the appropriate distribution.
The way we solve this problem by observing we can take functions $C_T \mathcal C^{1 - \beta}$ and compute their generalised derivative and then perform some smoothing procedure on the result of this.
Effectively, the approximated solution of the SDE will be the result of performing the *Euler-Maruyama* method over the SDE

$$
dX_t = P_h [\partial_x B(t, X_t)] dt + W_t,
$$

where $B(t, X_t) \in C_T \mathcal C^{1-\beta}$, and $P_h$ is the heat semigroup with variance $h$.

In particular we are interested in some function $B$ with a rough behaviour in an $\mathbb R$, although for practical purposes we want a said function defined on an interval $[-L, L]$. One such function is *a single sample path* of the well known fractional Brownian motion (fBm) denoted by $B^H$, where $H$ is the so called *Hurst parameter*. It is known that for $H>0$, the function $B^H(\cdot, x)$ is $\alpha$-Hölder continuos for any $\alpha < H$.

Notice that $P_h [\partial_x B^H](y) = p_h \ast [\partial_x B^H](y) = [\partial_x p_h] \ast B^H(y)$, so we can compute the approximated distributional drift by finding the derivative of the heat kernel, which is a very smooth function, and then performing the confolution of the result of it with the fBm we generate.

Once the drift is found, we can proceeed by performing the Euler scheme.

<!--For further details look at the paper on [ArXiV](https://arxiv.org/).-->

## Usage

There are three main files here:

- `dsdes.py` where all the functions are defined
- `error.py` where we can test the functions in the file above
- `plots.py` which creates the plots of the paper

If you want to test the numerical methods you could just run the file `error.py`.
The output you will get is the plot of the convergence rate for the Euler-Maruyama scheme.

Using this file I would recommend to just change the parameter `beta` to whichever parameter you want to explore, recall that in theory $\beta \in (0, 1/2)$, so any results you can get with $\beta$ out of that range are potentially nonsense.

You can also modify the dictionary `time_steps` in order to have different amounts of time steps or add more approximations, just have the following in mind:

- The point here is that we need to use the Euler-Maruyama method to approximate the real solution of the SDE because there is not an closed form solution to this SDE, so the first `key` of the dictionary is `real`, this corresponds to the amount of time steps we will use to compute our proxy of the real solutions.
- The following `keys` are `approx1`, `approx2`, etc. This is for the amount of approximations that you want to compute. So if you want an extra approximation you must modify three tuples, namely `keys`, `time_steps` and `error` on `lines 21, 23, 26` and then go to `line 112` and add an extra computation of error just as they are added there. You can see the example of this below.

## Example: Adding an extra approximation

```{python}
# Change lines 21, 23 and 26 for the following
keys = ('real', 'approx1', 'approx2', 'approx3', 'approx4', 'approx5', 'approx6')  # line 21

time_steps_tuple = (2**15, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14)  # line 23

error_keys = ('e1', 'e2', 'e3', 'e4', 'e5', 'e6')  # line 26

# Then go to line 112 and add a new line with the following
strong_error['e6'] = np.abs(solution['real'] - solution['approx6'])  # this will be line 113
```

Those changes should allow you to run the code and get an extra approximation, optionally you can also change the first element from the `time_stesp_tuple` from `2**15` to `2**16` to have a finer real solution.
