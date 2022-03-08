# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:05:37 2022

@author: mmlmcj

@title: Numerical Schemes for an SDE with distributional drift
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.random import default_rng
rng = default_rng()

class DistributionalSDE:
    """
    
    We have an SDE as in A Numerical Scheme for Stochastic Differential
    Equations with Distributional Drift by De Angelis, Germain and Issoglio
    (https://arxiv.org/abs/1906.11026)
    
    """
    def __init__ (self, n_haar, h_hurst, k_interval, m_time_steps):
        """
        

        Parameters
        ----------
        n_haar : int
            For a Haar series expansion to approximate a distributional drift
            as in https://arxiv.org/abs/1906.11026., this parameter corresponds
            to the parameter N from equation (13) in the aforementioned paper.
        h_hurst : float
            The Hurst coefficient for a fractional Brownian motion.
        k_interval : float
            The half lenght of the interval where the fBm and the solution of
            the SDE will be supported as in section 7.1 from
            (https://arxiv.org/abs/1906.11026)
        m_time_steps : int
            The amount of time steps that the Euler-Maruyama method will have
            as required in (https://arxiv.org/abs/1906.11026)

        Returns
        -------
        None.

        """
        self.n_haar = n_haar
        self.h_hurst = h_hurst
        self.k_interval = k_interval
        self.m_time_steps = m_time_steps
    
    def fbm (self):
        """
        Generates a fractional Brownian motion
        (https://en.wikipedia.org/wiki/Fractional_Brownian_motion)
        using the Cholesky decomposition method
        (https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Method_1_of_simulation)

        Returns
        -------
        fbm_array : TYPE
            A single path of a fractional Brownian motion, this is an array of
            dimension (1, 2^(self.n_haar+1)).

        """
        points_fbm = 2**(self.n_haar+1)
        x = np.linspace(start=1/points_fbm, stop=1., num=points_fbm, endpoint=True)
        g = rng.standard_normal(size=points_fbm)
        xg, yg = np.meshgrid(x, x, sparse=False, indexing='ij')
        correlation_fbm = 0.5*(np.abs(xg)**(2*self.h_hurst) + np.abs(yg)**(2*self.h_hurst) - np.abs(xg - yg)**(2*self.h_hurst))
        cholesky_fbm = np.linalg.cholesky(a=correlation_fbm)
        fbm_array = np.matmul(cholesky_fbm, g)
        fbm_array = np.insert(arr=fbm_array, obj=0, values=0)
        return fbm_array
        