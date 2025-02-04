import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import time
from hw1 import *


### 1) Example:
print("************************* \n ******* Example ******* \n *************************")

np.random.seed(123)
n_countries = 30
A = np.random.normal(1, 0.1, size=(n_countries, 1))
L = np.random.normal(5, 0.1, size=(n_countries, 1))
tau = np.random.normal(5, 0.4, size=(n_countries, n_countries))
sigma = 5

w = Armington_solver(A, L, tau, sigma, verbose=True)

print(w.T)


### 2) Unit Test

np.random.seed(123)
n_countries = 100
A_test = np.ones((n_countries, 1))
L_test = np.random.normal(1, 0.1, size=(n_countries, 1))**2
L_test[0] = 1 #normalization

tau_test = np.ones((n_countries, n_countries))
sigma_test = 5

print("************************* \n ******* Unit Test ******* \n ************************")
print(unit_test(A_test, L_test, sigma_test))


### 3) Gains From Trade
print("************************************ \n ******* Gains From Trade Computation *******\n ************************************")
np.random.seed(123)
n_countries = 10
A_test2 = np.ones((n_countries, 1))
L_test2 = np.random.normal(2, 0.1, size=(n_countries, 1))**2
L_test2[0] = 1 #normalization

tau_test2 = np.ones((n_countries, n_countries))
sigma_test2 = 10


print(f'Real income relative to the autarkic level: \n {GFT(A_test2, L_test2, tau_test2, sigma_test2).T}')

