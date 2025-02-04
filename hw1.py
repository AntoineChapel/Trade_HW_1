import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import time
from hw1 import *

def Armington_solver(A: np.ndarray,
                     L: np.ndarray,
                     tau: np.ndarray,
                     sigma: float,
                     damp: float = 0.9,
                     verbose = True,
                     normalize = True) -> np.ndarray:
  """
  Solves the Armington model for N countries
  Takes as input:
    * A (np.ndarray) : a vector of productivities of dimension N x 1
    * L (np.ndarray) : a vector of labor endowment of dimension N x 1
    * tau (np.ndarray) : a matrix of distances of dimension N x N, such that the
           entry on row i, column j, tau_{ij} is the bilateral trade cost
           between countries i and j
    * sigma (float) : elasticity parameter (scalar)
    * damp (float in [0, 1) ) : parametrizes the convex combination step of the
                                iterative procedure. With damp=0 (not 
                                recommended), full updating.
  """

  tic = time.time()

  A = A.reshape(-1, 1)
  L = L.reshape(-1, 1)

  assert A.shape == L.shape, "The two arrays A and L should have the same size"

  N = A.shape[0]
  assert tau.shape[0] == tau.shape[1], "The tau matrix should be square"
  assert tau.shape[0] == N, "The tau matrix should have dimension N x N"
  assert sigma > 0, "The elasticity parameter sigma should be strictly positive"
  assert damp >= 0 and damp < 1, "The damping parameter should be in [0, 1)"

  epsilon = sigma - 1

  A_jnp = (jnp.array(A)).reshape(-1, 1)
  L_jnp = (jnp.array(L)).reshape(-1, 1)
  tau_jnp = jnp.array(tau)
  T_jnp = A_jnp**epsilon

  tol = 1e-3
  max_iter = 1e5
  norm = 1e6
  iter_count = 0

  w = jnp.ones((N, 1))

  def D_j(T, w, tau, epsilon, j):
    return T.T@((w*(tau[:, j]).reshape(-1, 1))**(-epsilon))
  D_j_vectorized = vmap(D_j, in_axes=(None, None, None, None, 0))

  while norm > tol and iter_count < max_iter:
    Y = w * L_jnp
    lambda_D = (D_j_vectorized(T_jnp, w, tau_jnp, epsilon, jnp.arange(N))).reshape(1, -1)
    lambda_mat = (T_jnp.T*(w*tau_jnp)**(-epsilon)) / lambda_D
    Y_prime = lambda_mat@Y

    norm = jnp.max(jnp.abs(Y_prime - Y))
    Y = (damp*Y + (1-damp)*Y_prime).reshape(-1, 1)

    w = (Y/L_jnp).reshape(-1, 1)
    if normalize:
      w = w.at[0].set(1) #normalization enforcement: w_1 = 1
    iter_count += 1

    if verbose==True and iter_count%20==0:
      print(f"Iteration {iter_count}, norm: {norm}")

  tac = time.time()

  if iter_count == max_iter:
    print("Maximum number of iterations reached")
  else:
    print(f"Convergence reached in {iter_count} iterations and {tac - tic} seconds.")

  return np.array(w)


def unit_test(A, L, sigma, tol=1e-2):
  n_countries = A.shape[0]
  closed_form = L**(-1/sigma)

  tau = np.ones((n_countries, n_countries))
  equilibrium_w = Armington_solver(A, L, tau, sigma, verbose=True, damp=0.9)

  return jnp.max(jnp.abs(equilibrium_w - closed_form)) < tol



def GFT(A, L, tau, sigma, damp = 0.9):
  A = A.reshape(-1, 1)
  L = L.reshape(-1, 1)

  w_FT = Armington_solver(A, L, tau, sigma, verbose=False, normalize=True, damp=damp).reshape(-1, 1)
  
  P_FT = (jnp.sum(((w_FT*tau/A)**(1-sigma)), axis=0)**(1/(1-sigma))).reshape(-1, 1)

  real_income_FT = (w_FT*L)/P_FT
  real_income_autarky = (A*L)

  return np.array(real_income_FT)/np.array(real_income_autarky)