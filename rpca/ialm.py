'''Inexact augmented Lagrange multiplier method for robust PCA.'''
import numpy as __np


def fit(
  D,
  lambda_=None,
  epsilon1=1e-7,
  epsilon2=1e-5,
  mu=None,
  rho=1.6,
  max_iter=1000,
  verbose=True,
):
  '''
  Fit the robust PCA model using inexact augmented Lagrange multiplier method.

  Parameters
  ----------
  D : np.ArrayLike
    `m` x `n` matrix of observations/data.
  lambda_ : float, default=1 / np.sqrt(m)
    Weight on sparse error term in the cost function.
  epsilon1 : float, default=1e-7
    Tolerance for stopping criterion.
  epsilon2 : float, default=1e-5
    Tolerance for the other stopping criterion.
  mu : float, default=1.25 / norm_two
    Small positive scalar.
  rho : float, default=1.6
    `mu` update parameter, which should be greater than 1.
  max_iter : int, default=1000
    Maximum number of iterations.
  verbose : bool, default=True
    Printing verbose messages.

  Returns
  -------
  A : np.ndarray
    Low rank matrix.
  E : np.ndarray
    Sparse matrix.

  References
  ----------
  The function implements Algorithm 5 of the paper [1]_.

  .. [1] Z. Lin, M. Chen, and Y. Ma, "The Augmented Lagrange Multiplier Method
     for Exact Recovery of Corrupted Low-Rank Matrices," 2010. doi:
     https://doi.org/10.48550/arXiv.1009.5055.

  Examples
  --------
  >>> import numpy as np
  >>>
  >>> import rpca.ialm
  >>>
  >>> RNG = np.random.default_rng()
  >>> D = RNG.random((20, 20))
  >>> A, E = rpca.ialm.fit(D)  # doctest: +SKIP
  '''
  D = __np.asarray(D)

  m, n = D.shape
  if lambda_ is None:
    lambda_ = 1 / __np.sqrt(m)

  D = __np.float64(D)
  Y = __np.copy(D)
  norm_two = __np.linalg.norm(Y, 2)
  norm_inf = __np.linalg.norm(Y, __np.inf) / lambda_
  dual_norm = __np.max([norm_two, norm_inf])
  Y = Y / dual_norm

  A = __np.zeros_like(D)
  E = __np.zeros_like(D)
  d_norm = __np.linalg.norm(D, 'fro')
  tol_proj = epsilon2 * d_norm

  if mu is None:
    mu = 1.25 / norm_two

  iter_ = 0
  converged = False
  stop_criterion = True
  sv = 10

  if verbose:
    print(
      'lambda_',
      lambda_,
      'epsilon1',
      epsilon1,
      'epsilon2',
      epsilon2,
      'mu',
      mu,
      'rho',
      rho,
      'max_iter',
      max_iter,
      sep='\t',
    )

  while not converged:
    iter_ += 1

    temp_T = D - A + (1 / mu) * Y
    temp_E = (
      __np.maximum(temp_T - lambda_ / mu, 0) +
      __np.minimum(temp_T + lambda_ / mu, 0)
    )

    U, S, V = __np.linalg.svd(D - temp_E + (1 / mu) * Y, full_matrices=False)

    svp = __np.count_nonzero(S > 1 / mu)

    if svp < sv:
      sv = __np.min([svp + 1, n])
    else:
      sv = __np.min([svp + round(.05 * n), n])

    A = __np.dot(__np.dot(U[:, :svp], __np.diag(S[:svp] - 1 / mu)), V[:svp, :])

    Z = D - A - temp_E
    Y += mu * Z
    if mu * __np.linalg.norm(E - temp_E, 'fro') < tol_proj:
      mu *= rho
      converged = True

    E = temp_E

    stop_criterion = __np.linalg.norm(Z, 'fro') / d_norm

    converged = converged and stop_criterion < epsilon1

    if verbose:
      print(
        '#svd',
        iter_,
        'r(A)',
        __np.linalg.matrix_rank(A),
        '|E|_0',
        __np.count_nonzero(E),
        'stopCriterion',
        stop_criterion,
        sep='\t',
      )

    if not converged and iter_ >= max_iter:
      if verbose:
        print('max iter reached')
      break

  return A, E
