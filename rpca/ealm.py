import numpy as __np


def fit(
  D,
  lambda_=None,
  epsilon1=1e-7,
  epsilon2=1e-6,
  mu=None,
  rho=6.,
  max_iter=1000,
  verbose=True,
):
  '''
  Exact augmented Lagrange multiplier method for robust PCA.

  Parameters
  ----------
  D : np.ndarray
    `m` x `n` matrix of observations/data.
  lambda_ : float, default=1 / np.sqrt(m)
    Weight on sparse error term in the cost function.
  epsilon1 : float, default=1e-7
    Tolerance for stopping criterion.
  epsilon2 : float, default=1e-6
    Tolerance for inner loop stopping criterion.
  mu : float, default=0.5 / norm_two
    Small positive scalar.
  rho : float, default=6.
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
  The function implements Algorithm 4 of [1]_.

  .. [1] Lin, Zhouchen, Minming Chen, and Yi Ma. "The Augmented Lagrange
     Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices."
  '''
  m, n = D.shape
  if lambda_ is None:
    lambda_ = 1 / __np.sqrt(m)

  D = __np.float64(D)
  Y = __np.sign(D)
  norm_two = __np.linalg.norm(Y, 2)
  norm_inf = __np.linalg.norm(Y, __np.inf) / lambda_
  dual_norm = __np.max([norm_two, norm_inf])
  Y = Y / dual_norm

  A = __np.zeros_like(D)
  E = __np.zeros_like(D)
  d_norm = __np.linalg.norm(D, 'fro')
  tol_proj = epsilon2 * d_norm

  if mu is None:
    mu = 0.5 / norm_two

  iter_ = 0
  converged = False
  svd_count = 0
  sv = 5
  svp = sv

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

    # solve the primal problem by alternative projection
    primal_converged = False
    sv += round(n * 0.1)

    while not primal_converged:
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
        sv = __np.min([svp + round(0.05 * n), n])

      temp_A = __np.dot(
        __np.dot(U[:, :svp], __np.diag(S[:svp] - 1 / mu)),
        V[:svp, :],
      )

      primal_converged = (
        __np.linalg.norm(A - temp_A, 'fro') < tol_proj and
        __np.linalg.norm(E - temp_E, 'fro') < tol_proj
      )

      A = temp_A
      E = temp_E
      svd_count += 1

    Z = D - A - E
    Y += mu * Z
    mu *= rho

    # stop criterion
    stop_criterion = __np.linalg.norm(Z, 'fro') / d_norm
    converged = stop_criterion < epsilon1

    if verbose:
      print(
        'Iteration',
        iter_,
        '#svd',
        svd_count,
        'r(A)',
        svp,
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
