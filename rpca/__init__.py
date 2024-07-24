'''
RPCA: Python implementation of robust principal component analysis.

Robust principal component analysis (robust PCA, RPCA) is a modification of
principal component analysis (PCA) which works well with respect to grossly
corrupted observations. The package implements robust PCA in exact alternating
Lagrangian multipliers (EALM) algorithm and inexact alternating Lagrangian
multipliers (IALM) algorithm.

References
----------
The package implements Algorithm 4 & 5 of the paper [1]_.

.. [1] Z. Lin, M. Chen, and Y. Ma, "The Augmented Lagrange Multiplier Method for
   Exact Recovery of Corrupted Low-Rank Matrices," 2010. doi:
   https://doi.org/10.48550/arXiv.1009.5055.

Examples
--------
>>> import numpy as np
>>>
>>> import rpca.ealm
>>> import rpca.ialm
>>>
>>> RNG = np.random.default_rng()
>>> D = RNG.random((20, 20))
>>> A0, E0 = rpca.ealm.fit(D)  # doctest: +SKIP
>>> A1, E1 = rpca.ialm.fit(D)  # doctest: +SKIP
'''
