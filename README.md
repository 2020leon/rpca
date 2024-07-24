# RPCA

[Robust principal component analysis] (robust PCA, RPCA) is a modification of principal component analysis (PCA) which works well with respect to grossly corrupted observations. The package implements robust PCA in exact alternating Lagrangian multipliers (EALM) algorithm and inexact alternating Lagrangian multipliers (IALM) algorithm. The implementation refers [the paper][Lin] and [its MATLAB implementation][MATLAB packages].

## Installation

Use the package manager [`pip`] to install.

```shell
pip install git+https://github.com/2020leon/rpca.git@v1.1.0
```

## Usage

```python
import numpy as np

import rpca.ealm
import rpca.ialm

RNG = np.random.default_rng()
D = RNG.random((20, 20))
A0, E0 = rpca.ealm.fit(D)
A1, E1 = rpca.ialm.fit(D)
```

## Contributing

Contributing is welcome!

## License

MIT

## References

- [Robust principal component analysis - Wikipedia][Robust principal component analysis]
- [Z. Lin, M. Chen, and Y. Ma, "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices," 2010. doi: https://doi.org/10.48550/arXiv.1009.5055.][Lin]
- [MATLAB packages]

[Robust principal component analysis]: https://en.wikipedia.org/wiki/Robust_principal_component_analysis
[Lin]: https://doi.org/10.48550/arXiv.1009.5055
[MATLAB packages]: https://people.eecs.berkeley.edu/~yima/matrix-rank/sample_code.html
[`pip`]: https://pip.pypa.io/en/stable/
