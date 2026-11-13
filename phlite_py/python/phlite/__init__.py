"""Persistent homology computed with phlite.

The top level computes the persistence of a filtered chain complex you build (`ChainComplex`), and the submodules
compute it for kinds of data: `phlite.cloud` (point clouds and distance matrices), `phlite.network` (weighted
digraphs) and `phlite.image` (images). Every pipeline takes an `Algorithm`; `defaults()` lists what each fills in for
the options an `Algorithm` leaves unset.

The classes and functions are implemented in Rust, in the extension module `phlite._phlite`, which this package
re-exports (the type stubs alongside describe them).
"""

from . import _phlite
from ._phlite import *  # noqa: F403
from ._phlite import __version__
from . import cloud, image, network

__all__ = list(_phlite.__all__)
