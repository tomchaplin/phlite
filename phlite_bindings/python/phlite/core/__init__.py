# Re-exports the extension module `phlite.core._core` (built from phlite_bindings, see src/core), whose stubs are
# alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _core
from ._core import *  # noqa: F403
from ._core import __version__, _register_pipeline  # noqa: F401 (the latter for the other packages)

from . import spectral  # noqa: E402
from .spectral import SpectralSequence  # noqa: E402

__doc__ = _core.__doc__
__all__ = [name for name in _core.__all__ if not name.startswith("_")] + ["SpectralSequence"]
