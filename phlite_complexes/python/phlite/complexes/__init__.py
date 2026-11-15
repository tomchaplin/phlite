# Re-exports the extension module `phlite.complexes._complexes` (built from phlite_complexes, see src/python), whose
# stubs are alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _complexes
from ._complexes import *  # noqa: F403

__doc__ = _complexes.__doc__
__all__ = [name for name in _complexes.__all__ if not name.startswith("_")]
