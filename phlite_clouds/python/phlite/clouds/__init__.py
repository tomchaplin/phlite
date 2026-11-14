# Re-exports the extension module `phlite.clouds._clouds` (built from phlite_clouds, see src/python), whose stubs are
# alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _clouds
from ._clouds import *  # noqa: F403

__doc__ = _clouds.__doc__
__all__ = [name for name in getattr(_clouds, "__all__", []) if not name.startswith("_")]
