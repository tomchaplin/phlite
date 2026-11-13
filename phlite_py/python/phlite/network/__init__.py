# Re-exports the extension's submodule `phlite._phlite.network` (see phlite_py/src/python/network), whose stubs are
# alongside. New functions there appear here without any change.
from .._phlite import network as _native
from .._phlite.network import *  # noqa: F403

__doc__ = _native.__doc__
__all__ = list(_native.__all__)
