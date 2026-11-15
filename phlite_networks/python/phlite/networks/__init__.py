# Re-exports the extension module `phlite.networks._networks` (built from phlite_networks, see src/python), whose stubs are
# alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _networks
from ._networks import *  # noqa: F403

__doc__ = _networks.__doc__
__all__ = [name for name in getattr(_networks, "__all__", []) if not name.startswith("_")]
