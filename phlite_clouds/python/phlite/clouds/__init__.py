# Re-exports the extension module `phlite.clouds._clouds` (built from phlite_clouds, see src/python), whose stubs are
# alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _clouds
from ._clouds import *  # noqa: F403

from . import zigzags  # noqa: E402
from .zigzags import cloud_zigzag, sliding_windows  # noqa: E402

__doc__ = _clouds.__doc__
__all__ = [name for name in getattr(_clouds, "__all__", []) if not name.startswith("_")] + [
    "cloud_zigzag",
    "sliding_windows",
]
