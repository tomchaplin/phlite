# Re-exports the extension module `phlite.networks._networks` (built from phlite_networks, see src/python), whose stubs are
# alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _networks
from ._networks import *  # noqa: F403

from . import magnitudes  # noqa: E402
from .magnitudes import MagnitudeHomology  # noqa: E402


def magnitude(graph, edges=None, /, l_max=None, *, k_max=None, pairs=None, modulus=None, by_pair=False, integral=False):
    return magnitudes.wrap(
        _networks.magnitude(graph, edges, l_max, k_max=k_max, pairs=pairs, modulus=modulus, by_pair=by_pair,
                            integral=integral),
        modulus=modulus, by_pair=by_pair, integral=integral,
    )


magnitude.__doc__ = _networks.magnitude.__doc__

__doc__ = _networks.__doc__
__all__ = [name for name in getattr(_networks, "__all__", []) if not name.startswith("_")] + ["MagnitudeHomology"]
