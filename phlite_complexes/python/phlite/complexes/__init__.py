# Re-exports the extension module `phlite.complexes._complexes` (built from phlite_complexes, see src/python), whose
# stubs are alongside. `phlite` itself is a namespace package shared by every phlite package, so has no __init__.py.
from . import _complexes
from ._complexes import *  # noqa: F403

__doc__ = _complexes.__doc__
__all__ = [name for name in _complexes.__all__ if not name.startswith("_")]


def _simplicial_images(cells, vertex_map):
    """For `FilteredMap.simplicial`: the image of each simplex under the map of vertices, as a chain `{simplex: sign}`
    (empty if two vertices go to the same one)."""
    out = []
    for cell in cells:
        image = [vertex_map[v] for v in cell]
        if len(set(image)) < len(image):
            out.append({})
            continue
        order = sorted(range(len(image)), key=image.__getitem__)
        # The sign of the sorting permutation: (-1)^inversions
        inversions = sum(1 for i in range(len(order)) for j in range(i + 1, len(order)) if order[i] > order[j])
        out.append({tuple(image[i] for i in order): -1 if inversions % 2 else 1})
    return out
