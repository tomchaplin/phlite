"""Zigzag persistence of networks that change: `network_zigzag`.

```python
import networkx as nx
from phlite.networks import network_zigzag

# Contact networks day by day, each day's clique complex joined to the next's by their union
bars = network_zigzag([G_monday, G_tuesday, G_wednesday], max_dim=1)
# Digraphs: directed flag complexes (directed cliques), joined by intersections
bars = network_zigzag(digraphs, complex="directed_flag", join="intersection")
```
"""

from __future__ import annotations

from typing import Literal

from phlite.core import ZigzagBar, ZigzagBarcode

from ._networks import _network_zigzag

__all__ = ["network_zigzag"]


def _snapshot(graph):
    """`(nodes, edges)` of a networkx graph, a `(nodes, edges)` tuple, or a sequence of edges."""
    if hasattr(graph, "nodes") and hasattr(graph, "edges"):
        return list(graph.nodes), [(a, b) for a, b, *_ in graph.edges]
    if isinstance(graph, tuple) and len(graph) == 2:
        nodes, edges = graph
        return list(nodes), [(e[0], e[1]) for e in edges]
    return [], [(e[0], e[1]) for e in graph]


def network_zigzag(
    snapshots,
    max_dim: int = 1,
    *,
    complex: Literal["clique", "directed_flag"] = "clique",
    join: Literal["union", "intersection"] = "union",
    algorithm=None,
) -> ZigzagBarcode:
    """The zigzag barcode of a sequence of networks on shared nodes, each one's complex joined to the next by their
    union (`K_0 -> K_0 ∪ K_1 <- K_1 -> ...`) or intersection (`K_0 <- K_0 ∩ K_1 -> K_1 <- ...`).

    A node or edge in several snapshots is the same cell in each (nodes are matched by label), so a component or loop
    persists while it overlaps itself from one snapshot to the next (with unions), or while what it shares with the
    next carries it (with intersections). Computed by FastZigzag (Dey and Hou).

    Path homology has no zigzag here: its chains (the ∂-invariant paths) have no basis of cells and don't respect
    unions, so FastZigzag doesn't apply; the directed flag complex is the nearest cellular relative.

    Args:
        snapshots: The networks, each a networkx `Graph`/`DiGraph`, a `(nodes, edges)` tuple, or a sequence of
            `(source, target)` edges (nodes any hashable labels; self-loops ignored).
        max_dim: The largest homology dimension.
        complex: `'clique'` (the clique complex of the undirected graph) or `'directed_flag'` (a simplex
            `(x_0, ..., x_k)` for each directed clique, `x_i -> x_j` for `i < j`, as flagser).
        join: `'union'` or `'intersection'`.
        algorithm: The reduction algorithm (a `phlite.core.Algorithm`, default `Algorithm.serial()`).
            Pipeline defaults (`'networks.network_zigzag'`): columns='pivot', cache=False, store_r=True.

    Returns:
        A `phlite.core.ZigzagBarcode` over levels: snapshot `i` is level `2 i` and the join of snapshots `i` and
        `i + 1` level `2 i + 1`, each interval the levels where its class is alive (both ends inclusive).
    """
    labels: dict = {}

    def label(v) -> int:
        return labels.setdefault(v, len(labels))

    parsed = []
    for graph in snapshots:
        nodes, edges = _snapshot(graph)
        parsed.append(([label(v) for v in nodes], [(label(a), label(b)) for a, b in edges]))
    if not parsed:
        return ZigzagBarcode([], 0)
    bars = _network_zigzag(parsed, max_dim, complex=complex, join=join, algorithm=algorithm)
    return ZigzagBarcode((ZigzagBar(d, b, e, True, True) for d, b, e in bars), 2 * len(parsed) - 1)
