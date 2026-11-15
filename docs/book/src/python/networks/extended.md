# Extended persistence (`phlite.networks.extended`)

```python
from phlite.networks import extended

barcode = extended(G, node_weight="height")         # of the graph
barcode = extended(G, node_weight="height", dim=2)  # of its flag complex, up to triangles
barcode.ordinary[0], barcode.relative[1], barcode.extended_plus[0], barcode.extended_minus[1]
barcode.to_gudhi()   # as GUDHI's SimplexTree.extended_persistence returns it
```

Ordinary persistence of the lower-star filtration of a function \\(f\\) on the nodes leaves the homology of the whole
graph **essential**: a component is born at its minimum and never dies, and so is a loop. **Extended persistence**
(Cohen-Steiner, Edelsbrunner and Harer) goes up through the sublevel sets \\(K_{f \\le a}\\), then back down through
the pairs \\((K, K_{f \\ge a})\\), so that every class dies. Each bar is one of four kinds:

| Diagram | Born | Dies | For a graph |
|---|---|---|---|
| `ordinary` | going up | going up | a component that merges into an older one (as ordinary persistence) |
| `relative` | going down | going down | the same for \\(-f\\), one dimension up (dimensions start at 1, as GUDHI's) |
| `extended_plus` | going up | going down, higher | a component, from its minimum to its maximum |
| `extended_minus` | going up | going down, lower | a loop, from its highest point to its lowest |

So the extended bars measure the whole graph's features: each loop's height range, which ordinary persistence can't
see. This makes them popular summaries of graphs in machine learning (PersLay, for example).

The complex is the graph itself, or with `dim=2` and up its flag complex (every clique of at most `dim + 1` nodes).
Edge weights aren't used, only the node values. Each diagram is a list, by dimension, of sorted `(birth, death)`
pairs. Bars of length zero are left out, as they are by GUDHI with its default `min_persistence=0`.

## How it's computed

As GUDHI does, by **coning**: the complex \\(K\\), then a cone vertex \\(\\omega\\), and the cone \\(\\omega * \\sigma\\)
on every simplex \\(\\sigma\\), with boundary \\(\\sigma - \\omega * \\partial\\sigma\\). The cells enter in this order:

1. the cone vertex;
2. \\(K\\), by increasing \\(\\max f\\) (the lower-star filtration);
3. the cones, by decreasing \\(\\min f\\) (the upper-star filtration).

This explicit complex is reduced by `phlite_complexes` (with its defaults, `'networks.extended'`: pivot columns and
storing R), over \\(\\mathbb{Z}/p\\) for `modulus` 2, 3, 5, 7, 11 or 13. Each bar's kind follows from where its birth
and death cells are. The cone vertex's component is the only essential class, and is left out.

GUDHI rescales the values to put the two halves in \\([-2, -1]\\) and \\([1, 2]\\), so its values are within rounding
of the exact ones phlite reports.

## Checked against

GUDHI's `SimplexTree.extend_filtration` and `extended_persistence` on random graphs and their flag complexes (integer
values, so lots of ties, and floats), over \\(\\mathbb{Z}/2\\) and \\(\\mathbb{Z}/3\\), for every method and
algorithm. It is also checked against invariants: each component is an Ext⁺ bar from its minimum to its maximum, and
there are no more loops than the graph's cycle rank (`python-tests/tests/test_networks_extended.py`).
