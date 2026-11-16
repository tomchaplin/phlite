# Extended persistence

```python
from phlite.complexes import ChainComplex

# Simplices as tuples of vertices, faces before cofaces, with signed boundaries
chain = ChainComplex.from_cells(
    simplices,
    dimension=lambda s: len(s) - 1,
    boundary=lambda s: {s[:i] + s[i + 1:]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {},
)
filtered = chain.filtration(lambda s: height[s[0]] if len(s) == 1 else None, extend=True)  # lower-star of vertex values
barcode = filtered.extended_persistence()
barcode.ordinary, barcode.relative, barcode.extended_plus, barcode.extended_minus  # each a list by dimension
barcode.bars        # every bar, with the cells that give its birth and death
barcode.to_gudhi()  # as GUDHI's SimplexTree.extended_persistence returns it
```

Ordinary persistence of a filtration of a complex \\(K\\) leaves the homology of \\(K\\) itself **essential**: a
component of a mesh is born at its lowest point and never dies, and so is every loop. **Extended persistence**
(Cohen-Steiner, Edelsbrunner and Harer) goes up through the filtration, then back down through the pairs
\\((K, K^{\\ge a})\\), so that every class dies. Each bar is one of four kinds:

| Diagram | Born | Dies | Example |
|---|---|---|---|
| `ordinary` | going up | going up | a component that merges into an older one (as ordinary persistence) |
| `relative` | going down | going down | the same for the reverse filtration, one dimension up (dimensions start at 1, as GUDHI's) |
| `extended_plus` | going up | going down, higher | a component, from its lowest point to its highest |
| `extended_minus` | going up | going down, lower | a loop, from its highest point to its lowest |

The extended bars measure the whole complex: each loop's range of heights, say, which ordinary persistence can't see.

## The reverse filtration

Going down needs a second value \\(g\\) on each cell, no more than its faces' (\\(g(\\tau) \\ge g(\\sigma)\\) for a face
\\(\\tau\\) of \\(\\sigma\\)), so that each \\(K^{\\ge a} = \\{\\sigma : g(\\sigma) \\ge a\\}\\) is a subcomplex.
`phlite.complexes` takes \\(g(\\sigma) = \\min_{\\tau \\subseteq \\sigma} f(\\tau)\\), the largest subcomplex of each
superlevel set, which works for any filtration \\(f\\). For the classical setting, values on the vertices (heights on a
mesh, a function on a graph's nodes), filter the complex by the lower-star filtration of the values (each cell at its
highest vertex, as `filtration(..., extend=True)` makes it from values on the vertices alone); \\(g\\) is then the
upper-star filtration, each cell at its lowest vertex.

The computation is in the core, `phlite::extended`: it cones any filtered complex, given its boundary matrix, its
transpose and \\(g\\), without listing the cone, so any pipeline can offer extended persistence with whatever reverse
filtration is natural for it. `phlite.complexes` is the first.

### Graphs

Extended persistence of a function on a graph's nodes (degrees, centralities, heat kernel signatures: the standard
summaries of graphs in machine learning, as in PersLay) is that of the graph as a 1-dimensional complex:

```python
import networkx as nx
from phlite.complexes import ChainComplex

nodes = list(G.nodes)
cells = [(v,) for v in nodes] + [tuple(e) for e in G.edges]
chain = ChainComplex.from_cells(
    cells,
    dimension=lambda c: len(c) - 1,
    boundary=lambda c: {(c[0],): -1, (c[1],): 1} if len(c) == 2 else {},
)
degree = dict(G.degree)
barcode = chain.filtration(lambda c: degree[c[0]] if len(c) == 1 else None, extend=True).extended_persistence()
```

## How it's computed

By coning, as GUDHI does: a cone vertex \\(\\omega\\), then \\(K\\) in its filtration order, then the cone
\\(\\omega * \\sigma\\) on every cell in decreasing order of \\(g\\), with boundary
\\(\\partial(\\omega * \\sigma) = \\sigma - \\omega * \\partial\\sigma\\) (and \\(\\partial(\\omega * v) = v - \\omega\\) for
a vertex). The core wraps the complex's boundary and coboundary into the coned complex's, which any algorithm reduces
(by homology or cohomology, over any field), and reads each bar's kind off where its cells are. The cone vertex's
component, the only essential class, is left out, and so are bars of length zero (as GUDHI's, with its default
`min_persistence=0`). Values are exact: GUDHI rescales them into \\([-2, -1]\\) and \\([1, 2]\\), so its are within
rounding of these.

## Checked against

GUDHI's `extend_filtration` and `extended_persistence`, on random graphs, their flag complexes and random
2-complexes with vertex values (integers, so lots of ties, and floats), over \\(\\mathbb{Z}/2\\) and \\(\\mathbb{Z}/3\\), by
every method and algorithm; and, for filtrations that aren't lower-star, which GUDHI can't do, against the cone built
by hand and reduced by ordinary persistence (`python-tests/tests/test_extended_persistence.py`). The core's coned
matrices pass the [contract](../design/contract.md) checker.
