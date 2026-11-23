# Magnitude homology

Magnitude homology (Hepworth and Willerton) categorifies the magnitude of a graph: for a digraph `G` with its
shortest-path distance, `MC_{k,l}` is free on the tuples `(x_0, ..., x_k)` of nodes, consecutive ones distinct, whose
distances add up to the *length* `l`, and the differential deletes inner nodes lying on a geodesic between their
neighbours. Its Euler characteristic in each length is the coefficient of `q^l` in the magnitude of `G`.

```python
import networkx as nx
from phlite.networks import magnitude, magnitude_path_spectral_sequence

G = nx.cycle_graph(6).to_directed()
mh = magnitude(G, l_max=6)            # over Z/2 by default
print(mh)                             # a table: k across, l down
mh.rank(2, 3)                         # rank of MH_{2,3}; mh[3][2] is the same ([l][k], a list of lists)
mh.magnitude_coefficients()           # sum_k (-1)^k rank MH_{k,l}

mh = magnitude(G, l_max=6, integral=True)
mh.group(3, 4)                        # e.g. "Z^2 ⊕ Z/2"
mh.torsion(3, 4)                      # {order: multiplicity}

magnitude(G, l_max=6, modulus=[2, 3, 251])                # several fields at once: a dict of results
magnitude(G, l_max=8, pairs=[(0, 3)], by_pair=True)       # only the tuples from 0 to 3 (much cheaper)
```

Graphs are networkx `DiGraph`s (undirected ones through `.to_directed()`) or `(n_vertices, edges)`; edge weights are
ignored (every edge has length 1). `magnitude` returns a `MagnitudeHomology` (a dict of them with `by_pair` or several
moduli), with `rank`, `group`, `torsion`, `magnitude_coefficients`, `to_array` and `table`.

## The magnitude-path spectral sequence

Filtering the reachability chain complex (tuples with a directed path from each node to the next) by length gives a
spectral sequence (Asao; Hepworth and Roff) whose `E^1` is magnitude homology and whose `E^2` is *bigraded path
homology*, with the path homology of the digraph on its diagonal, converging to reachability homology.
`magnitude_path_spectral_sequence(G, l_max)` is a `phlite.core.SpectralSequence` (see
[spectral sequences](../spectral.md)): pages one at a time, groups, generators (chains of tuples of nodes) and
differentials, printed as gramag prints them. Over a field its pages are read off a barcode; with `integral=True`
each entry is computed exactly over Z. Entries `E^r_{k,l}` with `l + r > l_max + 1` aren't determined by the tuples
listed, and are `None`.

## How

The magnitude complex splits into one piece for each pair `(s, t)` of first and last nodes, and phlite works piece by
piece, in parallel:

1. **Enumeration** from each source by depth-first search in increasing order, so each piece's tuples of each size are
   sorted and faces are found by binary search (a technique of **Perseus**). With chosen `pairs`, a tuple is only
   extended while a target is within reach of the length left.
2. **Collapse**: free faces and coreductions without fill-in, then Markowitz elimination of unit pairs (Perseus's
   second phase). Both keep the integral homology; processing the free pairs in queue order removes nearly every cell.
3. What is left, typically a few hundred cells, is reduced over each field asked for, or put in Smith normal form for
   the integral groups.

The Rust implementation is `phlite_networks::magnitude` (and `phlite_networks::mpss` for the spectral sequence).

## Checked against

* **Brute force** (the complex written out and reduced), **Perseus**'s integral groups (by universal coefficients and
  exactly), complete graphs (diagonal), cycles, and the Z/2 in `MH_{3,4}` of Kaneta and Yoshinaga's RP² graph.
* **gramag** and **Perseus 6** in the benchmarks below: every rank and torsion group agrees.
* For the spectral sequence: `E^1` is `magnitude`, the `E^2` diagonal is path homology (`phlite.networks.pph` with unit
  weights), and Hepworth and Roff's theorems on directed and bi-directed cycles, page by page.

`python-tests/tests/test_networks_magnitude.py`, and Rust tests in `phlite_networks`.

## Performance

Fresh processes, 4 threads each:

| graph | l ≤ | phlite | Perseus 6 | gramag 0.4.1 |
|---|---|---:|---:|---:|
| C_12 (all pairs) | 10 | 0.10 s | 0.08 s | 0.95 s / 143 MB |
| grid 4×4 | 6 | 0.03 s | 0.01 s | 0.15 s |
| G(20, 0.2) | 5 | 0.04 s | 0.02 s | 0.36 s |
| Kaneta–Yoshinaga RP², 15 vertices, pair (0̂, 1̂) | 8 | 0.06 s | 0.03 s | 7.47 s / 1324 MB |
| Kaneta–Yoshinaga RP³, pair (0̂, 1̂) | 7 | 0.05 s | 0.04 s | out of memory |

On large single pieces (the 20 million tuples of the 33-vertex RP² graph's `(0̂, 1̂)` piece to length 10) phlite and
Perseus take the same time (16 s), phlite with about 1.3× the memory. phlite is 8–100× faster than gramag, and runs
where gramag runs out of memory.

## Searching for torsion

Torsion in magnitude homology is known to exist (Kaneta and Yoshinaga; Sazdanovic and Summers) but is provably absent
in degrees `k <= 2` and on the diagonal. `phlite_networks/examples/magnitude_torsion_search.py` searches families of
graphs and digraphs over Z: so far, no connected graph on at most 9 nodes and no digraph on at most 6 nodes has
torsion in `MH_{3,4}`, `MH_{3,5}` or `MH_{4,5}`, and Kaneta and Yoshinaga's 15-node RP² graph is locally minimal (no
node or edge can be removed keeping torsion).
