# Path homology (`phlite.networks.pph`)

```python
from phlite.networks import pph

barcode = pph(G)                      # a networkx DiGraph, or pph(n_vertices, edges)
barcode.pairings[1]                   # (birth, death) of the finite H1 bars
```

Persistent path homology (Chowdhury and Mémoli) in dimensions 0 and 1: the regular path homology of the digraph in
which `a -> b` enters at the length of the shortest directed path from `a` to `b`, as grpphati's `PPH`. It is
[GrPPH](grpph.md) without the grounding, computed by the same matrices with no edge present from the start: so its H1
bars are born when their cycles appear in the shortest-path digraph, and needn't be cycles of the digraph itself. The
2-cells are those of path homology: double edges `a -> b -> a`, directed triangles, and long squares.

It takes the same arguments as `grpph`, and the same defaults (pipeline `networks.pph`).

Checked against grpphati's `PPH` pipeline (`python-tests/tests/test_networks_complexes.py`), with the invariants that
reversing every edge, relabelling the nodes and scaling the weights change it accordingly.
