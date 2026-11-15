# Path homology (`phlite.networks.pph`)

```python
from phlite.networks import pph

barcode = pph(G)                                  # a networkx DiGraph, or pph(n_vertices, edges); H0 and H1
barcode = pph(G, max_dim=3)                       # any dimension
barcode = pph(G, filtration="shortest_path")      # as grpphati's PPH
barcode.pairings[2]                               # (birth, death) of the finite H2 bars
```

Persistent path homology (Chowdhury and Mémoli, *Persistent path homology of directed networks*): at time `t` the
digraph `G_t` has the edges entering by `t`, and `PH_p` is the persistent homology of its chain complexes of
∂-invariant paths `Ω_•(G_t)`. With `filtration='weight'` (the default, Chowdhury and Mémoli's) an edge enters at its
weight; with `filtration='shortest_path'` the pair `a -> b` enters at the length of the shortest directed path from
`a` to `b`, as in grpphati's `PPH` and in [GrPPH](grpph.md) (which is this without the grounding).

## How

`Ω_p` has no basis of paths, so the cells are the *regular* paths (no node repeated consecutively), each entering
once all its edges have (its *allow time*, `+∞` if never). phlite reduces the boundary matrix of the regular path
complex (irregular faces dropped), rows and columns by allow time. A face can be allowed *after* its path (deleting
`x_i` needs the shortcut edge `x_{i-1} -> x_{i+1}`), so a pair `(σ, τ)` gives the bar `[t(σ), max(t(τ), t(σ)))`: the
only change from ordinary persistence, and Chowdhury and Mémoli's own algorithm. Since the matrix is still the boundary
of a chain complex, all of phlite applies: clearing, cohomology (the default, and much faster: by cohomology the basis
also holds the never-allowed faces of allowed paths, which are rows of the boundary), apparent pairs and
representatives, which are sums of allowed regular paths (tuples of nodes).

A path is a single `u64`: its dimension, then the nodes in a mixed radix (the first node, then for each later node its
position among the nodes other than its predecessor), so appending a node is `code * (n - 1) + digit`, and the allowed
paths are enumerated a dimension at a time in parallel. The number of regular paths grows as `n (n - 1)^p`, so higher
dimensions get expensive quickly on dense digraphs. In dimensions 0 and 1 of the shortest-path filtration, GrPPH's
matrices without the grounding give the same bars 2–2.5× faster (they only need the 2-cells of path homology: double
edges, directed triangles and long squares), so are used instead.

The Rust implementation is `phlite_networks::pph` (`Network`, `pph_barcode`).

## Checked against

* `pph-matlab`, Chowdhury and Mémoli's MATLAB code, under Octave (`phlite_networks/matlab_compare/compare.py`): 0
  mismatches on 761 structured networks, 150 random ones and 80 with bars in dimension 2, for every method, algorithm
  and field, when the code was written (as the separate crate `phlite_pph`; not yet rerun since it moved here).
* [PathHomFast](https://github.com/WeilabMSU/PathHom) (pure Python, a test dependency pinned to a commit), which
  builds filtration-compatible bases of `Ω_q(G_t)` explicitly: an independent algorithm. Fuzzed in dimensions 0–2
  (0–3 on up to 4 nodes), both filtrations, and on the 80 networks with bars in dimension 2 from the pph-matlab
  comparison (`higher_dim_networks.txt`): no differences.
* A brute force computing the persistent Betti numbers from `Ω_p(G_t)` by linear algebra, which (unlike pph-matlab)
  handles missing edges: in `matlab_compare/sparse_check.py`, and fuzzed in `python-tests/tests/test_networks_pph.py`
  in dimensions 0–2, both filtrations.
* grpphati's `PPH` pipeline (shortest-path filtration, dimension 1; `test_networks_complexes.py`), and in Rust the
  GrPPH oracle without grounding (`phlite_networks/tests/pph_vs_grpph.rs`).
* Chowdhury and Mémoli's theorem on cycle networks (`Dgm_1 = {(1, ⌈n/2⌉)}`), their bi-fan and bi-parallel examples,
  the coboundary being the boundary's transpose, and every method, algorithm and field agreeing (`tests/pph.rs`).
* Invariants: reversing every edge, relabelling the nodes, doubling the weights; representatives are cycles of
  regular paths allowed by the bar's birth.
