# phlite_networks

Persistent homology of networks (weighted graphs and digraphs), in Rust and Python (`pip install phlite-networks`,
or `pip install phlite` for every package). Part of [phlite](https://github.com/tomchaplin/phlite), which brings the
techniques of Ripser and its relatives to any filtration; see [the book](https://phlite.tomchaplin.xyz/book/) and
the [in-browser playground](https://phlite.tomchaplin.xyz/playground/), where you can draw a network and see each of
these barcodes.

```python
import networkx as nx
from phlite.networks import grpph, pph, directed_flag, flag

G = nx.DiGraph()
G.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "a", 1.0)])

barcode = grpph(G, representatives=True)   # grounded persistent path homology
barcode = pph(G, max_dim=2)                # persistent path homology
barcode = directed_flag(G, max_dim=2)      # the directed flag complex, as flagser
barcode.pairing_representatives[1]         # a cycle for each finite H1 bar, in your node labels
```

Every function takes a networkx graph or `(number of nodes, edge list)`, and `algorithm=`, `method=` and
`representatives=`; representatives come back in your node labels.

## Filtrations

| Function | What | Book |
|---|---|---|
| `grpph` | grounded persistent path homology [[1]](#1): a stable descriptor of weighted digraphs | [GrPPH](https://phlite.tomchaplin.xyz/book/python/networks/grpph.html) |
| `pph` | persistent path homology (Chowdhury and Mémoli [[2]](#2)) in any dimension, by edge weight or shortest paths | [PPH](https://phlite.tomchaplin.xyz/book/python/networks/pph.html) |
| `directed_flag` | the directed flag complex [[3]](#3), grounded or not, as flagser computes it | [Directed flag](https://phlite.tomchaplin.xyz/book/python/networks/directed_flag.html) |
| `ordered_tuples` | directed Rips: ordered tuples of nodes | [Ordered tuples](https://phlite.tomchaplin.xyz/book/python/networks/ordered_tuples.html) |
| `dowker` | Dowker complexes (sink or source) | [Dowker](https://phlite.tomchaplin.xyz/book/python/networks/dowker.html) |
| `flag` | flag (clique) complexes of undirected graphs, by edge weight, node values or shortest paths | [Flag](https://phlite.tomchaplin.xyz/book/python/networks/flag.html) |
| `magnitude` | magnitude homology (Hepworth and Willerton [[4]](#4)) over Z/p or Z, with its torsion | [Magnitude](https://phlite.tomchaplin.xyz/book/python/networks/magnitude.html) |
| `network_zigzag` | zigzag persistence of snapshots of a network (clique or directed flag complexes, joined by unions or intersections), by FastZigzag | [Zigzag](https://phlite.tomchaplin.xyz/book/python/zigzag.html) |
| `magnitude_path_spectral_sequence` | the magnitude-path spectral sequence (Asao; Hepworth and Roff [[5]](#5)), a page at a time | [Magnitude](https://phlite.tomchaplin.xyz/book/python/networks/magnitude.html) |

Extended persistence of a function on a graph's nodes is in `phlite.complexes`
([book](https://phlite.tomchaplin.xyz/book/python/extended.html)). In Rust, each is a module with a `*_barcode`
function; `phlite_networks_grpph` and `phlite_networks_pph` are command-line tools.

## Compared with other packages

* **[grpphati](https://github.com/tomchaplin/grpphati)** (GrPPH and PPH, with PHAT or LoPHAT as its backend): phlite is 24–105× faster on
  GrPPH and 50–95× faster on PPH, in a fraction of the memory (grpphati needs 7 GB for a ring of 100 nodes, phlite
  31 MB).
* **[flagser](https://github.com/luetge/flagser)** (through [pyflagser](https://github.com/giotto-ai/pyflagser)), the
  reference for directed flag complexes and the source of the deduplicating heap phlite offers as `'dynamic'` columns: phlite is 1.4–1.9× faster (50 s against 87 s on G(500, 0.05) up to dimension 2),
  but flagser uses 2–4× less memory.
* **[GUDHI](https://gudhi.inria.fr)** (flag complexes of graphs, by `SimplexTree`): phlite is 3–18× faster, by edge
  weight, node values or shortest paths.
* **[PathHomFast](https://github.com/WeilabMSU/PathHom)** and Chowdhury and Mémoli's
  [MATLAB code](https://github.com/samirchowdhury/pph-matlab) compute persistent path homology too: phlite is fuzzed
  against PathHomFast and was compared with the MATLAB code, and computes it in any dimension, by homology or
  cohomology.

* **[gramag](https://github.com/tomchaplin/gramag)** and **[Perseus](https://people.maths.ox.ac.uk/nanda/perseus/)**
  (magnitude homology): phlite is 2–100× faster than gramag (2–20× on the random digraphs in `BENCHMARKS.md`, more
  on larger graphs), and runs where gramag runs out of memory; level with
  Perseus in time (whose coreductions and elimination phlite borrows, with its sorted enumeration), with about 1.3×
  its memory. Every rank and torsion group agrees with both.

Ordered tuples and Dowker complexes have, as far as we know, no other implementation to compare with; they are tested
against brute force. All of the numbers are in [the benchmarks](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md).

## References

<a id="1">[1]</a>
Chaplin, Thomas, Heather A. Harrington, and Ulrike Tillmann. "Grounded persistent path homology: a stable, topological
descriptor for weighted digraphs." arXiv:2210.11274 (2022).

<a id="2">[2]</a>
Chowdhury, Samir, and Facundo Mémoli. "Persistent path homology of directed networks." Proceedings of the 29th Annual
ACM-SIAM Symposium on Discrete Algorithms (2018). arXiv:1701.00565.

<a id="3">[3]</a>
Lütgehetmann, Daniel, et al. "Computing persistent homology of directed flag complexes." Algorithms 13.1 (2020): 19.

<a id="4">[4]</a>
Hepworth, Richard, and Simon Willerton. "Categorifying the magnitude of a graph." Homology, Homotopy and Applications
19.2 (2017): 31-60.

<a id="5">[5]</a>
Hepworth, Richard, and Emily Roff. "Bigraded path homology and the magnitude-path spectral sequence."
arXiv:2404.06689 (2024).
