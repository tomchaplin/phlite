# Example notebooks

These are [marimo](https://marimo.io) notebooks, in `docs/notebooks/`, each published with its outputs as a page of its own (the links below). Each draws
the dataset, its barcode and representatives (with Altair), and ends with an animation of the filtration: the complex
growing on the left, and on the right the barcode with the bars alive at that moment highlighted. To run one yourself, and change it, open it in the environment with the
package installed:

```sh
cd python-tests
uv run --project latest --group docs marimo edit ../docs/notebooks/phlite_image.py   # phlite notebooks
uv run --group docs marimo edit ../docs/notebooks/rips_points.py                     # phlite_rips and phlite_grpph
```

## An image (`phlite`)

The cubical complex of a 64 × 64 image with three dark rings, built from Python tuples, with the lower-star filtration
from pixel values (`extend=True`). The representatives of the three longest H1 bars trace the rings.

**[Open the notebook →](../notebooks/phlite_image.html)**

## A Delaunay complex (`phlite`)

Two noisy circles, their Delaunay triangulation filtered by edge length, over Z/3: a few hundred simplices instead of
the Rips complex's millions, and the same two loops.

**[Open the notebook →](../notebooks/phlite_delaunay.html)**

## Vietoris–Rips (`phlite_rips`)

A noisy figure eight: its two loops, drawn from their representatives, and a few algorithm configurations timed.

**[Open the notebook →](../notebooks/rips_points.html)**

## A digraph (`phlite_grpph`)

Directed cycles of different lengths, two parallel paths and a bidirected edge: grounded persistent path homology tells
them apart, and each representative is highlighted on the graph.

**[Open the notebook →](../notebooks/grpph_digraph.html)**
