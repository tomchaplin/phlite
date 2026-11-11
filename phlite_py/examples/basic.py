"""Persistent homology of a small filtered complex: a hollow tetrahedron (a 2-sphere)."""

import phlite

# PHAT/LoPHAT-style (dimension, boundary) pairs; cells enter in index order
boundaries = [
    (0, []),
    (0, []),
    (0, []),
    (0, []),
    (1, [0, 1]),
    (1, [0, 2]),
    (1, [1, 2]),
    (1, [0, 3]),
    (1, [1, 3]),
    (1, [2, 3]),
    (2, [4, 7, 8]),
    (2, [5, 7, 9]),
    (2, [6, 8, 9]),
    (2, [4, 5, 6]),
]
chain = phlite.ChainComplex.from_columns(boundaries)
barcode = chain.filtration().persistence(representatives=True)
print(chain)
print(barcode)
print("pairings:", barcode.pairings)
print("essential:", barcode.essential)
print("representatives:", {bar.birth: bar.representative for bar in barcode.bars})

# The same chain complex with real filtration values: diagrams are in terms of these
values = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
barcode = chain.filtration(values).persistence()
for dim, diagram in enumerate(barcode.diagrams()):
    print(f"H_{dim}:", diagram.tolist())
