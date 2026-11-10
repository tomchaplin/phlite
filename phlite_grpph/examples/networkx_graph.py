# Compute GrPPH of a networkx digraph, with representatives given in terms of the node labels.

import networkx as nx
from phlite_grpph import grpph

G = nx.DiGraph()
G.add_weighted_edges_from(
    [
        ("home", "shop", 10.0),
        ("home", "park", 10.0),
        ("shop", "work", 10.0),
        ("park", "work", 10.0),
        ("home", "work", 20.0),
        ("home", "cafe", 5.0),
        ("cafe", "gym", 10.0),
        ("gym", "work", 5.0),
    ]
)

barcode = grpph(G, representatives=True)
print(barcode)
for (birth, death), cycle in zip(barcode.pairings[1], barcode.pairing_representatives[1]):
    print(f"H_1 bar ({birth}, {death}) represented by {cycle}")
