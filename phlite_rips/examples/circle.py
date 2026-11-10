# Persistent homology of points sampled from a noisy circle (plus a small cluster),
# with a representative cycle for the big loop.

import numpy as np
from phlite_rips import rips

rng = np.random.default_rng(0)
theta = rng.uniform(0, 2 * np.pi, 60)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
circle += rng.normal(scale=0.05, size=circle.shape)
cluster = rng.normal(loc=(3.0, 0.0), scale=0.05, size=(10, 2))
points = np.concatenate([circle, cluster])

distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

barcode = rips(distances, max_dim=1, representatives=True)
print(barcode)

# The longest H_1 bar is the circle
lengths = [death - birth for birth, death in barcode.pairings[1]]
longest = int(np.argmax(lengths))
print("Longest H_1 bar:", barcode.pairings[1][longest])
loop = barcode.pairing_representatives[1][longest]
print(f"Represented by a cycle of {len(loop)} edges, e.g. {loop[:3]}")

# Every point is in some edge of the loop at most twice (it's a cycle)
assert all(len(edge) == 2 for edge in loop)
degree = np.zeros(len(points), dtype=int)
for a, b in loop:
    degree[a] += 1
    degree[b] += 1
assert np.all(degree % 2 == 0)

# H_0: the circle and the cluster are separated by a gap of about 2
print("Longest finite H_0 bar:", max(barcode.pairings[0], key=lambda bar: bar[1]))
print("Essential H_0:", barcode.essential[0])
