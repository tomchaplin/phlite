"""Structured inputs for the benchmarks, with long bars in their persistent homology, unlike uniformly random ones.

Point clouds (for Rips), from [tadasets](https://github.com/scikit-tda/tadasets) where it has the shape:

    torus     a torus in R^3 (tube radius 1 around a circle of radius 2): two long H1 bars and one long H2 bar
    sphere    a 2-sphere in R^3: one long H2 bar
    circles   a wedge of three circles in the plane, meeting at the origin: three long H1 bars
    spheres   a wedge of two 2-spheres in R^3, touching at a point: two long H2 bars

each sampled uniformly (by area, so evenly over the shape) with Gaussian noise. Every function takes a seed, so the
inputs are reproducible.
"""

import numpy as np
import tadasets

# Standard deviation of the noise added to every coordinate, relative to the shapes' radius 1
NOISE = 0.05


def torus(n, seed):
    return tadasets.torus(n=n, c=2, a=1, noise=NOISE, seed=seed, uniform=True)


def sphere(n, seed):
    return tadasets.dsphere(n=n, d=2, r=1, noise=NOISE, seed=seed)


def circles(n, seed, k=3):
    """A wedge of `k` unit circles in the plane, all through the origin, at evenly spaced angles."""
    parts = []
    for i, size in enumerate(_split(n, k)):
        angle = 2 * np.pi * i / k
        centre = np.array([np.cos(angle), np.sin(angle)])
        parts.append(tadasets.dsphere(n=size, d=1, r=1, noise=NOISE, seed=seed + i) + centre)
    return np.concatenate(parts)


def spheres(n, seed):
    """A wedge of two unit 2-spheres in R^3, touching at the origin."""
    a, b = _split(n, 2)
    left = tadasets.dsphere(n=a, d=2, r=1, noise=NOISE, seed=seed) + np.array([-1.0, 0, 0])
    right = tadasets.dsphere(n=b, d=2, r=1, noise=NOISE, seed=seed + 1) + np.array([1.0, 0, 0])
    return np.concatenate([left, right])


SHAPES = {"torus": torus, "sphere": sphere, "circles": circles, "spheres": spheres}


def points(shape, n, seed=0):
    """`n` points sampled from `shape` (see the module docs), as an `(n, dimension)` array."""
    return np.asarray(SHAPES[shape](n, seed), dtype=np.float64)


def distance_matrix(points):
    """Euclidean distances, rounded to float32 so that every implementation sees the same values."""
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return d.astype(np.float32).astype(np.float64)


def _split(n, k):
    """`n` split into `k` nearly equal parts."""
    return [n // k + (i < n % k) for i in range(k)]
