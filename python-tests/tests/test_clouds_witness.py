"""phlite.clouds.witness and witness_complex against GUDHI's EuclideanWitnessComplex and
EuclideanStrongWitnessComplex: the same simplices with the same values (α²), and the same barcodes."""

import math

import numpy as np
import pytest

from phlite.clouds import witness, witness_complex

gudhi = pytest.importorskip("gudhi")


def reference(landmarks, witnesses, strong, max_alpha_square, dimension):
    cls = gudhi.EuclideanStrongWitnessComplex if strong else gudhi.EuclideanWitnessComplex
    tree = cls(landmarks=landmarks, witnesses=witnesses).create_simplex_tree(
        max_alpha_square=max_alpha_square, limit_dimension=dimension
    )
    return tree


@pytest.mark.parametrize("strong", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_against_gudhi(strong, dim):
    rng = np.random.default_rng(dim + 10 * strong)
    for trial in range(4):
        points = rng.random((80, dim))
        landmarks = points[rng.choice(80, 10, replace=False)]
        for bound in (0.03, math.inf):
            tree = reference(landmarks, points, strong, bound, 2)
            theirs = {tuple(s): v for s, v in tree.get_filtration()}
            ours = dict(witness_complex(landmarks, points, 2, strong=strong, max_alpha_square=None if bound == math.inf else bound))
            assert ours.keys() == theirs.keys(), (trial, bound)
            for s, v in ours.items():
                assert v == pytest.approx(theirs[s], abs=1e-12)
            # Barcodes (over Z/2), dimensions 0 and 1
            tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
            barcode = witness(landmarks, points, 1, strong=strong, max_alpha_square=None if bound == math.inf else bound)
            for k in range(2):
                mine = sorted([tuple(p) for p in barcode.pairings[k]] + [(b, math.inf) for b in barcode.essential[k]])
                gudhi_bars = sorted(
                    (float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(k) if d > b
                )
                assert len(mine) == len(gudhi_bars)
                for (b1, d1), (b2, d2) in zip(mine, gudhi_bars):
                    assert b1 == pytest.approx(b2, abs=1e-12) and (d1 == d2 or d1 == pytest.approx(d2, abs=1e-12))


def test_circle_landmarks():
    # Landmarks round a circle, witnesses densely on it: one long H_1 bar
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    points = np.c_[np.cos(t), np.sin(t)]
    barcode = witness(points[::20], points, 1)
    assert len(barcode.essential[1]) + sum(1 for b, d in barcode.pairings[1] if d - b > 0.1) == 1
