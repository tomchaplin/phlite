"""Zigzags of point clouds (`phlite.clouds.cloud_zigzag`), networks (`phlite.networks.network_zigzag`) and videos
joined by intersections, built in Rust on `phlite_complexes::zigzag::ComplexZigzag`, against the same complexes built
in Python and pushed through `phlite.complexes.Zigzag.from_complexes` (itself checked against Betti numbers and map
ranks in test_complexes_zigzag.py)."""

import itertools

import numpy as np
import pytest

from phlite.complexes import Zigzag

clouds = pytest.importorskip("phlite.clouds")
networks = pytest.importorskip("phlite.networks")


def noisy_circle(n, seed=0, turns=2):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, turns * 2 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.standard_normal((n, 2))


def relabel(simplices, labels):
    return [tuple(sorted(labels[v] for v in s)) for s in simplices]


def python_zigzag(frames, complex_of, join):
    """from_complexes on each frame's complex (on its labels), cut to dimension <= 1 bars."""
    ks = [relabel(complex_of(points), labels) for labels, points in frames]
    return Zigzag.from_complexes(ks, join=join).barcode(levels=True)


def same(rust, python, max_dim=1):
    assert sorted((b.dimension, b.birth, b.death) for b in rust) == sorted(
        (b.dimension, b.birth, b.death) for b in python if b.dimension <= max_dim
    )
    assert rust.n_positions == python.n_positions


@pytest.mark.parametrize("join", ["union", "intersection"])
@pytest.mark.parametrize("filtration", ["rips", "cech", "alpha", "delaunay_cech"])
def test_cloud_filtrations_against_python(filtration, join):
    points = noisy_circle(30, seed=1)
    frames = clouds.sliding_windows(points, 12, step=3)
    t = {"rips": 0.6, "cech": 0.3, "alpha": 0.3, "delaunay_cech": 0.3}[filtration]

    def complex_of(pts):
        if filtration == "rips":
            listed = clouds.rips_complex(points=pts, dimension=2, threshold=t)
        elif filtration == "cech":
            listed = clouds.cech_complex(pts, 2, max_radius=t)
        elif filtration == "alpha":
            listed = clouds.alpha_complex(pts)
        else:
            listed = clouds.delaunay_cech_complex(pts)
        return [s for s, v in listed if v <= t and len(s) <= 3]

    rust = clouds.cloud_zigzag(frames, t, 1, filtration=filtration, join=join)
    same(rust, python_zigzag(frames, complex_of, join))


def test_cloud_forms():
    """Moving points (arrays), (labels, points) pairs and a distance matrix on the labels agree."""
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points = np.c_[np.cos(t), np.sin(t)]  # neighbours 0.31 apart, next-but-one 0.62
    rng = np.random.default_rng(3)
    moving = [points + 0.02 * rng.standard_normal(points.shape) for _ in range(4)]
    a = clouds.cloud_zigzag(moving, 0.45)
    b = clouds.cloud_zigzag([(list(range(20)), m) for m in moving], 0.45)
    assert a == b
    # The circle's loop persists throughout
    assert all(a.rank(level, 1) == 1 for level in range(a.n_positions))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    windows = clouds.sliding_windows(points, 8, 2)
    assert clouds.cloud_zigzag([labels for labels, _ in windows], 0.5, distances=distances) == clouds.cloud_zigzag(
        windows, 0.5
    )
    # Witness complexes run (landmarks each frame's points)
    w = clouds.cloud_zigzag(windows, 0.1, filtration="witness", witnesses=points)
    assert w.n_positions == 2 * len(windows) - 1
    with pytest.raises(ValueError):
        clouds.cloud_zigzag(windows, 0.5, filtration="witness")
    with pytest.raises(ValueError):
        clouds.cloud_zigzag(windows, 0.5, join="sideways")
    assert len(clouds.cloud_zigzag([], 0.5)) == 0


def cliques(nodes, edges, top=3):
    adj = {frozenset(e) for e in edges if e[0] != e[1]}
    nodes = sorted(set(nodes) | {v for e in edges for v in e})
    out = []
    for k in range(1, top + 1):
        for c in itertools.combinations(nodes, k):
            if all(frozenset(p) in adj for p in itertools.combinations(c, 2)):
                out.append(c)
    return out


@pytest.mark.parametrize("join", ["union", "intersection"])
def test_network_cliques_against_python(join):
    rng = np.random.default_rng(4)
    snapshots = []
    for _ in range(5):
        edges = [(int(a), int(b)) for a, b in itertools.combinations(range(8), 2) if rng.random() < 0.35]
        snapshots.append((list(range(8)), edges))
    rust = networks.network_zigzag(snapshots, 1, join=join)
    ks = [cliques(n, e) for n, e in snapshots]
    same(rust, Zigzag.from_complexes(ks, join=join).barcode(levels=True))


def test_network_directed_flag_and_inputs():
    cycle = [("a", "b"), ("b", "c"), ("c", "a")]
    transitive = [("a", "b"), ("b", "c"), ("a", "c")]
    # A directed 3-cycle has no directed 2-clique: a loop; a transitive triangle is filled
    zz = networks.network_zigzag([cycle, transitive], complex="directed_flag")
    assert zz.rank(0, 1) == 1 and zz.rank(2, 1) == 0
    # As clique complexes both are filled triangles
    assert networks.network_zigzag([cycle, transitive]).rank(0, 1) == 0
    nx = pytest.importorskip("networkx")
    g = nx.DiGraph(cycle)
    h = nx.DiGraph(transitive)
    assert networks.network_zigzag([g, h], complex="directed_flag") == zz
    assert networks.network_zigzag([(["a", "b", "c"], cycle), (["a", "b", "c"], transitive)], complex="directed_flag") == zz


def test_video_intersections():
    """Joined by intersections, a drifting blob still keeps one bar (consecutive frames overlap), and the frames'
    levels have the same Betti numbers as joined by unions."""
    images = pytest.importorskip("phlite.images")
    frames = np.ones((6, 12, 12))
    for i in range(6):
        frames[i, 4:8, i : i + 4] = 0.0
    zz = images.video_zigzag(frames, 0.5, join="intersection")
    assert [tuple(b) for b in zz] == [(0, 0, 10, True, True)]
    rng = np.random.default_rng(5)
    video = rng.random((5, 9, 9))
    u = images.video_zigzag(video, 0.4)
    n = images.video_zigzag(video, 0.4, join="intersection")
    for level in range(0, 9, 2):
        for k in range(2):
            assert u.rank(level, k) == n.rank(level, k)
