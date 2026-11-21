# Perseus-inspired performance work: log

Branch `perseus-techniques` (off `main` at 4015166), started 2026-11-21, worked unattended. Local commits only, not pushed.

The plan (from comparing phlite 0.2.0 with Perseus 6, commit cbd0017): port the techniques that generalise.

| Phase | Technique | Status |
|---|---|---|
| 0 | Perseus as a benchmark reference | dependency added (14e5664); suites below |
| 1 | Union-find for incidence dimensions (H0, dual top dimension) | done for `phlite_complexes` (493f996); alpha below |
| 2a | Edge collapse for flag filtrations (Rips, flag) | see below |
| 2b | Active H1 enumeration (RedZeD) | see below |
| 3 | Morse pre-reduction | see below |
| 4 | Small experiments | see below |

Timings are on the 8-core i7-1185G7 laptop, nothing else running, best of N.

## Phase 0: Perseus in the test environments (14e5664)

Both `python-tests` environments install `perseus-tda` from `viditnanda/perseus@cbd0017` (subdirectory `python`),
since PyPI's 6.0.0 predates its current engines. It builds from source (C++17, ~3 minutes).

## Phase 1: union-find for incidence dimensions (493f996)

**What.** `phlite::union_find::IncidenceReduction`: reduce, one at a time and in reduction order, columns with at most
two entries. It is a union-find whose nodes carry a *gain* (a linear functional relative to the component's root), so it
gives exactly the pivots of the reduction **over any field**: a column with one entry, or a cycle of columns that are
independent (which happens off Z/2, e.g. the dual graph of a Klein bottle or RP² over Z/3), *grounds* its component
(the virtual "outside" node). Proptests check it against a hand-written reduction over Z/2 and Z/3 with arbitrary
signs (5000 cases each).

`Clearing::with_incidence_dimensions(dims)` reduces those dimensions with it instead of adding columns, keeping the
clearing bookkeeping (cleared columns are skipped; the pairs clear the next dimension; zero-length pairs are left out of
the diagram as apparent ones are). Only for `diagram()` (no V), and serial clearing only.

`phlite_complexes` (cohomology without representatives, Clearing or ParallelClearing):
* dimension 0 by union-find on the edges (whenever every 1-cell has at most two faces: any simplicial or cubical
  complex), with the killing edges left out of the basis (as Rips and cubical do by hand);
* the top dimension `d - 1` through `with_incidence_dimensions` when every `(d - 1)`-cell has at most two cofaces
  (grids, surface meshes, Delaunay triangulations), checked in the same pass that builds the basis.

**Why a config on Clearing rather than a trait on the matrix.** A trait method would have to be forwarded by every
adaptor (`with_basis`, `reverse`, `with_transpose`, `&M`, ...); the pipeline already knows which of its dimensions
qualify, and checks it in a pass it makes anyway. Dimension 0 by cohomology needs the transpose (the coboundary of a
vertex isn't an incidence column), so the pipeline does it with `IncidenceReduction` directly, which is a dozen lines.

**Tests.** New proptests in `phlite_complexes`: random pieces of Freudenthal triangulations of 2D and 3D grids,
optionally wrapped (tori) or flipped (Klein bottles), lower-star values with ties, over Z/2 and Z/3, every `max_dim`,
with Clearing (merged; store R), Standard and ParallelClearing, against the standard algorithm (1000+ cases); the dual
path applies in ~99% of them. Plus random simplicial complexes. All existing tests pass.

**Results** (`phlite_complexes/examples/reduce_file`, `bench_complexes` inputs dumped to files, best of 7, ms):

| case | before | after | speed-up |
|---|---:|---:|---:|
| grid 500 | 68.9 | 24.3 | 2.8× |
| grid 1000 | 426.3 | 125.7 | 3.4× |
| rings 500 | 46.0 | 19.7 | 2.3× |
| rings 1000 | 231.8 | 92.8 | 2.5× |
| torus 300 | 21.1 | 11.5 | 1.8× |
| torus 700 | 158.1 | 69.2 | 2.3× |
| rips 150 | 20.4 | 21.1 | 0.97× |
| rips 200 | 43.0 | 46.3 | 0.93× |
| cells 3000 | 2.5 | 2.2 | ≈ |
| dense 200 | 1.2 | 1.2 | ≈ |

On grid 1000 dimension 0 alone took 426 → 173 ms and the dual top dimension the rest (→ 126). The first version emitted
every zero-length pair (2M bars on grid 1000) and made extra passes over all cells, which cost 10–30% on Rips
complexes; leaving zero-length pairs out and reusing the one basis pass brought that to 3–8% (close to noise).

Alpha and Delaunay–Čech reduce through `phlite_complexes`, so they get it for free (every `(d - 1)`-simplex of a
Delaunay triangulation has at most two cofaces). `phlite_clouds/examples/alpha_timing.rs`, reduction only, best of 5:

| points | before | after | speed-up |
|---|---:|---:|---:|
| 100k in the plane | 201.5 ms | 52.0 ms | 3.9× |
| 1M in the plane | 1554.6 ms | 373.5 ms | 4.2× |
| 20k in space | 76.7 ms | 64.8 ms | 1.2× |
| 100k in space | 554.8 ms | 387.1 ms | 1.4× |

**Found on the way (not fixed): building alpha complexes is the real bottleneck.** 100k points in the plane take
6.5 s to triangulate and filter, and 1M take 209 s (superlinear), against 0.37 s to reduce. Worth profiling separately.

## Phase 2a: edge collapse (ddae894, then the Python option)

**What.** `phlite_clouds::rips::collapse::collapse_edges(distances, threshold)`: the backward shift-and-trim pass of
Glisse–Pritam, with Perseus's word-parallel tests. Edges are processed latest first, keeping two bitset adjacency
matrices: `now` (edges not yet processed, at their original times, which are all ≤ the current edge's) and `fin`
(every edge not removed). For an edge `uv`:
* common neighbours at its time = `now[u] & now[v]`; a dominator is a common neighbour adjacent to all the others.
  The last dominator found for an edge at `u` or `v` is tried first (one row), then each common neighbour with an
  early exit (a non-dominator fails within a word or two). The first version intersected over all common neighbours,
  which on 1000 random points took 660 ms of a 750 ms collapse; trying candidates one by one took it to 24 ms.
* if the dominator is already adjacent to every vertex that will *ever* be a common neighbour
  (`fin[u] & fin[v]`), the edge is dominated forever: removed ("trim");
* otherwise the vertices that join later are listed with their join times, and the edge moves to the first time it
  isn't dominated ("shift"), hopping from dominator to dominator.

Everything works in rank space (an edge's time is the rank of the edge whose time it took), with a dense rank table up
to 4096 points (64 MB) and a hash map beyond; the first version hashed every lookup and was 3–8× slower on the
structured shapes.

`rips_barcode_with(..., collapse)` uses it from `max_dim` 1, without representatives and without births (weighted
Rips, lower-star flag), and keeps the collapsed graph only if at least 10% of the edges went (Perseus's rule). It then
runs the usual sparse Rips on it. **Behaviour change:** bar values are identical but birth/death *simplices* come from
the collapsed filtration (so the pairing test now runs without collapse); `Algorithm.with_collapse(False)` gives the
original filtration's. Python: `Algorithm.with_collapse(None | bool)`, default on for `clouds.rips` and
`networks.flag_implicit`, refused elsewhere (registry, spec dict, repr, docstrings, book table, stubs, `test_core`).

Perseus uses collapse only from `max_dim` 2 and a special H1 engine below that. For phlite, collapse also wins at
`max_dim` 1: the reduction after collapsing is negligible, so the time is the collapse itself.

**Tests.** Proptest (4000 cases): random clouds of up to 30 points in the cube, lengths rounded to 3 or 10 levels
(heavy ties) or not, `max_dim` 1–3, thresholds 0.3 / 0.6 / ∞: the Rips barcode of the collapsed graph equals the
original's (as sorted (dim, birth, death) with zero-length bars dropped). Plus a circle. Existing Rips and flag tests
pass.

**Results** (`phlite_clouds/examples/rips_collapse.rs`, distance matrices of the `bench_clouds_rips` inputs,
`phlite.clouds.rips`' defaults: serial clearing, heap columns, union-find; best of 3 except the 1000-point cases):

| case | edges before → after | no collapse | collapse | speed-up |
|---|---|---:|---:|---:|
| cube n=400 dim≤2 | 59,144 → 3,326 | 548 ms | 19 ms | 29× |
| torus n=400 dim≤2 | 53,907 → 8,114 | 529 ms | 65 ms | 8.1× |
| sphere n=400 dim≤2 | 74,011 → 20,873 | 7023 ms | 467 ms | 15× |
| spheres n=400 dim≤2 | 54,024 → 17,343 | 2496 ms | 294 ms | 8.5× |
| cube n=1000 dim≤2 | 395,271 → 10,655 | 9397 ms | 126 ms | 75× |
| torus n=1000 dim≤2 | 352,328 → 47,103 | 8897 ms | 949 ms | 9.4× |
| circles n=1000 dim≤1 | 345,798 → 6,016 | 1293 ms | 79 ms | 16× |
| cube n=1000 dim≤1 | 395,271 → 10,655 | 186 ms | 90 ms | 2.1× |
| circles n=2000 dim≤1 | 1,397,636 → 14,141 | 15736 ms | 531 ms | 30× |
| cube n=2000 dim≤1 | 1,541,461 → 26,845 | 1034 ms | 542 ms | 1.9× |

For comparison, Perseus 6 (F2, measured this morning): cube 1000 dim≤2 0.12 s, torus 1000 dim≤2 0.40 s, sphere 400
0.30 s, circles 2000 dim≤1 1.49 s, cube 2000 dim≤1 0.50 s. So phlite now matches or beats it on these except the
torus and sphere, where the remaining time is the collapse's scans (most edges there are shifted, not removed).

Collapse alone on 2000 cube points (1.5M edges) is ~440 ms: ~60 ms sorting edges, ~20 ms building the rank table,
the rest the main loop (~280 ns per edge).

### Repeated collapse passes

A second pass over the collapsed (sparse) graph is cheap (17–34 ms) and removes a lot more on structured shapes (an
edge's dominator can appear only after other edges moved): torus 1000 47,103 → 24,415 edges. `rips_barcode_with` now
repeats passes while each removes at least a tenth of what is left. Best of 3, end to end:

| case | one pass | repeated | Perseus 6 (F2) |
|---|---:|---:|---:|
| cube n=400 dim≤2 | 19 ms | 14 ms | 22 ms |
| torus n=400 dim≤2 | 65 ms | 35 ms | 40 ms |
| sphere n=400 dim≤2 | 467 ms | 252 ms | 299 ms |
| spheres n=400 dim≤2 | 294 ms | 189 ms | 186 ms |
| cube n=1000 dim≤2 | 126 ms | 98 ms | 123 ms |
| torus n=1000 dim≤2 | 949 ms | 404 ms | 395 ms |
| circles n=1000 dim≤1 | 79 ms | 77 ms | 174 ms |
| circles n=2000 dim≤1 | 531 ms | 463 ms | 1494 ms |
| cube n=2000 dim≤1 | 542 ms | 482 ms | 497 ms |

### Which filtrations collapse applies to (Tom asked)

Edge collapse is a property of *flag* filtrations (a flag complex is the clique complex of its graph, so a dominated
edge can go without changing the homotopy type). It works through the `Distances` trait, so every flag pipeline gets
it: Rips (dense, sparse, from points) and `phlite.networks.flag` (implicit), and repeating passes is valid for the
same reason. Not yet with vertex births (weighted / DTM Rips, lower-star flag from node values): the published
proofs assume every vertex appears at 0, so it is skipped there (an extension looks plausible, unchecked). It does
not apply to non-flag complexes (alpha, Delaunay–Čech, cubical, explicit) or directed flag complexes. The general
analogue there is collapsing within ties (Morse), tried below without a gain.

## Negative results

* **Ordering ties by a discrete gradient** (`Ties::Morse` for `phlite_complexes`: within each run of equal values,
  the reverse of a greedy collapse, so matched pairs become apparent pairs). Correct (proptests on grids and random
  complexes over Z/2 and Z/3) and it pairs more cells (3D smoothed noise: 11,215 → 6,276 bars recorded), but the
  barcode took the same time on 3D cubical and Freudenthal grids, the 2D grid, the torus mesh and Rips complexes, and
  `filter` twice as long. With union-find at both ends the middle reductions are already cheap (16–35 ms for 0.5–0.65M
  cells). Not committed; the patch is `perf-results/morse-ties-experiment.patch`.
* **3D alpha construction** (100k points, 2.9M simplices, 3.9 s against 0.31 s to reduce): triangulation 0.59 s
  (axiolid already Hilbert-sorts), faces 0.40 s, facet lookups 0.61 s, alpha values 1.35 s, building the filtered
  complex (validation, sort, transpose) 0.96 s. Spread out, so left alone.

# Part 2: magnitude homology, its torsion, and the magnitude-path spectral sequence

Tom's follow-up requests (2026-11-21, mid-morning): implement directed graph magnitude homology in phlite, at least
as fast as gramag; look at the literature on torsion and make a search possible; then the magnitude-path spectral
sequence (over fields now, integers later).

## Magnitude homology: `phlite.networks.magnitude` (e294b58, fac6b6f, 10d0012)

`MC_{k,l}` is free on tuples `(x_0, ..., x_k)`, `x_i ≠ x_{i+1}`, of total (shortest-path) length `l`; the
differential deletes inner vertices on a geodesic. It splits into pieces `MC_{*,l}(s, t)`.

* **Enumeration** per source by depth-first search in increasing order, so each piece's tuples of each size are
  sorted and faces are found by binary search (Perseus's trick). With chosen pairs, a tuple is only extended while a
  target is within reach of the length left (gramag can't restrict its search). Parallel over sources, or, with few
  sources, over each source's second vertex (concatenating subtrees in order keeps them sorted) and its pieces.
* **Collapse** (fill-in-free: free faces and coreductions) on compressed rows with live counts, then **Markowitz
  elimination** of unit pairs over Z (Perseus's second phase), with a work list for the free pairs and a heap for
  the rest. Both keep the integral homology. **The order matters enormously:** collapsing with a stack (last in, first
  out) removed 20% of the 1.9M cells of the RP² piece below; a queue removes nearly all of them, taking the piece from
  5.6 s to 1.4 s.
* What is left (typically a few hundred cells) is reduced by `phlite_complexes` over each field asked for (one
  collapse serves several moduli), or put in **Smith normal form** for the integral groups (`integral=True`).

Python: `magnitude(G or (n, edges), l_max=, k_max=, pairs=, modulus=int | list, by_pair=, integral=)`.

Tests: brute force (Rust), Perseus's integral groups by universal coefficients and exactly (Hypothesis, 1000+
examples), complete graphs are diagonal, the 5-cycle, and the Z/2 in `MH_{3,4}` of Kaneta–Yoshinaga's RP² graph.

**Against gramag and Perseus** (fresh processes, 4 threads each; phlite through Python, so ~20 ms of startup):

| graph | l ≤ | phlite | Perseus 6 | gramag 0.4.1 |
|---|---|---:|---:|---:|
| C_12 (all pairs) | 10 | 0.10 s | 0.08 s | 0.95 s / 143 MB |
| grid 4×4 | 6 | 0.03 s | 0.01 s | 0.15 s |
| G(20, 0.2) | 5 | 0.04 s | 0.02 s | 0.36 s |
| KY RP² 15 vertices, all pairs | 6 | 0.06 s | 0.04 s | 0.42 s |
| KY RP² 15 vertices, pair (0̂,1̂) | 8 | 0.06 s | 0.03 s | 7.47 s / 1324 MB |
| KY RP³, pair (0̂,1̂) | 7 | 0.05 s | 0.04 s | out of memory (Perseus's run) |

Large single pieces (Rust example, 4 threads; Perseus integral, 4 threads):

| piece | phlite | Perseus 6 |
|---|---:|---:|
| KY RP² (15 v), (0̂,1̂), l ≤ 10 (2M tuples) | 1.41 s / 234 MB | 1.37 s / 196 MB |
| KY RP² (33 v), (0̂,1̂), l ≤ 10 (20M tuples) | 16.1 s / 2.2 GB | 16.9 s / 1.5 GB |
| KY L(3,1), (0̂,1̂), l ≤ 9 | 6.4 s / 0.9 GB | 7.2 s / 0.7 GB |
| KY RP³, (0̂,1̂), l ≤ 9 | 2.7 s / 0.4 GB | 2.8 s / 0.3 GB |

So: 8–100× faster than gramag (and it runs where gramag runs out of memory), on par with Perseus in time, ~1.3×
its memory. Every rank and torsion group agreed with Perseus. Integral results reproduce Perseus's published torsion,
e.g. RP² (0̂,1̂) to l ≤ 10: Z/2 in (3,4), (5,6), (7,8), (9,10), (Z/2)⁴ in (8,10).

## The literature on torsion (what is known, what a search can address)

* Hepworth–Willerton asked whether magnitude homology can have torsion; Kaneta–Yoshinaga (Bull. LMS 2021) showed
  `H_{n-2}` of the order complex of a ranked poset `P` embeds in `MH_{n, rk+2}(0̂, 1̂)` of the Hasse diagram of `P̂`:
  RP² gives Z/2 in `MH_{3,4}`. Sazdanovic–Summers (JHRS 2021): every finitely generated abelian group appears
  (lens spaces: Z/p^r in `MH_{3,2n+3}`), and 2-torsion anywhere on the second diagonal. Martin–Sazdanovic
  (arXiv 2503.11976): torsion in Eulerian magnitude homology, explicit 3- and 5-torsion, and torsion not from
  manifolds; open problem: which graph structures produce (or propagate) torsion, e.g. whether torsion at `(k, l)`
  persists along its diagonal (RP²'s doesn't). Caputi–Collari (BLMS 2024): torsion is bounded for bounded genus.
* **Proved torsion-free**: degrees `k ≤ 2` (Gomi, for metric spaces; Ivanov–Mukoseev, for digraphs) and the main
  diagonal `k = l`. So the first possible torsion is in `MH_{3,l}`, `l ≥ 4`.
* **What a search can say**: the smallest graph (or digraph) with torsion, and in which bidegrees torsion first
  appears in small graphs. Known small examples are Kaneta–Yoshinaga graphs: 15 vertices (cellular RP², the hemicube
  with a bottom and top) and 33 (simplicial RP²). The minimal finite model of RP² has 13 points (Barmak–Minian), so
  the KY route can't go below 15; anything smaller would be a new kind of example.

`phlite_networks/examples/magnitude_torsion_search.py` searches families (networkx's atlas; nauty's `geng` and
`directg`, exhaustive; random graphs, digraphs and layered Hasse-like digraphs), over Z by default.

## The magnitude-path spectral sequence: `phlite.networks.magnitude_path_spectral_sequence` (1acacdd)

The spectral sequence of the length filtration of the reachability complex (all tuples, differential deleting any
vertex, endpoints included): `E^1` = magnitude homology, `E^2` = bigraded path homology (Hepworth–Roff, arXiv
2404.06689) with path homology on its diagonal (Asao), converging to reachability homology. Every page is read off
phlite's persistence barcode of that filtration: a bar `[b, d)` in degree `k` gives a generator of `E^r` at `(k, b)`
and `(k+1, d)` for `r ≤ d − b`. Exact for `l + r ≤ l_max + 1` (others `None`). The complex doesn't split over pairs
(endpoint deletions), so it is for small graphs. Tests: `E^1` = `magnitude`, `E^2` diagonal = path homology (against
`pph`, weights 1), pages only shrink, directed cycles. Over fields only; the integral version would need the
persistence of a filtered complex over Z (pivots that aren't units): the Smith-form machinery above is a start.
