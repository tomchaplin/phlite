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

### Search results (integral, exhaustive unless stated)

| family | count | l ≤ | torsion |
|---|---:|---:|---|
| connected graphs, ≤ 7 nodes (atlas) | 996 | 8 | none |
| connected graphs, 8 nodes (geng) | 11,117 | 6 | none |
| connected digraphs, ≤ 5 nodes (directg, arcs either or both ways) | 9,578 | 6 | none |
| connected digraphs, 6 nodes | 1,530,843 | 5 | none |
| connected graphs, 9 nodes (geng) | 261,080 | 5 | none |

(8-node graphs and ≤ 5-node digraphs were run with the field comparison over Z/2, Z/3, Z/5 against Z/251; the rest
over Z.) So no graph on ≤ 9 nodes and no digraph on ≤ 6 nodes has torsion in `MH_{3,4}`, `MH_{3,5}` or `MH_{4,5}`
(and, for graphs on ≤ 8 nodes, up to `l = 6`; on ≤ 7 nodes, up to `l = 8`). The random families (layered Hasse-like
digraphs, random digraphs) were stopped to free the machine for benchmarks; rerun with the script.

**Local minimality of the known examples** (`phlite_networks/examples/magnitude_torsion_minimise.py`: delete nodes,
then edges, while some torsion up to `l = 6` survives): Kaneta–Yoshinaga's 15-node graph of the hemicube RP²
(undirected, torsion (Z/2)² in `MH_{3,4}`) and its one-way Hasse digraph (31 arcs, Z/2 in `MH_{3,4}` from the pair
(0̂, 1̂) only) are both locally minimal: no single node or edge (arc) can go.


### MPSS: checked against Hepworth and Roff's computations

The spectral sequence reproduces, page by page, Hepworth–Roff's Theorem 8.2 (directed m-cycles, m = 3, 4, 5: rank
m on E¹ and 1 on E² in bidegrees (2i, mi) and (2i+1, mi+1), E² = … = E^{m−1}, E^m a single generator at (0, 0)) and
Theorem 9.3 (bi-directed cycles C_{m,n} for six (m, n): E² rank 1 at (0,0), (1,1), (2, max(m,n)), then E^{max}
trivial). Both are Rust tests in `phlite_networks/src/mpss.rs`.

## Python benchmark suites at the end of the day

The full tables (references, Perseus, phlite serial and parallel, with memory) are in
`perf-results/benchmarks-2026-11-21.md`, generated by `benchmarks/report.py` from the Rips, sparse Rips, alpha, flag
and cubical suites. The explicit complexes suite was stopped part-way when the machine ran low on memory, so it isn't
there. Headlines, against the numbers committed before today (`python-tests/BENCHMARKS.md`, same machine), phlite
serial:

| case | before | now | fastest reference |
|---|---:|---:|---|
| Rips n=400 dim≤2 | 518.0 ms | 20.0 ms | Perseus 14.8 ms |
| Rips sphere n=400 dim≤2 | 6.91 s | 265.8 ms | Perseus 298.8 ms |
| Rips n=1000 dim≤2 | 8.54 s | 118.3 ms | Perseus 108.7 ms |
| Rips torus n=1000 dim≤2 | 7.83 s | 415.5 ms | Perseus 389.9 ms |
| Rips circles n=2000 dim≤1 | 14.88 s | 502.3 ms | Perseus 1.52 s |
| alpha plane n=100,000 | — | 512.7 ms | GUDHI 1.48 s |
| alpha plane n=1,000,000 | — | 8.50 s | GUDHI 22.44 s |
| flag weight n=300 | 10.6 ms | 9.8 ms | GUDHI 100.5 ms (Perseus 1.15 s) |
| sparse Rips n=2000 | 38.5 ms | 23.7 ms* | giotto-ph 46.6 ms |
| sparse Rips n=50,000 dim≤1 | 941.7 ms | 977.6 ms* | giotto-ph 968.8 ms |

\* after capping collapse at 8,192 points (611bf69); the suite run itself showed the regression above that size
(4.0 s / 677 MB at 50,000 points), which is what prompted the cap.

Cubical is unchanged (nothing here touched `phlite_images`), and on 2D T-construction images Perseus's planar engine
is now 6–13× faster than phlite (smooth 1024²: 27.7 ms against 340 ms); in 3D 3.5–4×. That is the biggest gap left,
and the planar engine is image-specific; the generic route would be Morse pre-reduction of the implicit cubical
complex (the tie-ordering experiment on explicit complexes didn't pay, because union-find already covers both ends).
The parallel flag pipeline's slowness (277 ms against 10 ms serial at n = 300) predates today.

## Union-find everywhere (a667d50, 64b49b5)

Every union-find in the workspace is now `phlite::union_find::IncidenceReduction`: explicit complexes (H0 and top
dimension), Rips H0, cubical H0 and the dual top dimension, and (new) H0 of the lazy network complexes (directed flag,
ordered tuples, Dowker), which is on by default in those three pipelines and the web playground. Getting the generic
union-find to parity with the hand-written cubical ones took a trailing ground node (instead of a grounded bit or array),
the classic path-halving loop, `#[inline(always)]` on `join`, and reversed numbering with `u32` keys on the dual. The
lazy network complexes barely move (directed flag on 1000 sparse nodes 18.8 → 15.8 ms; elsewhere neutral): dimension 0
isn't where they spend their time.

## Cubical Morse: volumes by a filtered discrete gradient (3c11e7f)

`phlite_images::cubical::morse`. For the T-construction with three axes (cohomology, no representatives, union_find
on): each pixel's *block* (the faces of its top cell that enter with it) is matched by a greedy collapse inside the
block, cached by the block's shape (a 27-bit set; 3–9 thousand shapes on the benchmark images), giving a filtered
discrete gradient; the Morse boundaries follow gradient paths with memoised flows; the Morse complex is reduced by
`phlite_complexes` (union-find at both ends). The matching is perfect on the benchmark images (smooth 128³: 400,053
critical cells = 2 × 200,026 finite bars + 1 essential), so all of the remaining cost is implementation:

* Caching blocks by the 26-bit neighbour mask missed (524k distinct masks for 2M pixels); by block shape, it hits.
* HashMaps for flows and indices cost 4 s; one `u32` of state per cell (partner + the partner's axis mask, critical
  index, or memoised flow span) and spans into an arena made it ~1 s, and carrying axis masks removed every division
  from the traversal (another 20%).
* Blocks are disjoint, so the gradient runs over pixels in parallel (index order, critical cells sorted afterwards);
  flows into each dimension only touch that dimension, so the three boundary dimensions run in parallel.

| 3D T-construction | previous default (serial / parallel) | Morse (serial / parallel) | Perseus (1 thread) |
|---|---:|---:|---:|
| smooth 128³ | 3.05 / 2.34 s | 1.24 / 0.73 s | 0.77 s |
| noise 128³ | 6.06 / 21.2 s | 2.88 / 2.89 s | — |
| noise 64³ | 0.57 / 1.02 s | 0.22 / 0.20 s | — |
| noise 32³ | 25 / 22 ms | 23 / 19 ms | 17 ms |

What's left in 3D: the Morse boundaries dominate (noise 128³: 1.5 s serial of 2.9 s, dimension 2 most of it), and
`ParallelClearing` on the explicit Morse complex is *slower* than serial (1.8 s against 0.6 s on noise 128³), which is
why parallel doesn't help there; that is a general weakness of parallel clearing on explicit complexes, not specific to
Morse, so I didn't override the user's algorithm.

Not used in 2D, where it loses: union-find at both ends already leaves no middle dimension, and the gradient pass
alone costs more (smooth 1024²: 1.18 s with Morse against 0.37 s). The 2D gap to Perseus (6–13×) is its planar engine
(union-find with a sorted sweep and no cell enumeration at all); our 2D time is mostly building the two sorted cell
lists for the union-finds (`dimension_cells`), which is where to look next for 2D.

## Phase 4: both experiments negative (no code kept except a benchmark case)

**Size-switched working column.** `HybridColumns { threshold }`: a `MergeCol` until more than `threshold` entries
have been added, then drained into a `BitCol` kept between columns (patch kept at the scratchpad as
`hybrid-columns.patch`). Swept over the dumped explicit, cubical and Freudenthal complexes (clearing, `store_r`), with
and without union-find, thresholds 64–8192. Pivot columns win or tie everywhere except the tiny `dense200` (1.2 ms at
threshold 64 against 1.9 ms); e.g. without union-find grid500 68.8 ms pivot / 76.0 ms hybrid(512) / 103 ms merged,
cubical3d-noise-40 45.6 / 53.5 / 67.4 ms. The bit column's random slot accesses aren't the cost I expected them to be,
and merging small columns doesn't beat it. Pivot is already the default for explicit complexes, and cubical images no
longer reduce much (union-find at both ends, Morse in 3D), so not merged.

**Forward cohomology (annotations) for Linial–Meshulam.** Not implemented, because phlite doesn't trail there: on
Y_2(n, p) with triangles at 2–5 levels, phlite (serial, pivot, cohomology) is 5–7× faster than Perseus (F2, one
thread): n=100 p=0.1: 0.11 s vs 0.58 s; n=150 p=0.05: 2.0 s vs 11.8 s; n=200 p=0.03: 5.1 s vs 36.1 s. Column
representation matters enormously here (n=200: merged columns 178 s against pivot's 5.1 s). Added `linial-meshulam`
cases to `bench_complexes` (n=100 p=0.1, and n=150 p=0.05 as large): at n=100 phlite 90 ms, PHAT twist 181 ms,
Perseus 541 ms, lophat 7.1 s.

## Integral spectral sequences and persistent groups (8df1ca3)

Over Z a filtered complex has no barcode (it doesn't decompose into intervals), so `phlite::integral` works with the
subgroups themselves, as lattices: `Z^r_p = {x ∈ F_p : dx ∈ F_{p-r}}` is a kernel, and
`E^r_p = Z^r_p / (Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1})` and `H^{i,j}_k = Z_k(F_i) / d{x ∈ F_j : dx ∈ F_i}` are quotients,
read off Smith normal forms. Exact, dense, `i128` with checked arithmetic (`Overflow`, never a wrong answer),
least-modulus pivots throughout. Checks: RP² (H_1 = Z/2 at every step, and as E² = E^∞ when filtered by dimension),
homology against the classical Smith-form formula on random complexes, free ranks against the field pages over Z/251
on 150 random filtered complexes.

`phlite.networks.magnitude_path_spectral_sequence(..., integral=True)` (Rust `mpss_integral`) returns each page as
`magnitude(..., integral=True)` does: E¹ equals integral magnitude homology and the free ranks the pages over Z/251
(Rust and Python tests). It is dense over the whole reachability complex, so ~20× slower per unit of length:
bidirected C6 to length 5 in 0.3 s, to 6 in 7 s (the field version: 1–2 ms).

**Search for torsion on later pages** (all connected digraphs, nauty `geng -c | directg`): none on 3 or 4 nodes up to
length 5 (pages 1–5); 5 nodes up to length 4: see below.

**Not done, needs a decision:** explicit complexes. `phlite.complexes.ChainComplex` reduces integer coefficients mod p
on input and doesn't keep them, so integral pages or persistent groups there need either the integer columns kept in
`ChainComplex` (memory for every complex, only used by the integral path) or a separate entry point taking columns,
coefficients and values. Also open: making it faster than dense (sparse elimination with the magnitude module's
collapse-then-Markowitz approach, which only applies where the complex splits, e.g. per degree and step).
