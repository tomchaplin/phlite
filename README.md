<div align="center">

<img src="https://github.com/tomchaplin/phlite/raw/main/phlite_logo.svg" alt="phlite logo" width="300" role="img">

<h1>phlite</h1>

<span>/flaɪt/</span>

Persistent homology that's light on memory usage.

[![crates.io](https://img.shields.io/crates/v/phlite)](https://crates.io/crates/phlite)
[![docs.rs](https://img.shields.io/docsrs/phlite?label=Docs.rs)](https://docs.rs/phlite/latest/phlite/)

</div>

## Background

The current state-of-the-art for computing persistent homology (PH) is [Ripser](#1); according to their repository:

> Currently, Ripser outperforms other codes \[...\] by a factor of more than 40 in computation time and a factor of more than 15 in memory efficiency.

The importance of memory efficieny cannot be overstated:
* memory usage is typically the limiting factor for PH (especially on consumer-grade hardware);
* allocating memory is time-consuming, thus limiting applications even on large computers.

The excellent memory efficiency of Ripser has made it, and by extension the Vietoris-Rips (VR) filtration, the go-to suggestion for many applications.
Since Ripser is ruthlessly optimised for the VR filtration, it has been regularly forked and adapted to work for other filtations (for example Flagser [[3]](#3) and Cubical Ripser [[4]](#4)).

The goal of this library is to separate out the components of Ripser that are specialised to the VR filtration from those that are applicable more boardly.
As such, __we provide a framework for developing fast, memory-efficient software to compute the persistent homology of novel filtrations__.

### Related Work

Ripser has been ported to a number of languages, including Ripser.py for Python [[5]](#5) and Ripserer.jl for Julia [[6]](#6).
The latter exposes APIs that achieve the same aims as `phlite`, namely:

> Ripserer is designed to be easily extended with new simplex or filtration types.

For those familiar with the Julia language, we strongly recommend this library.

## Usage

At its core, `phlite` is a framework for implementing lazy oracles for sparse matrices $D$ where the rows and columns can be indexed by arbitrary (ordered types).
Unlike Ripser, `phlite` is _batteries not included_.
In order to use the library, you must
* choose appropriate types to index your matrix;
* implement a matrix oracle which can report the non-zero entries in a given column;
* compute an ordered basis for the column space of $D$ (typically in reverse filtration order).

Given such an implementation, `phlite` provides memory-efficient methods for computing $R = D V$ decompositions and hence computing persistent homology.
In line with Ripser's approach, the output of these methods is an oracle for the matrix $V$, from which $R$ can be readily computed.
In particular, when the implementation of $D$ satisfies sufficient constraints, `phlite` provides an implementation of the _clearing_ optimisation (first introduced in [[2]](#2)), both serially and in parallel (following [[8]](#8)).
If you can also provide the transpose of $D$ (e.g. the boundary matrix, for a coboundary matrix), the reduction recognises _apparent pairs_ [[1]](#1), which need no reduction or storage.
`phlite::barcode` turns all of this into persistence barcodes, optionally with a representative cycle for every bar.

Most of `phlite`'s performance comes from optimisations we have adapted from Ripser [[1]](#1) and giotto-ph [[9]](#9):
* from Ripser: computing cohomology with clearing, storing $V$ rather than $R$ and recomputing columns on demand,
  _apparent_ and _emergent_ pairs (finishing a column as soon as its unclaimed pivot is found, by enumerating cofacets in order and
  stopping at the first with the simplex's own diameter), finding apparent pairs with early-stopping facet and cofacet searches,
  leaving the simplices in zero-persistence apparent pairs out of the columns to reduce, and truncating the Vietoris-Rips filtration at the enclosing radius;
* from giotto-ph: the parallel structure of each dimension's reduction (classifying all columns in parallel and giving per-column state
  only to the few that need reducing, which are then reduced with the lock-free algorithm of [[8]](#8)), building the columns to reduce
  in parallel, and truncating at the enclosing radius by default;
* from OAT [[10]](#10): reducing a column by lazily merging sorted columns, rather than pushing every entry into one heap
  (see `phlite::columns::MergedColumns`), which keeps the heap small enough to stay in cache when columns are long.

`phlite`'s contribution is making these available for any filtration, behind generic traits, rather than just Vietoris-Rips.

With these, `phlite_rips` is several times faster than Ripser serially and on par with giotto-ph [[9]](#9) in parallel,
using a fraction of Ripser's memory, on the following benchmark.

**Test case:** $n$ points sampled uniformly from the unit cube $[0, 1]^3$ (seed 0), with Euclidean distances (rounded to `float32`, so every implementation sees identical input),
computing persistent homology in dimensions 0, 1 and 2 over $\mathbb{Z}/2$ by cohomology.
We compare `phlite_rips` (`Algorithm.serial()`) with [ripser.py](https://github.com/scikit-tda/ripser.py) (serial),
and `phlite_rips` (`Algorithm.parallel(threads=4)`) with giotto-ph (`n_threads=4`).
Times are the best of 2 runs and peak memory is the extra resident memory used during the computation,
each run in a fresh process on one 8-core Linux machine; all implementations produced the same barcodes.

| $n$ | Ripser (serial) | `phlite_rips` (serial) | giotto-ph (4 threads) | `phlite_rips` (4 threads) |
|---|---|---|---|---|
| 200 | 0.52s / 55 MB | 0.16s / 0.8 MB | 0.065s / 1.5 MB | 0.077s / 2.6 MB |
| 300 | 2.20s / 209 MB | 0.57s / 3.5 MB | 0.22s / 7.2 MB | 0.27s / 7.2 MB |
| 400 | 5.10s / 389 MB | 1.10s / 4.4 MB | 0.56s / 7.3 MB | 0.50s / 8.4 MB |

These are a single family of inputs on a single machine, so treat them as indicative.
To reproduce, run `just bench-rips --sizes 200 300 400 --max-dim 2 --threads 4 --repeats 2` in the `python-tests` folder
(see its README for the other implementations and options).

For more details, please refer to the documentation on `docs.rs`.
If you need support implementing `phlite` traits for your specific application, please get in touch!

## Crates

* `phlite` - Core library, matrix oracle traits, reduction algorithms, barcodes with representatives.
* `phlite_rips` - Implementation of VR filtration, à la Ripser. Rust library + basic CLI + Python bindings via PyO3.
* `phlite_grpph` - Implementation of GrPPH [[7]](#7). Rust library + Python bindings via PyO3.
* `phlite_py` - Python bindings (the `phlite` Python package) for explicit sparse filtered chain complexes, à la PHAT/LoPHAT.
* `phlite_pyutils` - Shared helpers for the Python bindings (the `Barcode` result type and argument handling).

The `python-tests` folder holds fuzz tests and benchmarks of the Python bindings against ripser, giotto-ph and grpphati.

## TODO

### High Priority

- [x] Investigate apparent and emergent pairs
- [ ] Documentation all traits and APIs
- [ ] Tests (unit + integration + property)
- [ ] Explain Rips implementation in docs
- [x] Add a way to compute involuted PH representatives (see `phlite::barcode`)
- [x] Implement lockfree reduction algo
- [ ] Implement dynamic priority queues (as done in Flagser)

### Medium priority

- [x] Improve handling of binary-heap column addition (the bottleneck for GrPPH)
- [ ] Compute H_0 of Rips with union-find (the bottleneck for large Rips inputs)

### Low priority

- [x] Optional parallel construction of Rips basis
- [ ] Add optional logging
- [ ] Write some Rust examples
- [ ] Ripser comaptible CLI?
- [ ] Web interface for Rips?
- [x] Python bindings?
- [ ] Serialisation of V matrices
- [ ] Implement magnitude homology
- [ ] Implement directed flag complex
- [ ] Add suggested citation

## References

<a id="1">[1]</a>
Bauer, Ulrich. "Ripser: efficient computation of Vietoris–Rips persistence barcodes." Journal of Applied and Computational Topology 5.3 (2021): 391-423. [Github repo.](https://github.com/Ripser/ripser)

<a id="2">[2]</a>
Chen, Chao, and Michael Kerber. "Persistent homology computation with a twist." Proceedings 27th European workshop on computational geometry. Vol. 11. 2011.

<a id="3">[3]</a>
Lütgehetmann, Daniel, et al. "Computing persistent homology of directed flag complexes." Algorithms 13.1 (2020): 19. [Github repo.](https://github.com/luetge/flagser)

<a id="4">[4]</a>
Kaji, Shizuo, Takeki Sudo, and Kazushi Ahara. "Cubical ripser: Software for computing persistent homology of image and volume data." arXiv preprint arXiv:2005.12692 (2020). [Github repo.](https://github.com/shizuo-kaji/CubicalRipser_3dim)

<a id="5">[5]</a>
Tralie, Christopher, Nathaniel Saul, and Rann Bar-On. "Ripser. py: A lean persistent homology library for python." Journal of Open Source Software 3.29 (2018): 925. [Github repo.](https://github.com/scikit-tda/ripser.py)

<a id="6">[6]</a>
Čufar, Matija. "Ripserer. jl: flexible and efficient persistent homology computation in Julia." Journal of Open Source Software 5.54 (2020): 2614. [Github repo.](https://github.com/mtsch/Ripserer.jl) [Docs.](https://mtsch.github.io/Ripserer.jl/dev/)

<a id="7">[7]</a>
Chaplin, T., Harrington, H.A. and Tillmann, U., 2022.
Grounded persistent path homology: a stable, topological descriptor for weighted digraphs.
arXiv preprint [arXiv:2210.11274](https://arxiv.org/abs/2210.11274).

<a id="8">[8]</a>
Morozov, Dmitriy, and Arnur Nigmetov. "Towards lockfree persistent homology." Proceedings of the 32nd ACM Symposium on Parallelism in Algorithms and Architectures. 2020. [DOI.](https://doi.org/10.1145/3350755.3400244)

<a id="9">[9]</a>
Pérez, Julián Burella, et al. "giotto-ph: A python library for high-performance computation of persistent homology of Vietoris-Rips filtrations." arXiv preprint arXiv:2107.05412 (2021). [Github repo.](https://github.com/giotto-ai/giotto-ph)

<a id="10">[10]</a>
Open Applied Topology (OAT). [Github repo.](https://github.com/OpenAppliedTopology/oat_rust)
