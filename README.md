<div align="center">

<h1>phlite</h1>

Persistent homology that's light on memory usage.

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
In particular, when the implementation of $D$ satisfies sufficient constraints, `phlite` provides an implementation of the _clearing_ optimisation (first introduced in [[2]](#2)).

For more details, please refer to the documentation.

## Crates

* `phlite` - Core library, matrix oracle traits, reduction algorithms.
* `phlite_rips` - Implementation of VR filtration, à la Ripser. Rust library + basic CLI.
* `phlite_grpph` - Implementation of GrPPH [[7]](#7). Rust library + Python bindings via PyO3.

## TODO

### High Priority

- [ ] Investigate apparent and emergent pairs
- [ ] Documentation all traits and APIs
- [ ] Compute involuted persistent homology cycle representatives
- [ ] Tests (unit + integration + property)
- [ ] Explain Rips implementation in docs

### Medium priority

- [ ] Improve handling of binary-heap column addition
- [ ] Improve organisation of clearing algorithm (implement as a builder?)
- [ ] Add reverse filtration and reverse basis adaptors?
- [ ] Implement lockfree reduction algo

### Low priority

- [ ] Add optional logging
- [ ] Write some Rust examples
- [ ] Ripser comaptible CLI?
- [ ] Web interface for Rips?
- [ ] Python bindings?
- [ ] Serialisation of V matrices
- [ ] Implement magnitude homology


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
