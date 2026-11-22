# Changelog

Every crate and Python package shares one version. Versions follow [Semantic Versioning](https://semver.org): before
1.0, a minor version may change public APIs.

## 0.3.0

Persistence beyond a single filtration (maps, zigzags, the integers, spectral sequences), magnitude homology, more
filtrations, and faster cubical, Rips and flag pipelines. Every technique is behind a Rust trait and has a Python API;
each is tested against the reference implementations named below and benchmarked in `BENCHMARKS.md`.

### Maps of filtrations

- **Images, kernels and cokernels** of the map on persistent homology induced by an inclusion (Cohen-Steiner,
  Edelsbrunner, Harer and Morozov), generic over boundary oracles: `phlite::image_kernel`;
  `FilteredComplex.image_kernel_cokernel`; `phlite.clouds.rips_image_kernel` for Rips filtrations of a subset. Faster
  than phimaker on every input tried.
- **Six-packs**: the barcodes of the domain, codomain, image, kernel, cokernel and pair, with representatives
  (`phlite::six_pack`; `FilteredComplex.six_pack`, `FilteredMap.six_pack`, `rips_six_pack`, `rips_map_six_pack`;
  `phlite.core.SixPack`).

### Zigzag persistence

- **FastZigzag** (Dey and Hou) on phlite's reductions: `phlite_complexes::zigzag::zigzag_barcode`;
  `phlite.complexes.zigzag` on `(simplex, insert)` operations or cells of any kind with their faces, and the chainable
  `Zigzag` builder (`insert`, `delete`, `move_to`, `from_complexes`, `rips_windows`). Tested against fzz and Dionysus 2;
  within 1.6× of fzz reducing the same way, and 17–385× faster than Dionysus.
- **Zigzags of any complex**: the trait `ZigzagComplex` and `ComplexZigzag`, joining consecutive complexes by unions or
  intersections (`Join`), with barcodes by level (`LevelBar`); for point clouds (`phlite.clouds.cloud_zigzag`,
  `sliding_windows`: Rips in any metric or of a distance matrix, Čech, alpha, Delaunay–Čech and witness complexes, or
  any `PointFiltration`), networks (`phlite.networks.network_zigzag`: clique and directed flag complexes) and videos
  (`phlite.images.video_zigzag`, `CubicalCut`).
- **Cells with lifetimes**: `timed_zigzag` (Rust and `phlite.complexes.timed_zigzag`), each cell living over intervals of
  time, with bars in time (`TimedZigzagBarcode`). Tested against Dionysus 2 given the same times.

### Persistence over the integers and spectral sequences

- **Every filtration over the integers** (`modulus=0`): the barcode over Q and the torsion of each bar, from
  `phlite::integral` (Smith forms); `rips_complex` and `cubical_complex` export filtrations with signed boundaries.
- **One algebra for the integers**, in `phlite`: the torsion primes are checked with phlite's own clearing
  reduction over a prime field chosen at run time (`phlite::fields::Fp`); and the elimination of unit pairs over Z
  behind magnitude homology (as Perseus 6's) is `phlite::integral::elimination`, which persistent groups now use too,
  pairing only cells that enter at the same step (Mischaikow and Nanda), so they are computed from far smaller
  complexes.
- **Spectral sequences** of filtered complexes, a page at a time and lazily, over a field or the integers, with groups,
  generators, classes of chains and the differentials (`FilteredComplex.spectral_sequence`,
  `phlite.core.SpectralSequence`). Tested against Kenzo, spectral-sequence-engine and PerMaViss.

### Magnitude homology

- **Magnitude homology** of digraphs over Z/p or the integers, with its torsion (`phlite.networks.magnitude`,
  `MagnitudeHomology`), by Markowitz elimination: 8–100× faster than gramag, and level with Perseus.
- **The magnitude-path spectral sequence** (Asao; Hepworth and Roff), every page as groups with generators
  (`magnitude_path_spectral_sequence`), tested against Hepworth and Roff's computations for directed and bi-directed
  cycles.

### New filtrations

- **Čech complexes** in any dimension, by minimal enclosing balls (`phlite.clouds.cech`, `cech_complex`).
- **Witness complexes**, relaxed and strong, as GUDHI's (`phlite.clouds.witness`, `witness_complex`).
- **Periodic cubical complexes**, any axes wrapped round, as GUDHI's `PeriodicCubicalComplex`
  (`cubical(image, periodic=...)`).
- **Binary image filtrations**, as giotto-tda's: height, radial, dilation, erosion, signed distance and density
  (`phlite.images.filtrations`).

### Performance

- **Edge collapse** for Rips and flag filtrations in dimensions 2 and up (`Algorithm.with_collapse`), repeated while a
  pass removes a tenth of what is left.
- **Union-find** for every dimension whose columns have at most two entries, on one shared implementation
  (`phlite::union_find`), including dimension 0 of lazy network complexes.
- **Cubical volumes** in the T-construction by a filtered discrete gradient (cubical Morse theory).
- **Alpha complexes in the plane** bulk-load their Delaunay triangulation.
- **Perseus 6** joins the references in the tests and benchmarks (Rips, cubical, flag and explicit complexes).

### Playground

- Zigzags on every tab: sliding windows and moving points (Rips, Čech, alpha, Delaunay–Čech, witness), videos
  (examples, uploads or the camera) as the Images tab's zigzag, temporal networks (clique and directed flag complexes
  over windows of time), and cells with lifetimes on the Chain complexes tab.
- Six-packs of a selected subset of points, magnitude homology and the magnitude-path spectral sequence (groups and
  generators, drawn on the graph), periodic and binary-filtered images, the integers and spectral sequences of your own
  complexes.
- One way to choose what to compute on every tab, algorithm options folded away, shorter notes.

### Changes

- `phlite_images::zigzag::video_zigzag` takes a `Join` (before the algorithm).
- `MagnitudeHomology` replaces the dicts `magnitude(..., integral=True)` returned (it still compares equal to them).
- Calls that run in parallel outside a parallel algorithm's thread pool (images, kernels and cokernels, six-packs,
  witness complexes, magnitude homology) now use a pool of their own, so a process forked after them, as Python's
  `multiprocessing` does on Linux, no longer hangs.

## 0.2.0

The first release of the `phlite.*` packages: `phlite.core`, `phlite.complexes`, `phlite.clouds`, `phlite.networks`
and `phlite.images`, and the `phlite` metapackage.
