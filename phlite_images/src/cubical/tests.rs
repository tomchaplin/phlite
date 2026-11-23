//! Tests of the cubical complexes: the geometry exhaustively on small grids, the optimised hooks against brute force,
//! and barcodes against a naive reduction written from the definitions (independent of the numbering of cells).

use std::collections::{BTreeMap, BTreeSet};

use ordered_float::NotNan;
use phlite::{
    barcode::Algorithm,
    columns::{HeapColumns, MergedColumns, SortedColumns},
    fields::{NonZeroCoefficient, Z2, Z3},
    matrices::{HasRowFiltration, MatrixOracle},
    reduction::{Clearing, Standard},
};

use super::{
    cohomology::CubicalCoboundary, cubical_barcode, homology::CubicalBoundary, Construction, Cube,
    CubicalBar, CubicalComplex, Filtration, Grid, Method,
};

/// A small deterministic random number generator (xorshift), so the tests need no dependencies.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Every cell of `grid`.
fn cells<const D: usize>(grid: &Grid<D>) -> impl Iterator<Item = Cube> {
    (0..grid.n_cells()).map(Cube)
}

/// The boundary of `cube` from its doubled coordinates, from the definition: `sum_i (-1)^i (upper - lower)` along the
/// `i`-th odd axis.
fn boundary_from_coordinates<const D: usize>(grid: &Grid<D>, cube: Cube) -> BTreeMap<Cube, Z3> {
    let coordinates = grid.coordinates(cube);
    let mut out = BTreeMap::new();
    let mut i = 0;
    for a in 0..D {
        if coordinates[a] % 2 == 1 {
            let sign = |negative: bool| {
                if negative {
                    Z3::one().additive_inverse()
                } else {
                    Z3::one()
                }
            };
            let mut upper = coordinates;
            upper[a] += 1;
            let mut lower = coordinates;
            lower[a] -= 1;
            out.insert(grid.cube_at(upper).unwrap(), sign(i % 2 == 1));
            out.insert(grid.cube_at(lower).unwrap(), sign(i % 2 == 0));
            i += 1;
        }
    }
    out
}

/// Grids of every shape with up to `max` vertices along each axis.
fn small_grids<const D: usize>(max: u32) -> impl Iterator<Item = Grid<D>> {
    let total = (max as usize).pow(D as u32);
    (0..total).map(move |mut i| {
        let vertices = std::array::from_fn(|_| {
            let n = (i % max as usize) as u32 + 1;
            i /= max as usize;
            n
        });
        Grid::new(vertices).unwrap()
    })
}

/// A complex with every cell present (all pixels zero), for the geometry tests, on a grid with `vertices`.
fn flat_complex<const D: usize>(vertices: [u32; D]) -> CubicalComplex<D> {
    let shape = vertices.map(|n| n as usize);
    CubicalComplex::new(
        vec![0.0; shape.iter().product()],
        shape,
        Construction::V,
        Filtration::Sublevel,
    )
    .unwrap()
}

fn check_geometry<const D: usize>(max: u32) {
    for grid in small_grids::<D>(max) {
        let complex = flat_complex(grid.vertices());
        let boundary = CubicalBoundary::<Z3, D>::new(&complex);
        let coboundary = CubicalCoboundary::<Z3, D>::new(&complex);
        // The doubled grid has a cell at every point, and numbers them 0..n_cells
        let side: u32 = grid.vertices().iter().map(|&n| 2 * n - 1).product();
        assert_eq!(grid.n_cells(), side);
        let mut seen = BTreeSet::new();
        for cube in cells(&grid) {
            let coordinates = grid.coordinates(cube);
            assert_eq!(grid.cube_at(coordinates), Some(cube));
            assert!(seen.insert(coordinates));
            let dimension = coordinates.iter().filter(|&&c| c % 2 == 1).count();
            assert_eq!(grid.dimension(cube), dimension);
            assert!(grid.cells_of_dimension(dimension).contains(&cube.0));
            // The boundary from the definition
            let column: Vec<(Z3, Cube)> = boundary.column(cube).collect();
            let expected = boundary_from_coordinates(&grid, cube);
            assert_eq!(
                column
                    .iter()
                    .map(|(c, f)| (*f, *c))
                    .collect::<BTreeMap<_, _>>(),
                expected
            );
            // Faces come in decreasing order, and every face has a smaller number (a lower dimension)
            assert!(
                column.windows(2).all(|w| w[0].1 > w[1].1),
                "{coordinates:?}"
            );
            assert!(column.iter().all(|(_, face)| *face < cube));
            // Cofaces come in increasing order
            let cofaces: Vec<(Z3, Cube)> = coboundary.column(cube).collect();
            assert!(
                cofaces.windows(2).all(|w| w[0].1 < w[1].1),
                "{coordinates:?}"
            );
            // Boundary of the boundary is zero
            let mut sum: BTreeMap<Cube, Option<Z3>> = BTreeMap::new();
            for (c, face) in &column {
                for (d, face_of_face) in boundary.column(*face) {
                    let entry = sum.entry(face_of_face).or_insert(None);
                    *entry = (*c * d) + *entry;
                }
            }
            assert!(sum.values().all(Option::is_none), "{coordinates:?}");
        }
        // The coboundary is the transpose of the boundary
        let mut transpose: BTreeMap<Cube, BTreeMap<Cube, Z3>> = BTreeMap::new();
        for cube in cells(&grid) {
            for (c, face) in boundary.column(cube) {
                transpose.entry(face).or_default().insert(cube, c);
            }
        }
        for cube in cells(&grid) {
            let cofaces: BTreeMap<Cube, Z3> =
                coboundary.column(cube).map(|(c, f)| (f, c)).collect();
            assert_eq!(cofaces, transpose.remove(&cube).unwrap_or_default());
        }
    }
}

#[test]
fn geometry_1d() {
    check_geometry::<1>(6);
}

#[test]
fn geometry_2d() {
    check_geometry::<2>(5);
}

#[test]
fn geometry_3d() {
    check_geometry::<3>(4);
}

#[test]
fn geometry_4d() {
    check_geometry::<4>(3);
}

/// A random image of small integers (lots of ties), sometimes with infinite and NaN pixels.
fn random_image<const D: usize>(rng: &mut Rng, max_side: u64) -> (Vec<f64>, [usize; D]) {
    let shape = std::array::from_fn(|_| 1 + rng.below(max_side) as usize);
    let n: usize = shape.iter().product();
    let levels = 1 + rng.below(5);
    let special = rng.below(3) == 0;
    let pixels = (0..n)
        .map(|_| {
            if special {
                match rng.below(10) {
                    0 => return f64::NAN,
                    1 => return f64::INFINITY,
                    2 => return f64::NEG_INFINITY,
                    _ => {}
                }
            }
            rng.below(levels) as f64
        })
        .collect();
    (pixels, shape)
}

/// The value of the cell at doubled `coordinates` straight from the definitions, in the sublevel filtration of the
/// (possibly negated) image, `+inf` for cells not in the complex.
fn value_from_definition<const D: usize>(
    pixels: &[f64],
    shape: [usize; D],
    construction: Construction,
    coordinates: [u32; D],
) -> f64 {
    // The pixels around the cell, per axis
    let around: Vec<Vec<usize>> = (0..D)
        .map(|a| {
            let c = coordinates[a] as usize;
            match construction {
                Construction::V => (c / 2..=c.div_ceil(2)).collect(),
                // Top cells are at odd coordinates 2p + 1: those within one step of c
                Construction::T => (c.saturating_sub(1)..=c + 1)
                    .filter(|t| t % 2 == 1 && t / 2 < shape[a])
                    .map(|t| t / 2)
                    .collect(),
            }
        })
        .collect();
    let mut values = vec![];
    let mut index = [0usize; D];
    loop {
        let mut offset = 0;
        for a in 0..D {
            offset = offset * shape[a] + around[a][index[a]];
        }
        let value = pixels[offset];
        values.push(if value.is_nan() { f64::INFINITY } else { value });
        let mut a = 0;
        while a < D {
            index[a] += 1;
            if index[a] < around[a].len() {
                break;
            }
            index[a] = 0;
            a += 1;
        }
        if a == D {
            break;
        }
    }
    match construction {
        Construction::V => values.into_iter().fold(f64::NEG_INFINITY, f64::max),
        Construction::T => values.into_iter().fold(f64::INFINITY, f64::min),
    }
}

/// The barcode (per dimension: sorted (birth, death) pairs, death `inf` for essential bars, zero-length bars dropped),
/// by the standard algorithm on the whole boundary matrix, with every cell given by its doubled coordinates and the
/// cells sorted by (value, dimension, coordinates).
fn naive_barcode<const D: usize>(
    pixels: &[f64],
    shape: [usize; D],
    construction: Construction,
    filtration: Filtration,
) -> Vec<Vec<(f64, f64)>> {
    let sign = if filtration == Filtration::Superlevel {
        -1.0
    } else {
        1.0
    };
    let pixels: Vec<f64> = pixels.iter().map(|p| sign * p).collect();
    if pixels.is_empty() {
        return vec![vec![]; D];
    }
    let side: Vec<u32> = (0..D)
        .map(|a| match construction {
            Construction::V => 2 * shape[a] as u32 - 1,
            Construction::T => 2 * shape[a] as u32 + 1,
        })
        .collect();
    // Every cell of the complex
    let mut cells: Vec<(NotNan<f64>, usize, [u32; D])> = vec![];
    let total: u32 = side.iter().product();
    for mut i in 0..total {
        let coordinates: [u32; D] = std::array::from_fn(|a| {
            let c = i % side[a];
            i /= side[a];
            c
        });
        let value = value_from_definition(&pixels, shape, construction, coordinates);
        if value < f64::INFINITY {
            let dimension = coordinates.iter().filter(|&&c| c % 2 == 1).count();
            cells.push((NotNan::new(value).unwrap(), dimension, coordinates));
        }
    }
    cells.sort();
    let position: BTreeMap<[u32; D], usize> = cells
        .iter()
        .enumerate()
        .map(|(i, (_, _, c))| (*c, i))
        .collect();
    // Reduce over Z/2, columns as sorted sets of positions
    let mut reduced: Vec<BTreeSet<usize>> = Vec::with_capacity(cells.len());
    let mut pivots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut paired = BTreeSet::new();
    let mut bars = vec![vec![]; D + 1];
    for (j, (value, dimension, coordinates)) in cells.iter().enumerate() {
        let mut column = BTreeSet::new();
        for a in 0..D {
            if coordinates[a] % 2 == 1 {
                for step in [coordinates[a] - 1, coordinates[a] + 1] {
                    let mut face = *coordinates;
                    face[a] = step;
                    let face = position[&face];
                    assert!(face < j, "a face must come first");
                    column.insert(face);
                }
            }
        }
        while let Some(&other) = column.last().and_then(|pivot| pivots.get(pivot)) {
            column = column
                .symmetric_difference(&reduced[other])
                .copied()
                .collect();
        }
        if let Some(&pivot) = column.last() {
            pivots.insert(pivot, j);
            paired.insert(pivot);
            paired.insert(j);
            let birth = cells[pivot].0.into_inner();
            if birth != value.into_inner() {
                bars[*dimension - 1].push((sign * birth, sign * value.into_inner()));
            }
        }
        reduced.push(column);
    }
    for (j, (value, dimension, _)) in cells.iter().enumerate() {
        if !paired.contains(&j) {
            bars[*dimension].push((sign * value.into_inner(), f64::INFINITY));
        }
    }
    bars.truncate(D);
    for dim in &mut bars {
        dim.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    bars
}

/// `bars` in the form of [`naive_barcode`].
fn summary<const D: usize>(
    complex: &CubicalComplex<D>,
    bars: &[CubicalBar],
) -> Vec<Vec<(f64, f64)>> {
    let mut out = vec![vec![]; D];
    for bar in bars {
        let death = bar.death.map_or(f64::INFINITY, |(_, d)| d.into_inner());
        out[complex.dimension(bar.birth)].push((bar.birth_value.into_inner(), death));
    }
    for dim in &mut out {
        dim.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    out
}

/// Every algorithm (with the given columns), cached or not.
macro_rules! for_each_algorithm {
    ($columns:expr, $algorithm:ident, $body:block) => {
        for cache in [false, true] {
            let clearing = Clearing::new().with_columns($columns).cache_columns(cache);
            for $algorithm in [
                Algorithm::Standard(Standard::new().with_columns($columns).cache_columns(cache)),
                Algorithm::Clearing(clearing.clone()),
                Algorithm::Clearing(clearing.clone().store_r(true)),
                Algorithm::Clearing(clearing.clone().skip_zero_apparent_pairs(false)),
                Algorithm::ParallelClearing(clearing.clone().parallel()),
            ] {
                $body
            }
        }
    };
}

/// Check every representative is a cycle (over Z/2) whose youngest cell is the birth, in decreasing filtration order.
fn check_representatives<const D: usize>(complex: &CubicalComplex<D>, bars: &[CubicalBar]) {
    let boundary = CubicalBoundary::<Z2, D>::new(complex);
    for bar in bars {
        let representative = bar
            .representative
            .as_ref()
            .expect("representatives were requested");
        // A finite bar's representative leads with its birth, and is in decreasing filtration order (an essential bar's
        // is a column of V, in decreasing order of cells, which contains its birth)
        if bar.death.is_some() {
            assert_eq!(representative.first().map(|(_, c)| *c), Some(bar.birth));
            let key = |cube: Cube| (boundary.filtration_value(cube), cube);
            assert!(representative.windows(2).all(|w| key(w[0].1) > key(w[1].1)));
        } else {
            assert!(representative.iter().any(|(_, c)| *c == bar.birth));
        }
        let mut sum = BTreeSet::new();
        for (_, cell) in representative {
            assert_eq!(complex.dimension(*cell), complex.dimension(bar.birth));
            for (_, face) in boundary.column(*cell) {
                if !sum.remove(&face) {
                    sum.insert(face);
                }
            }
        }
        assert!(sum.is_empty(), "not a cycle: {bar:?}");
    }
}

fn check_barcodes<const D: usize>(seed: u64, cases: usize, max_side: u64) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let (pixels, shape) = random_image::<D>(&mut rng, max_side);
        for construction in [Construction::V, Construction::T] {
            for filtration in [Filtration::Sublevel, Filtration::Superlevel] {
                let expected = naive_barcode(&pixels, shape, construction, filtration);
                let complex =
                    CubicalComplex::new(pixels.clone(), shape, construction, filtration).unwrap();
                let context =
                    format!("case {case}: {shape:?} {construction:?} {filtration:?} {pixels:?}");
                // Every algorithm and method, with the default columns
                for_each_algorithm!(MergedColumns, algorithm, {
                    for method in [Method::Cohomology, Method::Homology] {
                        for representatives in [false, true] {
                            let bars = cubical_barcode(
                                &complex,
                                D,
                                method,
                                algorithm.clone(),
                                representatives,
                            );
                            assert_eq!(
                                summary(&complex, &bars),
                                expected,
                                "{context} {method:?} {algorithm:?}"
                            );
                            if representatives {
                                check_representatives(&complex, &bars);
                            }
                        }
                    }
                });
                // The other column representations, with clearing
                let algorithm = Algorithm::Clearing(Clearing::new().with_columns(HeapColumns));
                let bars = cubical_barcode(&complex, D, Method::Cohomology, algorithm, true);
                assert_eq!(summary(&complex, &bars), expected, "{context}");
                let bars = cubical_barcode(
                    &complex,
                    D,
                    Method::Cohomology,
                    Algorithm::Clearing(Clearing::new().with_columns(SortedColumns)),
                    false,
                );
                assert_eq!(summary(&complex, &bars), expected, "{context}");
            }
        }
    }
}

#[test]
fn barcodes_1d() {
    check_barcodes::<1>(1, 60, 8);
}

#[test]
fn barcodes_2d() {
    check_barcodes::<2>(2, 60, 5);
}

#[test]
fn barcodes_3d() {
    check_barcodes::<3>(3, 25, 3);
}

#[test]
fn barcodes_4d() {
    check_barcodes::<4>(4, 4, 2);
}

/// Faces never enter after their cofaces, in either construction, and the early-stopping hooks agree with brute force.
fn check_hooks<const D: usize>(seed: u64, cases: usize, max_side: u64) {
    let mut rng = Rng(seed);
    for _ in 0..cases {
        let (pixels, shape) = random_image::<D>(&mut rng, max_side);
        for construction in [Construction::V, Construction::T] {
            let complex =
                CubicalComplex::new(pixels.clone(), shape, construction, Filtration::Sublevel)
                    .unwrap();
            let Some(grid) = complex.grid() else { continue };
            let boundary = CubicalBoundary::<Z2, D>::new(&complex);
            let coboundary = CubicalCoboundary::<Z2, D>::new(&complex);
            for cube in cells(grid) {
                let value = complex.internal_value_of(cube);
                assert_eq!(
                    value,
                    value_from_definition(
                        &complex.pixels,
                        shape,
                        construction,
                        grid.coordinates(cube)
                    )
                );
                for (_, face) in boundary.column(cube) {
                    assert!(complex.internal_value_of(face) <= value);
                }
                if value == f64::INFINITY {
                    continue;
                }
                let entry = |e: phlite::columns::ColumnEntry<NotNan<f64>, Cube, Z2>| {
                    (e.filtration_value, e.row_index)
                };
                // Cohomology's pivot, the oldest coface
                let cofaces: Vec<_> = coboundary.column_with_filtration(cube).map(entry).collect();
                assert_eq!(
                    coboundary.column_min(cube).map(entry),
                    cofaces.iter().copied().min()
                );
                assert_eq!(
                    boundary.first_column_in_row(cube),
                    cofaces.iter().copied().min().map(|(_, c)| c)
                );
                assert_eq!(
                    boundary.last_column_in_row(cube),
                    cofaces.iter().copied().max().map(|(_, c)| c)
                );
                // Homology's pivot, the youngest face
                let faces: Vec<_> = boundary.column_with_filtration(cube).map(entry).collect();
                assert_eq!(
                    boundary.column_pivot(cube).map(entry),
                    faces.iter().copied().max()
                );
                assert_eq!(
                    coboundary.last_column_in_row(cube),
                    faces.iter().copied().max().map(|(_, c)| c)
                );
                assert_eq!(
                    coboundary.first_column_in_row(cube),
                    faces.iter().copied().min().map(|(_, c)| c)
                );
            }
        }
    }
}

#[test]
fn hooks_match_brute_force() {
    check_hooks::<1>(5, 50, 8);
    check_hooks::<2>(6, 50, 6);
    check_hooks::<3>(7, 20, 4);
}

/// The reduction bases leave out exactly the cells in zero-persistence apparent pairs, serially and in parallel, and
/// in smooth images that is most of them.
#[test]
fn reduction_bases_leave_out_zero_apparent_pairs() {
    use super::{build_bases, build_reduction_bases};
    use phlite::matrices::{ColBasis, SplitByDimension};
    let n = 24;
    let pixels: Vec<f64> = (0..n * n)
        .map(|i| (((i / n) as f64 / 4.0).sin() + ((i % n) as f64 / 5.0).cos()).round())
        .collect();
    for construction in [Construction::V, Construction::T] {
        let complex =
            CubicalComplex::new(pixels.clone(), [n, n], construction, Filtration::Sublevel)
                .unwrap();
        let full = build_bases(&complex, 2, false, false);
        let serial = build_reduction_bases(&complex, 2, false);
        let parallel = build_reduction_bases(&complex, 2, true);
        let as_vec = |b: phlite::matrices::BasisSlice<'_, Cube>| {
            (0..b.size()).map(|i| b.element(i)).collect::<Vec<_>>()
        };
        for dim in 0..=2 {
            let expected: Vec<_> = as_vec((&full).in_dimension(dim))
                .into_iter()
                .filter(|cube| !complex.in_zero_apparent_pair(*cube))
                .collect();
            assert_eq!(as_vec((&serial).in_dimension(dim)), expected);
            assert_eq!(as_vec((&parallel).in_dimension(dim)), expected);
        }
        assert!(
            4 * serial.size() < full.size(),
            "{} of {}",
            serial.size(),
            full.size()
        );
    }
}

/// Small images, checked by hand.
#[test]
fn hand_checked() {
    let bars = |pixels: Vec<f64>, shape: [usize; 2], construction, filtration| {
        let complex = CubicalComplex::new(pixels, shape, construction, filtration).unwrap();
        let bars = cubical_barcode(&complex, 1, Method::Cohomology, Algorithm::default(), true);
        let mut out: Vec<_> = bars
            .iter()
            .map(|bar| {
                (
                    complex.dimension(bar.birth),
                    bar.birth_value.into_inner(),
                    bar.death.map(|(_, d)| d.into_inner()),
                    complex.coordinates(bar.birth),
                    bar.death.map(|(d, _)| complex.coordinates(d)),
                )
            })
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    };
    use Construction::{T, V};
    use Filtration::{Sublevel, Superlevel};
    // A ring of dark pixels around a bright one: a loop born at 1, when the ring closes (at its last edge in the
    // order on cells, the bottom-right one), and filled at 5, when the centre comes in (with the last of its squares)
    let ring = vec![1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(
        bars(ring.clone(), [3, 3], V, Sublevel),
        vec![
            (0, 1.0, None, [0, 0], None),
            (1, 1.0, Some(5.0), [4, 3], Some([3, 3])),
        ]
    );
    // In the T-construction the ring's top cells already surround the centre's, so the loop is the same
    assert_eq!(
        bars(ring.clone(), [3, 3], T, Sublevel),
        vec![
            (0, 1.0, None, [0, 0], None),
            (1, 1.0, Some(5.0), [4, 3], Some([3, 3])),
        ]
    );
    // Superlevel sets: the bright centre is born first, and the ring merges it at 1 (bars run downwards)
    assert_eq!(
        bars(ring, [3, 3], V, Superlevel),
        vec![(0, 5.0, None, [2, 2], None)]
    );
    // A diagonal: the two dark pixels touch only at a corner, which joins them in the T-construction but not the V
    let diagonal = vec![0.0, 1.0, 1.0, 0.0];
    assert_eq!(
        bars(diagonal.clone(), [2, 2], V, Sublevel),
        vec![
            (0, 0.0, None, [0, 0], None),
            (0, 0.0, Some(1.0), [2, 2], Some([0, 1])),
        ]
    );
    assert_eq!(
        bars(diagonal, [2, 2], T, Sublevel),
        vec![(0, 0.0, None, [0, 0], None)]
    );
    // A NaN pixel is a hole: in the V-construction it leaves a loop of edges that never fills
    let holed = vec![0.0, 0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        bars(holed.clone(), [3, 3], V, Sublevel),
        vec![(0, 0.0, None, [0, 0], None), (1, 0.0, None, [4, 3], None)]
    );
    assert_eq!(
        bars(holed, [3, 3], T, Superlevel),
        vec![(0, 0.0, None, [0, 0], None), (1, 0.0, None, [4, 3], None)]
    );
}

/// Images without pixels have no cells and no bars.
#[test]
fn empty_images() {
    for shape in [[0, 3], [2, 0]] {
        for construction in [Construction::V, Construction::T] {
            let complex =
                CubicalComplex::new(vec![], shape, construction, Filtration::Sublevel).unwrap();
            assert_eq!(complex.n_cells(), 0);
            let bars = cubical_barcode(&complex, 1, Method::Cohomology, Algorithm::default(), true);
            assert!(bars.is_empty());
        }
    }
    assert!(CubicalComplex::new(vec![0.0], [2, 2], Construction::V, Filtration::Sublevel).is_err());
}

/// Every column representation but pivot columns (which need row ranks, which cubical matrices don't have) agrees.
#[test]
fn every_column_representation() {
    let mut rng = Rng(11);
    for _ in 0..20 {
        let (pixels, shape) = random_image::<2>(&mut rng, 6);
        let complex =
            CubicalComplex::new(pixels.clone(), shape, Construction::V, Filtration::Sublevel)
                .unwrap();
        let expected = naive_barcode(&pixels, shape, Construction::V, Filtration::Sublevel);
        for_each_algorithm!(HeapColumns, algorithm, {
            let bars = cubical_barcode(&complex, 1, Method::Cohomology, algorithm, false);
            assert_eq!(summary(&complex, &bars), expected);
        });
        for_each_algorithm!(SortedColumns, algorithm, {
            let bars = cubical_barcode(&complex, 1, Method::Homology, algorithm, false);
            assert_eq!(summary(&complex, &bars), expected);
        });
    }
}

/// The T-construction by the filtered discrete gradient (the default in three dimensions for cohomology without
/// representatives) against the plain reduction, on larger images than the naive check can afford, with ties, missing and
/// infinite pixels, every `max_dim`.
fn check_morse<const D: usize>(seed: u64, cases: usize, max_side: u64) {
    let mut rng = Rng(seed);
    for case in 0..cases {
        let (pixels, shape) = random_image::<D>(&mut rng, max_side);
        for filtration in [Filtration::Sublevel, Filtration::Superlevel] {
            let complex =
                CubicalComplex::new(pixels.clone(), shape, Construction::T, filtration).unwrap();
            for max_dim in 0..D {
                let algorithm = || Algorithm::Clearing(Clearing::new().with_columns(MergedColumns));
                let plain = super::filtration_barcode(
                    &complex,
                    max_dim,
                    Method::Cohomology,
                    algorithm(),
                    false,
                    true,
                    false,
                    false,
                );
                for parallel in [false, true] {
                    let morse = super::filtration_barcode(
                        &complex,
                        max_dim,
                        Method::Cohomology,
                        algorithm(),
                        false,
                        true,
                        true,
                        parallel,
                    );
                    assert_eq!(
                        summary(&complex, &morse),
                        summary(&complex, &plain),
                        "case {case}: {shape:?} {filtration:?} max_dim {max_dim} parallel {parallel} {pixels:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn morse_matches_plain_reduction() {
    check_morse::<3>(13, 150, 9);
}
