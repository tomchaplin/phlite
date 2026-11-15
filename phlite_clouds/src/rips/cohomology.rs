use std::marker::PhantomData;

use ordered_float::NotNan;
use phlite::values::INFINITY;

use crate::rips::{
    build_rips_bases_with_threshold, youngest_facet, Distances, RipsIndex, RipsInt, Vertices,
};
use phlite::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
};

use super::RipsBasis;

/// The Rips coboundary oracle, with simplices indexed by `RipsIndex<I>` (see [`RipsInt`]), and distances from `D`: a
/// dense matrix by default, or a [`SparseDistances`](super::SparseDistances), or a reference to either.
pub struct RipsCoboundary<
    CF: NonZeroCoefficient + Invertible,
    I: RipsInt = usize,
    D: Distances = Vec<Vec<NotNan<f64>>>,
> {
    distances: D,
    /// Cofacets with a larger diameter are not in the complex
    threshold: NotNan<f64>,
    phantom: PhantomData<(CF, I)>,
}

impl<CF: NonZeroCoefficient + Invertible, I: RipsInt, D: Distances> RipsCoboundary<CF, I, D> {
    /// The coboundary oracle on its own, without a column basis attached (see [`build_rips_coboundary_matrix`]).
    pub fn new(distances: D) -> Self {
        Self {
            distances,
            threshold: INFINITY,
            phantom: PhantomData,
        }
    }

    /// Only include simplices with diameter at most `threshold` (the default is no limit).
    pub fn with_threshold(mut self, threshold: NotNan<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    fn n_points(&self) -> usize {
        self.distances.n_points()
    }
}

/// Iterates over the cofacets of a simplex, in increasing order of [`RipsIndex`] (equivalently, of the inserted vertex).
///
/// The vertices to try inserting come from `candidates`, in increasing order (every point for a dense matrix, or the
/// neighbours of one of the simplex's vertices for a sparse one); those already in the simplex are skipped.
/// A cofacet's index is the simplex's index with one extra base-(n + 1) digit inserted, so rather than decoding and re-encoding
/// we track the digits below the insertion position (`low`) and above it (`high`) as the insertion position moves up.
struct CoboundaryIterator<CF: NonZeroCoefficient, I: RipsInt, C> {
    base: Vertices,
    radix: I,
    candidates: C,
    insertion_position: usize,
    /// The value of the digits of the simplex's index below `insertion_position`
    low: I,
    /// The simplex's index, shifted down by `insertion_position` digits
    high: I,
    /// `radix ^ insertion_position`
    place: I,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient, I: RipsInt, C: Iterator<Item = usize>> CoboundaryIterator<CF, I, C> {
    /// The cofacets of `simplex` (whose vertices are `vertices`) adding a vertex from `candidates`
    fn new(simplex: RipsIndex<I>, vertices: Vertices, n_points: usize, candidates: C) -> Self {
        Self {
            base: vertices,
            radix: I::from_usize(n_points + 1),
            candidates,
            insertion_position: 0,
            low: I::ZERO,
            high: simplex.0,
            place: I::ONE,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient, I: RipsInt, C: Iterator<Item = usize>> Iterator
    for CoboundaryIterator<CF, I, C>
{
    // Coefficient, RipsIndex, inserted vertex
    type Item = (CF, RipsIndex<I>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_vertex = self.candidates.next()?;
            // Move the insertion position past the simplex's vertices below next_vertex: each such digit moves from
            // high to low (candidates increase, so the position only ever moves up)
            while let Some(&vertex_at_insert_pos) = self.base.get(self.insertion_position) {
                if vertex_at_insert_pos >= next_vertex {
                    break;
                }
                self.low += I::from_usize(vertex_at_insert_pos + 1) * self.place;
                self.high /= self.radix;
                self.insertion_position += 1;
                self.place *= self.radix;
            }
            if self.base.get(self.insertion_position) == Some(&next_vertex) {
                // Vertex already appears in simplex, try the next
                continue;
            }

            // Insert the digit for next_vertex at insertion_position, shifting the higher digits up by one
            let index = RipsIndex(
                self.low + ((self.high * self.radix) + I::from_usize(next_vertex + 1)) * self.place,
            );
            let coeff = if self.insertion_position % 2 == 0 {
                CF::one()
            } else {
                CF::one().additive_inverse()
            };
            return Some((coeff, index, next_vertex));
        }
    }
}

// Rips boundary matrix, anti-transposed
impl<CF: NonZeroCoefficient + Invertible, I: RipsInt, D: Distances> MatrixOracle
    for RipsCoboundary<CF, I, D>
{
    type CoefficientField = CF;

    // NOTE: We do NOT reverse the order of rips indices but only the order of filtration values
    // This is still a valid order but it will not yield the same pairing as RipsBoundaryMatrix
    // TODO: Should we reverse the index to allow for better testing?
    type ColT = RipsIndex<I>;

    type RowT = RipsIndex<I>;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.column_with_filtration(col)
            .map(|entry| (entry.coeff, entry.row_index))
    }
}

impl<CF: NonZeroCoefficient + Invertible, I: RipsInt, D: Distances> HasRowFiltration
    for RipsCoboundary<CF, I, D>
{
    // Reverse filtration to anti-transpose
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        row.vertices(self.n_points()).diameter(&self.distances)
    }

    // The filtration value of a cofacet is the max of the simplex's diameter and the distances to the inserted vertex
    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        let n_points = self.n_points();
        let col_vertices = col.vertices(n_points);
        let candidates = self
            .distances
            .cofacet_candidates(self.distances.sparsest(&col_vertices), 0);
        let coboundary_iterator = CoboundaryIterator::new(col, col_vertices, n_points, candidates);
        let max_pd_amongst_col = col_vertices.diameter(&self.distances);

        coboundary_iterator.filter_map(move |(coeff, row_index, inserted_vertex)| {
            let max_to_inserted = col_vertices
                .iter()
                .map(|v| self.distances.distance(*v, inserted_vertex))
                .max()
                .unwrap();

            let mut filtration_value = max_pd_amongst_col.max(max_to_inserted);
            if D::HAS_BIRTHS {
                filtration_value = filtration_value.max(self.distances.birth(inserted_vertex));
            }

            (filtration_value <= self.threshold).then_some(ColumnEntry {
                filtration_value,
                row_index,
                coeff,
            })
        })
    }

    // The pivot of the anti-transpose (which is what we reduce) is the smallest entry of the coboundary.
    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        oldest_cofacet(&self.distances, self.threshold, col)
    }

    // Row tau has entries in the columns of its facets, so the last is its youngest facet.
    // This is what detects apparent pairs in the anti-transpose (which reverses the order).
    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        youngest_facet(row, &self.distances).map(|(_diameter, facet)| facet)
    }
}

/// The oldest cofacet of `simplex` (smallest in (diameter, index) order) with diameter at most `threshold`, as a coboundary entry.
///
/// Cofacets are produced in increasing index order and none can appear before the simplex itself,
/// so the first cofacet with the same diameter as the simplex is the oldest and we can stop there.
/// In Rips filtrations most pairs have zero persistence, so this usually stops early.
pub(crate) fn oldest_cofacet<CF: NonZeroCoefficient, I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    threshold: NotNan<f64>,
    simplex: RipsIndex<I>,
) -> Option<ColumnEntry<NotNan<f64>, RipsIndex<I>, CF>> {
    let vertices = simplex.vertices(distances.n_points());
    let diameter = vertices.diameter(distances);
    oldest_cofacet_given(distances, threshold, simplex, vertices, diameter)
}

/// As [`oldest_cofacet`], when the vertices and diameter of `simplex` are already known
pub(crate) fn oldest_cofacet_given<CF: NonZeroCoefficient, I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    threshold: NotNan<f64>,
    simplex: RipsIndex<I>,
    vertices: Vertices,
    diameter: NotNan<f64>,
) -> Option<ColumnEntry<NotNan<f64>, RipsIndex<I>, CF>> {
    let n_points = distances.n_points();
    let candidates = distances.cofacet_candidates(distances.sparsest(&vertices), 0);
    let mut best: Option<ColumnEntry<NotNan<f64>, RipsIndex<I>, CF>> = None;
    for (coeff, row_index, inserted_vertex) in
        CoboundaryIterator::<CF, I, _>::new(simplex, vertices, n_points, candidates)
    {
        let filtration_value = vertices
            .iter()
            .map(|v| distances.distance(*v, inserted_vertex))
            .max()
            .unwrap()
            .max(diameter);
        let filtration_value = if D::HAS_BIRTHS {
            filtration_value.max(distances.birth(inserted_vertex))
        } else {
            filtration_value
        };
        if filtration_value > threshold {
            continue;
        }
        // On ties keep the earlier, smaller, index
        if best
            .as_ref()
            .is_none_or(|current| filtration_value < current.filtration_value)
        {
            best = Some(ColumnEntry {
                filtration_value,
                row_index,
                coeff,
            });
            if filtration_value == diameter {
                break;
            }
        }
    }
    best
}

pub type RipsCoboundaryAllDims<CF, I = usize, D = Vec<Vec<NotNan<f64>>>> =
    MatrixWithBasis<RipsCoboundary<CF, I, D>, RipsBasis<I>>;

pub fn build_rips_coboundary_matrix<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> RipsCoboundaryAllDims<CF> {
    build_rips_coboundary_matrix_with_threshold(
        distances,
        max_dim,
        INFINITY,
    )
}

/// As [`build_rips_coboundary_matrix`], but only the simplices with diameter at most `threshold`, from any
/// [`Distances`].
pub fn build_rips_coboundary_matrix_with_threshold<CF: Invertible, D: Distances>(
    distances: D,
    max_dim: usize,
    threshold: NotNan<f64>,
) -> RipsCoboundaryAllDims<CF, usize, D> {
    // Pass in the Reverse functor to revere filtration order on columns in basis
    let basis = build_rips_bases_with_threshold(&distances, max_dim, threshold);
    RipsCoboundaryAllDims {
        matrix: RipsCoboundary::new(distances).with_threshold(threshold),
        basis,
    }
}
// TODO: Write some proper tests
#[cfg(test)]
mod tests {
    use phlite::values::ZERO;
    use ordered_float::NotNan;

    use crate::rips::{
        cohomology::{build_rips_coboundary_matrix, RipsCoboundary},
        RipsIndex,
    };
    use phlite::{
        fields::Z2,
        matrices::{combinators::product, HasRowFiltration, MatrixOracle},
        reduction::{Clearing, Reduction, ReductionAlgorithm, Standard},
    };

    /// The coboundary and boundary meet phlite's contract, hooks included, and are each other's transposes
    #[test]
    fn meets_the_contract() {
        use crate::rips::homology::build_rips_boundary_matrix;
        use phlite::matrices::contract::{check_coboundary, check_split_basis, check_transposes, Direction};
        for max_dim in 0..=2 {
            let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix(), max_dim);
            assert_eq!(check_coboundary(&coboundary), Ok(()));
            assert_eq!(check_split_basis(&coboundary, Direction::Coboundary), Ok(()));
            let boundary = build_rips_boundary_matrix::<Z2>(distance_matrix(), max_dim);
            assert_eq!(check_transposes(&coboundary, &boundary.matrix), Ok(()));
        }
    }

    /// The optimised `column_with_filtration` must agree with computing each row's filtration value from scratch
    #[test]
    fn test_column_with_filtration_matches_filtration_value() {
        use phlite::matrices::{ColBasis, HasSplitColBasis};
        let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix(), 2);
        for dim in 0..=2 {
            let basis = coboundary.sub_basis(dim);
            for i in 0..basis.size() {
                let col = basis.element(i);
                let fast: Vec<_> = coboundary
                    .matrix
                    .column_with_filtration(col)
                    .map(|entry| (entry.row_index, entry.filtration_value))
                    .collect();
                let slow: Vec<_> = coboundary
                    .matrix
                    .column(col)
                    .map(|(_coeff, row)| (row, coboundary.matrix.filtration_value(row)))
                    .collect();
                assert_eq!(fast, slow, "column {col:?}");
            }
        }
    }

    /// Skipping zero-persistence apparent pairs must drop exactly those pairs from the diagram, serially and in parallel
    #[test]
    fn test_skip_zero_apparent_pairs() {
        use crate::rips::homology::RipsBoundary;
        use phlite::reduction::Clearing;
        // Twelve points on a noisy circle, plus integer distances (lots of ties)
        let points: Vec<(f64, f64)> = (0..12)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 12.0;
                let r = 1.0 + 0.1 * ((i * 7 % 5) as f64 - 2.0);
                (r * t.cos(), r * t.sin())
            })
            .collect();
        let circle: Vec<Vec<NotNan<f64>>> = points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| {
                        NotNan::new(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()).unwrap()
                    })
                    .collect()
            })
            .collect();
        let ties: Vec<Vec<NotNan<f64>>> = (0..9)
            .map(|i| {
                (0..9)
                    .map(|j| {
                        NotNan::new(if i == j {
                            0.0
                        } else {
                            ((i * j + i + j) % 3) as f64
                        })
                        .unwrap()
                    })
                    .collect()
            })
            .collect();
        for distances in [circle, ties] {
            let boundary = RipsBoundary::<Z2>::new(distances.clone());
            let coboundary = build_rips_coboundary_matrix::<Z2>(distances, 2);
            let with_t = coboundary.by_ref().with_transpose(&boundary);
            let anti = with_t.by_ref().reverse();
            let full = Clearing::new()
                .skip_zero_apparent_pairs(false)
                .reduce(&anti)
                .diagram
                .sorted();
            for skipped in [
                Clearing::new().reduce(&anti).diagram.sorted(),
                Clearing::new().parallel().reduce(&anti).diagram.sorted(),
                Clearing::new()
                    .skip_zero_apparent_pairs(true)
                    .reduce(&anti)
                    .diagram
                    .sorted(),
            ] {
                assert_eq!(skipped.essential, full.essential);
                assert!(skipped
                    .pairings
                    .iter()
                    .all(|pair| full.pairings.contains(pair)));
                let dropped: Vec<_> = full
                    .pairings
                    .iter()
                    .filter(|pair| !skipped.pairings.contains(pair))
                    .collect();
                assert!(!dropped.is_empty(), "nothing was skipped");
                for (row, col) in &full.pairings {
                    let zero = anti.filtration_value(*row) == anti.filtration_value(*col);
                    let apparent = anti.first_column_in_row(*row) == Some(*col);
                    assert_eq!(
                        skipped.pairings.contains(&(*row, *col)),
                        !(zero && apparent),
                        "{row:?} {col:?}"
                    );
                }
            }
        }
    }

    /// Cofacet and facet indices are computed arithmetically; check them against encoding the vertex sets directly
    #[test]
    fn test_index_arithmetic_matches_encoding() {
        use crate::rips::homology::RipsBoundary;
        use phlite::fields::{NonZeroCoefficient, Z3};
        use phlite::matrices::{ColBasis, HasSplitColBasis};
        // Over Z3 so that the signs are checked too
        let sign = |position: usize| {
            if position.is_multiple_of(2) {
                Z3::one()
            } else {
                Z3::one().additive_inverse()
            }
        };
        let n = 9;
        let distances: Vec<Vec<NotNan<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| NotNan::new(if i == j { 0.0 } else { (i + j) as f64 }).unwrap())
                    .collect()
            })
            .collect();
        let coboundary = build_rips_coboundary_matrix::<Z3>(distances.clone(), 3);
        let boundary = RipsBoundary::<Z3>::new(distances);
        let mut checked = 0;
        for dim in 0..=3 {
            let basis = coboundary.sub_basis(dim);
            for i in 0..basis.size() {
                let simplex = basis.element(i);
                let vertices = simplex.to_vec(n);
                assert_eq!(vertices.len(), dim + 1);
                assert_eq!(simplex.dimension(n), dim);
                // Cofacets: insert each vertex not already present, in increasing order, with sign (-1)^position
                let expected: Vec<(Z3, RipsIndex)> = (0..n)
                    .filter(|v| !vertices.contains(v))
                    .map(|v| {
                        let mut cofacet = vertices.clone();
                        cofacet.push(v);
                        cofacet.sort();
                        let position = cofacet.iter().position(|w| *w == v).unwrap();
                        (
                            sign(position),
                            RipsIndex::<usize>::from_indices(cofacet.into_iter(), n).unwrap(),
                        )
                    })
                    .collect();
                let got: Vec<_> = coboundary.matrix.column(simplex).collect();
                assert_eq!(got, expected, "cofacets of {vertices:?}");
                // Facets: remove each vertex in turn
                let expected: Vec<(Z3, RipsIndex)> = if dim == 0 {
                    vec![]
                } else {
                    (0..=dim)
                        .map(|i| {
                            let facet = vertices
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, v)| *v);
                            (sign(i), RipsIndex::<usize>::from_indices(facet, n).unwrap())
                        })
                        .collect()
                };
                let got: Vec<_> = boundary.column(simplex).collect();
                assert_eq!(got, expected, "facets of {vertices:?}");
                checked += 1;
            }
        }
        assert_eq!(checked, 9 + 36 + 84 + 126);
    }

    /// The early-stopping first/last column lookups (which find apparent pairs) must agree with brute force
    #[test]
    fn test_first_last_column_in_row_match_brute_force() {
        use crate::rips::homology::RipsBoundary;
        use phlite::matrices::{ColBasis, HasSplitColBasis};
        let n = 7;
        let ties: Vec<Vec<NotNan<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        NotNan::new(if i == j {
                            0.0
                        } else {
                            ((i * j + i + j) % 4) as f64
                        })
                        .unwrap()
                    })
                    .collect()
            })
            .collect();
        for dists in [ties, distance_matrix()] {
            for threshold in [f64::INFINITY, 2.0] {
                let threshold = NotNan::new(threshold).unwrap();
                let coboundary = super::build_rips_coboundary_matrix_with_threshold::<Z2, _>(
                    dists.clone(),
                    3,
                    threshold,
                );
                let boundary = RipsBoundary::<Z2>::new(dists.clone()).with_threshold(threshold);
                for dim in 0..=3 {
                    let basis = coboundary.sub_basis(dim);
                    for i in 0..basis.size() {
                        let simplex = basis.element(i);
                        // Row `simplex` of the coboundary has entries in its facets' columns
                        let youngest_facet = boundary
                            .column(simplex)
                            .map(|(_, facet)| (coboundary.matrix.filtration_value(facet), facet))
                            .max()
                            .map(|(_, facet)| facet);
                        assert_eq!(
                            coboundary.matrix.last_column_in_row(simplex),
                            youngest_facet,
                            "{simplex:?}"
                        );
                        // Row `simplex` of the boundary has entries in its cofacets' columns
                        let oldest_cofacet = coboundary
                            .matrix
                            .column_with_filtration(simplex)
                            .min()
                            .map(|entry| entry.row_index);
                        assert_eq!(
                            boundary.first_column_in_row(simplex),
                            oldest_cofacet,
                            "{simplex:?}"
                        );
                    }
                }
            }
        }
    }

    /// `column_min` stops early, relying on cofacets coming out in increasing index order; check it against brute force
    #[test]
    fn test_column_min_matches_brute_force() {
        use phlite::matrices::{ColBasis, HasSplitColBasis};
        // Integer distances, so lots of ties, and a non-metric
        let n = 7;
        let distances: Vec<Vec<NotNan<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        NotNan::new(if i == j {
                            0.0
                        } else {
                            ((i * j + i + j) % 4) as f64
                        })
                        .unwrap()
                    })
                    .collect()
            })
            .collect();
        for dists in [distances, distance_matrix()] {
            let coboundary = build_rips_coboundary_matrix::<Z2>(dists, 2);
            for dim in 0..=2 {
                let basis = coboundary.sub_basis(dim);
                for i in 0..basis.size() {
                    let col = basis.element(i);
                    let entries: Vec<_> = coboundary.matrix.column_with_filtration(col).collect();
                    assert!(
                        entries.windows(2).all(|w| w[0].row_index < w[1].row_index),
                        "cofacets of {col:?} not in increasing order"
                    );
                    let brute = entries
                        .into_iter()
                        .min()
                        .map(|e| (e.filtration_value, e.row_index));
                    let fast = coboundary
                        .matrix
                        .column_min(col)
                        .map(|e| (e.filtration_value, e.row_index));
                    assert_eq!(fast, brute, "column {col:?}");
                }
            }
        }
    }

    #[test]
    fn test_coboundary() {
        let distance_matrix = distance_matrix();
        let coboundary: RipsCoboundary<Z2> = RipsCoboundary::new(distance_matrix);

        let smplx = RipsIndex::<usize>::from_indices(vec![1, 2, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 1);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::<usize>::from_indices(vec![0, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 2);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::<usize>::from_indices(vec![1].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 3);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }
    }

    fn distance_matrix() -> Vec<Vec<NotNan<f64>>> {
        let zero = ZERO;
        let one = NotNan::new(1.0).unwrap();
        let sqrt2 = NotNan::new(2.0_f64.sqrt()).unwrap();
        let distance_matrix: Vec<Vec<NotNan<f64>>> = vec![
            vec![zero, one, sqrt2, one],
            vec![one, zero, one, sqrt2],
            vec![sqrt2, one, zero, one],
            vec![one, sqrt2, one, zero],
        ];
        distance_matrix
    }

    #[test]
    fn test_rips_total_coboundary() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 2;

        // Compute column basis
        let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix, max_dim);
        let coboundary = coboundary.reverse();
        // Compute reduction matrix
        let Reduction { v, diagram } = Standard::new().reduce(&coboundary);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).0;
            let dim = idx.0.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1 .0.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).0;
            let death_f = coboundary.filtration_value(tup.0).0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // 2-dimensional void is killed without having to compute basis for C_3
        assert_eq!(diagram.pairings.len(), 7);
        assert_eq!(diagram.essential.len(), 1);
    }

    /// A coboundary matrix maps dimension d to d + 1, so clearing should detect an ascending order,
    /// and the diagram should be the same whatever order is used.
    #[test]
    fn test_clearing_dimension_order() {
        for max_dim in 1..=2 {
            let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix(), max_dim);
            let coboundary = coboundary.reverse();
            assert_eq!(
                Clearing::new().dimension_order(&coboundary),
                (0..=max_dim).collect::<Vec<_>>()
            );
            // Keep every pair, to compare with the standard algorithm
            let full = Clearing::new().skip_zero_apparent_pairs(false);
            let detected = full.clone().reduce(&coboundary).diagram.sorted();
            let standard = Standard::new().reduce(&coboundary).diagram.sorted();
            // Descending is the wrong way round for a coboundary matrix: no clearing happens, but the diagram is still right
            let wrong_order = full
                .with_dimension_order((0..=max_dim).rev())
                .reduce(&coboundary)
                .diagram
                .sorted();
            for diagram in [&standard, &wrong_order] {
                assert_eq!(detected.essential, diagram.essential);
                assert_eq!(detected.pairings, diagram.pairings);
            }
            // Skipping zero-persistence apparent pairs (the default) drops some pairs, but never makes their births
            // essential, in either order
            for order in [
                (0..=max_dim).collect::<Vec<_>>(),
                (0..=max_dim).rev().collect(),
            ] {
                let skipped = Clearing::new()
                    .with_dimension_order(order)
                    .reduce(&coboundary)
                    .diagram
                    .sorted();
                assert_eq!(detected.essential, skipped.essential);
                assert!(skipped
                    .pairings
                    .iter()
                    .all(|pair| detected.pairings.contains(pair)));
            }
        }
    }

    #[test]
    fn test_parallel_clearing_matches_serial() {
        // Rips cohomology never enumerates the row space, so this checks the label-keyed pivot table on a
        // matrix whose rows are implicit (RipsIndex labels) as well as on grpph's
        for max_dim in 1..=2 {
            let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix(), max_dim);
            let coboundary = coboundary.reverse();
            let serial = Clearing::new().reduce(&coboundary).diagram.sorted();
            // Descending is the wrong way round for a coboundary matrix, the diagram should still be right
            let wrong_order = Clearing::new()
                .with_dimension_order((0..=max_dim).rev())
                .parallel()
                .reduce(&coboundary)
                .diagram
                .sorted();
            assert_eq!(serial.essential, wrong_order.essential);
            assert_eq!(serial.pairings, wrong_order.pairings);
            for _ in 0..20 {
                let parallel = Clearing::new()
                    .parallel()
                    .reduce(&coboundary)
                    .diagram
                    .sorted();
                assert_eq!(serial.essential, parallel.essential);
                assert_eq!(serial.pairings, parallel.pairings);
                #[cfg(feature = "lock-free")]
                {
                    let lock_free = Clearing::new()
                        .parallel()
                        .storage(phlite::reduction::Storage::LockFree)
                        .reduce(&coboundary)
                        .diagram
                        .sorted();
                    assert_eq!(serial.essential, lock_free.essential);
                    assert_eq!(serial.pairings, lock_free.pairings);
                }
            }
        }
    }

    #[test]
    fn test_clearing() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 1;

        // Compute column basis
        let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix, max_dim);
        let coboundary = coboundary.reverse();
        // Compute reduction matrix, in increasing dimension
        let Reduction { v, diagram } = Clearing::new()
            .skip_zero_apparent_pairs(false)
            .reduce(&coboundary);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).0;
            let dim = idx.0.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1 .0.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).0;
            let death_f = coboundary.filtration_value(tup.0).0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // Ignored 2-dimensional void
        assert_eq!(diagram.pairings.len(), 6);
        assert_eq!(diagram.essential.len(), 1);
    }
}
