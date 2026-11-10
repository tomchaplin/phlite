use std::marker::PhantomData;

use ordered_float::NotNan;

use crate::{build_rips_bases_with_threshold, RipsIndex, Vertices};
use phlite::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
};

use super::RipsBasis;

// TODO: distances only needs to be a reference

pub struct RipsCoboundary<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    /// Cofacets with a larger diameter are not in the complex
    threshold: NotNan<f64>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsCoboundary<CF> {
    /// The coboundary oracle on its own, without a column basis attached (see [`build_rips_coboundary_matrix`]).
    pub fn new(distances: Vec<Vec<NotNan<f64>>>) -> Self {
        Self {
            distances,
            threshold: NotNan::new(f64::INFINITY).unwrap(),
            phantom: PhantomData,
        }
    }

    /// Only include simplices with diameter at most `threshold` (the default is no limit).
    pub fn with_threshold(mut self, threshold: NotNan<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    fn n_points(&self) -> usize {
        self.distances.len()
    }
}

/// Iterates over the cofacets of a simplex, in increasing order of [`RipsIndex`] (equivalently, of the inserted vertex).
///
/// A cofacet's index is the simplex's index with one extra base-(n + 1) digit inserted, so rather than decoding and re-encoding
/// we track the digits below the insertion position (`low`) and above it (`high`) as the insertion position moves up.
struct CoboundaryIterator<CF: NonZeroCoefficient> {
    base: Vertices,
    radix: usize,
    next_vertex: usize,
    max_vertex: usize,
    insertion_position: usize,
    /// The value of the digits of the simplex's index below `insertion_position`
    low: usize,
    /// The simplex's index, shifted down by `insertion_position` digits
    high: usize,
    /// `radix ^ insertion_position`
    place: usize,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient> CoboundaryIterator<CF> {
    fn new(simplex: RipsIndex, n_points: usize) -> Self {
        Self {
            base: simplex.vertices(n_points),
            radix: n_points + 1,
            next_vertex: 0,
            max_vertex: n_points - 1,
            insertion_position: 0,
            low: 0,
            high: simplex.0,
            place: 1,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> Iterator for CoboundaryIterator<CF> {
    // Coefficient, RipsIndex, inserted vertex
    type Item = (CF, RipsIndex, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // By the end of this loop self.next_vertex and self.insertion_position
        // will be setup correctly so that the next simplex
        // can be obtained by inserting self.next_vertex into index self.insertion_position (which may be the end fo self.base)
        loop {
            if self.next_vertex > self.max_vertex {
                // Ran out of vertices to try and insert
                return None;
            }
            let Some(&vertex_at_insert_pos) = self.base.get(self.insertion_position) else {
                // The vertex will have to go at the end
                break;
            };

            match vertex_at_insert_pos.cmp(&self.next_vertex) {
                std::cmp::Ordering::Less => {
                    // next_vertex has to go afterwards, so this digit moves from high to low
                    self.low += (vertex_at_insert_pos + 1) * self.place;
                    self.high /= self.radix;
                    self.insertion_position += 1;
                    self.place *= self.radix;
                }
                std::cmp::Ordering::Equal => {
                    // Vertex already appears in simplex, try and insert next vertex
                    self.next_vertex += 1;
                }
                std::cmp::Ordering::Greater => {
                    // We have found the correct insertion position
                    break;
                }
            }
        }

        // Insert the digit for next_vertex at insertion_position, shifting the higher digits up by one
        let index =
            RipsIndex(self.low + ((self.high * self.radix) + self.next_vertex + 1) * self.place);
        let coeff = if self.insertion_position % 2 == 0 {
            CF::one()
        } else {
            CF::one().additive_inverse()
        };

        let inserted_vertex = self.next_vertex;
        self.next_vertex += 1;
        Some((coeff, index, inserted_vertex))
    }
}

// Rips boundary matrix, anti-transposed
impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsCoboundary<CF> {
    type CoefficientField = CF;

    // NOTE: We do NOT reverse the order of rips indices but only the order of filtration values
    // This is still a valid order but it will not yield the same pairing as RipsBoundaryMatrix
    // TODO: Should we reverse the index to allow for better testing?
    type ColT = RipsIndex;

    type RowT = RipsIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.column_with_filtration(col)
            .map(|entry| (entry.coeff, entry.row_index))
    }
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundary<CF> {
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
        let coboundary_iterator = CoboundaryIterator::new(col, n_points);
        let col_vertices = col.vertices(n_points);
        let max_pd_amongst_col = col_vertices.diameter(&self.distances);

        coboundary_iterator.filter_map(move |(coeff, row_index, inserted_vertex)| {
            let max_to_inserted = col_vertices
                .iter()
                .map(|v| self.distances[*v][inserted_vertex])
                .max()
                .unwrap();

            let filtration_value = max_pd_amongst_col.max(max_to_inserted);

            (filtration_value <= self.threshold).then_some(ColumnEntry {
                filtration_value,
                row_index,
                coeff,
            })
        })
    }

    // The pivot of the anti-transpose (which is what we reduce) is the smallest entry of the coboundary.
    // Cofacets are produced in increasing index order and none can appear before the simplex itself,
    // so the first cofacet with the same diameter as the simplex is the smallest and we can stop there.
    // In Rips filtrations most pairs have zero persistence, so this usually stops early.
    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        let n_points = self.n_points();
        let col_vertices = col.vertices(n_points);
        let diameter = col_vertices.diameter(&self.distances);
        let mut best: Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> =
            None;
        for (coeff, row_index, inserted_vertex) in CoboundaryIterator::new(col, n_points) {
            let filtration_value = col_vertices
                .iter()
                .map(|v| self.distances[*v][inserted_vertex])
                .max()
                .unwrap()
                .max(diameter);
            if filtration_value > self.threshold {
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
}

pub type RipsCoboundaryAllDims<CF> = MatrixWithBasis<RipsCoboundary<CF>, RipsBasis>;

pub fn build_rips_coboundary_matrix<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> RipsCoboundaryAllDims<CF> {
    build_rips_coboundary_matrix_with_threshold(
        distances,
        max_dim,
        NotNan::new(f64::INFINITY).unwrap(),
    )
}

/// As [`build_rips_coboundary_matrix`], but only the simplices with diameter at most `threshold`.
pub fn build_rips_coboundary_matrix_with_threshold<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
) -> RipsCoboundaryAllDims<CF> {
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
    use ordered_float::NotNan;

    use crate::{
        cohomology::{build_rips_coboundary_matrix, RipsCoboundary},
        RipsIndex,
    };
    use phlite::{
        fields::Z2,
        matrices::{combinators::product, HasRowFiltration, MatrixOracle},
        reduction::{Clearing, Reduction, ReductionAlgorithm, Standard},
    };

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
        use crate::homology::RipsBoundary;
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
            let with_t = (&coboundary).with_transpose(&boundary);
            let anti = (&with_t).reverse();
            let full = Clearing::new().reduce(&anti).diagram;
            for skipped in [
                Clearing::new()
                    .skip_zero_apparent_pairs()
                    .reduce(&anti)
                    .diagram,
                Clearing::new()
                    .parallel()
                    .skip_zero_apparent_pairs()
                    .reduce(&anti)
                    .diagram,
            ] {
                assert_eq!(skipped.essential, full.essential);
                assert!(skipped.pairings.is_subset(&full.pairings));
                let dropped: Vec<_> = full.pairings.difference(&skipped.pairings).collect();
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
        use crate::homology::RipsBoundary;
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
                            RipsIndex::from_indices(cofacet.into_iter(), n).unwrap(),
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
                            (sign(i), RipsIndex::from_indices(facet, n).unwrap())
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

        let smplx = RipsIndex::from_indices(vec![1, 2, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 1);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::from_indices(vec![0, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 2);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::from_indices(vec![1].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).collect();
        assert_eq!(coboundary_vec.len(), 3);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }
    }

    fn distance_matrix() -> Vec<Vec<NotNan<f64>>> {
        let zero = NotNan::new(0.0).unwrap();
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
            let detected = Clearing::new().reduce(&coboundary).diagram;
            let standard = Standard::new().reduce(&coboundary).diagram;
            // Descending is the wrong way round for a coboundary matrix: no clearing happens, but the diagram is still right
            let wrong_order = Clearing::new()
                .with_dimension_order((0..=max_dim).rev())
                .reduce(&coboundary)
                .diagram;
            for diagram in [&standard, &wrong_order] {
                assert_eq!(detected.essential, diagram.essential);
                assert_eq!(detected.pairings, diagram.pairings);
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
            let serial = Clearing::new().reduce(&coboundary).diagram;
            // Descending is the wrong way round for a coboundary matrix, the diagram should still be right
            let wrong_order = Clearing::new()
                .with_dimension_order((0..=max_dim).rev())
                .parallel()
                .reduce(&coboundary)
                .diagram;
            assert_eq!(serial.essential, wrong_order.essential);
            assert_eq!(serial.pairings, wrong_order.pairings);
            for _ in 0..20 {
                let parallel = Clearing::new().parallel().reduce(&coboundary).diagram;
                assert_eq!(serial.essential, parallel.essential);
                assert_eq!(serial.pairings, parallel.pairings);
                #[cfg(feature = "lock-free")]
                {
                    let lock_free = Clearing::new()
                        .parallel()
                        .storage(phlite::reduction::Storage::LockFree)
                        .reduce(&coboundary)
                        .diagram;
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
        let Reduction { v, diagram } = Clearing::new().reduce(&coboundary);
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
