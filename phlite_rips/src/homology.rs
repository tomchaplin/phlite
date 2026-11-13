use super::RipsBasis;
use crate::{build_rips_bases_with_threshold, cohomology::oldest_cofacet, RipsIndex, RipsInt};
use ordered_float::NotNan;
use phlite::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
};
use std::marker::PhantomData;

// TUTORIAL:
// Now we get into the nitty-gritty of implementing the boundary matrix.
// Note, we do not store the column basis in the following stuct.
// Instead, we just store the distance matrix since given a simplex, this all we need to compute its boundary.
// We will attach the column basis later on!
//
// We add the phantom because we want to be generic over coefficient type
// but they don't appear anywhere else in the struct so the compiler must be appeased.

pub struct RipsBoundary<CF: NonZeroCoefficient + Invertible, I: RipsInt = usize> {
    distances: Vec<Vec<NotNan<f64>>>,
    /// Simplices with a larger diameter are not in the complex (only matters for finding cofacets)
    threshold: NotNan<f64>,
    phantom: PhantomData<(CF, I)>,
}

impl<CF: NonZeroCoefficient + Invertible, I: RipsInt> RipsBoundary<CF, I> {
    /// The boundary oracle on its own, without a column basis attached (see [`build_rips_boundary_matrix`]).
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

// TUTORIAL:
// Now we implement `MatrixOracle` - this is where we implicity describe the boundary matrix.
// We have to choose:
// * The coefficient type - we are generic over this;
// * The index type for rows and columns - we use `RipsIndex` for both
//   (generic over the integer it is stored in, `usize` by default).
//
// Finally, we have to describe each column by providing an iterator over its non-zero entries.
// An entry is described by the row index and the value of the coefficient.
// There is no requirement on the order in which the entries are produced.

impl<CF: NonZeroCoefficient + Invertible, I: RipsInt> MatrixOracle for RipsBoundary<CF, I> {
    type CoefficientField = CF;

    type ColT = RipsIndex<I>;

    type RowT = RipsIndex<I>;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // Removing the vertex in position i (i.e. the i-th base-(n + 1) digit of the index) gives a facet with sign (-1)^i.
        // We compute each facet's index arithmetically rather than decoding and re-encoding the vertices.
        let radix = I::from_usize(self.n_points() + 1);
        let index = col.0;
        let n_vertices = col.dimension(self.n_points()) + 1;
        // A vertex has an empty boundary (removing its only vertex leaves nothing)
        let n_facets = if n_vertices > 1 { n_vertices } else { 0 };
        (0..n_facets).scan(I::ONE, move |place, i| {
            // place = radix^i
            let low = index % *place;
            let high = index / *place / radix;
            let facet = RipsIndex(low + high * *place);
            *place = place.wrapping_mul(radix);
            let coeff = if i % 2 == 0 {
                CF::one()
            } else {
                CF::one().additive_inverse()
            };
            Some((coeff, facet))
        })
    }
}

// TUTORIAL:
// Since we have stored the distance matrix, this boundary matrix also naturally has a filtration on the columns.
// We describe this here by implementing `HasRowFiltration`.
// In your application, it might make sense to implement a boundary matrix without a row filtration
// then wrap it in an additional struct that has the filtration information.
// In these cases, consider using `phlite::matricies::MatrixRef::with_filtration`.

impl<CF: NonZeroCoefficient + Invertible, I: RipsInt> HasRowFiltration for RipsBoundary<CF, I> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        row.vertices(self.n_points()).diameter(&self.distances)
    }

    // Row sigma has entries in the columns of its cofacets, so the first is its oldest cofacet
    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        oldest_cofacet::<CF, I>(&self.distances, self.threshold, row).map(|entry| entry.row_index)
    }

    // TODO:Can we override column with filtration to compute more efficiently?
}

// TUTORIAL:
// Now we provide a type that encapsulates our boundary matrix (which has a built-in row filtration) alongside our column basis.
// For this we use `MatrixWithBasis` which is provided by `phltie`.
// We also provide a helper function for constructing such a thing directly from a distance matrix.

pub type RipsBoundaryAllDims<CF, I = usize> = MatrixWithBasis<RipsBoundary<CF, I>, RipsBasis<I>>;

pub fn build_rips_boundary_matrix<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> RipsBoundaryAllDims<CF> {
    build_rips_boundary_matrix_with_threshold(
        distances,
        max_dim,
        NotNan::new(f64::INFINITY).unwrap(),
    )
}

/// As [`build_rips_boundary_matrix`], but only the simplices with diameter at most `threshold`.
/// (The boundary of such a simplex is automatically within the threshold.)
pub fn build_rips_boundary_matrix_with_threshold<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
) -> RipsBoundaryAllDims<CF> {
    let basis = build_rips_bases_with_threshold(&distances, max_dim, threshold);
    RipsBoundaryAllDims {
        matrix: RipsBoundary::new(distances).with_threshold(threshold),
        basis,
    }
}

// TUTORIAL:
// Now it's time to choose your own adventure!
// 1. Head over to cohomology.rs to see how we implement the coboundary matrix.
//    The structure is pretty similar but implementation is a bit more complicated and can be optimised.
// 2. Head over to main.rs to see how we use the reduction algorithms.
//    We actually decompose the (reversed) coboundary matrix since its more efficient.
//    However, rewriting main.rs to use the boundary matrix could be instructive!

// TODO: Write some proper tests
#[cfg(test)]
mod tests {

    use ordered_float::NotNan;

    use crate::homology::build_rips_boundary_matrix;
    use phlite::{
        fields::Z2,
        matrices::{combinators::product, HasRowFiltration},
        reduction::{Clearing, Reduction, ReductionAlgorithm, Standard},
    };

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
    fn test_rips_total_boundary() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 2;

        // Compute column basis
        let boundary = build_rips_boundary_matrix::<Z2>(distance_matrix, max_dim);
        // Compute reduction matrix
        let Reduction { v, diagram } = Standard::new().reduce(&boundary);
        let _r = &product(&boundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = boundary.filtration_value(*idx);
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.0.dimension(n_points);
            let idx_tup = (tup.0, tup.1);
            let birth_f = boundary.filtration_value(tup.0);
            let death_f = boundary.filtration_value(tup.1);
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        assert_eq!(diagram.pairings.len(), 6);
        // Additional essential idx because we don't fill in a 2-void
        assert_eq!(diagram.essential.len(), 2);
    }

    /// Ten points on a noisy circle, so that there is some H_1 and some cancellation in H_2
    fn circle_distances() -> Vec<Vec<NotNan<f64>>> {
        let points: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 10.0;
                let r = 1.0 + 0.1 * ((i * 7 % 5) as f64 - 2.0);
                (r * t.cos(), r * t.sin())
            })
            .collect();
        points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| {
                        NotNan::new(((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()).unwrap()
                    })
                    .collect()
            })
            .collect()
    }

    /// A boundary matrix maps dimension d to d - 1, so clearing should detect a descending order,
    /// and the diagram should agree with the standard algorithm whatever order is used.
    #[test]
    fn test_clearing_on_boundary_matrix() {
        for max_dim in 1..=2 {
            let boundary = build_rips_boundary_matrix::<Z2>(circle_distances(), max_dim);
            assert_eq!(
                Clearing::new().dimension_order(&boundary),
                (0..=max_dim).rev().collect::<Vec<_>>()
            );

            let standard = Standard::new().reduce(&boundary).diagram.sorted();
            // Keep every pair, to compare with the standard algorithm
            let full = Clearing::new().skip_zero_apparent_pairs(false);
            let detected = full.clone().reduce(&boundary).diagram.sorted();
            // Ascending is the wrong way round for a boundary matrix: no clearing happens, but the diagram is still right
            let wrong_order = full
                .with_dimension_order(0..=max_dim)
                .reduce(&boundary)
                .diagram
                .sorted();
            for diagram in [&detected, &wrong_order] {
                assert_eq!(standard.essential, diagram.essential);
                assert_eq!(standard.pairings, diagram.pairings);
            }
            // Skipping zero-persistence apparent pairs (the default) drops some pairs, but never makes their births
            // essential, in either order
            for order in [
                (0..=max_dim).collect::<Vec<_>>(),
                (0..=max_dim).rev().collect(),
            ] {
                let skipped = Clearing::new()
                    .with_dimension_order(order)
                    .reduce(&boundary)
                    .diagram
                    .sorted();
                assert_eq!(standard.essential, skipped.essential);
                assert!(skipped
                    .pairings
                    .iter()
                    .all(|pair| standard.pairings.contains(pair)));
            }
        }
    }
}
