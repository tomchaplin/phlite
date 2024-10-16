use super::MultiDimRipsBasisWithFilt;
use crate::{build_rips_bases, max_pairwise_distance, RipsIndex};
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

pub struct RipsBoundary<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundary<CF> {
    fn n_points(&self) -> usize {
        self.distances.len()
    }
}

// TUTORIAL:
// Now we implement `MatrixOracle` - this is where we implicity describe the boundary matrix.
// We have to choose:
// * The coefficient type - we are generic over this;
// * The index type for rows and columns - we use `RipsIndex` for both.
//
// Finally, we have to describe each column by providing an iterator over its non-zero entries.
// An entry is described by the row index and the value of the coefficient.
// There is no requirement on the order in which the entries are produced.

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsBoundary<CF> {
    type CoefficientField = CF;

    type ColT = RipsIndex;

    type RowT = RipsIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let index_as_vec = col.to_vec(self.n_points());
        let parity = |i: usize| {
            if i % 2 == 0 {
                CF::one()
            } else {
                CF::one().additive_inverse()
            }
        };
        let n_points = self.n_points();
        // TODO: Return early if n_points == 0
        (0..(index_as_vec.len())).filter_map(move |i| {
            // Iterator over all but the ith vertices
            let smplx = index_as_vec
                .iter()
                .enumerate()
                .filter(move |(j, _v)| *j != i)
                .map(|(_j, v)| v)
                .copied();
            let index = RipsIndex::from_indices(smplx, n_points)?;
            let coeff = parity(i);
            Some((coeff, index))
        })
    }
}

// TUTORIAL:
// Since we have stored the distance matrix, this boundary matrix also naturally has a filtration on the columns.
// We describe this here by implementing `HasRowFiltration`.
// In your application, it might make sense to implement a boundary matrix without a row filtration
// then wrap it in an additional struct that has the filtration information.
// In these cases, consider using `phlite::matricies::MatrixRef::with_filtration`.

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsBoundary<CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        let as_vec = row.to_vec(self.n_points());
        max_pairwise_distance(&as_vec, &self.distances)
    }

    // TODO:Can we override column with filtration to compute more efficiently?
}

// TUTORIAL:
// Now we provide a type that encapsulates our boundary matrix (which has a built-in row filtration) alongside our column basis.
// For this we use `MatrixWithBasis` which is provided by `phltie`.
// We also provide a helper function for constructing such a thing directly from a distance matrix.

pub type RipsBoundaryAllDims<CF> = MatrixWithBasis<RipsBoundary<CF>, MultiDimRipsBasisWithFilt>;

pub fn build_rips_boundary_matrix<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> RipsBoundaryAllDims<CF> {
    let basis = build_rips_bases(&distances, max_dim);
    RipsBoundaryAllDims {
        matrix: RipsBoundary {
            distances,
            phantom: PhantomData,
        },
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
        reduction::standard_algo_with_diagram,
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
        let (v, diagram) = standard_algo_with_diagram(&boundary, false);
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
}
