use std::{convert::identity, marker::PhantomData};

use ordered_float::NotNan;

use crate::{build_rips_bases, max_pairwise_distance, RipsIndex};
use phlite::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{adaptors::MatrixWithBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

use super::MultiDimRipsBasisWithFilt;

pub struct RipsBoundary<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundary<CF> {
    fn n_points(&self) -> usize {
        self.distances.len()
    }
}

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsBoundary<CF> {
    type CoefficientField = CF;

    type ColT = RipsIndex;

    type RowT = RipsIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
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
        Ok((0..(index_as_vec.len())).filter_map(move |i| {
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
        }))
    }
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsBoundary<CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, phlite::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        Ok(max_pairwise_distance(&as_vec, &self.distances))
    }

    // TODO:Can we override column with filtration to compute more efficiently?
}

pub struct RipsBoundaryAllDims<CF: Invertible>(
    MatrixWithBasis<RipsBoundary<CF>, MultiDimRipsBasisWithFilt<NotNan<f64>>>,
);

impl<CF: Invertible> MatrixOracle for RipsBoundaryAllDims<CF> {
    type CoefficientField = CF;

    type ColT = <MatrixWithBasis<
        RipsBoundary<CF>,
        MultiDimRipsBasisWithFilt<NotNan<f64>>,
    > as MatrixOracle>::ColT;

    type RowT = <MatrixWithBasis<
        RipsBoundary<CF>,
        MultiDimRipsBasisWithFilt<NotNan<f64>>,
    > as MatrixOracle>::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
        self.0.column(col)
    }
}

impl<CF: Invertible> HasRowFiltration for RipsBoundaryAllDims<CF> {
    type FiltrationT = <MatrixWithBasis<
        RipsBoundary<CF>,
        MultiDimRipsBasisWithFilt<NotNan<f64>>,
    > as HasRowFiltration>::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, phlite::PhliteError> {
        self.0.filtration_value(row)
    }
}

impl<CF: Invertible> HasColBasis for RipsBoundaryAllDims<CF> {
    type BasisT = <MatrixWithBasis<
        RipsBoundary<CF>,
        MultiDimRipsBasisWithFilt<NotNan<f64>>,
    > as HasColBasis>::BasisT;

    fn basis(&self) -> &Self::BasisT {
        self.0.basis()
    }
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundaryAllDims<CF> {
    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let basis = build_rips_bases(&distances, max_dim, identity);
        RipsBoundaryAllDims(MatrixWithBasis {
            matrix: RipsBoundary {
                distances,
                phantom: PhantomData,
            },
            basis,
        })
    }
}

// TODO: Write some proper tests
#[cfg(test)]
mod tests {

    use ordered_float::NotNan;

    use crate::homology::RipsBoundaryAllDims;
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
        let boundary = RipsBoundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix
        let (v, diagram) = standard_algo_with_diagram(&boundary, false);
        let _r = &product(&boundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = boundary.filtration_value(*idx).unwrap();
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.0.dimension(n_points);
            let idx_tup = (tup.0, tup.1);
            let birth_f = boundary.filtration_value(tup.0).unwrap();
            let death_f = boundary.filtration_value(tup.1).unwrap();
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        assert_eq!(diagram.pairings.len(), 6);
        // Additional essential idx because we don't fill in a 2-void
        assert_eq!(diagram.essential.len(), 2);
    }
}
