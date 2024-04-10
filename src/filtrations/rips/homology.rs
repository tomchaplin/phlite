use std::{convert::identity, marker::PhantomData};

use ordered_float::NotNan;

use crate::{
    fields::{Invertible, NonZeroCoefficient},
    filtrations::rips::{build_rips_bases, max_pairwise_distance, RipsIndex},
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
};

use super::{MultiDimRipsBasisWithFilt, SingleDimRipsBasisWithFilt};

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
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
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

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        Ok(max_pairwise_distance(&as_vec, &self.distances))
    }

    // TODO:Can we override column with filtration to compute more efficiently?
}

pub type RipsBoundarySingleDim<'a, CF> =
    MatrixWithBasis<&'a RipsBoundary<CF>, &'a SingleDimRipsBasisWithFilt<NotNan<f64>>>;

pub type RipsBoundaryAllDims<CF> =
    MatrixWithBasis<RipsBoundary<CF>, MultiDimRipsBasisWithFilt<NotNan<f64>>>;

impl<CF: NonZeroCoefficient + Invertible> RipsBoundaryAllDims<CF> {
    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let basis = build_rips_bases(&distances, max_dim, identity);
        RipsBoundaryAllDims {
            matrix: RipsBoundary {
                distances,
                phantom: PhantomData,
            },
            basis,
        }
    }

    pub fn dimension_matrix(&self, dim: usize) -> RipsBoundarySingleDim<CF> {
        RipsBoundarySingleDim {
            matrix: &self.matrix,
            basis: &self.basis.0[dim],
        }
    }
}

// TODO: Write some proper tests
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        filtrations::rips::homology::RipsBoundaryAllDims,
        matrices::{combinators::product, ColBasis, HasColBasis, HasRowFiltration, MatrixRef},
        reduction::standard_algo,
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
        let ensemble = RipsBoundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix
        let v = standard_algo(&ensemble);
        let r = &product(&ensemble, &v);

        // Read off diagram
        let mut essential_idxs = HashSet::new();
        for i in 0..r.basis().size() {
            let mut r_i = r.build_bhcol(i).unwrap();
            if r_i.pop_pivot().is_none() {
                essential_idxs.insert(ensemble.basis().element(i));
            }
        }

        let mut pairings = vec![];
        for i in 0..r.basis().size() {
            let mut r_i = r.build_bhcol(i).unwrap();
            if let Some(piv) = r_i.pop_pivot() {
                let death_idx = ensemble.basis().element(i);
                let death_t = ensemble.matrix.filtration_value(death_idx).unwrap();
                pairings.push((piv.row_index, death_idx, piv.filtration_value, death_t));
                essential_idxs.remove(&piv.row_index);
            }
        }

        // Report
        println!("Essential:");
        for idx in essential_idxs {
            let f_val = ensemble.filtration_value(idx).unwrap();
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in pairings {
            let dim = tup.0.dimension(n_points);
            let idx_tup = (tup.0, tup.1);
            let birth_f = tup.2;
            let death_f = tup.3;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
    }
}
