use std::marker::PhantomData;

use ordered_float::NotNan;

use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matricies::{BasisElement, FiniteOrderedColBasis, HasRowFiltration, MatrixOracle},
};

// TODO: Custom derive Debug
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RipsIndex(usize);

impl BasisElement for RipsIndex {}

impl RipsIndex {
    /// Produces from least significant to most significant
    fn to_vec(self, n_points: usize) -> Vec<usize> {
        let mut vec = vec![];
        let mut working = self.0;
        while working > 0 {
            let remainder = working.rem_euclid(n_points + 1);
            vec.push(remainder - 1);
            working = working.div_euclid(n_points + 1);
        }
        vec
    }

    fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
        let inner = indices
            .enumerate()
            .map(|(i, coeff)| (n_points + 1).pow(i as u32) * (coeff + 1))
            .sum();
        if inner > 0 {
            Some(Self(inner))
        } else {
            None
        }
    }
}

struct RipsBoundaryMatrix<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundaryMatrix<CF> {
    fn n_points(&self) -> usize {
        self.distances.len()
    }
}

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsBoundaryMatrix<CF> {
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

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsBoundaryMatrix<CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        let n = as_vec.len();
        let max_pairwise_distance = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                let v_i = as_vec[i];
                let v_j = as_vec[j];
                self.distances[v_i][v_j]
            })
            .max()
            .unwrap();
        Ok(max_pairwise_distance)
    }

    // TODO: Override colun with filtration because we can compute max more efficinetly
}

// Represents the boundary matrix for a given dimension
// Column basis is sorted by (filtration, index)
#[derive(Clone, Copy)]
pub struct RipsBoundaryMatrixWithBasis<'a, CF: NonZeroCoefficient + Invertible> {
    oracle: &'a RipsBoundaryMatrix<CF>,
    // bases[i] = basis for the dimension i simplices, sorted by (filtration, index) - ascending
    basis: &'a Vec<(NotNan<f64>, RipsIndex)>,
}

impl<'a, CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsBoundaryMatrixWithBasis<'a, CF> {
    type CoefficientField = CF;
    type ColT = usize;
    type RowT = RipsIndex;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
    {
        self.oracle.column(self.basis[col].1)
    }
}

impl<'a, CF: NonZeroCoefficient + Invertible> HasRowFiltration
    for RipsBoundaryMatrixWithBasis<'a, CF>
{
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<'a, CF: NonZeroCoefficient + Invertible> FiniteOrderedColBasis
    for RipsBoundaryMatrixWithBasis<'a, CF>
{
    fn n_cols(&self) -> usize {
        self.basis.len()
    }
}

pub struct RipsBoundaryEnsemble<CF: NonZeroCoefficient + Invertible> {
    oracle: RipsBoundaryMatrix<CF>,
    bases: Vec<Vec<(NotNan<f64>, RipsIndex)>>,
}

fn add_cofacets(
    bases: &mut Vec<Vec<(NotNan<f64>, RipsIndex)>>,
    simplex_as_vec: &mut Vec<usize>,
    fil_value: NotNan<f64>,
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) {
    let n_vertices = distances.len();
    let max_element = simplex_as_vec.last().unwrap();
    let prev_dimension = simplex_as_vec.len() - 1;
    let new_dimension = prev_dimension + 1;
    if *max_element == n_vertices - 1 {
        // Cannot extend
        return;
    }
    for v in (max_element + 1)..n_vertices {
        // Compute filtration value
        let new_filtration_value = simplex_as_vec
            .iter()
            .map(|w| distances[*w][v])
            .max()
            .unwrap();
        let new_filtration_value = new_filtration_value.max(fil_value);
        // Build up coface
        simplex_as_vec.push(v);
        // Insert into basis
        let index = RipsIndex::from_indices(simplex_as_vec.iter().copied(), n_vertices).unwrap();
        bases[new_dimension].push((new_filtration_value, index));
        // Recurse to cofacets
        if new_dimension < max_dim {
            add_cofacets(
                bases,
                simplex_as_vec,
                new_filtration_value,
                distances,
                max_dim,
            );
        }
        // Remove v ready for next coface
        simplex_as_vec.pop();
    }
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundaryEnsemble<CF> {
    pub fn dimension_matrix<'a>(&'a self, dim: usize) -> RipsBoundaryMatrixWithBasis<'a, CF> {
        RipsBoundaryMatrixWithBasis {
            oracle: &self.oracle,
            basis: &self.bases[dim],
        }
    }

    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let mut bases = vec![vec![]; max_dim + 1];

        let n_points = distances.len();
        // Build up the basis
        for v in 0..(distances.len()) {
            let mut simplex = vec![v];
            bases[0].push((
                NotNan::new(0.0_f64).unwrap(),
                RipsIndex::from_indices(vec![v].into_iter(), n_points).unwrap(),
            ));
            add_cofacets(
                &mut bases,
                &mut simplex,
                NotNan::new(0 as f64).unwrap(),
                &distances,
                max_dim,
            );
        }

        for dim in 0..=max_dim {
            // Sort by filtration value then index
            bases[dim].sort_unstable();
        }

        Self {
            oracle: RipsBoundaryMatrix {
                distances,
                phantom: PhantomData,
            },
            bases,
        }
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        matricies::{product, FiniteOrderedColBasis, HasRowFiltration},
        reduction::inefficient_reduction,
    };

    use super::{RipsBoundaryEnsemble, RipsIndex};

    #[test]
    fn test_rips_index() {
        let vec = vec![5, 12, 7];
        let index = RipsIndex::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);

        // Preserves trailing 0
        let vec = vec![5, 12, 0];
        let index = RipsIndex::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);
    }

    #[test]
    fn test_rips() {
        let zero = NotNan::new(0.0).unwrap();
        let one = NotNan::new(1.0).unwrap();
        let sqrt2 = NotNan::new(2.0_f64.sqrt()).unwrap();
        let distance_matrix: Vec<Vec<NotNan<f64>>> = vec![
            vec![zero, one, sqrt2, one],
            vec![one, zero, one, sqrt2],
            vec![sqrt2, one, zero, one],
            vec![one, sqrt2, one, zero],
        ];

        let ensemble = RipsBoundaryEnsemble::<Z2>::build(distance_matrix, 2);
        println!("{:?}", ensemble.bases);

        assert_eq!(ensemble.bases[0].len(), 4);
        assert_eq!(ensemble.bases[1].len(), 6);
        assert_eq!(ensemble.bases[2].len(), 4);

        let d0 = ensemble.dimension_matrix(0);
        let d1 = ensemble.dimension_matrix(1);
        let d2 = ensemble.dimension_matrix(2);
        let v0 = inefficient_reduction(&d0);
        let v1 = inefficient_reduction(&d1);
        let v2 = inefficient_reduction(&d2);

        let r0 = product(&d0, &v0);
        let r1 = product(&d1, &v1);
        let r2 = product(&d2, &v2);

        println!("R0");
        for i in 0..r0.n_cols() {
            let r0_i = r0.build_bhcol(i).unwrap().to_sorted_vec();
            println!("Col {i} : {r0_i:?}");
        }

        println!("R1");
        for i in 0..r1.n_cols() {
            let r1_i = r1.build_bhcol(i).unwrap().to_sorted_vec();
            println!("Col {i} : {r1_i:?}");
        }

        println!("R2");
        for i in 0..r2.n_cols() {
            let r2_i = r2.build_bhcol(i).unwrap().to_sorted_vec();
            println!("Col {i} : {r2_i:?}");
        }

        let mut essential_idxs = HashSet::new();
        for i in 0..r1.n_cols() {
            let mut r1_i = r1.build_bhcol(i).unwrap();
            if r1_i.pop_pivot().is_none() {
                essential_idxs.insert(d1.basis[i].1);
            }
        }

        let mut pairings = vec![];
        for i in 0..r2.n_cols() {
            let mut r2_i = r2.build_bhcol(i).unwrap();
            if let Some(piv) = r2_i.pop_pivot() {
                pairings.push((
                    piv.row_index,
                    d2.basis[i].1,
                    piv.filtration_value,
                    d2.basis[i].0,
                ));
                essential_idxs.remove(&piv.row_index);
            }
        }
        println!("Essential:");
        for idx in essential_idxs {
            println!("{idx:?}");
        }
        println!("Pairings:");
        for tup in pairings {
            println!("{tup:?}");
        }
    }
}
