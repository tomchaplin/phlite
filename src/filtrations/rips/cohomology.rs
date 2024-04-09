use std::{cmp::Reverse, iter, marker::PhantomData};

use ordered_float::NotNan;

use crate::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    filtrations::rips::{build_rips_bases, max_pairwise_distance, RipsIndex},
    matrices::{FiniteOrderedColBasis, HasRowFiltration, MatrixOracle},
};

// TODO: distances only needs to be a reference

struct RipsCoboundary<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsCoboundary<CF> {
    fn n_points(&self) -> usize {
        self.distances.len()
    }
}

struct CoboundaryIterator<CF: NonZeroCoefficient> {
    base: Vec<usize>,
    next_vertex: usize,
    max_vertex: usize,
    insertion_position: usize,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient> CoboundaryIterator<CF> {
    fn new(base: Vec<usize>, n_points: usize) -> Self {
        Self {
            base,
            next_vertex: 0,
            max_vertex: n_points - 1,
            insertion_position: 0,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> Iterator for CoboundaryIterator<CF> {
    // Coefficient, RipsIndex, inserted vertex
    type Item = (CF, RipsIndex, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // By the ened of this loop self.next_vertex and self.insertion_position
        // will be setup correctly so that the next simplex
        // can be obtained by inserting self.next_vertex into index self.insertion_position (which may be the end fo self.base)
        // TODO: Change into a match statement
        loop {
            // Ran out of vertices to try and insert
            if self.next_vertex > self.max_vertex {
                return None;
            }
            if self.insertion_position >= self.base.len() {
                // The vertex will have to go at the end
                break;
            }
            if self.base[self.insertion_position] > self.next_vertex {
                // We have found the correct insertion position
                break;
            }
            if self.base[self.insertion_position] == self.next_vertex {
                // Vertex already appears in simplex, try next one
                // TODO: Can I also increment insertion_position here?
                self.next_vertex += 1;
            }
            if self.base[self.insertion_position] < self.next_vertex {
                self.insertion_position += 1;
            }
        }

        let (vertices_prior, vertices_after) = self.base.split_at(self.insertion_position);
        let new_simplex = vertices_prior
            .iter()
            .chain(iter::once(&self.next_vertex))
            .chain(vertices_after.iter())
            .copied();
        let index = RipsIndex::from_indices(new_simplex, self.max_vertex + 1).unwrap();
        // TODO: Double check that this is correct
        let coeff = if self.next_vertex % 2 == 0 {
            CF::one()
        } else {
            CF::one().additive_inverse()
        };

        self.next_vertex += 1;
        Some((coeff, index, self.next_vertex))
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
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
    {
        let n_points = self.n_points();
        Ok(CoboundaryIterator::new(col.to_vec(n_points), n_points)
            .map(|(coeff, idx, _inserted_vertex)| (coeff, idx)))
    }
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundary<CF> {
    // Reverse filtration to anti-transpose
    type FiltrationT = Reverse<NotNan<f64>>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        Ok(Reverse(max_pairwise_distance(&as_vec, &self.distances)))
    }

    // We can speed this up by pre-computing the max of most elements
    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<
        impl Iterator<Item = Result<crate::columns::ColumnEntry<Self>, crate::PhliteError>>,
        crate::PhliteError,
    > {
        let n_points = self.n_points();
        let coboundary_iterator = CoboundaryIterator::new(col.to_vec(n_points), n_points);

        let col_vertices = col.to_vec(self.n_points());

        let max_pd_amongst_col = max_pairwise_distance(&col_vertices, &self.distances);

        Ok(
            coboundary_iterator.map(move |(coeff, row_index, inserted_vertex)| {
                let max_to_inserted = col_vertices
                    .iter()
                    .map(|v| self.distances[*v][inserted_vertex])
                    .max()
                    .unwrap();

                let filtration_value = max_pd_amongst_col.max(max_to_inserted);
                let filtration_value = Reverse(filtration_value);

                Ok(ColumnEntry {
                    filtration_value,
                    row_index,
                    coeff,
                })
            }),
        )
    }
}

pub struct RipsCoboundaryAllDims<CF: NonZeroCoefficient + Invertible> {
    oracle: RipsCoboundary<CF>,
    bases: Vec<Vec<(Reverse<NotNan<f64>>, RipsIndex)>>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsCoboundaryAllDims<CF> {
    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let bases = build_rips_bases(&distances, max_dim, Reverse);
        Self {
            oracle: RipsCoboundary {
                distances,
                phantom: PhantomData,
            },
            bases,
        }
    }

    pub fn dimension_matrix<'a>(&'a self, dim: usize) -> RipsCoboundarySingleDim<'a, CF> {
        RipsCoboundarySingleDim {
            oracle: &self.oracle,
            basis: &self.bases[dim],
        }
    }

    pub fn total_index_to_dim_index(&self, total_index: usize) -> (usize, usize) {
        let mut working = total_index;
        let mut dim = 0;
        while working >= self.bases[dim].len() {
            working -= self.bases[dim].len();
            dim += 1;
        }
        (dim, working)
    }

    pub fn basis_at_total_index(&self, total_index: usize) -> (Reverse<NotNan<f64>>, RipsIndex) {
        let (dim, local_index) = self.total_index_to_dim_index(total_index);
        self.bases[dim][local_index]
    }
}

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsCoboundaryAllDims<CF> {
    type CoefficientField = CF;

    type ColT = usize;

    type RowT = RipsIndex;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
    {
        let (dim, local_index) = self.total_index_to_dim_index(col);
        self.oracle.column(self.bases[dim][local_index].1)
    }
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundaryAllDims<CF> {
    type FiltrationT = Reverse<NotNan<f64>>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<CF: NonZeroCoefficient + Invertible> FiniteOrderedColBasis for RipsCoboundaryAllDims<CF> {
    fn n_cols(&self) -> usize {
        self.bases.iter().map(|basis| basis.len()).sum()
    }
}

// Represents the boundary matrix for a given dimension
// Column basis is sorted by (filtration, index)
#[derive(Clone, Copy)]
pub struct RipsCoboundarySingleDim<'a, CF: NonZeroCoefficient + Invertible> {
    oracle: &'a RipsCoboundary<CF>,
    // bases[i] = basis for the dimension i simplices, sorted by (filtration, index) - ascending
    basis: &'a Vec<(Reverse<NotNan<f64>>, RipsIndex)>,
}

impl<'a, CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsCoboundarySingleDim<'a, CF> {
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

impl<'a, CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundarySingleDim<'a, CF> {
    type FiltrationT = Reverse<NotNan<f64>>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<'a, CF: NonZeroCoefficient + Invertible> FiniteOrderedColBasis
    for RipsCoboundarySingleDim<'a, CF>
{
    fn n_cols(&self) -> usize {
        self.basis.len()
    }
}

// TODO: Write some proper tests
#[cfg(test)]
mod tests {
    use std::{collections::HashSet, marker::PhantomData};

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        filtrations::rips::{
            cohomology::{RipsCoboundary, RipsCoboundaryAllDims},
            RipsIndex,
        },
        matrices::{combinators::product, FiniteOrderedColBasis, HasRowFiltration, MatrixOracle},
        reduction::standard_algo,
    };

    #[test]
    fn test_coboundary() {
        let distance_matrix = distance_matrix();
        let coboundary: RipsCoboundary<Z2> = RipsCoboundary {
            distances: distance_matrix,
            phantom: PhantomData,
        };

        let smplx = RipsIndex::from_indices(vec![1, 2, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).unwrap().collect();
        assert_eq!(coboundary_vec.len(), 1);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::from_indices(vec![0, 3].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).unwrap().collect();
        assert_eq!(coboundary_vec.len(), 2);
        println!("{coboundary_vec:?}");
        for entry in coboundary_vec {
            let vertices = entry.1.to_vec(4);
            println!("{vertices:?}")
        }

        let smplx = RipsIndex::from_indices(vec![1].into_iter(), 4).unwrap();
        let coboundary_vec: Vec<_> = coboundary.column(smplx).unwrap().collect();
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
        let ensemble = RipsCoboundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix
        let v = standard_algo(&ensemble);
        let r = product(&ensemble, &v);

        // Read off diagram
        let mut essential_idxs = HashSet::new();
        for i in 0..r.n_cols() {
            let mut r_i = r.build_bhcol(i).unwrap();
            if r_i.pop_pivot().is_none() {
                essential_idxs.insert(ensemble.basis_at_total_index(i).1);
            }
        }

        let mut pairings = vec![];
        for i in 0..r.n_cols() {
            let mut r_i = r.build_bhcol(i).unwrap();
            if let Some(piv) = r_i.pop_pivot() {
                let (death_t, death_idx) = ensemble.basis_at_total_index(i);
                pairings.push((piv.row_index, death_idx, piv.filtration_value, death_t));
                essential_idxs.remove(&piv.row_index);
            }
        }

        // Report
        println!("Essential:");
        for idx in essential_idxs {
            let f_val = ensemble.filtration_value(idx).unwrap().0;
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in pairings {
            let dim = tup.0.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = tup.3 .0;
            let death_f = tup.2 .0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
    }
}
