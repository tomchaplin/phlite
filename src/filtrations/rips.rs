use std::{cmp::Reverse, iter, marker::PhantomData};

use ordered_float::NotNan;

use crate::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matricies::{BasisElement, FiniteOrderedColBasis, HasRowFiltration, MatrixOracle},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RipsIndex(usize);

impl BasisElement for RipsIndex {}

impl RipsIndex {
    /// Produces from least significant to most significant
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        let mut vec = vec![];
        let mut working = self.0;
        while working > 0 {
            let remainder = working.rem_euclid(n_points + 1);
            vec.push(remainder - 1);
            working = working.div_euclid(n_points + 1);
        }
        vec
    }

    pub fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
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

    pub fn dimension(self, n_points: usize) -> usize {
        self.to_vec(n_points).len() - 1
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

fn max_pairwise_distance(vertices: &Vec<usize>, distances: &Vec<Vec<NotNan<f64>>>) -> NotNan<f64> {
    let n_vertices = vertices.len();
    (0..n_vertices)
        .flat_map(|i| (0..n_vertices).map(move |j| (i, j)))
        .map(|(i, j)| {
            let v_i = vertices[i];
            let v_j = vertices[j];
            distances[v_i][v_j]
        })
        .max()
        .unwrap()
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsBoundaryMatrix<CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        Ok(max_pairwise_distance(&as_vec, &self.distances))
    }

    // TODO:Can we override column with filtration to compute more efficiently?
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

fn add_cofacets<F, O>(
    bases: &mut Vec<Vec<(O, RipsIndex)>>,
    simplex_as_vec: &mut Vec<usize>,
    fil_value: NotNan<f64>,
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    functor: &F,
) where
    F: Fn(NotNan<f64>) -> O,
{
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
        bases[new_dimension].push((functor(new_filtration_value), index));
        // Recurse to cofacets
        if new_dimension < max_dim {
            add_cofacets(
                bases,
                simplex_as_vec,
                new_filtration_value,
                distances,
                max_dim,
                functor,
            );
        }
        // Remove v ready for next coface
        simplex_as_vec.pop();
    }
}

fn build_rips_bases<F, O>(
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    functor: F,
) -> Vec<Vec<(O, RipsIndex)>>
where
    F: Fn(NotNan<f64>) -> O,
    O: Clone + Ord,
{
    let mut bases = vec![vec![]; max_dim + 1];

    let n_points = distances.len();
    // Build up the basis
    for v in 0..(distances.len()) {
        let mut simplex = vec![v];
        bases[0].push((
            functor(NotNan::new(0.0_f64).unwrap()),
            RipsIndex::from_indices(vec![v].into_iter(), n_points).unwrap(),
        ));
        add_cofacets(
            &mut bases,
            &mut simplex,
            NotNan::new(0 as f64).unwrap(),
            &distances,
            max_dim,
            &functor,
        );
    }

    for dim in 0..=max_dim {
        // Sort by filtration value then index
        bases[dim].sort_unstable();
    }
    bases
}

impl<CF: NonZeroCoefficient + Invertible> RipsBoundaryEnsemble<CF> {
    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let bases = build_rips_bases(&distances, max_dim, |x| x);
        Self {
            oracle: RipsBoundaryMatrix {
                distances,
                phantom: PhantomData,
            },
            bases,
        }
    }

    pub fn dimension_matrix<'a>(&'a self, dim: usize) -> RipsBoundaryMatrixWithBasis<'a, CF> {
        RipsBoundaryMatrixWithBasis {
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

    pub fn basis_at_total_index(&self, total_index: usize) -> (NotNan<f64>, RipsIndex) {
        let (dim, local_index) = self.total_index_to_dim_index(total_index);
        self.bases[dim][local_index]
    }
}

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsBoundaryEnsemble<CF> {
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

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsBoundaryEnsemble<CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<CF: NonZeroCoefficient + Invertible> FiniteOrderedColBasis for RipsBoundaryEnsemble<CF> {
    fn n_cols(&self) -> usize {
        self.bases.iter().map(|basis| basis.len()).sum()
    }
}

struct RipsCoboundaryMatrix<CF: NonZeroCoefficient + Invertible> {
    distances: Vec<Vec<NotNan<f64>>>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsCoboundaryMatrix<CF> {
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
impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsCoboundaryMatrix<CF> {
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

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundaryMatrix<CF> {
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

pub struct RipsCoboundaryEnsemble<CF: NonZeroCoefficient + Invertible> {
    oracle: RipsCoboundaryMatrix<CF>,
    bases: Vec<Vec<(Reverse<NotNan<f64>>, RipsIndex)>>,
}

impl<CF: NonZeroCoefficient + Invertible> RipsCoboundaryEnsemble<CF> {
    pub fn build(distances: Vec<Vec<NotNan<f64>>>, max_dim: usize) -> Self {
        let bases = build_rips_bases(&distances, max_dim, Reverse);
        Self {
            oracle: RipsCoboundaryMatrix {
                distances,
                phantom: PhantomData,
            },
            bases,
        }
    }

    // pub fn dimension_matrix<'a>(&'a self, dim: usize) -> RipsBoundaryMatrixWithBasis<'a, CF> {
    //     RipsBoundaryMatrixWithBasis {
    //         oracle: &self.oracle,
    //         basis: &self.bases[dim],
    //     }
    // }

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

impl<CF: NonZeroCoefficient + Invertible> MatrixOracle for RipsCoboundaryEnsemble<CF> {
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

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundaryEnsemble<CF> {
    type FiltrationT = Reverse<NotNan<f64>>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, crate::PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<CF: NonZeroCoefficient + Invertible> FiniteOrderedColBasis for RipsCoboundaryEnsemble<CF> {
    fn n_cols(&self) -> usize {
        self.bases.iter().map(|basis| basis.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, marker::PhantomData};

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        filtrations::rips::RipsCoboundaryEnsemble,
        matricies::{product, FiniteOrderedColBasis, HasRowFiltration, MatrixOracle},
        reduction::inefficient_reduction,
    };

    use super::{RipsBoundaryEnsemble, RipsCoboundaryMatrix, RipsIndex};

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
        let ensemble = RipsBoundaryEnsemble::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix
        let v = inefficient_reduction(&ensemble);
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

    #[test]
    fn test_coboundary() {
        let distance_matrix = distance_matrix();
        let coboundary: RipsCoboundaryMatrix<Z2> = RipsCoboundaryMatrix {
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

    #[test]
    fn test_rips_total_coboundary() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 2;

        // Compute column basis
        let ensemble = RipsCoboundaryEnsemble::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix
        let v = inefficient_reduction(&ensemble);
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
