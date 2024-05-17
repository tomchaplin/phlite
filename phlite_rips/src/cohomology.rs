use std::{convert::identity, iter, marker::PhantomData};

use ordered_float::NotNan;

use crate::{build_rips_bases, max_pairwise_distance, RipsIndex};
use phlite::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
    PhliteError,
};

use super::MultiDimRipsBasisWithFilt;

// TODO: distances only needs to be a reference

pub struct RipsCoboundary<CF: NonZeroCoefficient + Invertible> {
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

// TODO: Check whether it is better to produce cobundaries in the opposite order for heap insertion
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
            let Some(vertex_at_insert_pos) = self.base.get(self.insertion_position) else {
                // The vertex will have to go at the end
                break;
            };

            let comparison = vertex_at_insert_pos.cmp(&self.next_vertex);

            match comparison {
                std::cmp::Ordering::Less => {
                    // next_vertex has to go afterwards
                    self.insertion_position += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Vertex already appears in simplex, try and insert next vertex
                    // TODO: Can I also increment insertion_position here?
                    self.next_vertex += 1;
                }
                std::cmp::Ordering::Greater => {
                    // We have found the correct insertion position
                    break;
                }
            }
        }

        let (vertices_prior, vertices_after) = self.base.split_at(self.insertion_position);
        let new_simplex = vertices_prior
            .iter()
            .chain(iter::once(&self.next_vertex))
            .chain(vertices_after.iter())
            .copied();
        let index = RipsIndex::from_indices(new_simplex, self.max_vertex + 1).unwrap();
        let coeff = if self.insertion_position % 2 == 0 {
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
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
        let n_points = self.n_points();
        Ok(CoboundaryIterator::new(col.to_vec(n_points), n_points)
            .map(|(coeff, idx, _inserted_vertex)| (coeff, idx)))
    }
}

impl<CF: NonZeroCoefficient + Invertible> HasRowFiltration for RipsCoboundary<CF> {
    // Reverse filtration to anti-transpose
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, phlite::PhliteError> {
        let as_vec = row.to_vec(self.n_points());
        Ok(max_pairwise_distance(&as_vec, &self.distances))
    }

    // We can speed this up by pre-computing the max of most elements
    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = ColumnEntry<Self>>, PhliteError> {
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

                ColumnEntry {
                    filtration_value,
                    row_index,
                    coeff,
                }
            }),
        )
    }
}

pub type RipsCoboundaryAllDims<CF> =
    MatrixWithBasis<RipsCoboundary<CF>, MultiDimRipsBasisWithFilt<NotNan<f64>>>;

pub fn build_rips_coboundary_matrix<CF: Invertible>(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> RipsCoboundaryAllDims<CF> {
    // Pass in the Reverse functor to revere filtration order on columns in basis
    let basis = build_rips_bases(&distances, max_dim, identity);
    RipsCoboundaryAllDims {
        matrix: RipsCoboundary {
            distances,
            phantom: PhantomData,
        },
        basis,
    }
}
// TODO: Write some proper tests
#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use ordered_float::NotNan;

    use crate::{
        cohomology::{build_rips_coboundary_matrix, RipsCoboundary},
        RipsIndex,
    };
    use phlite::{
        fields::Z2,
        matrices::{combinators::product, HasRowFiltration, MatrixOracle, MatrixRef},
        reduction::{standard_algo_with_diagram, ClearedReductionMatrix},
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
        let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix, max_dim);
        let coboundary = coboundary.reverse();
        // Compute reduction matrix
        let (v, diagram) = standard_algo_with_diagram(&coboundary, false);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).unwrap().0;
            let dim = idx.0.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1 .0.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).unwrap().0;
            let death_f = coboundary.filtration_value(tup.0).unwrap().0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // 2-dimensional void is killed without having to compute basis for C_3
        assert_eq!(diagram.pairings.len(), 7);
        assert_eq!(diagram.essential.len(), 1);
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
        let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&coboundary, 0..=max_dim);
        let _r = product(&coboundary, &v);

        // Report
        println!("Essential:");
        for idx in diagram.essential.iter() {
            let f_val = coboundary.filtration_value(*idx).unwrap().0;
            let dim = idx.0.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in diagram.pairings.iter() {
            let dim = tup.1 .0.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).unwrap().0;
            let death_f = coboundary.filtration_value(tup.0).unwrap().0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // Ignored 2-dimensional void
        assert_eq!(diagram.pairings.len(), 6);
        assert_eq!(diagram.essential.len(), 1);
    }
}
