pub mod cohomology;
pub mod homology;

use ordered_float::NotNan;
use phlite::matrices::{BasisElement, ColBasis, SplitByDimension};

// TUTORIAL:
// First we implement the types that index into our boundary an coboundary matrix

// TODO:
// Swap to lexicographic numbering scheme described in Ripser paper

/// Represents a simplex by storing the elements as an integer with base `b` where `b = n_points + 1`.
/// Each digit in the integer corresponds to an element in the simplex.
///
/// The elements are always represented in increasing order.
/// We add on `1` to each digit to ensure that a leading `0` is not deleted.
/// Therefore the simplex `{0, 1}` is represented by `RipsIndex(12)` where the base for `12` depends on the number of points in your point cloud.
/// Note this is fairly inefficient since as simplicies `{0, 1} == {1, 0}` but we only ever use `RipsIndex(12)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RipsIndex(usize);

// TUTORIAL:
// We must manually implement this trait to mark `RipsIndex` as a indexing type for matrices

impl BasisElement for RipsIndex {}

// TUTORIAL:
// Now we implement some helper methods to convert our indexing type into the corresponding simplices

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

    /// NOTE: The indicies should be provided in ascending order!
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

// TUTORIAL:
// Now we decide on the type for our column bases.
// We have one type for the simplices in a given dimension and another type for the collection of all dimensions.
//
// In most applications your bases should look fairly similar to this.
// Don't bother reversing any filtration orders for cohomology because you can just use [`MatrixOracle::reverse`]

// TUTORIAL:
// The single-dimensional basis is just a `Vec` of `RipsIndex` stored alongside the simplex's filtration value (so that we can quickly sort).
// We wrap it in a new type.

// Wrapper around the default structure that contains the basis for Rips homology
#[derive(Clone)]
pub struct SingleDimRipsBasisWithFilt(Vec<(NotNan<f64>, RipsIndex)>);

// TUTORIAL:
// We must implement `ColBasis` on the wrapper, to retrieve elements from the structure.

impl ColBasis for SingleDimRipsBasisWithFilt {
    type ElemT = RipsIndex;

    fn element(&self, index: usize) -> Self::ElemT {
        self.0[index].1
    }

    fn size(&self) -> usize {
        self.0.len()
    }
}

// TUTORIAL:
// The multi-dimensional basis is just a `Vec` of single-dimensional bases, wrapped in a new type.
// Again we have to implement `ColBasis`.
// Technically there is a choice on how we interleave simplices in each dimension.
// However, it doesn't change PH so we just flatten the vector of vectors.

#[derive(Clone)]
pub struct MultiDimRipsBasisWithFilt(Vec<SingleDimRipsBasisWithFilt>);

impl ColBasis for MultiDimRipsBasisWithFilt {
    type ElemT = RipsIndex;

    fn element(&self, index: usize) -> Self::ElemT {
        let mut working = index;
        let mut dim = 0;
        while working >= self.0[dim].size() {
            working -= self.0[dim].size();
            dim += 1;
        }
        self.0[dim].element(working)
    }

    fn size(&self) -> usize {
        self.0.iter().map(|basis| basis.size()).sum()
    }
}

// TUTORIAL:
// The multi-dimensional basis contains a number of single-dimensional bases.
// You must provide access to each of these "sub-bases".

impl SplitByDimension for MultiDimRipsBasisWithFilt {
    type SubBasisT = SingleDimRipsBasisWithFilt;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        &self.0[dimension]
    }
}

// TUTORIAL:
// Now we provide some functions for building up a basis for Rips (co)homology
// This will be quite different depending on your application.
// Remember: You only need to construct and sort a basis for the column space of each matrix.
// If you are computing cohomology then to compute H^K just compute a basis for C_K.

fn add_cofacets(
    bases: &mut Vec<SingleDimRipsBasisWithFilt>,
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
        bases[new_dimension].0.push((new_filtration_value, index));
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

pub fn build_rips_bases(
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
) -> MultiDimRipsBasisWithFilt {
    let mut bases = vec![SingleDimRipsBasisWithFilt(vec![]); max_dim + 1];

    let n_points = distances.len();
    // Build up the basis
    for v in 0..(distances.len()) {
        let mut simplex = vec![v];
        bases[0].0.push((
            NotNan::new(0.0_f64).unwrap(),
            RipsIndex::from_indices(vec![v].into_iter(), n_points).unwrap(),
        ));
        add_cofacets(
            &mut bases,
            &mut simplex,
            NotNan::new(0 as f64).unwrap(),
            distances,
            max_dim,
        );
    }

    for basis in bases.iter_mut().take(max_dim + 1) {
        // Sort by filtration value then index
        basis.0.sort_unstable();
    }
    MultiDimRipsBasisWithFilt(bases)
}

pub(crate) fn max_pairwise_distance(
    vertices: &[usize],
    distances: &[Vec<NotNan<f64>>],
) -> NotNan<f64> {
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

// TUTORIAL:
// Now head over to homology.rs to implement the boundary matrix

#[cfg(test)]
mod tests {
    use crate::RipsIndex;

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
}
