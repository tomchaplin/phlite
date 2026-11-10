pub mod cohomology;
pub mod homology;
#[cfg(feature = "python")]
mod python;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar},
    fields::Z2,
    matrices::{BasisElement, MatrixOracle, SplitBasis},
};

use crate::{
    cohomology::{build_rips_coboundary_matrix, RipsCoboundary},
    homology::{build_rips_boundary_matrix, RipsBoundary},
};

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
// All we need is a basis for the simplices of every dimension, with an efficient way to get at the simplices of a single dimension.
// `phlite` provides `SplitBasis` for exactly this: all of the simplices are stored in one flat `Vec`,
// along with a table of offsets marking where each dimension starts.
// Its implementation of `SplitByDimension` hands out slices of that `Vec` by value, so
// we never need to build (or own) a separate basis type per dimension.
//
// In most applications your bases should look fairly similar to this.
// Don't bother reversing any filtration orders for cohomology because you can just use [`MatrixOracle::reverse`]
// (this works on the views that `SplitBasis` hands out, so no `unsafe` is required anywhere).

/// The column basis for Rips (co)homology: all simplices up to some dimension, split by dimension.
pub type RipsBasis = SplitBasis<RipsIndex>;

// TUTORIAL:
// Now we provide some functions for building up a basis for Rips (co)homology
// This will be quite different depending on your application.
// Remember: You only need to construct and sort a basis for the column space of each matrix.
// If you are computing cohomology then to compute H^K just compute a basis for C_K.
//
// While building we keep each simplex's filtration value alongside it (so that we can sort),
// then throw the filtration values away when we assemble the final `SplitBasis`.

type SimplicesWithFilt = Vec<(NotNan<f64>, RipsIndex)>;

fn add_cofacets(
    bases: &mut [SimplicesWithFilt],
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

pub fn build_rips_bases(distances: &Vec<Vec<NotNan<f64>>>, max_dim: usize) -> RipsBasis {
    let mut bases: Vec<SimplicesWithFilt> = vec![vec![]; max_dim + 1];

    let n_points = distances.len();
    // Build up the basis
    for v in 0..(distances.len()) {
        let mut simplex = vec![v];
        bases[0].push((
            NotNan::new(0.0_f64).unwrap(),
            RipsIndex::from_indices(vec![v].into_iter(), n_points).unwrap(),
        ));
        if max_dim > 0 {
            add_cofacets(
                &mut bases,
                &mut simplex,
                NotNan::new(0 as f64).unwrap(),
                distances,
                max_dim,
            );
        }
    }

    for basis in bases.iter_mut() {
        // Sort by filtration value then index
        basis.sort_unstable();
    }
    // Drop the filtration values and flatten into a single allocation
    SplitBasis::from_dimensions(
        bases
            .into_iter()
            .map(|dim| dim.into_iter().map(|(_filtration, index)| index)),
    )
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

// ====== High-level API ============================================

pub use phlite::barcode::Method;

/// A bar in the Rips barcode, see [`phlite::barcode::Bar`].
pub type RipsBar = Bar<RipsIndex, NotNan<f64>, Z2>;

/// Whether every simplex of dimension at most `max_dim` on `n_points` points can be represented by a [`RipsIndex`].
pub fn rips_index_fits(n_points: usize, max_dim: usize) -> bool {
    u32::try_from(max_dim + 1)
        .ok()
        .and_then(|exponent| (n_points + 1).checked_pow(exponent))
        .is_some()
}

/// Compute the Vietoris-Rips barcode of a distance matrix in dimensions `0..=max_dim`.
///
/// * `method` chooses between reducing the coboundary or boundary matrix.
/// * `algorithm` is the reduction algorithm to use.
/// * If `representatives` is true then every bar comes with a representative cycle.
///
/// Bars are returned in no particular order, and zero-length bars are included.
///
/// # Panics
/// If the simplices needed do not fit in a [`RipsIndex`], see [`rips_index_fits`].
pub fn rips_barcode(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_>,
    representatives: bool,
) -> Vec<RipsBar> {
    let n_points = distances.len();
    // Each matrix is given its transpose, so that the reduction can skip apparent pairs
    match method {
        Method::Cohomology => {
            assert!(rips_index_fits(n_points, max_dim));
            let boundary = RipsBoundary::<Z2>::new(distances.clone());
            let coboundary = build_rips_coboundary_matrix::<Z2>(distances, max_dim);
            let coboundary = (&coboundary).with_transpose(&boundary);
            if representatives {
                barcode::cohomology_with_involution(coboundary, &boundary, algorithm)
            } else {
                barcode::cohomology(coboundary, algorithm)
            }
        }
        Method::Homology => {
            // Bars in dimension max_dim die at simplices of dimension max_dim + 1
            assert!(rips_index_fits(n_points, max_dim + 1));
            let coboundary = RipsCoboundary::<Z2>::new(distances.clone());
            let boundary = build_rips_boundary_matrix::<Z2>(distances, max_dim + 1);
            let boundary = (&boundary).with_transpose(&coboundary);
            let mut bars = barcode::homology(boundary, algorithm, representatives);
            // Nothing can kill the top dimensional simplices, so they look essential but are not bars we computed
            bars.retain(|bar| bar.birth.dimension(n_points) <= max_dim);
            bars
        }
    }
}

#[cfg(feature = "python")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

#[cfg(test)]
mod tests {
    use crate::RipsIndex;

    /// `rips_barcode` attaches transposes (enabling apparent pairs); that must not change any bar
    #[test]
    fn apparent_pairs_do_not_change_barcode() {
        use crate::{cohomology::build_rips_coboundary_matrix, rips_barcode, Method};
        use ordered_float::NotNan;
        use phlite::{
            barcode::{self, Algorithm},
            fields::Z2,
            reduction::{Clearing, Standard},
        };
        // Twelve points on a noisy circle
        let points: Vec<(f64, f64)> = (0..12)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 12.0;
                let r = 1.0 + 0.1 * ((i * 7 % 5) as f64 - 2.0);
                (r * t.cos(), r * t.sin())
            })
            .collect();
        let distances: Vec<Vec<NotNan<f64>>> = points
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
        let summary = |bars: Vec<crate::RipsBar>| {
            let mut out: Vec<_> = bars
                .into_iter()
                .map(|bar| {
                    (
                        bar.birth,
                        bar.death.map(|(d, _)| d),
                        bar.representative.is_some(),
                    )
                })
                .collect();
            out.sort();
            out
        };
        let coboundary = build_rips_coboundary_matrix::<Z2>(distances.clone(), 2);
        let expected = summary(barcode::cohomology(&coboundary, Algorithm::default()));
        for algorithm in [
            Algorithm::Standard(Standard::new()),
            Algorithm::Clearing(Clearing::new()),
            Algorithm::ParallelClearing(Clearing::new().parallel()),
        ] {
            let bars = rips_barcode(
                distances.clone(),
                2,
                Method::Cohomology,
                algorithm.clone(),
                false,
            );
            assert_eq!(summary(bars), expected, "{algorithm:?}");
            // With representatives the involution's debug assertions check them against the pairing
            let bars = rips_barcode(
                distances.clone(),
                2,
                Method::Cohomology,
                algorithm.clone(),
                true,
            );
            assert_eq!(bars.len(), expected.len());
            // Homology pairs simplices the same way (the pairing doesn't depend on the method)
            let bars = rips_barcode(distances.clone(), 2, Method::Homology, algorithm, false);
            let mut hom: Vec<_> = bars
                .into_iter()
                .map(|bar| (bar.birth, bar.death.map(|(d, _)| d)))
                .collect();
            hom.sort();
            let cohom: Vec<_> = expected.iter().map(|(b, d, _)| (*b, *d)).collect();
            assert_eq!(hom, cohom);
        }
    }

    #[test]
    fn test_bases_with_max_dim_0() {
        use ordered_float::NotNan;
        use phlite::matrices::ColBasis;
        let one = NotNan::new(1.0).unwrap();
        let zero = NotNan::new(0.0).unwrap();
        let basis = crate::build_rips_bases(&vec![vec![zero, one], vec![one, zero]], 0);
        assert_eq!(basis.n_dimensions(), 1);
        assert_eq!(basis.size(), 2);
    }

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
