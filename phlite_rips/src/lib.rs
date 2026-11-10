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
    cohomology::{build_rips_coboundary_matrix_with_threshold, RipsCoboundary},
    homology::{build_rips_boundary_matrix_with_threshold, RipsBoundary},
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
    /// The vertices, in increasing order. Prefer [`vertices`](Self::vertices), which doesn't allocate.
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        self.vertices(n_points).to_vec()
    }

    /// The vertices, in increasing order, decoded onto the stack.
    pub fn vertices(self, n_points: usize) -> Vertices {
        let radix = n_points + 1;
        let mut vertices = Vertices {
            buf: [0; MAX_VERTICES],
            len: 0,
        };
        let mut working = self.0;
        while working > 0 {
            vertices.buf[vertices.len] = working % radix - 1;
            vertices.len += 1;
            working /= radix;
        }
        vertices
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
        let radix = n_points + 1;
        let mut working = self.0;
        let mut n_vertices = 0;
        while working > 0 {
            n_vertices += 1;
            working /= radix;
        }
        n_vertices - 1
    }
}

/// A simplex has at most this many vertices: an index has one base-(n + 1) digit per vertex, with n + 1 >= 2, and fits in a `usize`.
const MAX_VERTICES: usize = usize::BITS as usize;

/// The vertices of a simplex, in increasing order, stored on the stack (see [`RipsIndex::vertices`]).
#[derive(Clone, Copy)]
pub struct Vertices {
    buf: [usize; MAX_VERTICES],
    len: usize,
}

impl std::ops::Deref for Vertices {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.buf[..self.len]
    }
}

impl Vertices {
    /// The diameter of the simplex: the largest distance between two of its vertices (0 for a vertex).
    pub(crate) fn diameter(&self, distances: &[Vec<NotNan<f64>>]) -> NotNan<f64> {
        max_pairwise_distance(self, distances)
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

#[allow(clippy::too_many_arguments)]
fn add_cofacets(
    bases: &mut [SimplicesWithFilt],
    simplex_as_vec: &mut Vec<usize>,
    index: RipsIndex,
    fil_value: NotNan<f64>,
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
) {
    let n_vertices = distances.len();
    let max_element = simplex_as_vec.last().unwrap();
    let prev_dimension = simplex_as_vec.len() - 1;
    let new_dimension = prev_dimension + 1;
    if *max_element == n_vertices - 1 {
        // Cannot extend
        return;
    }
    // The new vertex is the largest, so it is the most significant digit of the cofacet's index
    let top_digit = (n_vertices + 1).pow(new_dimension as u32);
    for v in (max_element + 1)..n_vertices {
        // Compute filtration value
        let new_filtration_value = simplex_as_vec
            .iter()
            .map(|w| distances[*w][v])
            .max()
            .unwrap();
        let new_filtration_value = new_filtration_value.max(fil_value);
        // Cofaces only get bigger, so skip the whole subtree
        if new_filtration_value > threshold {
            continue;
        }
        // Build up coface
        simplex_as_vec.push(v);
        // Insert into basis
        let cofacet = RipsIndex(index.0 + (v + 1) * top_digit);
        bases[new_dimension].push((new_filtration_value, cofacet));
        // Recurse to cofacets
        if new_dimension < max_dim {
            add_cofacets(
                bases,
                simplex_as_vec,
                cofacet,
                new_filtration_value,
                distances,
                max_dim,
                threshold,
            );
        }
        // Remove v ready for next coface
        simplex_as_vec.pop();
    }
}

pub fn build_rips_bases(distances: &Vec<Vec<NotNan<f64>>>, max_dim: usize) -> RipsBasis {
    build_rips_bases_with_threshold(distances, max_dim, NotNan::new(f64::INFINITY).unwrap())
}

/// As [`build_rips_bases`], but only the simplices with diameter at most `threshold`.
pub fn build_rips_bases_with_threshold(
    distances: &Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
) -> RipsBasis {
    let mut bases: Vec<SimplicesWithFilt> = vec![vec![]; max_dim + 1];

    // Build up the basis
    for v in 0..(distances.len()) {
        let mut simplex = vec![v];
        let vertex = RipsIndex(v + 1);
        bases[0].push((NotNan::new(0.0_f64).unwrap(), vertex));
        if max_dim > 0 {
            add_cofacets(
                &mut bases,
                &mut simplex,
                vertex,
                NotNan::new(0 as f64).unwrap(),
                distances,
                max_dim,
                threshold,
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

/// The youngest facet of `simplex` (largest in (diameter, index) order) and its diameter, or `None` for a vertex.
///
/// Removing each vertex in turn (smallest first) gives the facets in decreasing index order.
/// No facet is bigger than the simplex, so the first with the same diameter is the youngest and we can stop there;
/// for a simplex with at least three vertices that happens as soon as we keep its longest edge, so almost immediately.
pub(crate) fn youngest_facet(
    simplex: RipsIndex,
    distances: &[Vec<NotNan<f64>>],
) -> Option<(NotNan<f64>, RipsIndex)> {
    let n_points = distances.len();
    let radix = n_points + 1;
    let vertices = simplex.vertices(n_points);
    if vertices.len() < 2 {
        return None;
    }
    let diameter = vertices.diameter(distances);
    let index = simplex.0;
    let mut best: Option<(NotNan<f64>, RipsIndex)> = None;
    let mut place = 1usize;
    for removed in 0..vertices.len() {
        let facet = RipsIndex(index % place + index / place / radix * place);
        let mut facet_diameter = NotNan::new(0.0).unwrap();
        for (i, &v) in vertices.iter().enumerate() {
            if i == removed {
                continue;
            }
            for (j, &w) in vertices.iter().enumerate().skip(i + 1) {
                if j != removed {
                    facet_diameter = facet_diameter.max(distances[v][w]);
                }
            }
        }
        // On ties keep the earlier, larger, index
        if best.is_none_or(|(current, _)| facet_diameter > current) {
            best = Some((facet_diameter, facet));
            if facet_diameter == diameter {
                break;
            }
        }
        place = place.wrapping_mul(radix);
    }
    best
}

/// The enclosing radius `min_i max_j d(i, j)`: at this diameter the Rips complex is a cone on some point,
/// so all homology (apart from one component) has died and nothing changes afterwards.
/// Infinite if there are no points.
pub fn enclosing_radius(distances: &[Vec<NotNan<f64>>]) -> NotNan<f64> {
    distances
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, d)| *d)
                .max()
                .unwrap_or(NotNan::new(0.0).unwrap())
        })
        .min()
        .unwrap_or(NotNan::new(f64::INFINITY).unwrap())
}

/// The largest distance between two distinct vertices (so 0 for a single vertex, whatever the diagonal of `distances`).
pub(crate) fn max_pairwise_distance(
    vertices: &[usize],
    distances: &[Vec<NotNan<f64>>],
) -> NotNan<f64> {
    let mut max = NotNan::new(0.0).unwrap();
    for (i, &v) in vertices.iter().enumerate() {
        let row = &distances[v];
        for &w in &vertices[i + 1..] {
            max = max.max(row[w]);
        }
    }
    max
}

// TUTORIAL:
// Now head over to homology.rs to implement the boundary matrix

// ====== High-level API ============================================

pub use phlite::barcode::Method;

/// A bar in the Rips barcode, see [`phlite::barcode::Bar`].
pub type RipsBar = Bar<RipsIndex, NotNan<f64>, Z2>;

/// Whether every simplex of dimension at most `max_dim` on `n_points` points can be represented by a [`RipsIndex`].
pub fn rips_index_fits(n_points: usize, max_dim: usize) -> bool {
    // There are no simplices with more vertices than points
    let max_dim = max_dim.min(n_points.saturating_sub(1));
    u32::try_from(max_dim + 1)
        .ok()
        .and_then(|exponent| (n_points + 1).checked_pow(exponent))
        .is_some()
}

/// Compute the Vietoris-Rips barcode of a distance matrix in dimensions `0..=max_dim`,
/// for the filtration up to diameter `threshold` (use infinity for no limit).
///
/// * `method` chooses between reducing the coboundary or boundary matrix.
/// * `algorithm` is the reduction algorithm to use.
/// * If `representatives` is true then every bar comes with a representative cycle.
///
/// Simplices with diameter above the [`enclosing_radius`] are never built, since by then the complex is a cone
/// and nothing changes; this doesn't change the barcode.
/// Bars of a filtration truncated at `threshold` that are still alive at `threshold` are essential.
/// Bars are returned in no particular order, and zero-length bars are omitted.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` (which both methods need) do not fit in a [`RipsIndex`], see [`rips_index_fits`].
pub fn rips_barcode(
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    algorithm: Algorithm<'_>,
    representatives: bool,
) -> Vec<RipsBar> {
    let n_points = distances.len();
    let threshold = threshold.min(enclosing_radius(&distances));
    // Both methods need simplices of dimension max_dim + 1 (as cofacets, or the top of the boundary matrix)
    assert!(rips_index_fits(n_points, max_dim + 1));
    // Each matrix is given its transpose, so that the reduction can skip apparent pairs
    match method {
        Method::Cohomology => {
            let boundary = RipsBoundary::<Z2>::new(distances.clone());
            let coboundary =
                build_rips_coboundary_matrix_with_threshold::<Z2>(distances, max_dim, threshold);
            let coboundary = (&coboundary).with_transpose(&boundary);
            if representatives {
                barcode::cohomology_with_involution(coboundary, algorithm)
            } else {
                barcode::cohomology(coboundary, algorithm)
            }
        }
        Method::Homology => {
            let coboundary = RipsCoboundary::<Z2>::new(distances.clone()).with_threshold(threshold);
            let boundary =
                build_rips_boundary_matrix_with_threshold::<Z2>(distances, max_dim + 1, threshold);
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
                NotNan::new(f64::INFINITY).unwrap(),
                Method::Cohomology,
                algorithm.clone(),
                false,
            );
            assert_eq!(summary(bars), expected, "{algorithm:?}");
            // With representatives the involution's debug assertions check them against the pairing
            let bars = rips_barcode(
                distances.clone(),
                2,
                NotNan::new(f64::INFINITY).unwrap(),
                Method::Cohomology,
                algorithm.clone(),
                true,
            );
            assert_eq!(bars.len(), expected.len());
            // Homology pairs simplices the same way (the pairing doesn't depend on the method)
            let bars = rips_barcode(
                distances.clone(),
                2,
                NotNan::new(f64::INFINITY).unwrap(),
                Method::Homology,
                algorithm,
                false,
            );
            let mut hom: Vec<_> = bars
                .into_iter()
                .map(|bar| (bar.birth, bar.death.map(|(d, _)| d)))
                .collect();
            hom.sort();
            let cohom: Vec<_> = expected.iter().map(|(b, d, _)| (*b, *d)).collect();
            assert_eq!(hom, cohom);
        }
    }

    /// Truncating at the enclosing radius must not change the barcode, but should shrink the complex
    #[test]
    fn enclosing_radius_truncation_is_exact() {
        use crate::{
            build_rips_bases, build_rips_bases_with_threshold,
            cohomology::build_rips_coboundary_matrix, enclosing_radius, rips_barcode, Method,
        };
        use ordered_float::NotNan;
        use phlite::{
            barcode::{self, Algorithm},
            fields::Z2,
            matrices::ColBasis,
        };
        // Points spread over a square, so the enclosing radius is well below the diameter
        let points: Vec<(f64, f64)> = (0..30)
            .map(|i| (((i * 7) % 11) as f64, ((i * 5) % 13) as f64))
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
        let r = enclosing_radius(&distances);
        assert!(
            build_rips_bases_with_threshold(&distances, 2, r).size()
                < build_rips_bases(&distances, 2).size()
        );
        let summary = |bars: Vec<crate::RipsBar>| {
            let mut out: Vec<_> = bars
                .into_iter()
                .map(|bar| (bar.birth_value, bar.death.map(|(_, d)| d)))
                .collect();
            out.sort();
            out
        };
        let untruncated = build_rips_coboundary_matrix::<Z2>(distances.clone(), 2);
        let expected = summary(barcode::cohomology(&untruncated, Algorithm::default()));
        for method in [Method::Cohomology, Method::Homology] {
            let bars = rips_barcode(
                distances.clone(),
                2,
                NotNan::new(f64::INFINITY).unwrap(),
                method,
                Algorithm::default(),
                true,
            );
            assert_eq!(summary(bars), expected, "{method:?}");
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
