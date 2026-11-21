//! Vietoris-Rips persistent homology, implemented with [`phlite`].
//!
//! The quickest way in is [`rips_barcode`], which computes the barcode of a distance matrix (optionally with representatives),
//! by either cohomology or homology and with any of `phlite`'s reduction algorithms.
//! In Python, this is `phlite.clouds.rips`.
//!
//! # How it is put together
//!
//! * Simplices are indexed by [`RipsIndex`]: the vertices, in increasing order, as the digits of a base-`(n + 1)` number.
//!   Cofacet and facet indices are computed arithmetically, by inserting or removing one digit.
//! * [`cohomology::RipsCoboundary`] and [`homology::RipsBoundary`] are the coboundary and boundary oracles,
//!   each of which is the other's transpose.
//!   They override the [optional performance hooks](phlite::matrices#optional-performance-hooks):
//!   cofacets are enumerated in increasing index order and none can be smaller than the simplex itself,
//!   so the first cofacet with the simplex's own diameter is its oldest one, and similarly for facets.
//!   Since nearly every Rips pair has zero persistence, these searches nearly always stop early.
//! * [`build_reduction_bases`] builds the column bases, omitting every simplex in a zero-persistence apparent pair
//!   (see [`in_zero_apparent_pair`]); for Rips these are nearly all of them.
//!   The full bases are available from [`build_rips_bases`] and friends.
//! * Simplices with diameter above the [`enclosing_radius`] are never built, since by then the complex is a cone,
//!   and a smaller `threshold` truncates the filtration as in ripser.
//!
//! These are, broadly, the optimisations that ripser and giotto-ph make.
//!
//! # Tutorial
//!
//! The source of this module doubles as a tutorial for implementing a filtration with `phlite`
//! (see the comments marked `TUTORIAL`): start in `src/rips/mod.rs`, then `homology.rs`, `cohomology.rs` and
//! `src/bin/rips.rs`.
//! The tutorial builds the full bases, which is the simplest way to use `phlite`;
//! [`rips_barcode`] shows the optimised version.

pub mod cohomology;
pub mod collapse;
pub mod distances;
pub mod explicit;
pub mod homology;

pub use distances::{distance_to_measure, Distances, Metric, SparseDistances, Weighted};

use ordered_float::NotNan;
use phlite::values::{INFINITY, MAX_FINITE, ZERO};
use phlite::{
    barcode::{self, Algorithm, Bar},
    columns::ColumnRepresentation,
    fields::{NonZeroCoefficient, Z2},
    matrices::{BasisElement, MatrixOracle, SplitBasis},
};

use crate::rips::{cohomology::RipsCoboundary, homology::RipsBoundary};

// TUTORIAL:
// First we implement the types that index into our boundary an coboundary matrix

// TODO:
// Swap to lexicographic numbering scheme described in Ripser paper

/// The unsigned integer type that a [`RipsIndex`] is stored in: `usize` (the default), `u32` or `u64`.
///
/// A simplex with `k` vertices on `n` points needs `(n + 1)^k` to fit (see [`RipsIndex::fits`]), so a wider type allows
/// more points in high dimensions, at the cost of more memory per index. `usize` is 32 bits on `wasm32`, so the browser
/// build uses `u64` there.
pub trait RipsInt:
    Copy
    + Ord
    + std::hash::Hash
    + std::fmt::Debug
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Rem<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
    const ZERO: Self;
    const ONE: Self;
    /// `x` as this type (`x` is a vertex, digit or radix, so always fits when the simplices do).
    fn from_usize(x: usize) -> Self;
    /// `self` as a `usize` (only called on a single digit, which is at most the number of points).
    fn to_usize(self) -> usize;
    fn checked_pow(self, exp: u32) -> Option<Self>;
    fn wrapping_mul(self, other: Self) -> Self;
    fn pow(self, exp: u32) -> Self;
}

macro_rules! impl_rips_int {
    ($($t:ty),*) => {$(
        impl RipsInt for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline]
            fn from_usize(x: usize) -> Self {
                x as Self
            }
            #[inline]
            fn to_usize(self) -> usize {
                self as usize
            }
            #[inline]
            fn checked_pow(self, exp: u32) -> Option<Self> {
                <$t>::checked_pow(self, exp)
            }
            #[inline]
            fn wrapping_mul(self, other: Self) -> Self {
                <$t>::wrapping_mul(self, other)
            }
            #[inline]
            fn pow(self, exp: u32) -> Self {
                <$t>::pow(self, exp)
            }
        }
    )*};
}
impl_rips_int!(usize, u32, u64);

/// Represents a simplex by storing the elements as an integer with base `b` where `b = n_points + 1`.
/// Each digit in the integer corresponds to an element in the simplex.
///
/// The elements are always represented in increasing order.
/// We add on `1` to each digit to ensure that a leading `0` is not deleted.
/// Therefore the simplex `{0, 1}` is represented by `RipsIndex(12)` where the base for `12` depends on the number of points in your point cloud.
/// Note this is fairly inefficient since as simplicies `{0, 1} == {1, 0}` but we only ever use `RipsIndex(12)`.
///
/// The integer is a `usize` by default; see [`RipsInt`] for the alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RipsIndex<I: RipsInt = usize>(I);

// TUTORIAL:
// We must manually implement this trait to mark `RipsIndex` as a indexing type for matrices

impl<I: RipsInt> BasisElement for RipsIndex<I> {}

// TUTORIAL:
// Now we implement some helper methods to convert our indexing type into the corresponding simplices

impl<I: RipsInt> RipsIndex<I> {
    /// Whether every simplex of dimension at most `max_dim` on `n_points` points can be represented.
    pub fn fits(n_points: usize, max_dim: usize) -> bool {
        // There are no simplices with more vertices than points
        let max_dim = max_dim.min(n_points.saturating_sub(1));
        u32::try_from(max_dim + 1)
            .ok()
            .and_then(|exponent| {
                let radix = n_points.checked_add(1)?;
                // The radix itself must fit before we can raise it to a power
                if I::from_usize(radix).to_usize() != radix {
                    return None;
                }
                I::from_usize(radix).checked_pow(exponent)
            })
            .is_some()
    }

    /// The vertices, in increasing order. Prefer [`vertices`](Self::vertices), which doesn't allocate.
    pub fn to_vec(self, n_points: usize) -> Vec<usize> {
        self.vertices(n_points).to_vec()
    }

    /// The vertices, in increasing order, decoded onto the stack.
    pub fn vertices(self, n_points: usize) -> Vertices {
        let radix = I::from_usize(n_points + 1);
        let mut vertices = Vertices {
            buf: [0; MAX_VERTICES],
            len: 0,
        };
        let mut working = self.0;
        while working > I::ZERO {
            vertices.buf[vertices.len] = (working % radix).to_usize() - 1;
            vertices.len += 1;
            working /= radix;
        }
        vertices
    }

    /// NOTE: The indicies should be provided in ascending order!
    pub fn from_indices(indices: impl Iterator<Item = usize>, n_points: usize) -> Option<Self> {
        let radix = I::from_usize(n_points + 1);
        let mut inner = I::ZERO;
        for (i, coeff) in indices.enumerate() {
            inner += radix.pow(i as u32) * I::from_usize(coeff + 1);
        }
        if inner > I::ZERO {
            Some(Self(inner))
        } else {
            None
        }
    }

    pub fn dimension(self, n_points: usize) -> usize {
        let radix = I::from_usize(n_points + 1);
        let mut working = self.0;
        let mut n_vertices = 0;
        while working > I::ZERO {
            n_vertices += 1;
            working /= radix;
        }
        n_vertices - 1
    }
}

/// A simplex has at most this many vertices.
///
/// A simplex with `k` vertices needs `n >= k` points, so its index is at least `k (k + 1)^(k - 1)`
/// (the vertices `0, ..., k - 1` as base-`(k + 1)` digits); for that to fit in a `usize` we need `k <= 15` (64-bit) or `k <= 9` (32-bit).
/// Keeping the buffer small matters, since [`Vertices`] is copied around a lot.
pub(crate) const MAX_VERTICES: usize = 16;
const _: () = assert!(
    MAX_VERTICES >= 15,
    "Vertices must hold the largest simplex whose index fits in 64 bits"
);

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
    #[inline]
    pub(crate) fn diameter<D: Distances + ?Sized>(&self, distances: &D) -> NotNan<f64> {
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
pub type RipsBasis<I = usize> = SplitBasis<RipsIndex<I>>;

// TUTORIAL:
// Now we provide some functions for building up a basis for Rips (co)homology
// This will be quite different depending on your application.
// Remember: You only need to construct and sort a basis for the column space of each matrix.
// If you are computing cohomology then to compute H^K just compute a basis for C_K.
//
// While building we keep each simplex's filtration value alongside it (so that we can sort),
// then throw the filtration values away when we assemble the final `SplitBasis`.

type SimplicesWithFilt<I> = Vec<(NotNan<f64>, RipsIndex<I>)>;

#[allow(clippy::too_many_arguments)]
fn add_cofacets<I: RipsInt, D: Distances + ?Sized>(
    bases: &mut [SimplicesWithFilt<I>],
    simplex_as_vec: &mut Vec<usize>,
    index: RipsIndex<I>,
    fil_value: NotNan<f64>,
    distances: &D,
    max_dim: usize,
    threshold: NotNan<f64>,
) {
    let n_vertices = distances.n_points();
    let max_element = *simplex_as_vec.last().unwrap();
    let prev_dimension = simplex_as_vec.len() - 1;
    let new_dimension = prev_dimension + 1;
    if max_element == n_vertices - 1 {
        // Cannot extend
        return;
    }
    // The new vertex is the largest, so it is the most significant digit of the cofacet's index
    let top_digit = I::from_usize(n_vertices + 1).pow(new_dimension as u32);
    for v in distances.cofacet_candidates(distances.sparsest(simplex_as_vec), max_element + 1) {
        // Compute filtration value
        let new_filtration_value = simplex_as_vec
            .iter()
            .map(|w| distances.distance(*w, v))
            .max()
            .unwrap();
        let mut new_filtration_value = new_filtration_value.max(fil_value);
        if D::HAS_BIRTHS {
            new_filtration_value = new_filtration_value.max(distances.birth(v));
        }
        // Cofaces only get bigger, so skip the whole subtree
        if new_filtration_value > threshold {
            continue;
        }
        // Build up coface
        simplex_as_vec.push(v);
        // Insert into basis
        let cofacet = RipsIndex(index.0 + I::from_usize(v + 1) * top_digit);
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

pub fn build_rips_bases<I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    max_dim: usize,
) -> RipsBasis<I> {
    build_rips_bases_with_threshold(distances, max_dim, INFINITY)
}

/// As [`build_rips_bases`], but only the simplices with diameter at most `threshold`.
pub fn build_rips_bases_with_threshold<I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    max_dim: usize,
    threshold: NotNan<f64>,
) -> RipsBasis<I> {
    let mut bases: Vec<SimplicesWithFilt<I>> = vec![vec![]; max_dim + 1];

    // Build up the basis
    for v in 0..(distances.n_points()) {
        let birth = distances.birth(v);
        // Points born after the threshold never appear (but, as in ripser, unweighted points, born at 0, always do)
        if birth > threshold && (D::HAS_BIRTHS || birth > ZERO) {
            continue;
        }
        let mut simplex = vec![v];
        let vertex = RipsIndex(I::from_usize(v + 1));
        bases[0].push((birth, vertex));
        if max_dim > 0 {
            add_cofacets(
                &mut bases,
                &mut simplex,
                vertex,
                birth,
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
pub(crate) fn youngest_facet<I: RipsInt, D: Distances + ?Sized>(
    simplex: RipsIndex<I>,
    distances: &D,
) -> Option<(NotNan<f64>, RipsIndex<I>)> {
    let vertices = simplex.vertices(distances.n_points());
    let diameter = vertices.diameter(distances);
    youngest_facet_given(simplex, &vertices, diameter, distances)
}

/// As [`youngest_facet`], when the vertices and diameter of `simplex` are already known
fn youngest_facet_given<I: RipsInt, D: Distances + ?Sized>(
    simplex: RipsIndex<I>,
    vertices: &Vertices,
    diameter: NotNan<f64>,
    distances: &D,
) -> Option<(NotNan<f64>, RipsIndex<I>)> {
    let radix = I::from_usize(distances.n_points() + 1);
    if vertices.len() < 2 {
        return None;
    }
    let index = simplex.0;
    let mut best: Option<(NotNan<f64>, RipsIndex<I>)> = None;
    let mut place = I::ONE;
    for removed in 0..vertices.len() {
        let facet = RipsIndex(index % place + index / place / radix * place);
        let mut facet_diameter = if D::HAS_BIRTHS {
            vertices
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != removed)
                .map(|(_, &v)| distances.birth(v))
                .max()
                .unwrap_or(ZERO)
        } else {
            ZERO
        };
        for (i, &v) in vertices.iter().enumerate() {
            if i == removed {
                continue;
            }
            for (j, &w) in vertices.iter().enumerate().skip(i + 1) {
                if j != removed {
                    facet_diameter = facet_diameter.max(distances.distance(v, w));
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
/// Infinite if there are no points, or if no point is joined to every other.
pub fn enclosing_radius<D: Distances + ?Sized>(distances: &D) -> NotNan<f64> {
    distances.enclosing_radius()
}

/// Whether `simplex` is in a zero-persistence apparent pair (on either side), with respect to the complex of simplices
/// with diameter at most `threshold`.
///
/// Such simplices need no reduction and only give zero-length bars, so [`build_reduction_bases`] omits them.
/// If `check_cofacets` is false then we only check whether `simplex` is the *death* of such a pair,
/// which avoids looking at cofacets (e.g. if they might not fit in a [`RipsIndex`]).
pub fn in_zero_apparent_pair<I: RipsInt, D: Distances + ?Sized>(
    simplex: RipsIndex<I>,
    distances: &D,
    threshold: NotNan<f64>,
    check_cofacets: bool,
) -> bool {
    // Decode each simplex and compute its diameter only once
    let n_points = distances.n_points();
    let vertices = simplex.vertices(n_points);
    let diameter = vertices.diameter(distances);
    // The birth: its oldest cofacet has the same diameter, and it is that cofacet's youngest facet
    if check_cofacets {
        let cofacet = cohomology::oldest_cofacet_given::<Z2, I, _>(
            distances, threshold, simplex, vertices, diameter,
        );
        if let Some(cofacet) = cofacet {
            if cofacet.filtration_value == diameter {
                let cofacet_vertices = cofacet.row_index.vertices(n_points);
                let youngest =
                    youngest_facet_given(cofacet.row_index, &cofacet_vertices, diameter, distances);
                if youngest.map(|(_, facet)| facet) == Some(simplex) {
                    return true;
                }
            }
        }
    }
    // The death: its youngest facet has the same diameter, and it is that facet's oldest cofacet
    if let Some((facet_diameter, facet)) =
        youngest_facet_given(simplex, &vertices, diameter, distances)
    {
        if facet_diameter == diameter {
            let facet_vertices = facet.vertices(n_points);
            let oldest = cohomology::oldest_cofacet_given::<Z2, I, _>(
                distances,
                threshold,
                facet,
                facet_vertices,
                facet_diameter,
            );
            if oldest.map(|entry| entry.row_index) == Some(simplex) {
                return true;
            }
        }
    }
    false
}

/// The cofacets of `simplex` (with diameter `diameter`) obtained by adding a vertex larger than all of its own,
/// with diameter at most `threshold`. Every simplex is such a cofacet of exactly one facet.
fn cofacets_above<I: RipsInt, D: Distances + ?Sized>(
    simplex: RipsIndex<I>,
    diameter: NotNan<f64>,
    distances: &D,
    threshold: NotNan<f64>,
) -> impl Iterator<Item = (NotNan<f64>, RipsIndex<I>)> + '_ {
    let n_points = distances.n_points();
    let vertices = simplex.vertices(n_points);
    // The new vertex is the largest, so it is the most significant digit of the cofacet's index
    let top_digit = I::from_usize(n_points + 1).pow(vertices.len() as u32);
    let first = vertices.last().map_or(0, |v| v + 1);
    let candidates = distances.cofacet_candidates(distances.sparsest(&vertices), first);
    candidates.filter_map(move |v| {
        let cofacet_diameter = vertices
            .iter()
            .map(|w| distances.distance(*w, v))
            .max()
            .unwrap_or(diameter)
            .max(diameter);
        let cofacet_diameter = if D::HAS_BIRTHS {
            cofacet_diameter.max(distances.birth(v))
        } else {
            cofacet_diameter
        };
        (cofacet_diameter <= threshold).then(|| {
            (
                cofacet_diameter,
                RipsIndex(simplex.0 + I::from_usize(v + 1) * top_digit),
            )
        })
    })
}

/// Column bases for reducing a Rips (co)boundary matrix: the simplices of each dimension up to `max_dim` with diameter
/// at most `threshold`, in filtration order, but **omitting every simplex in a zero-persistence apparent pair**.
///
/// Those simplices need no reduction (the clearing algorithms recompute apparent pairs when needed,
/// given [`first_column_in_row`](phlite::matrices::HasRowFiltration::first_column_in_row)) and only give zero-length bars,
/// and for Rips filtrations they are nearly all the simplices; this is what ripser and giotto-ph do.
/// Only the remaining simplices are stored and sorted, and the top dimension is never stored in full.
///
/// Only use these with [`Clearing`](phlite::reduction::Clearing) or its parallel version (on a matrix with a transpose):
/// the [`Standard`](phlite::reduction::Standard) algorithm doesn't look for apparent pairs, so needs every simplex.
///
/// Set `check_top_cofacets` to false if cofacets of the top dimension might not fit in a [`RipsIndex`]
/// (then top-dimensional births of zero-persistence apparent pairs are kept, which is harmless).
/// If `parallel`, build on the current rayon thread pool.
pub fn build_reduction_bases<I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    max_dim: usize,
    threshold: NotNan<f64>,
    check_top_cofacets: bool,
    parallel: bool,
) -> RipsBasis<I> {
    build_reduction_bases_from(distances, max_dim, threshold, check_top_cofacets, parallel, false, &[])
}

/// As [`build_reduction_bases`], but with dimension 0 empty if `skip_vertices` (it is found by union-find instead),
/// and leaving out the edges in `excluded` (sorted: those that kill a component, which clearing would leave out).
fn build_reduction_bases_from<I: RipsInt, D: Distances + ?Sized>(
    distances: &D,
    max_dim: usize,
    threshold: NotNan<f64>,
    check_top_cofacets: bool,
    parallel: bool,
    skip_vertices: bool,
    excluded: &[RipsIndex<I>],
) -> RipsBasis<I> {
    use rayon::prelude::*;
    let keep = |dim: usize| {
        let check_cofacets = dim < max_dim || check_top_cofacets;
        move |&(_, simplex): &(NotNan<f64>, RipsIndex<I>)| {
            !(dim == 0 && skip_vertices)
                && !(dim == 1 && excluded.binary_search(&simplex).is_ok())
                && !in_zero_apparent_pair(simplex, distances, threshold, check_cofacets)
        }
    };
    let sorted_indices = |mut simplices: Vec<(NotNan<f64>, RipsIndex<I>)>| {
        if parallel {
            simplices.par_sort_unstable();
        } else {
            simplices.sort_unstable();
        }
        simplices
            .into_iter()
            .map(|(_, simplex)| simplex)
            .collect::<Vec<_>>()
    };

    // All the simplices of the current dimension (needed to generate the next), then those we keep
    let mut all: Vec<(NotNan<f64>, RipsIndex<I>)> = (0..distances.n_points())
        .map(|v| (distances.birth(v), RipsIndex(I::from_usize(v + 1))))
        // Points born after the threshold never appear (but, as in ripser, unweighted points, born at 0, always do)
        .filter(|&(birth, _)| birth <= threshold || (!D::HAS_BIRTHS && birth <= ZERO))
        .collect();
    let mut dims = vec![];
    for dim in 0..=max_dim {
        let kept: Vec<_> = if parallel {
            all.par_iter().copied().filter(keep(dim)).collect()
        } else {
            all.iter().copied().filter(keep(dim)).collect()
        };
        dims.push(sorted_indices(kept));
        if dim == max_dim {
            break;
        }
        let generate = |&(diameter, simplex): &(NotNan<f64>, RipsIndex<I>)| {
            cofacets_above(simplex, diameter, distances, threshold)
        };
        if dim + 1 == max_dim {
            // Generate and filter the top dimension in one go, so that it is never stored in full
            let kept: Vec<_> = if parallel {
                all.par_iter()
                    .flat_map_iter(generate)
                    .filter(keep(max_dim))
                    .collect()
            } else {
                all.iter()
                    .flat_map(generate)
                    .filter(keep(max_dim))
                    .collect()
            };
            dims.push(sorted_indices(kept));
            break;
        }
        all = if parallel {
            all.par_iter().flat_map_iter(generate).collect()
        } else {
            all.iter().flat_map(generate).collect()
        };
    }
    SplitBasis::from_dimensions(dims)
}

/// The largest distance between two distinct vertices (so 0 for a single vertex, whatever the diagonal of `distances`).
#[inline]
pub(crate) fn max_pairwise_distance<D: Distances + ?Sized>(
    vertices: &[usize],
    distances: &D,
) -> NotNan<f64> {
    let mut max = if D::HAS_BIRTHS {
        vertices
            .iter()
            .map(|&v| distances.birth(v))
            .max()
            .unwrap_or(ZERO)
    } else {
        ZERO
    };
    for (i, &v) in vertices.iter().enumerate() {
        for &w in &vertices[i + 1..] {
            max = max.max(distances.distance(v, w));
        }
    }
    max
}

// TUTORIAL:
// Now head over to homology.rs to implement the boundary matrix

// ====== High-level API ============================================

pub use phlite::barcode::Method;

/// A bar in the Rips barcode, see [`phlite::barcode::Bar`].
pub type RipsBar<I = usize> = Bar<RipsIndex<I>, NotNan<f64>, Z2>;

/// Whether every simplex of dimension at most `max_dim` on `n_points` points can be represented by a [`RipsIndex`].
///
/// This is for the default `usize` index; see [`RipsIndex::fits`] for the others.
pub fn rips_index_fits(n_points: usize, max_dim: usize) -> bool {
    RipsIndex::<usize>::fits(n_points, max_dim)
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
/// An infinite distance is an edge that never appears (as is a missing entry of a [`SparseDistances`]), so no
/// simplex containing it is built.
///
/// `distances` is any [`Distances`]: a dense matrix (`Vec<Vec<NotNan<f64>>>`), a [`SparseDistances`], or a reference
/// to either.
/// Bars are returned in no particular order, and zero-length bars are omitted.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` (which both methods need) do not fit in a [`RipsIndex`], see [`rips_index_fits`].
///
/// Simplices are indexed by `RipsIndex<I>` (with `I = usize` unless you choose otherwise, e.g.
/// `rips_barcode::<_, u64, _>(...)` for more points on 32-bit targets); it must fit every simplex up to dimension
/// `max_dim + 1` (see [`RipsIndex::fits`]).
pub fn rips_barcode<R: ColumnRepresentation, I: RipsInt, D: Distances>(
    distances: D,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<RipsBar<I>> {
    #[allow(unreachable_patterns)]
    let parallel = matches!(algorithm, Algorithm::ParallelClearing(_));
    rips_barcode_with(distances, max_dim, threshold, method, algorithm, representatives, true, parallel, true)
}

/// As [`rips_barcode`], choosing whether (by cohomology) to find dimension 0 by union-find (`union_find`, which
/// [`rips_barcode`] does) rather than by reducing it, whether to build the bases (and run the union-find's sorts)
/// in parallel on the current rayon thread pool (`parallel_setup`, which [`rips_barcode`] does only for the parallel
/// algorithm), whatever the algorithm, and whether to [collapse the edges](collapse) first (`collapse`, which
/// [`rips_barcode`] does).
///
/// Edges are collapsed only when it can help and is safe: from `max_dim` 2 (in dimensions 0 and 1 the reduction is
/// cheaper than collapsing, as Perseus 6 found too), without representatives, and when the points have no births. The
/// collapsed edges are used only if at least a tenth of the edges went (when collapsing moves many edges but removes
/// few, the reduction can get slower).
///
/// # Panics
/// As [`rips_barcode`]: if simplices of dimension `max_dim + 1` don't fit in a [`RipsIndex`].
#[allow(clippy::too_many_arguments)]
pub fn rips_barcode_with<R: ColumnRepresentation, I: RipsInt, D: Distances>(
    distances: D,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    union_find: bool,
    parallel_setup: bool,
    collapse: bool,
) -> Vec<RipsBar<I>> {
    let n_points = distances.n_points();
    // A simplex with an infinite diameter has an edge that never appears, so it never appears either
    let largest_finite = MAX_FINITE;
    let threshold = threshold
        .min(enclosing_radius(&distances))
        .min(largest_finite);
    if collapse && max_dim >= 2 && !representatives && !D::HAS_BIRTHS {
        let (collapsed, stats) = collapse::collapse_edges(&distances, threshold);
        if stats.edges_after * 10 <= stats.edges_before * 9 {
            return rips_barcode_with(
                collapsed,
                max_dim,
                threshold,
                method,
                algorithm,
                representatives,
                union_find,
                parallel_setup,
                false,
            );
        }
    }
    // Both methods need simplices of dimension max_dim + 1 (as cofacets, or the top of the boundary matrix)
    assert!(
        RipsIndex::<I>::fits(n_points, max_dim + 1),
        "simplices of dimension {} on {n_points} points don't fit in a RipsIndex (see RipsIndex::fits)",
        max_dim + 1
    );
    let parallel = parallel_setup;
    // The standard algorithm can't find the owners of omitted apparent pairs, so give it every simplex
    let filter = !matches!(algorithm, Algorithm::Standard(_));
    let bases = |max_dim: usize, check_top_cofacets: bool, skip_vertices: bool, excluded: &[RipsIndex<I>]| {
        if filter {
            build_reduction_bases_from(
                &distances,
                max_dim,
                threshold,
                check_top_cofacets,
                parallel,
                skip_vertices,
                excluded,
            )
        } else {
            without(build_rips_bases_with_threshold(&distances, max_dim, threshold), skip_vertices, excluded)
        }
    };
    // Each matrix is given its transpose, so that the reduction can find apparent pairs,
    // and its basis omits the zero-persistence ones entirely. Both share the one set of distances.
    match method {
        Method::Cohomology if union_find && !(representatives && threshold < enclosing_radius(&distances)) => {
            // Dimension 0 by union-find, as Ripser does. The edges that kill a component are exactly the pivots of
            // the reduced dimension-0 coboundary, which clearing would leave out of dimension 1, so leave them out of
            // the basis. (With representatives, the involution finds those of essential classes above dimension 0 by
            // reducing their columns, which needs those edges: there are none unless the threshold is below the
            // enclosing radius, where the complex is a cone; otherwise reduce dimension 0 as usual, below.)
            let (mut bars, deaths) = connected_components(&distances, threshold, representatives, parallel);
            if max_dim >= 1 {
                let basis = bases(max_dim, true, true, &deaths);
                let boundary = RipsBoundary::<Z2, I, _>::new(&distances).with_threshold(threshold);
                let coboundary = RipsCoboundary::<Z2, I, _>::new(&distances)
                    .with_threshold(threshold)
                    .with_basis(basis);
                let coboundary = coboundary.by_ref().with_transpose(&boundary);
                bars.extend(if representatives {
                    barcode::cohomology_with_involution(coboundary, algorithm)
                } else {
                    barcode::cohomology(coboundary, algorithm)
                });
            }
            bars
        }
        Method::Cohomology => {
            let basis = bases(max_dim, true, false, &[]);
            let boundary = RipsBoundary::<Z2, I, _>::new(&distances).with_threshold(threshold);
            let coboundary = RipsCoboundary::<Z2, I, _>::new(&distances)
                .with_threshold(threshold)
                .with_basis(basis);
            let coboundary = coboundary.by_ref().with_transpose(&boundary);
            if representatives {
                barcode::cohomology_with_involution(coboundary, algorithm)
            } else {
                barcode::cohomology(coboundary, algorithm)
            }
        }
        Method::Homology => {
            // Cofacets of the top dimension (max_dim + 1) might not fit in a RipsIndex, so don't look at them
            let basis = bases(max_dim + 1, false, false, &[]);
            let coboundary = RipsCoboundary::<Z2, I, _>::new(&distances).with_threshold(threshold);
            let boundary = RipsBoundary::<Z2, I, _>::new(&distances)
                .with_threshold(threshold)
                .with_basis(basis);
            let boundary = boundary.by_ref().with_transpose(&coboundary);
            let mut bars = barcode::homology(boundary, algorithm, representatives);
            // Nothing can kill the top dimensional simplices, so they look essential but are not bars we computed
            bars.retain(|bar| bar.birth.dimension(n_points) <= max_dim);
            bars
        }
    }
}

/// `basis` without dimension 0 if `skip_vertices`, and without the edges in `excluded` (sorted).
fn without<I: RipsInt>(basis: RipsBasis<I>, skip_vertices: bool, excluded: &[RipsIndex<I>]) -> RipsBasis<I> {
    use phlite::matrices::SplitByDimension;
    let dims: Vec<Vec<RipsIndex<I>>> = (0..SplitByDimension::n_dimensions(&&basis))
        .map(|dim| {
            let slice = (&basis).in_dimension(dim).0;
            match dim {
                0 if skip_vertices => vec![],
                1 => slice.iter().copied().filter(|e| excluded.binary_search(e).is_err()).collect(),
                _ => slice.to_vec(),
            }
        })
        .collect();
    SplitBasis::from_dimensions(dims)
}

/// The barcode in dimension 0, by union-find on the edges in filtration order (by diameter, then index, as the
/// reductions order simplices), with the elder rule: when an edge joins two components, the one whose oldest vertex
/// (by birth, then index) is younger dies, so the pairs are those a reduction would find. Zero-length bars are left
/// out. Also returns the edges that kill a component (zero-length bars' included), sorted.
///
/// With `representatives`, a finite bar's is its birth vertex plus the oldest vertex of the component it joins (born
/// earlier, so the chain is born at the birth, and joined to it at the death and not before); an essential bar's, its
/// birth vertex.
fn connected_components<I: RipsInt, D: Distances>(
    distances: &D,
    threshold: NotNan<f64>,
    representatives: bool,
    parallel: bool,
) -> (Vec<RipsBar<I>>, Vec<RipsIndex<I>>) {
    use rayon::prelude::*;
    let n = distances.n_points();
    let zero = ZERO;
    // Points born after the threshold never appear (but, as in ripser, unweighted points, born at 0, always do)
    let present: Vec<bool> = (0..n)
        .map(|v| {
            let birth = distances.birth(v);
            birth <= threshold || (!D::HAS_BIRTHS && birth <= zero)
        })
        .collect();
    let edge = |i: usize, j: usize| RipsIndex(I::from_usize(i + 1) + I::from_usize(n + 1) * I::from_usize(j + 1));
    let edges_of = |i: usize| {
        let present = &present;
        distances
            .neighbours(i, i + 1)
            .filter(move |&j| j != i && present[j])
            .filter_map(move |j| {
                let value = max_pairwise_distance(&[i, j], distances);
                (value <= threshold).then(|| (value, edge(i, j), i as u32, j as u32))
            })
    };
    let mut edges: Vec<(NotNan<f64>, RipsIndex<I>, u32, u32)> = if parallel {
        (0..n).into_par_iter().filter(|&i| present[i]).flat_map_iter(edges_of).collect()
    } else {
        (0..n).filter(|&i| present[i]).flat_map(edges_of).collect()
    };
    let by_key = |a: &(NotNan<f64>, RipsIndex<I>, u32, u32), b: &(NotNan<f64>, RipsIndex<I>, u32, u32)| {
        (a.0, a.1).cmp(&(b.0, b.1))
    };
    if parallel {
        edges.par_sort_unstable_by(by_key);
    } else {
        edges.sort_unstable_by(by_key);
    }
    let vertex = |v: u32| RipsIndex(I::from_usize(v as usize + 1));
    let birth = |v: u32| distances.birth(v as usize);
    // Each vertex's parent, next to its birth, so that comparing roots needs no other lookup
    let mut nodes: Vec<(u32, NotNan<f64>)> = (0..n as u32).map(|v| (v, birth(v))).collect();
    fn find(nodes: &mut [(u32, NotNan<f64>)], mut v: u32) -> u32 {
        while nodes[v as usize].0 != v {
            let grandparent = nodes[nodes[v as usize].0 as usize].0;
            nodes[v as usize].0 = grandparent;
            v = grandparent;
        }
        v
    }
    let rep = |cells: &[u32]| {
        representatives.then(|| cells.iter().map(|&v| (<Z2 as NonZeroCoefficient>::one(), vertex(v))).collect())
    };
    let mut bars = vec![];
    let mut deaths = vec![];
    for (value, index, a, b) in edges {
        let (ra, rb) = (find(&mut nodes, a), find(&mut nodes, b));
        if ra == rb {
            continue;
        }
        let (older, younger) = if (nodes[ra as usize].1, ra) < (nodes[rb as usize].1, rb) { (ra, rb) } else { (rb, ra) };
        nodes[younger as usize].0 = older;
        deaths.push(index);
        let birth_value = nodes[younger as usize].1;
        if birth_value != value {
            bars.push(Bar {
                birth: vertex(younger),
                birth_value,
                death: Some((index, value)),
                representative: rep(&[younger, older]),
            });
        }
    }
    for v in 0..n as u32 {
        if present[v as usize] && nodes[v as usize].0 == v {
            bars.push(Bar { birth: vertex(v), birth_value: birth(v), death: None, representative: rep(&[v]) });
        }
    }
    if parallel {
        deaths.par_sort_unstable();
    } else {
        deaths.sort_unstable();
    }
    (bars, deaths)
}

#[cfg(test)]
mod tests {
    use phlite::values::{INFINITY, ZERO};
    use crate::rips::RipsIndex;

    /// `rips_barcode` attaches transposes (enabling apparent pairs); that must not change any bar
    #[test]
    fn apparent_pairs_do_not_change_barcode() {
        // Without edge collapse, which changes which simplices pair (not the values)
        use crate::rips::{cohomology::build_rips_coboundary_matrix, rips_barcode_with, Method};
        use ordered_float::NotNan;
        use phlite::{
            barcode::{self, Algorithm},
            columns::{HeapColumns, MergedColumns, PivotColumns, SortedColumns},
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
        let summary = |bars: Vec<crate::rips::RipsBar>| {
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
        // Each representation gives a different type of algorithm, so loop over each separately
        macro_rules! check_algorithms {
            ($columns:expr) => {
                for algorithm in [true, false].into_iter().flat_map(|cache| {
                    let clearing = Clearing::new().with_columns($columns).cache_columns(cache);
                    [
                        Algorithm::Standard(Standard::new().with_columns($columns).cache_columns(cache)),
                        Algorithm::ParallelClearing(clearing.clone().parallel()),
                        Algorithm::Clearing(clearing.clone().store_r(true)),
                        Algorithm::Clearing(clearing),
                    ]
                }) {  let bars = rips_barcode_with::<_, usize, _>(
                distances.clone(),
                2,
                INFINITY,
                Method::Cohomology,
                algorithm.clone(),
                false,
                true,
                false,
                false,
            );
            assert_eq!(summary(bars), expected, "{algorithm:?}");
            // With representatives the involution's debug assertions check them against the pairing
            let bars = rips_barcode_with::<_, usize, _>(
                distances.clone(),
                2,
                INFINITY,
                Method::Cohomology,
                algorithm.clone(),
                true,
                true,
                false,
                false,
            );
            assert_eq!(bars.len(), expected.len());
            // Homology pairs simplices the same way (the pairing doesn't depend on the method)
            let bars = rips_barcode_with::<_, usize, _>(
                distances.clone(),
                2,
                INFINITY,
                Method::Homology,
                algorithm,
                false,
                true,
                false,
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
            };
        }
        check_algorithms!(MergedColumns);
        check_algorithms!(HeapColumns);
        check_algorithms!(SortedColumns);
        check_algorithms!(PivotColumns);
    }

    /// Truncating at the enclosing radius must not change the barcode, but should shrink the complex
    #[test]
    fn enclosing_radius_truncation_is_exact() {
        use crate::rips::{
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
            build_rips_bases_with_threshold::<usize, _>(&distances, 2, r).size()
                < build_rips_bases::<usize, _>(&distances, 2).size()
        );
        let summary = |bars: Vec<crate::rips::RipsBar>| {
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
            let bars = rips_barcode::<_, usize, _>(
                distances.clone(),
                2,
                INFINITY,
                method,
                Algorithm::default(),
                true,
            );
            assert_eq!(summary(bars), expected, "{method:?}");
        }
    }

    /// The reduction bases are the full bases minus the zero-persistence apparent pairs, built the same way in parallel
    #[test]
    fn reduction_bases_omit_exactly_zero_apparent_pairs() {
        use crate::rips::{
            build_reduction_bases, build_rips_bases_with_threshold, in_zero_apparent_pair,
        };
        use ordered_float::NotNan;
        use phlite::matrices::{ColBasis, SplitByDimension};
        let points: Vec<(f64, f64)> = (0..14)
            .map(|i| (((i * 7) % 11) as f64, ((i * 5) % 13) as f64))
            .collect();
        let euclidean: Vec<Vec<NotNan<f64>>> = points
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
        let ties: Vec<Vec<NotNan<f64>>> = (0..9)
            .map(|i| {
                (0..9)
                    .map(|j| {
                        NotNan::new(if i == j {
                            0.0
                        } else {
                            ((i * j + i + j) % 3) as f64
                        })
                        .unwrap()
                    })
                    .collect()
            })
            .collect();
        for distances in [euclidean, ties] {
            for threshold in [f64::INFINITY, 4.0] {
                let threshold = NotNan::new(threshold).unwrap();
                for check_top_cofacets in [true, false] {
                    let full = build_rips_bases_with_threshold(&distances, 2, threshold);
                    let serial =
                        build_reduction_bases(&distances, 2, threshold, check_top_cofacets, false);
                    let parallel =
                        build_reduction_bases(&distances, 2, threshold, check_top_cofacets, true);
                    let as_vec = |b: phlite::matrices::BasisSlice<'_, crate::rips::RipsIndex>| {
                        (0..b.size()).map(|i| b.element(i)).collect::<Vec<_>>()
                    };
                    for dim in 0..=2 {
                        let expected: Vec<_> = as_vec((&full).in_dimension(dim))
                            .into_iter()
                            .filter(|s| {
                                !in_zero_apparent_pair(
                                    *s,
                                    &distances,
                                    threshold,
                                    dim < 2 || check_top_cofacets,
                                )
                            })
                            .collect();
                        assert_eq!(as_vec((&serial).in_dimension(dim)), expected, "dim {dim}");
                        assert_eq!(as_vec((&parallel).in_dimension(dim)), expected, "dim {dim}");
                    }
                    assert!(serial.size() < full.size());
                }
            }
        }
    }

    /// Storing the index in `u32`, `u64` or `usize` changes nothing but how many points fit
    #[test]
    fn index_width_does_not_change_barcode() {
        use crate::rips::{rips_barcode, Method, RipsBar, RipsIndex, RipsInt};
        use ordered_float::NotNan;
        use phlite::{
            barcode::Algorithm,
            columns::HeapColumns,
            reduction::{Clearing, Standard},
        };
        let points: Vec<(f64, f64)> = (0..14)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 14.0;
                let r = 1.0 + 0.15 * ((i * 5 % 7) as f64 - 3.0);
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
        fn summary<I: RipsInt>(
            bars: Vec<RipsBar<I>>,
            n: usize,
        ) -> Vec<(Vec<usize>, Option<Vec<usize>>, f64)> {
            let mut out: Vec<_> = bars
                .into_iter()
                .map(|bar| {
                    (
                        bar.birth.to_vec(n),
                        bar.death.map(|(d, _)| d.to_vec(n)),
                        bar.birth_value.into_inner(),
                    )
                })
                .collect();
            out.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out
        }
        let inf = INFINITY;
        for method in [Method::Cohomology, Method::Homology] {
            let run = |standard: bool| {
                let algorithm = |_: ()| -> Algorithm<'static, HeapColumns> {
                    if standard {
                        Algorithm::Standard(Standard::new().with_columns(HeapColumns))
                    } else {
                        Algorithm::Clearing(Clearing::new().with_columns(HeapColumns))
                    }
                };
                let a = summary(
                    rips_barcode::<_, usize, _>(
                        distances.clone(),
                        2,
                        inf,
                        method,
                        algorithm(()),
                        true,
                    ),
                    14,
                );
                let b = summary(
                    rips_barcode::<_, u32, _>(distances.clone(), 2, inf, method, algorithm(()), true),
                    14,
                );
                let c = summary(
                    rips_barcode::<_, u64, _>(distances.clone(), 2, inf, method, algorithm(()), true),
                    14,
                );
                assert_eq!(a, b);
                assert_eq!(a, c);
            };
            run(false);
            run(true);
        }
        // (n + 1)^4 must fit for H_2 (3-simplices): n = 254 is the most for u32, 65 534 for u64
        assert!(RipsIndex::<u32>::fits(254, 3));
        assert!(!RipsIndex::<u32>::fits(255, 3));
        assert!(RipsIndex::<u64>::fits(65_534, 3));
        assert!(!RipsIndex::<u64>::fits(65_535, 3));
    }

    /// `Vertices` holds at most `MAX_VERTICES` vertices; check no valid simplex has more
    #[test]
    fn max_vertices_is_enough() {
        use crate::rips::{rips_index_fits, RipsIndex};
        // The largest simplex on k points has k vertices: it fits for k = 15 but not 16
        assert!(rips_index_fits(15, 14));
        assert!(!rips_index_fits(16, 15));
        let simplex = RipsIndex::<usize>::from_indices(0..15, 15).unwrap();
        assert_eq!(simplex.vertices(15).len(), 15);
    }

    #[test]
    fn test_bases_with_max_dim_0() {
        use ordered_float::NotNan;
        use phlite::matrices::ColBasis;
        let one = NotNan::new(1.0).unwrap();
        let zero = ZERO;
        let basis =
            crate::rips::build_rips_bases::<usize, _>(&vec![vec![zero, one], vec![one, zero]], 0);
        assert_eq!(basis.n_dimensions(), 1);
        assert_eq!(basis.size(), 2);
    }

    /// A sparse matrix (absent = never appears) must give exactly the dense matrix's barcode (infinite entries), with
    /// every algorithm, method and threshold, and with representatives
    #[test]
    fn sparse_distances_give_the_dense_barcode() {
        use crate::rips::{rips_barcode, Method, SparseDistances};
        use ordered_float::NotNan;
        use phlite::{
            barcode::Algorithm,
            columns::{HeapColumns, MergedColumns},
            reduction::{Clearing, Standard},
        };
        // A small linear congruential generator: deterministic, and no dependency
        let mut state = 12345u64;
        let mut next = move |modulus: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % modulus
        };
        type Summary = Vec<(
            Vec<usize>,
            NotNan<f64>,
            Option<(Vec<usize>, NotNan<f64>)>,
            Option<Vec<Vec<usize>>>,
        )>;
        let summary = |bars: Vec<crate::rips::RipsBar>, n: usize| -> Summary {
            let mut out: Summary = bars
                .into_iter()
                .map(|bar| {
                    let rep = bar.representative.map(|rep| {
                        let mut rep: Vec<_> = rep.into_iter().map(|(_, s)| s.to_vec(n)).collect();
                        rep.sort();
                        rep
                    });
                    (
                        bar.birth.to_vec(n),
                        bar.birth_value,
                        bar.death.map(|(d, v)| (d.to_vec(n), v)),
                        rep,
                    )
                })
                .collect();
            out.sort();
            out
        };
        for trial in 0..60 {
            let n = 2 + next(9) as usize;
            // Small integer distances (lots of ties), with some edges missing
            let mut dense = vec![vec![ZERO; n]; n];
            let mut edges = vec![];
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = if next(4) == 0 {
                        f64::INFINITY
                    } else {
                        next(6) as f64
                    };
                    dense[i][j] = NotNan::new(d).unwrap();
                    dense[j][i] = NotNan::new(d).unwrap();
                    edges.push((i, j, d));
                }
            }
            let sparse = SparseDistances::from_edges(n, edges).unwrap();
            let threshold = NotNan::new(if trial % 3 == 0 { 3.0 } else { f64::INFINITY }).unwrap();
            for method in [Method::Cohomology, Method::Homology] {
                for representatives in [false, true] {
                    macro_rules! compare {
                        ($algorithm:expr) => {{
                            let want = summary(
                                rips_barcode::<_, usize, _>(
                                    dense.clone(),
                                    2,
                                    threshold,
                                    method,
                                    $algorithm,
                                    representatives,
                                ),
                                n,
                            );
                            let got = summary(
                                rips_barcode::<_, usize, _>(
                                    &sparse,
                                    2,
                                    threshold,
                                    method,
                                    $algorithm,
                                    representatives,
                                ),
                                n,
                            );
                            assert_eq!(
                                got,
                                want,
                                "trial {trial}, {method:?}, representatives {representatives}, {}",
                                stringify!($algorithm)
                            );
                        }};
                    }
                    compare!(Algorithm::Clearing(
                        Clearing::new().with_columns(HeapColumns)
                    ));
                    compare!(Algorithm::Clearing(
                        Clearing::new()
                            .with_columns(MergedColumns)
                            .cache_columns(true)
                    ));
                    compare!(Algorithm::Standard(
                        Standard::new().with_columns(HeapColumns)
                    ));
                    // (Which representative the parallel algorithm finds can depend on the threads' timing)
                    if !representatives {
                        compare!(Algorithm::ParallelClearing(
                            Clearing::new().with_columns(HeapColumns).parallel()
                        ));
                    }
                }
            }
        }
    }

    #[test]
    fn test_rips_index() {
        let vec = vec![5, 12, 7];
        let index = RipsIndex::<usize>::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);

        // Preserves trailing 0
        let vec = vec![5, 12, 0];
        let index = RipsIndex::<usize>::from_indices(vec.iter().copied(), 20).unwrap();
        let as_vec = index.to_vec(20);
        assert_eq!(vec, as_vec);
    }
}
