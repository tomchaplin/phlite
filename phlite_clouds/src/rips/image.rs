//! Image, kernel and cokernel persistence for the inclusion of the Rips complex of a subset of the points into that of
//! all of them (with the same distances), by [`phlite::image_kernel`] on the lazy Rips boundary oracle.

use ordered_float::NotNan;
use phlite::columns::MergedColumns;
use phlite::fields::Z2;
use phlite::image_kernel::{image_kernel_cokernel, ImageKernelCokernel};
use phlite::matrices::SplitByDimension;

use super::{build_rips_bases_with_threshold, Distances, RipsBoundary, RipsIndex};

/// The barcodes of the image, kernel and cokernel of `H(Rips(subset)) -> H(Rips(all))`, and of the pair, over Z/2, in
/// dimensions up to `max_dim` (simplices up to `max_dim + 1`), with diameters at most `threshold`. `in_subset[v]` says
/// whether point `v` is in the subset.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` don't fit a [`RipsIndex`] (see [`super::rips_index_fits`]).
pub fn rips_image_kernel_cokernel<D: Distances>(
    distances: &D,
    in_subset: &[bool],
    max_dim: usize,
    threshold: NotNan<f64>,
) -> ImageKernelCokernel<RipsIndex> {
    let n = distances.n_points();
    let basis = build_rips_bases_with_threshold::<usize, _>(distances, max_dim + 1, threshold);
    let cells: Vec<Vec<RipsIndex>> =
        (0..basis.n_dimensions()).map(|d| (&basis).in_dimension(d).0.to_vec()).collect();
    let boundary = RipsBoundary::<Z2, usize, &D>::new(distances).with_threshold(threshold);
    let in_subcomplex = |s: &RipsIndex| s.vertices(n).iter().all(|&v| in_subset[v]);
    image_kernel_cokernel(&boundary, cells, in_subcomplex, MergedColumns)
}
