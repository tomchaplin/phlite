//! Image, kernel and cokernel persistence for the inclusion of the Rips complex of a subset of the points into that of
//! all of them (with the same distances), by [`phlite::image_kernel`] on the lazy Rips boundary oracle.

use ordered_float::NotNan;
use phlite::columns::MergedColumns;
use phlite::fields::Z2;
use phlite::image_kernel::{image_kernel_cokernel, ImageKernelCokernel};
use phlite::matrices::SplitByDimension;
use phlite::six_pack::{FilteredChainMap, Inclusion, SixPack};

use super::{build_rips_bases_with_threshold, Distances, RipsBoundary, RipsIndex, RipsInt};

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

/// The six-pack of the inclusion of the Rips complex of a subset of the points into that of all of them (see
/// [`phlite::six_pack`]), over Z/2, with simplices up to dimension `max_dim + 1` (so bars of dimension at most
/// `max_dim` are right) and diameter at most `threshold`, with representatives if `representatives`.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` don't fit a [`RipsIndex<I>`] (see [`RipsIndex::fits`]).
pub fn rips_six_pack<I: RipsInt, D: Distances>(
    distances: &D,
    in_subset: &[bool],
    max_dim: usize,
    threshold: NotNan<f64>,
    representatives: bool,
) -> SixPack<RipsIndex<I>, RipsIndex<I>, NotNan<f64>, Z2> {
    let n = distances.n_points();
    let basis = build_rips_bases_with_threshold::<I, _>(distances, max_dim + 1, threshold);
    let cells: Vec<Vec<RipsIndex<I>>> = (0..basis.n_dimensions())
        .map(|d| (&basis).in_dimension(d).0.to_vec())
        .collect();
    let boundary = RipsBoundary::<Z2, I, &D>::new(distances).with_threshold(threshold);
    let in_subcomplex = |s: &RipsIndex<I>| s.vertices(n).iter().all(|&v| in_subset[v]);
    Inclusion {
        boundary: &boundary,
        cells,
        in_subcomplex,
    }
    .six_pack(MergedColumns, representatives, Some(max_dim))
}

/// The map of Rips filtrations induced by a map of points `f` from one point cloud to another that never increases
/// distances (`d(f x, f y) <= d(x, y)`), a simplicial map: a simplex goes to the simplex on its vertices' images, or to
/// zero if two of them go to the same point. Over Z/2, lazily on both Rips boundaries, with simplices up to dimension
/// `max_dim + 1` and diameter `threshold`.
pub struct RipsMap<'a, D1: Distances, D2: Distances> {
    domain: &'a D1,
    codomain: &'a D2,
    vertex_map: &'a [usize],
    max_dim: usize,
    threshold: NotNan<f64>,
}

impl<'a, D1: Distances, D2: Distances> RipsMap<'a, D1, D2> {
    /// The map sending point `i` of `domain` to point `vertex_map[i]` of `codomain`.
    ///
    /// # Errors
    /// With the first pair of points (within `threshold` of each other) whose images are further apart than they are.
    pub fn new(
        domain: &'a D1,
        codomain: &'a D2,
        vertex_map: &'a [usize],
        max_dim: usize,
        threshold: NotNan<f64>,
    ) -> Result<Self, (usize, usize)> {
        let n = domain.n_points();
        assert_eq!(vertex_map.len(), n, "an image for every point");
        assert!(
            vertex_map.iter().all(|&v| v < codomain.n_points()),
            "images are points of the codomain"
        );
        for i in 0..n {
            for j in (i + 1)..n {
                let d = domain.distance(i, j);
                let (a, b) = (vertex_map[i], vertex_map[j]);
                if d <= threshold && a != b && codomain.distance(a, b) > d {
                    return Err((i, j));
                }
            }
        }
        Ok(Self {
            domain,
            codomain,
            vertex_map,
            max_dim,
            threshold,
        })
    }

    /// Its six-pack (see [`phlite::six_pack`]), with representatives if `representatives`.
    pub fn six_pack(
        &self,
        representatives: bool,
    ) -> SixPack<RipsIndex, RipsIndex, NotNan<f64>, Z2> {
        FilteredChainMap::six_pack(self, MergedColumns, representatives, Some(self.max_dim))
    }
}

impl<'a, D1: Distances, D2: Distances> FilteredChainMap for RipsMap<'a, D1, D2> {
    type DomainCell = RipsIndex;
    type CodomainCell = RipsIndex;
    type Field = Z2;
    type Value = NotNan<f64>;
    type Domain = RipsBoundary<Z2, usize, &'a D1>;
    type Codomain = RipsBoundary<Z2, usize, &'a D2>;

    fn domain(&self) -> Self::Domain {
        RipsBoundary::new(self.domain).with_threshold(self.threshold)
    }

    fn codomain(&self) -> Self::Codomain {
        RipsBoundary::new(self.codomain).with_threshold(self.threshold)
    }

    fn domain_cells(&self) -> Vec<Vec<RipsIndex>> {
        let basis = build_rips_bases_with_threshold::<usize, _>(
            self.domain,
            self.max_dim + 1,
            self.threshold,
        );
        (0..basis.n_dimensions())
            .map(|d| (&basis).in_dimension(d).0.to_vec())
            .collect()
    }

    fn codomain_cells(&self) -> Vec<Vec<RipsIndex>> {
        let basis = build_rips_bases_with_threshold::<usize, _>(
            self.codomain,
            self.max_dim + 1,
            self.threshold,
        );
        (0..basis.n_dimensions())
            .map(|d| (&basis).in_dimension(d).0.to_vec())
            .collect()
    }

    fn apply(&self, cell: RipsIndex) -> Vec<(Z2, RipsIndex)> {
        let mut image: Vec<usize> = cell
            .vertices(self.domain.n_points())
            .iter()
            .map(|&v| self.vertex_map[v])
            .collect();
        let k = image.len();
        image.sort_unstable();
        image.dedup();
        if image.len() < k {
            return vec![];
        }
        let index = RipsIndex::from_indices(image.into_iter(), self.codomain.n_points())
            .expect("the image fits");
        vec![(Z2, index)]
    }
}
