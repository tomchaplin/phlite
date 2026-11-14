//! Cubical persistent homology of images (2D), volumes (3D) and grids of any dimension up to [`MAX_AXES`], implemented
//! with [`phlite`].
//!
//! The quickest way in is [`cubical_barcode`], which computes the barcode of a [`CubicalComplex`] (optionally with
//! representatives), by either cohomology or homology and with any of `phlite`'s reduction algorithms.
//! In Python, this is `phlite.images.cubical`.
//!
//! # Constructions
//!
//! An image gives a filtered cubical complex in one of two ways (see [`Construction`]):
//! * the **V-construction**: pixels (voxels) are the *vertices* of a grid, and every other cell enters with the
//!   largest value of its vertices (the lower-star filtration). The grid has `n_a` vertices along an axis of `n_a`
//!   pixels.
//! * the **T-construction**: pixels are the *top-dimensional cells* (squares, cubes) of a grid, and every other cell
//!   enters with the smallest value of the top cells it is a face of. The grid has `n_a + 1` vertices along an axis of
//!   `n_a` pixels.
//!
//! Either way the complex grows through the *sublevel* sets of the image (dark first), or with
//! [`Filtration::Superlevel`] through its superlevel sets (bright first).
//!
//! # Coordinates
//!
//! Cells are identified to users by their coordinates on the *doubled grid* (see [`CubicalComplex::coordinates`]), in
//! the order of the image's axes (`[row, column]` for a 2D numpy array): a coordinate is even where the cell is a
//! single vertex position along that axis, and odd where it spans two, so a cell's dimension is its number of odd
//! coordinates. In the V-construction pixel `p` is the vertex at `2 p`; in the T-construction it is the top cell at
//! `2 p + 1`. This is the convention of gudhi's cubical complexes.
//!
//! # Missing and infinite values
//!
//! A cell with value `+inf` (after negating, for superlevel sets) never enters the filtration, so it is left out of the
//! complex altogether, and NaN pixels are treated the same way, whichever the filtration: the complex is then the
//! cubical complex of the remaining pixels (in the V-construction, the cells all of whose vertices remain; in the
//! T-construction, the faces of the remaining top cells). Bars of the remaining complex that would die at `+inf` are
//! essential. A value of `-inf` (or `+inf` for superlevel sets) enters before everything else.
//!
//! # How it is put together
//!
//! Like `phlite_clouds::rips`, nothing is stored but the image:
//! * Cells are numbered by a [`Cube`] (a `u32`), densely and in a way that makes ties break correctly and lets
//!   faces and cofaces be enumerated in order of number (see [`grid`]).
//!   A cell's value is computed from the pixels whenever it is needed.
//! * [`cohomology::CubicalCoboundary`] and [`homology::CubicalBoundary`] are the coboundary and boundary oracles, each
//!   the other's transpose. They override the [optional performance hooks](phlite::matrices#optional-performance-hooks):
//!   no coface comes before a cell in the filtration, and cofaces are enumerated in order of number, so the first coface
//!   with the cell's own value is its oldest one (an *emergent pair*, as in Cubical Ripser), and similarly for faces.
//! * [`build_reduction_bases`] builds the column bases, leaving out every cell in a zero-persistence apparent pair
//!   (see [`CubicalComplex::in_zero_apparent_pair`]); in images with smooth regions these are most cells.

pub mod cohomology;
pub mod grid;
pub mod homology;
#[cfg(test)]
mod tests;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar},
    columns::ColumnRepresentation,
    fields::Z2,
    matrices::{MatrixOracle, SplitBasis},
};
use rayon::prelude::*;

pub use grid::{Cube, Grid, TooManyCells, MAX_AXES};
pub use phlite::barcode::Method;

use crate::cubical::{cohomology::CubicalCoboundary, grid::Decoded, homology::CubicalBoundary};

/// How pixels become cells, see the [module docs](self#constructions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Construction {
    /// Pixels are vertices, and a cell's value is the largest of its vertices' (the lower-star filtration).
    #[default]
    V,
    /// Pixels are top-dimensional cells, and a cell's value is the smallest of the top cells it is a face of.
    T,
}

/// Whether the complex grows through sublevel or superlevel sets of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filtration {
    /// By increasing value: cells enter at their value.
    #[default]
    Sublevel,
    /// By decreasing value: the sublevel filtration of the negated image, reported in the image's own values (so bars
    /// run downwards, from a larger birth to a smaller death).
    Superlevel,
}

/// The input can't be made into a cubical complex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CubicalError {
    /// The number of pixels isn't the product of the shape.
    WrongLength { expected: usize, found: usize },
    /// The complex has too many cells to number (see [`TooManyCells`]).
    TooManyCells(u64),
}

impl std::fmt::Display for CubicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubicalError::WrongLength { expected, found } => {
                write!(
                    f,
                    "expected {expected} pixels (the product of the shape), found {found}"
                )
            }
            CubicalError::TooManyCells(n) => write!(
                f,
                "the cubical complex would have {n} cells, but at most {} are supported",
                u32::MAX - 1
            ),
        }
    }
}

impl std::error::Error for CubicalError {}

/// The filtered cubical complex of an image with `D` axes: the image, and the [`Grid`] of its cells.
///
/// Values are kept per pixel, never per cell: a cell's value is computed from the (at most `2^D`) pixels around it
/// whenever it is needed.
#[derive(Clone, Debug)]
pub struct CubicalComplex<const D: usize> {
    /// The pixels in C order, negated for superlevel sets, with NaN made `+inf`: cells enter at these values
    pixels: Vec<f64>,
    /// Pixels along each axis
    shape: [usize; D],
    /// C-order strides of `pixels`
    pixel_strides: [usize; D],
    construction: Construction,
    filtration: Filtration,
    /// `None` if the image has no pixels (so the complex has no cells)
    grid: Option<Grid<D>>,
}

impl<const D: usize> CubicalComplex<D> {
    /// The cubical complex of the image with the given `shape` (pixels along each axis), whose `pixels` are in C order
    /// (row-major, as a contiguous numpy array).
    ///
    /// NaN pixels are missing (see the [module docs](self#missing-and-infinite-values)).
    ///
    /// # Errors
    /// If `pixels` doesn't have the length the shape says, or the complex has `u32::MAX` cells or more.
    ///
    /// # Panics
    /// If `D` is zero or more than [`MAX_AXES`].
    pub fn new(
        mut pixels: Vec<f64>,
        shape: [usize; D],
        construction: Construction,
        filtration: Filtration,
    ) -> Result<Self, CubicalError> {
        assert!(
            (1..=MAX_AXES).contains(&D),
            "images have between 1 and {MAX_AXES} axes"
        );
        let expected = shape.iter().product::<usize>();
        if pixels.len() != expected {
            return Err(CubicalError::WrongLength {
                expected,
                found: pixels.len(),
            });
        }
        for pixel in &mut pixels {
            *pixel = match filtration {
                _ if pixel.is_nan() => f64::INFINITY,
                Filtration::Sublevel => *pixel,
                Filtration::Superlevel => -*pixel,
            };
        }
        let mut pixel_strides = [0; D];
        let mut stride = 1;
        for a in (0..D).rev() {
            pixel_strides[a] = stride;
            stride *= shape[a];
        }
        let grid = if expected == 0 {
            None
        } else {
            let vertices = shape.map(|n| match construction {
                Construction::V => n,
                Construction::T => n + 1,
            });
            let too_many =
                || CubicalError::TooManyCells(vertices.iter().map(|&n| 2 * n as u64 - 1).product());
            let vertices = vertices
                .iter()
                .map(|&n| u32::try_from(n).map_err(|_| too_many()))
                .collect::<Result<Vec<_>, _>>()?;
            let grid = Grid::new(std::array::from_fn(|a| vertices[a]))
                .map_err(|TooManyCells(n)| CubicalError::TooManyCells(n))?;
            Some(grid)
        };
        Ok(Self {
            pixels,
            shape,
            pixel_strides,
            construction,
            filtration,
            grid,
        })
    }

    /// The number of pixels along each axis.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    pub fn construction(&self) -> Construction {
        self.construction
    }

    pub fn filtration(&self) -> Filtration {
        self.filtration
    }

    /// The grid of cells, or `None` if the image has no pixels.
    pub fn grid(&self) -> Option<&Grid<D>> {
        self.grid.as_ref()
    }

    /// The number of cells of the grid (including those left out because their value is `+inf`).
    pub fn n_cells(&self) -> usize {
        self.grid.as_ref().map_or(0, |grid| grid.n_cells() as usize)
    }

    #[inline]
    fn grid_ref(&self) -> &Grid<D> {
        self.grid
            .as_ref()
            .expect("an image without pixels has no cells")
    }

    /// The coordinates of `cube` on the doubled grid (see the [module docs](self#coordinates)).
    pub fn coordinates(&self, cube: Cube) -> [u32; D] {
        self.grid_ref().coordinates(cube)
    }

    /// The cell at `coordinates` on the doubled grid, or `None` if they are outside it.
    pub fn cube_at(&self, coordinates: [u32; D]) -> Option<Cube> {
        self.grid.as_ref()?.cube_at(coordinates)
    }

    /// The dimension of `cube`.
    pub fn dimension(&self, cube: Cube) -> usize {
        self.grid_ref().dimension(cube)
    }

    /// The value at which `cube` enters the filtration, in the image's own values (so for superlevel sets, the cube
    /// enters when the threshold falls to this value). NaN if it never does.
    pub fn value(&self, cube: Cube) -> f64 {
        self.to_image_value(self.internal_value(&self.grid_ref().decode(cube)))
    }

    /// Convert a filtration value (of the sublevel filtration of the possibly negated image) to the image's values.
    fn to_image_value(&self, value: f64) -> f64 {
        match self.filtration {
            _ if value == f64::INFINITY => f64::NAN,
            Filtration::Sublevel => value,
            Filtration::Superlevel => -value,
        }
    }

    /// The value at which `cell` enters the (sublevel) filtration: `+inf` if never.
    #[inline]
    pub(crate) fn internal_value(&self, cell: &Decoded<D>) -> f64 {
        // The (at most 2^D) pixels around the cell, as offsets into `pixels`
        let mut offsets = [0usize; 1 << MAX_AXES];
        let mut n = 1;
        match self.construction {
            Construction::V => {
                // The vertices: along an odd axis, the pixels at r and r + 1; along an even one, at r
                for a in 0..D {
                    let stride = self.pixel_strides[a];
                    let base = cell.reduced[a] as usize * stride;
                    if cell.is_odd(a) {
                        for i in 0..n {
                            offsets[n + i] = offsets[i] + base + stride;
                            offsets[i] += base;
                        }
                        n *= 2;
                    } else {
                        for offset in &mut offsets[..n] {
                            *offset += base;
                        }
                    }
                }
                offsets[..n]
                    .iter()
                    .map(|&i| self.pixels[i])
                    .fold(f64::NEG_INFINITY, f64::max)
            }
            Construction::T => {
                // The top cells: along an odd axis, the pixel at r; along an even one, those at r - 1 and r (if any)
                for a in 0..D {
                    let stride = self.pixel_strides[a];
                    let r = cell.reduced[a] as usize;
                    let below = (!cell.is_odd(a) && r > 0).then(|| (r - 1) * stride);
                    let at = (cell.is_odd(a) || r < self.shape[a]).then_some(r * stride);
                    match (below, at) {
                        (Some(below), Some(at)) => {
                            for i in 0..n {
                                offsets[n + i] = offsets[i] + at;
                                offsets[i] += below;
                            }
                            n *= 2;
                        }
                        (Some(only), None) | (None, Some(only)) => {
                            for offset in &mut offsets[..n] {
                                *offset += only;
                            }
                        }
                        (None, None) => unreachable!("every axis has a pixel"),
                    }
                }
                offsets[..n]
                    .iter()
                    .map(|&i| self.pixels[i])
                    .fold(f64::INFINITY, f64::min)
            }
        }
    }

    /// The value of the cell numbered `cube`, as [`internal_value`](Self::internal_value).
    #[inline]
    pub(crate) fn internal_value_of(&self, cube: Cube) -> f64 {
        self.internal_value(&self.grid_ref().decode(cube))
    }

    /// The oldest coface of `cell` (the smallest in the order on (value, number)) that is in the complex, and its value.
    ///
    /// Cofaces are enumerated in increasing order of number and none enters before `cell`, which has value `value`,
    /// so the first with that value is the oldest and we stop there.
    #[inline]
    pub(crate) fn oldest_coface(
        &self,
        cell: &Decoded<D>,
        value: f64,
    ) -> Option<(f64, grid::Neighbour<D>)> {
        let mut best: Option<(f64, grid::Neighbour<D>)> = None;
        for coface in self.grid_ref().cofaces(cell) {
            let coface_value = self.internal_value(&coface.decoded);
            // Ties keep the earlier, smaller, number; cells at +inf are not in the complex
            if coface_value < best.map_or(f64::INFINITY, |(v, _)| v) {
                best = Some((coface_value, coface));
                if coface_value == value {
                    break;
                }
            }
        }
        best
    }

    /// The youngest face of `cell` (the largest in the order on (value, number)), and its value; `None` for a vertex.
    ///
    /// Faces are enumerated in decreasing order of number and none enters after `cell`, which has value `value`, so
    /// the first with that value is the youngest and we stop there.
    #[inline]
    pub(crate) fn youngest_face(
        &self,
        cell: &Decoded<D>,
        value: f64,
    ) -> Option<(f64, grid::Neighbour<D>)> {
        let mut best: Option<(f64, grid::Neighbour<D>)> = None;
        for face in self.grid_ref().faces(cell) {
            let face_value = self.internal_value(&face.decoded);
            // Ties keep the earlier, larger, number
            if best.is_none_or(|(v, _)| face_value > v) {
                best = Some((face_value, face));
                if face_value == value {
                    break;
                }
            }
        }
        best
    }

    /// Whether `cube` (which must be in the complex) is in a zero-persistence apparent pair, as either its birth or its
    /// death.
    ///
    /// Such cells need no reduction and only give zero-length bars, so [`build_reduction_bases`] leaves them out: the
    /// clearing algorithms find them again when they need them, through
    /// [`first_column_in_row`](phlite::matrices::HasRowFiltration::first_column_in_row).
    pub fn in_zero_apparent_pair(&self, cube: Cube) -> bool {
        let grid = self.grid_ref();
        let cell = grid.decode(cube);
        let value = self.internal_value(&cell);
        self.in_zero_apparent_pair_given(cube, &cell, value)
    }

    /// As [`in_zero_apparent_pair`](Self::in_zero_apparent_pair), when the cell is already decoded and has value `value`.
    fn in_zero_apparent_pair_given(&self, cube: Cube, cell: &Decoded<D>, value: f64) -> bool {
        // The birth: its oldest coface has the same value, and it is that coface's youngest face
        if let Some((coface_value, coface)) = self.oldest_coface(cell, value) {
            if coface_value == value
                && self
                    .youngest_face(&coface.decoded, coface_value)
                    .is_some_and(|(_, face)| face.cube == cube)
            {
                return true;
            }
        }
        // The death: its youngest face has the same value, and it is that face's oldest coface
        if let Some((face_value, face)) = self.youngest_face(cell, value) {
            if face_value == value
                && self
                    .oldest_coface(&face.decoded, face_value)
                    .is_some_and(|(_, coface)| coface.cube == cube)
            {
                return true;
            }
        }
        false
    }
}

// ====== Bases ===========================================================

/// The column basis for cubical (co)homology: the cells of each dimension, in filtration order.
pub type CubicalBasis = SplitBasis<Cube>;

/// The cells of dimensions `0..=max_dim` that are in the complex (value below `+inf`), in filtration order (by value,
/// then number), leaving out those in zero-persistence apparent pairs if `skip_apparent`.
///
/// Leaving those out is only valid for the clearing algorithms, on a matrix that can find apparent pairs (as the
/// cubical matrices can): the [`Standard`](phlite::reduction::Standard) algorithm needs every cell.
/// If `parallel`, build on the current rayon thread pool.
pub fn build_bases<const D: usize>(
    complex: &CubicalComplex<D>,
    max_dim: usize,
    skip_apparent: bool,
    parallel: bool,
) -> CubicalBasis {
    build_bases_from(complex, 0, max_dim, skip_apparent, parallel, &[])
}

/// As [`build_bases`], but with no cells below dimension `min_dim` (those dimensions are empty), and leaving out every
/// cube in `excluded` (sorted).
fn build_bases_from<const D: usize>(
    complex: &CubicalComplex<D>,
    min_dim: usize,
    max_dim: usize,
    skip_apparent: bool,
    parallel: bool,
    excluded: &[Cube],
) -> CubicalBasis {
    let Some(grid) = complex.grid() else {
        return SplitBasis::from_dimensions((0..=max_dim).map(|_| std::iter::empty()));
    };
    let keep = |number: u32| -> Option<(NotNan<f64>, Cube)> {
        let cube = Cube(number);
        let cell = grid.decode(cube);
        let value = complex.internal_value(&cell);
        if value == f64::INFINITY
            || (skip_apparent && complex.in_zero_apparent_pair_given(cube, &cell, value))
            || excluded.binary_search(&cube).is_ok()
        {
            return None;
        }
        Some((NotNan::new(value).expect("values are never NaN"), cube))
    };
    let dimensions = (0..=max_dim).map(|dim| {
        let cells = if dim < min_dim {
            0..0
        } else {
            grid.cells_of_dimension(dim)
        };
        let mut kept: Vec<(NotNan<f64>, Cube)> = if parallel {
            cells.into_par_iter().filter_map(keep).collect()
        } else {
            cells.filter_map(keep).collect()
        };
        if parallel {
            kept.par_sort_unstable();
        } else {
            kept.sort_unstable();
        }
        kept.into_iter().map(|(_, cube)| cube).collect::<Vec<_>>()
    });
    SplitBasis::from_dimensions(dimensions.collect::<Vec<_>>())
}

/// Column bases for reducing a cubical (co)boundary matrix: the cells of dimensions `0..=max_dim`, in filtration
/// order, **leaving out every cell in a zero-persistence apparent pair** (see [`build_bases`]).
pub fn build_reduction_bases<const D: usize>(
    complex: &CubicalComplex<D>,
    max_dim: usize,
    parallel: bool,
) -> CubicalBasis {
    build_bases(complex, max_dim, true, parallel)
}

// ====== Dimension 0 ======================================================

/// The barcode in dimension 0, by union-find on the edges in filtration order (by value, then number, as the
/// reductions order cells), with the elder rule: when an edge joins two components, the one whose oldest vertex is
/// younger dies. So the pairs are those a reduction would find. Zero-length bars are left out. Also returns the edges
/// that kill a component (zero-length bars' included). Bars have no representatives.
fn connected_components<const D: usize>(
    complex: &CubicalComplex<D>,
    parallel: bool,
) -> (Vec<CubicalBar>, Vec<Cube>) {
    let Some(grid) = complex.grid() else {
        return (vec![], vec![]);
    };
    let value = |cube: Cube| complex.internal_value_of(cube);
    let key = |cube: Cube| (NotNan::new(value(cube)).expect("values are never NaN"), cube);
    let vertices = grid.cells_of_dimension(0);
    let first_vertex = vertices.start;
    let mut parent: Vec<u32> = vertices.clone().collect();
    let finite_edge = |number: u32| {
        let cube = Cube(number);
        let v = value(cube);
        (v != f64::INFINITY).then(|| (NotNan::new(v).expect("values are never NaN"), cube))
    };
    let mut edges: Vec<(NotNan<f64>, Cube)> = if parallel {
        grid.cells_of_dimension(1).into_par_iter().filter_map(finite_edge).collect()
    } else {
        grid.cells_of_dimension(1).filter_map(finite_edge).collect()
    };
    if parallel {
        edges.par_sort_unstable();
    } else {
        edges.sort_unstable();
    }
    // The root of a component is its oldest vertex
    fn find(parent: &mut [u32], first: u32, mut v: u32) -> u32 {
        while parent[(v - first) as usize] != v {
            let grandparent = parent[(parent[(v - first) as usize] - first) as usize];
            parent[(v - first) as usize] = grandparent;
            v = grandparent;
        }
        v
    }
    let mut bars = vec![];
    let mut deaths = vec![];
    for (edge_value, edge) in edges {
        let mut ends = grid.faces(&grid.decode(edge)).into_iter().map(|face| face.cube.0);
        let (a, b) = (ends.next().expect("an edge has two ends"), ends.next().expect("an edge has two ends"));
        let (ra, rb) = (find(&mut parent, first_vertex, a), find(&mut parent, first_vertex, b));
        if ra == rb {
            continue;
        }
        let (older, younger) = if key(Cube(ra)) < key(Cube(rb)) { (ra, rb) } else { (rb, ra) };
        parent[(younger - first_vertex) as usize] = older;
        deaths.push(edge);
        let birth_value = key(Cube(younger)).0;
        if birth_value != edge_value {
            bars.push(Bar {
                birth: Cube(younger),
                birth_value,
                death: Some((edge, edge_value)),
                representative: None,
            });
        }
    }
    for v in vertices {
        if parent[(v - first_vertex) as usize] == v && value(Cube(v)) != f64::INFINITY {
            bars.push(Bar {
                birth: Cube(v),
                birth_value: key(Cube(v)).0,
                death: None,
                representative: None,
            });
        }
    }
    (bars, deaths)
}

/// The barcode in the top dimension `D - 1`, by union-find on the dual: the cells of dimension `D - 1` in decreasing
/// filtration order (by value, then number), each joining the (one or two) top cells it is a face of, or a top cell and
/// the outside of the grid. Cells not in the complex count as entering at `+inf` (the whole grid is then a box, with no
/// homology in dimension `D - 1`). A component's root is its youngest top cell (the outside is younger than all);
/// when a cell joins two components, the one with the older root dies (in reverse), so that cell is the birth of a
/// class which that root kills. By duality, these are the pairs a reduction would find. Classes killed at `+inf` are
/// essential, classes born there never exist, and zero-length bars are left out. Bars have no representatives.
fn voids<const D: usize>(complex: &CubicalComplex<D>, parallel: bool) -> Vec<CubicalBar> {
    let Some(grid) = complex.grid() else {
        return vec![];
    };
    let key = |cube: Cube| (NotNan::new(complex.internal_value_of(cube)).expect("values are never NaN"), cube);
    let tops = grid.cells_of_dimension(D);
    if tops.is_empty() {
        // An axis with a single vertex: the complex is a box of lower dimension, with no homology in dimension D - 1
        return vec![];
    }
    let first_top = tops.start;
    let outside = tops.len() as u32;
    let mut parent: Vec<u32> = (0..=outside).collect();
    let mut cells: Vec<(NotNan<f64>, Cube)> = if parallel {
        grid.cells_of_dimension(D - 1).into_par_iter().map(|n| key(Cube(n))).collect()
    } else {
        grid.cells_of_dimension(D - 1).map(|n| key(Cube(n))).collect()
    };
    if parallel {
        cells.par_sort_unstable_by(|a, b| b.cmp(a));
    } else {
        cells.sort_unstable_by(|a, b| b.cmp(a));
    }
    fn find(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            let grandparent = parent[parent[v as usize] as usize];
            parent[v as usize] = grandparent;
            v = grandparent;
        }
        v
    }
    // Roots are compared by key, the outside last
    let root_key = |node: u32| (node != outside).then(|| key(Cube(first_top + node)));
    let younger = |a: u32, b: u32| match (root_key(a), root_key(b)) {
        (None, _) => true,
        (_, None) => false,
        (Some(ka), Some(kb)) => ka > kb,
    };
    let mut bars = vec![];
    for (value, cell) in cells {
        let mut sides = grid.cofaces(&grid.decode(cell)).into_iter().map(|c| c.cube.0 - first_top);
        let a = sides.next().expect("a cell of dimension D - 1 is a face of a top cell");
        let b = sides.next().unwrap_or(outside);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            continue;
        }
        let (survivor, dies) = if younger(ra, rb) { (ra, rb) } else { (rb, ra) };
        parent[dies as usize] = survivor;
        if value.into_inner() == f64::INFINITY {
            continue;
        }
        let (death_value, death) = key(Cube(first_top + dies));
        if death_value == value {
            continue;
        }
        bars.push(Bar {
            birth: cell,
            birth_value: value,
            death: (death_value.into_inner() != f64::INFINITY).then_some((death, death_value)),
            representative: None,
        });
    }
    bars
}

// ====== High-level API ===================================================

/// A bar in a cubical barcode, see [`phlite::barcode::Bar`]. Values are the image's own (see [`cubical_barcode`]).
pub type CubicalBar = Bar<Cube, NotNan<f64>, Z2>;

/// Compute the barcode of a cubical complex in dimensions `0..=max_dim`, over Z/2.
///
/// * `method` chooses between reducing the coboundary or boundary matrix.
/// * `algorithm` is the reduction algorithm to use.
/// * If `representatives` is true then every bar comes with a representative cycle.
///
/// Dimensions above `D - 1` have no homology (a cubical complex in `D` dimensions has none there), so `max_dim` is
/// capped at `D - 1`.
///
/// Bars are in the image's own values: for [`Filtration::Superlevel`], a bar is born at a larger value than it dies.
/// Cells with value `+inf` (after negating, for superlevel sets), and NaN pixels, are left out of the complex (see the
/// [module docs](self#missing-and-infinite-values)), so classes that would only die there are essential.
/// Bars are returned in no particular order, and zero-length bars are omitted.
pub fn cubical_barcode<R: ColumnRepresentation, const D: usize>(
    complex: &CubicalComplex<D>,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<CubicalBar> {
    filtration_barcode(complex, max_dim, method, algorithm, representatives, true)
}

/// As [`cubical_barcode`], but for superlevel sets only negates the values back to the image's if
/// `negate_superlevel`: otherwise the bars are those of the sublevel filtration of the negated image.
pub fn filtration_barcode<R: ColumnRepresentation, const D: usize>(
    complex: &CubicalComplex<D>,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    negate_superlevel: bool,
) -> Vec<CubicalBar> {
    if complex.grid().is_none() {
        return vec![];
    }
    let max_dim = max_dim.min(D - 1);
    // Build the basis in parallel only for the parallel algorithm, on whatever rayon pool we're called from
    #[allow(unreachable_patterns)]
    let parallel = matches!(algorithm, Algorithm::ParallelClearing(_));
    // The standard algorithm can't find the owners of omitted apparent pairs, so give it every cell
    let skip_apparent = !matches!(algorithm, Algorithm::Standard(_));
    let coboundary = CubicalCoboundary::<Z2, D>::new(complex);
    let boundary = CubicalBoundary::<Z2, D>::new(complex);
    // Each matrix is given its transpose, so that the reduction can find apparent pairs (the matrices find them
    // themselves, but the involution needs the transpose for representatives)
    let mut bars = match method {
        Method::Cohomology if representatives => {
            // The involution finds representatives of essential classes in dimension 1 by reducing edge columns, which
            // needs the edges that kill components: so reduce dimension 0 too
            let basis = build_bases(complex, max_dim, skip_apparent, parallel);
            let coboundary = coboundary.with_basis(basis);
            let coboundary = (&coboundary).with_transpose(boundary);
            barcode::cohomology_with_involution(coboundary, algorithm)
        }
        Method::Cohomology => {
            // Dimension 0 by union-find, as Ripser and Cubical Ripser do: reducing it is most of the work otherwise.
            // The edges that kill a component are exactly the pivots of the reduced dimension-0 coboundary, which
            // clearing would leave out of dimension 1, so leave them out of the basis.
            // And the top dimension (D - 1) by union-find on the dual, as Cubical Ripser does
            let (mut bars, mut deaths) = connected_components(complex, parallel);
            if D >= 2 && max_dim >= D - 1 {
                bars.extend(voids(complex, parallel));
            }
            if parallel {
                deaths.par_sort_unstable();
            } else {
                deaths.sort_unstable();
            }
            let middle = max_dim.min(D.saturating_sub(2));
            if middle >= 1 {
                let basis = build_bases_from(complex, 1, middle, skip_apparent, parallel, &deaths);
                let coboundary = coboundary.with_basis(basis);
                let coboundary = (&coboundary).with_transpose(boundary);
                bars.extend(barcode::cohomology(coboundary, algorithm));
            }
            bars
        }
        Method::Homology => {
            let basis = build_bases(complex, max_dim + 1, skip_apparent, parallel);
            let boundary = boundary.with_basis(basis);
            let boundary = (&boundary).with_transpose(coboundary);
            let mut bars = barcode::homology(boundary, algorithm, representatives);
            // Nothing can kill the cells of the top dimension of the basis, so they look essential but are not bars
            // we computed
            bars.retain(|bar| complex.dimension(bar.birth) <= max_dim);
            bars
        }
    };
    if negate_superlevel && complex.filtration == Filtration::Superlevel {
        let negate = |value: NotNan<f64>| -value;
        for bar in &mut bars {
            bar.birth_value = negate(bar.birth_value);
            if let Some((_, death)) = &mut bar.death {
                *death = negate(*death);
            }
        }
    }
    bars
}
