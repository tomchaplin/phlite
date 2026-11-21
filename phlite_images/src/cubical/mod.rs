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
//! * By cohomology without representatives, dimension 0 and the top dimension are found by union-find (on the edges
//!   and on the dual), and the T-construction of a volume (three axes) by a filtered discrete gradient instead, as
//!   Perseus does: only its critical cells are reduced (`morse`).

pub mod cohomology;
pub mod grid;
pub mod homology;
mod morse;
#[cfg(test)]
mod tests;

use ordered_float::NotNan;
use phlite::union_find::{IncidenceReduction, GROUND};
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
    /// The pixels again, for computing values: in the T-construction padded with `+inf` all round (so that every cell
    /// has a top cell either side along each axis it doesn't span); in the V-construction, empty (`pixels` is used)
    padded: Vec<f64>,
    /// C-order strides of the image values are computed from (`padded` for T, `pixels` for V)
    value_strides: [usize; D],
    /// For each cell type (bitmask of odd axes): the offsets of the pixels around a cell from its base pixel
    /// `type_shift[mask] + Σ reduced[a] * value_strides[a]`, the first `type_len[mask]` of them
    type_offsets: [[usize; 1 << MAX_AXES]; 1 << MAX_AXES],
    type_len: [u8; 1 << MAX_AXES],
    type_shift: [usize; 1 << MAX_AXES],
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
        // The pixels around each type of cell, at fixed offsets: V, the vertices, a step along each odd axis; T, the
        // top cells, a step along each even axis (padded, the base is the one below, and along odd axes the top cell
        // is the next one)
        let (padded, value_strides) = match construction {
            Construction::V => (vec![], pixel_strides),
            Construction::T => {
                let padded_shape = shape.map(|n| n + 2);
                let mut strides = [0; D];
                let mut stride = 1;
                for a in (0..D).rev() {
                    strides[a] = stride;
                    stride *= padded_shape[a];
                }
                let mut padded = vec![f64::INFINITY; stride];
                for (i, &value) in pixels.iter().enumerate() {
                    let index: usize = (0..D).map(|a| ((i / pixel_strides[a]) % shape[a] + 1) * strides[a]).sum();
                    padded[index] = value;
                }
                (padded, strides)
            }
        };
        let mut type_offsets = [[0usize; 1 << MAX_AXES]; 1 << MAX_AXES];
        let mut type_len = [0u8; 1 << MAX_AXES];
        let mut type_shift = [0usize; 1 << MAX_AXES];
        for mask in 0..1usize << D {
            let odd = |a: usize| mask >> a & 1 == 1;
            let spans = |a: usize| match construction {
                Construction::V => odd(a),
                Construction::T => !odd(a),
            };
            let mut len = 1;
            for a in (0..D).filter(|&a| spans(a)) {
                for i in 0..len {
                    type_offsets[mask][len + i] = type_offsets[mask][i] + value_strides[a];
                }
                len *= 2;
            }
            type_len[mask] = len as u8;
            type_shift[mask] = match construction {
                Construction::V => 0,
                Construction::T => (0..D).filter(|&a| odd(a)).map(|a| value_strides[a]).sum(),
            };
        }
        Ok(Self {
            pixels,
            shape,
            pixel_strides,
            construction,
            filtration,
            grid,
            padded,
            value_strides,
            type_offsets,
            type_len,
            type_shift,
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
        let mask = cell.mask as usize;
        let mut base = self.type_shift[mask];
        for a in 0..D {
            base += cell.reduced[a] as usize * self.value_strides[a];
        }
        let offsets = &self.type_offsets[mask][..self.type_len[mask] as usize];
        match self.construction {
            Construction::V => offsets.iter().map(|&o| self.pixels[base + o]).fold(f64::NEG_INFINITY, f64::max),
            Construction::T => offsets.iter().map(|&o| self.padded[base + o]).fold(f64::INFINITY, f64::min),
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

/// Whether the middle dimensions (`1..=max_dim`, below `D - 1`) may have essential classes, so that the involution
/// can't find their representatives without dimension 0 (it reduces their columns, which needs the edges that kill
/// components). Without excluded cells the complex is a box, with no homology above dimension 0.
fn middle_essentials<const D: usize>(complex: &CubicalComplex<D>, max_dim: usize) -> bool {
    D >= 3 && max_dim >= 1 && complex.pixels.contains(&f64::INFINITY)
}

// ====== Union-find at both ends ===========================================

/// The ranks of the pixels' (internal) values: every cell's value is one of its pixels', so its rank among the distinct
/// values orders cells as their values do, and fits a `u32`. `values[rank]` is the value; `+inf` has the last rank.
struct Ranks {
    /// Each pixel's rank, in C order
    pixels: Vec<u32>,
    /// The distinct values in increasing order, `+inf` last (whether or not a pixel has it)
    values: Vec<f64>,
}

impl Ranks {
    fn new(pixels: &[f64], parallel: bool) -> Self {
        // Sort the pixels by value (as integers that sort like the values, with -0.0 as 0.0), keeping their indices
        let sortable = |(i, v): (usize, &f64)| {
            let bits = (v + 0.0).to_bits();
            let bits = if bits >> 63 == 1 { !bits } else { bits | 1 << 63 };
            (u128::from(bits) << 32) | i as u128
        };
        let mut sorted: Vec<u128> = if parallel {
            pixels.par_iter().enumerate().map(sortable).collect()
        } else {
            pixels.iter().enumerate().map(sortable).collect()
        };
        if parallel {
            sorted.par_sort_unstable();
        } else {
            sorted.sort_unstable();
        }
        let mut ranks = vec![0u32; pixels.len()];
        let mut values: Vec<f64> = vec![];
        for key in sorted {
            let i = key as u32 as usize;
            let value = pixels[i] + 0.0;
            if value == f64::INFINITY {
                break;
            }
            if values.last() != Some(&value) {
                values.push(value);
            }
            ranks[i] = values.len() as u32 - 1;
        }
        values.push(f64::INFINITY);
        let infinite = values.len() as u32 - 1;
        for (rank, value) in ranks.iter_mut().zip(pixels) {
            if *value == f64::INFINITY {
                *rank = infinite;
            }
        }
        Self { pixels: ranks, values }
    }

    fn infinite(&self) -> u32 {
        self.values.len() as u32 - 1
    }
}

/// Along which axis a cell's neighbours (for union-find) lie, and in which grid.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Neighbours {
    /// An edge's two vertices
    Vertices,
    /// A cell of dimension `D - 1`'s one or two top cells (the other side being the outside)
    Tops,
}

/// The cells of dimension `k`, each with its rank and (if asked) its two neighbours, walked in order of number.
struct DimensionCells {
    /// The rank of each cell, indexed from the dimension's first cell
    ranks: Vec<u32>,
    /// The two neighbours of each cell (vertex or top cell indices, `u32::MAX` for the outside), if asked for
    neighbours: Vec<[u32; 2]>,
}

impl<const D: usize> CubicalComplex<D> {
    /// The rank of every cell of dimension `k` (see [`Ranks`]), indexed by its number less the dimension's first, and
    /// with `neighbours`, each cell's two neighbours.
    ///
    /// Cells of a type are numbered in C order, and the pixels around a cell are at fixed offsets from a base pixel
    /// (in the T-construction, of the image padded with `+inf`, so that every cell has a top cell either side along
    /// each axis it doesn't span), so this walks each type once, with no decoding.
    fn dimension_cells(&self, ranks: &Ranks, k: usize, neighbours: Option<Neighbours>) -> DimensionCells {
        let grid = self.grid_ref();
        let vertices = grid.vertices().map(|n| n as usize);
        let n_cells = grid.cells_of_dimension(k).len();
        let mut out = DimensionCells {
            ranks: Vec::with_capacity(n_cells),
            neighbours: Vec::with_capacity(if neighbours.is_some() { n_cells } else { 0 }),
        };
        let c_strides = |extents: [usize; D]| {
            let mut strides = [0; D];
            let mut stride = 1;
            for a in (0..D).rev() {
                strides[a] = stride;
                stride *= extents[a];
            }
            strides
        };
        // The image the offsets index, and its C-order strides
        let padded;
        let (image, strides): (&[u32], [usize; D]) = match self.construction {
            Construction::V => (&ranks.pixels, self.pixel_strides),
            Construction::T => {
                let strides = c_strides(self.shape.map(|n| n + 2));
                let mut image = vec![ranks.infinite(); self.shape.iter().map(|n| n + 2).product()];
                for (i, &rank) in ranks.pixels.iter().enumerate() {
                    let mut padded_index = 0;
                    for a in 0..D {
                        padded_index += ((i / self.pixel_strides[a]) % self.shape[a] + 1) * strides[a];
                    }
                    image[padded_index] = rank;
                }
                padded = image;
                (&padded, strides)
            }
        };
        let vertex_strides = c_strides(vertices);
        let top_strides = c_strides(vertices.map(|n| n.saturating_sub(1)));
        let mut masks: Vec<u32> = (0..1u32 << D).filter(|m| m.count_ones() as usize == k).collect();
        masks.sort_unstable();
        for mask in masks {
            let odd = |a: usize| mask >> a & 1 == 1;
            let extents: [usize; D] = std::array::from_fn(|a| vertices[a] - usize::from(odd(a)));
            if extents.contains(&0) {
                continue;
            }
            // V: the vertices, a step along each odd axis; T: the top cells, a step along each even axis (padded,
            // the base is the one below, and along odd axes the top cell is the next one)
            let spans = |a: usize| match self.construction {
                Construction::V => odd(a),
                Construction::T => !odd(a),
            };
            let mut offsets = vec![0usize];
            for a in (0..D).filter(|&a| spans(a)) {
                let more: Vec<usize> = offsets.iter().map(|o| o + strides[a]).collect();
                offsets.extend(more);
            }
            let mut base: usize = match self.construction {
                Construction::V => 0,
                Construction::T => (0..D).filter(|&a| odd(a)).map(|a| strides[a]).sum(),
            };
            // The neighbours: an edge's along its odd axis, a (D - 1)-cell's along its even axis
            let axis = (0..D).find(|&a| odd(a) == (neighbours == Some(Neighbours::Vertices))).unwrap_or(0);
            let neighbour_strides = match neighbours {
                Some(Neighbours::Vertices) | None => vertex_strides,
                Some(Neighbours::Tops) => top_strides,
            };
            let mut neighbour_base = 0usize;
            let total: usize = extents.iter().product();
            let mut reduced = [0usize; D];
            for _ in 0..total {
                let rank = match self.construction {
                    Construction::V => offsets.iter().map(|&o| image[base + o]).max(),
                    Construction::T => offsets.iter().map(|&o| image[base + o]).min(),
                };
                out.ranks.push(rank.expect("a cell has a pixel"));
                match neighbours {
                    None => {}
                    Some(Neighbours::Vertices) => out
                        .neighbours
                        .push([neighbour_base as u32, (neighbour_base + vertex_strides[axis]) as u32]),
                    Some(Neighbours::Tops) => {
                        let below = (reduced[axis] > 0).then(|| (neighbour_base - top_strides[axis]) as u32);
                        let above = (reduced[axis] + 1 < vertices[axis]).then_some(neighbour_base as u32);
                        out.neighbours.push([below.unwrap_or(u32::MAX), above.unwrap_or(u32::MAX)]);
                    }
                }
                // The next cell in C order
                let mut a = D - 1;
                loop {
                    reduced[a] += 1;
                    base += strides[a];
                    neighbour_base += neighbour_strides[a];
                    if reduced[a] < extents[a] {
                        break;
                    }
                    base -= extents[a] * strides[a];
                    neighbour_base -= extents[a] * neighbour_strides[a];
                    reduced[a] = 0;
                    if a == 0 {
                        break;
                    }
                    a -= 1;
                }
            }
        }
        out
    }
}

/// The indices of the cells with rank below `skip_from`, in increasing order of (rank, index): a least significant
/// digit radix sort on the ranks, 11 bits at a time (so the buckets stay in cache), which is stable, so cells with the
/// same rank stay in order of index.
fn sort_by_rank(ranks: &[u32], n_ranks: usize, skip_from: u32) -> Vec<u32> {
    const BITS: u32 = 11;
    const BUCKETS: usize = 1 << BITS;
    // (rank, index) packed, so that each pass moves one word
    let mut keys: Vec<u64> = ranks
        .iter()
        .enumerate()
        .filter(|&(_, &rank)| rank < skip_from)
        .map(|(i, &rank)| (u64::from(rank) << 32) | i as u64)
        .collect();
    let mut scratch = vec![0u64; keys.len()];
    let rank_bits = usize::BITS - n_ranks.leading_zeros();
    let mut shift = 32;
    while shift < 32 + rank_bits {
        let mut starts = [0usize; BUCKETS + 1];
        for &key in &keys {
            starts[((key >> shift) as usize & (BUCKETS - 1)) + 1] += 1;
        }
        for b in 0..BUCKETS {
            starts[b + 1] += starts[b];
        }
        for &key in &keys {
            let slot = &mut starts[(key >> shift) as usize & (BUCKETS - 1)];
            scratch[*slot] = key;
            *slot += 1;
        }
        std::mem::swap(&mut keys, &mut scratch);
        shift += BITS;
    }
    keys.into_iter().map(|key| key as u32).collect()
}

fn not_nan(value: f64) -> NotNan<f64> {
    NotNan::new(value).expect("values are never NaN")
}

fn one() -> Z2 {
    <Z2 as phlite::fields::NonZeroCoefficient>::one()
}

/// The barcode in dimension 0, by union-find on the edges in filtration order (by value, then number, as the
/// reductions order cells), with the elder rule: when an edge joins two components, the one whose oldest vertex is
/// younger dies. So the pairs are those a reduction would find. Zero-length bars are left out. With `deaths`, also
/// returns the edges that kill a component (zero-length bars' included), sorted.
///
/// With `representatives`, a finite bar's is its birth vertex plus the oldest vertex of the component it joins (born
/// earlier, so the chain is born at the birth, and joined to it at the death and not before); an essential bar's, its
/// birth vertex.
fn connected_components<const D: usize>(
    complex: &CubicalComplex<D>,
    ranks: &Ranks,
    vertex_ranks: &[u32],
    edges: &DimensionCells,
    representatives: bool,
    want_deaths: bool,
) -> (Vec<CubicalBar>, Vec<Cube>) {
    let grid = complex.grid_ref();
    let first_vertex = grid.cells_of_dimension(0).start;
    let first_edge = grid.cells_of_dimension(1).start;
    let value = |rank: u32| not_nan(ranks.values[rank as usize]);
    let order = sort_by_rank(&edges.ranks, ranks.values.len(), ranks.infinite());
    // Vertices by index from 0; the boundary of each edge by phlite's union-find, in which the root of a component is
    // its oldest vertex (by rank, then index) and the younger of two roots is the pivot, so dies
    let mut union_find = IncidenceReduction::<Z2, u32>::with_keys(vertex_ranks.iter().copied());
    let rep = |cells: &[u32]| representatives.then(|| cells.iter().map(|&v| (one(), Cube(first_vertex + v))).collect());
    let mut bars = vec![];
    let mut deaths = Vec::with_capacity(if want_deaths { vertex_ranks.len() } else { 0 });
    for edge in order {
        let [a, b] = edges.neighbours[edge as usize];
        let Some(younger) = union_find.join(a, Some(b)) else {
            continue;
        };
        // (The surviving root only matters for representatives)
        let older = if representatives { union_find.root(younger) } else { younger };
        let cube = Cube(first_edge + edge);
        if want_deaths {
            deaths.push(cube);
        }
        let (birth, death) = (union_find.key(younger), edges.ranks[edge as usize]);
        if birth != death {
            bars.push(Bar {
                birth: Cube(first_vertex + younger),
                birth_value: value(birth),
                death: Some((cube, value(death))),
                representative: rep(&[younger, older]),
            });
        }
    }
    for v in 0..vertex_ranks.len() as u32 {
        if union_find.is_free_root(v) && vertex_ranks[v as usize] != ranks.infinite() {
            bars.push(Bar {
                birth: Cube(first_vertex + v),
                birth_value: value(vertex_ranks[v as usize]),
                death: None,
                representative: rep(&[v]),
            });
        }
    }
    deaths.sort_unstable();
    (bars, deaths)
}

/// The barcode in the top dimension `D - 1`, by union-find on the dual: the cells of dimension `D - 1` in decreasing
/// filtration order (by value, then number), each joining the (one or two) top cells it is a face of, or a top cell and
/// the outside of the grid. Cells not in the complex count as entering at `+inf` (the whole grid is then a box, with no
/// homology in dimension `D - 1`). A component's root is its youngest top cell (the outside is younger than all);
/// when a cell joins two components, the one with the older root dies (in reverse), so that cell is the birth of a
/// class which that root kills. By duality, these are the pairs a reduction would find. Classes killed at `+inf` are
/// essential, classes born there never exist, and zero-length bars are left out.
///
/// With `representatives`, a bar's is the boundary of the top cells of the component that dies (the void): its faces
/// all entered no later than the cell joining it to the rest, which is the youngest of them (the birth), and it is a
/// boundary exactly once all of those top cells have entered, the last being the root (the death).
fn voids<const D: usize>(
    complex: &CubicalComplex<D>,
    ranks: &Ranks,
    faces: &DimensionCells,
    top_ranks: &[u32],
    representatives: bool,
) -> Vec<CubicalBar> {
    let grid = complex.grid_ref();
    if top_ranks.is_empty() {
        // An axis with a single vertex: the complex is a box of lower dimension, with no homology in dimension D - 1
        return vec![];
    }
    let first_top = grid.cells_of_dimension(D).start;
    let first_face = grid.cells_of_dimension(D - 1).start;
    let outside = top_ranks.len() as u32;
    let value = |rank: u32| not_nan(ranks.values[rank as usize]);
    // Top cells by index from 0, by phlite's union-find on the coboundary of the (D - 1)-cells: the root of a component is
    // its youngest top cell (by rank, then index), and the older of two roots is the pivot, so dies. The outside of
    // the grid, younger than everything, is the union-find's ground: a cell with one top coface grounds its component.
    // The union-find's rows are the top cells in reverse (row `last - t` for top cell `t`) with reversed ranks as keys, so
    // that its order on rows, by (key, row), is the reverse of (rank, top cell): the older of two roots is the later.
    let last_top = outside - 1;
    let row = |t: u32| last_top - t;
    let mut union_find = IncidenceReduction::<Z2, std::cmp::Reverse<u32>>::with_keys(
        top_ranks.iter().rev().map(|&rank| std::cmp::Reverse(rank)),
    );
    // Every cell, infinite ones too, in decreasing order
    let order = sort_by_rank(&faces.ranks, ranks.values.len(), u32::MAX);
    // With representatives, each component's top cells, as linked lists: `next[v]` follows `v`, and `last[root]`
    // is the root's list's last
    let (mut next, mut last): (Vec<u32>, Vec<u32>) = if representatives {
        (vec![u32::MAX; outside as usize + 1], (0..=outside).collect())
    } else {
        (vec![], vec![])
    };
    let mut bars = vec![];
    for cell in order.into_iter().rev() {
        // A cell with one top coface is joined to the outside: the ground
        let [a, b] = faces.neighbours[cell as usize];
        let dies = match (a < outside, b < outside) {
            (true, true) => union_find.join(row(a), Some(row(b))),
            (true, false) => union_find.join(row(a), None),
            (false, true) => union_find.join(row(b), None),
            (false, false) => None,
        };
        let Some(dies_row) = dies else {
            continue;
        };
        let dies = row(dies_row);
        let (birth, death) = (faces.ranks[cell as usize], union_find.key(dies_row).0);
        if birth != ranks.infinite() && birth != death {
            bars.push(Bar {
                birth: Cube(first_face + cell),
                birth_value: value(birth),
                death: (death != ranks.infinite()).then(|| (Cube(first_top + dies), value(death))),
                representative: representatives.then(|| void_boundary(complex, &next, dies, first_top, outside)),
            });
        }
        if representatives {
            // Append the dying list to the survivor's (grounded: joined to the outside, whose list collects them)
            let survivor = match union_find.root(dies_row) {
                GROUND => outside,
                root => row(root),
            };
            next[last[survivor as usize] as usize] = dies;
            last[survivor as usize] = last[dies as usize];
        }
    }
    bars
}

/// The boundary (over Z/2) of the top cells in the list starting at `root`, in decreasing filtration order.
fn void_boundary<const D: usize>(
    complex: &CubicalComplex<D>,
    next: &[u32],
    root: u32,
    first_top: u32,
    outside: u32,
) -> Vec<(Z2, Cube)> {
    let grid = complex.grid_ref();
    let mut faces: Vec<u32> = vec![];
    let mut node = root;
    while node != u32::MAX {
        // (The outside is never in a dying component)
        debug_assert!(node != outside);
        faces.extend(grid.faces(&grid.decode(Cube(first_top + node))).into_iter().map(|f| f.cube.0));
        node = next[node as usize];
    }
    // Faces shared by two of the top cells cancel
    faces.sort_unstable();
    let mut boundary: Vec<(NotNan<f64>, u32)> = vec![];
    let mut i = 0;
    while i < faces.len() {
        let mut j = i;
        while j < faces.len() && faces[j] == faces[i] {
            j += 1;
        }
        if (j - i) % 2 == 1 {
            boundary.push((not_nan(complex.internal_value_of(Cube(faces[i]))), faces[i]));
        }
        i = j;
    }
    boundary.sort_unstable_by(|a, b| b.cmp(a));
    boundary
        .into_iter()
        .map(|(_, cube)| (one(), Cube(cube)))
        .collect()
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
    #[allow(unreachable_patterns)]
    let parallel = matches!(algorithm, Algorithm::ParallelClearing(_));
    filtration_barcode(complex, max_dim, method, algorithm, representatives, true, true, parallel)
}

/// As [`cubical_barcode`], but for superlevel sets only negates the values back to the image's if
/// `negate_superlevel` (otherwise the bars are those of the sublevel filtration of the negated image), and choosing
/// whether, by cohomology, to find dimension 0 and the top dimension by union-find (`union_find`, which
/// [`cubical_barcode`] does) rather than by reducing them (and the T-construction with three axes by its Morse complex,
/// see the [module docs](self)), and whether to build the bases and run the union-finds'
/// sorts in parallel on the current rayon thread pool (`parallel_setup`, which [`cubical_barcode`] does only for the
/// parallel algorithm), whatever the algorithm.
pub fn filtration_barcode<R: ColumnRepresentation, const D: usize>(
    complex: &CubicalComplex<D>,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    negate_superlevel: bool,
    union_find: bool,
    parallel_setup: bool,
) -> Vec<CubicalBar> {
    if complex.grid().is_none() {
        return vec![];
    }
    let max_dim = max_dim.min(D - 1);
    let parallel = parallel_setup;
    // The standard algorithm can't find the owners of omitted apparent pairs, so give it every cell
    let skip_apparent = !matches!(algorithm, Algorithm::Standard(_));
    let coboundary = CubicalCoboundary::<Z2, D>::new(complex);
    let boundary = CubicalBoundary::<Z2, D>::new(complex);
    // Each matrix is given its transpose, so that the reduction can find apparent pairs (the matrices find them
    // themselves, but the involution needs the transpose for representatives)
    let mut bars = match method {
        Method::Cohomology if !union_find || (representatives && middle_essentials(complex, max_dim)) => {
            // The involution finds representatives of essential classes in the middle dimensions by reducing their
            // columns, which needs the edges that kill components: so reduce dimension 0 too
            let basis = build_bases(complex, max_dim, skip_apparent, parallel);
            let coboundary = coboundary.with_basis(basis);
            let coboundary = coboundary.by_ref().with_transpose(boundary);
            barcode::cohomology_with_involution(coboundary, algorithm)
        }
        Method::Cohomology
            if D == 3
                && complex.construction == Construction::T
                && !representatives
                && complex.grid_ref().n_cells() as usize <= morse::MAX_MORSE_CELLS =>
        {
            // The T-construction in three dimensions by a filtered discrete gradient: the Morse complex of its
            // critical cells, reduced explicitly (union-find at both ends included)
            let ranks = Ranks::new(&complex.pixels, parallel);
            morse::morse_barcode(
                complex,
                &ranks.pixels,
                &ranks.values,
                ranks.infinite(),
                max_dim,
                algorithm,
                parallel,
            )
        }
        Method::Cohomology => {
            // Dimension 0 by union-find, as Ripser and Cubical Ripser do: reducing it is most of the work otherwise.
            // The edges that kill a component are exactly the pivots of the reduced dimension-0 coboundary, which
            // clearing would leave out of dimension 1, so leave them out of the basis.
            // And the top dimension (D - 1) by union-find on the dual, as Cubical Ripser does
            // (With representatives, by the union-finds and, in between, by the involution: see `middle_essentials`.)
            let ranks = Ranks::new(&complex.pixels, parallel);
            let cells = |k: usize, neighbours| complex.dimension_cells(&ranks, k, neighbours);
            let vertex_ranks = cells(0, None).ranks;
            let edges = cells(1, Some(Neighbours::Vertices));
            let middle = max_dim.min(D.saturating_sub(2));
            let (mut bars, deaths) =
                connected_components(complex, &ranks, &vertex_ranks, &edges, representatives, middle >= 1);
            drop(edges);
            if D >= 2 && max_dim >= D - 1 {
                let faces = cells(D - 1, Some(Neighbours::Tops));
                let top_ranks = cells(D, None).ranks;
                bars.extend(voids(complex, &ranks, &faces, &top_ranks, representatives));
            }
            if middle >= 1 {
                let basis = build_bases_from(complex, 1, middle, skip_apparent, parallel, &deaths);
                let coboundary = coboundary.with_basis(basis);
                let coboundary = coboundary.by_ref().with_transpose(boundary);
                bars.extend(if representatives {
                    barcode::cohomology_with_involution(coboundary, algorithm)
                } else {
                    barcode::cohomology(coboundary, algorithm)
                });
            }
            bars
        }
        Method::Homology => {
            let basis = build_bases(complex, max_dim + 1, skip_apparent, parallel);
            let boundary = boundary.with_basis(basis);
            let boundary = boundary.by_ref().with_transpose(coboundary);
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
