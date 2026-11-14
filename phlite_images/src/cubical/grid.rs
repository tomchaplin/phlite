//! The geometry of a cubical grid: numbering its cells, and stepping between faces and cofaces.
//!
//! A grid with `n_a` vertices along each axis `a` has one cell for every point of the *doubled grid*: coordinates
//! `0 <= c_a <= 2 (n_a - 1)`, where an even coordinate is a vertex position along that axis and an odd one spans the
//! interval between two. A cell's *type* is the set of its odd axes (a bitmask), and its dimension the size of that set.
//! The faces of a cell are one step either way along each odd axis; its cofaces are one step either way along each even
//! axis (staying inside the grid).
//!
//! # Numbering
//!
//! Cells are numbered `0..n_cells` (see [`Cube`]) type by type, in increasing order of dimension and then of the type's
//! bitmask; within a type, by the C-order (row-major) index of the cell's *reduced coordinates* `c_a / 2`. So:
//! * the numbering is dense, so the reductions keep their bookkeeping in vectors (see
//!   [`BasisElement::DENSE_INDEX`]);
//! * every cell of dimension `k` comes before every cell of dimension `k + 1`, so ordering cells by filtration value and
//!   then number puts every face before its cofaces, even when they have the same value (which they often do);
//! * the cofaces of a cell, enumerated along each even axis in increasing order, one step down before one step up,
//!   come out in increasing order of number, and the faces, enumerated along each odd axis in increasing order, one
//!   step up before one step down, come out in decreasing order. This is
//!   what lets the matrices stop early when looking for pivots and apparent pairs.

use phlite::matrices::BasisElement;

/// The largest number of axes a grid can have: cell types are bitmasks over the axes, kept in arrays of this many bits.
pub const MAX_AXES: usize = 4;
/// The number of cell types of a grid with [`MAX_AXES`] axes.
const MAX_TYPES: usize = 1 << MAX_AXES;
/// At most this many faces or cofaces (two per axis).
pub(crate) const MAX_NEIGHBOURS: usize = 2 * MAX_AXES;

/// A cell of a cubical complex, numbered as described in the [module docs](self).
///
/// The number only means something together with its [`CubicalComplex`](super::CubicalComplex), which converts it to
/// coordinates ([`coordinates`](super::CubicalComplex::coordinates)) and back
/// ([`cube_at`](super::CubicalComplex::cube_at)). Cubes are ordered by number, which is how ties between equal
/// filtration values are broken: by dimension first, so faces enter before their cofaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cube(pub(crate) u32);

impl Cube {
    /// The cell's number (see the [module docs](self)).
    pub fn number(self) -> u32 {
        self.0
    }
}

impl BasisElement for Cube {
    const DENSE_INDEX: bool = true;

    #[inline]
    fn dense_index(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn from_dense_index(index: usize) -> Self {
        Cube(index as u32)
    }
}

/// The cells of one type (a set of odd axes): a box of cells, numbered from `offset` in C order.
#[derive(Clone, Copy, Debug)]
struct CellType<const D: usize> {
    /// The number of the first cell of this type
    offset: u32,
    /// The C-order strides of the box (in cell numbers)
    strides: [u32; D],
}

/// A cell, decoded from its number: its type and reduced coordinates (`c_a / 2` for each doubled coordinate `c_a`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Decoded<const D: usize> {
    /// The odd axes, as a bitmask
    pub(crate) mask: u32,
    pub(crate) reduced: [u32; D],
}

impl<const D: usize> Decoded<D> {
    /// Whether axis `axis` is odd (the cell spans an interval along it).
    #[inline]
    pub(crate) fn is_odd(&self, axis: usize) -> bool {
        self.mask >> axis & 1 == 1
    }

    /// The coordinates on the doubled grid.
    pub(crate) fn coordinates(&self) -> [u32; D] {
        std::array::from_fn(|a| 2 * self.reduced[a] + u32::from(self.is_odd(a)))
    }

    /// Whether the boundary coefficient between a cell and its face across `axis` is negative. The boundary of a cell
    /// with odd axes `a_0 < a_1 < ...` is `sum_i (-1)^i (upper face along a_i - lower face along a_i)`.
    ///
    /// `self` is either cell of the pair (removing or adding `axis` doesn't change the odd axes before it), and
    /// `step_up` says whether the face is the upper one.
    #[inline]
    fn negative(&self, axis: usize, step_up: bool) -> bool {
        let before = (self.mask & ((1 << axis) - 1)).count_ones();
        (before % 2 == 1) == step_up
    }
}

/// A face or coface of a cell, with its boundary coefficient's sign.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbour<const D: usize> {
    pub(crate) cube: Cube,
    pub(crate) decoded: Decoded<D>,
    /// Whether the coefficient is `-1` (rather than `+1`)
    pub(crate) negative: bool,
}

/// Up to [`MAX_NEIGHBOURS`] faces or cofaces of a cell, in the order they were found.
#[derive(Clone, Copy)]
pub(crate) struct Neighbours<const D: usize> {
    items: [Option<Neighbour<D>>; MAX_NEIGHBOURS],
    len: usize,
}

impl<const D: usize> Neighbours<D> {
    fn new() -> Self {
        Self {
            items: [None; MAX_NEIGHBOURS],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, neighbour: Neighbour<D>) {
        self.items[self.len] = Some(neighbour);
        self.len += 1;
    }
}

impl<const D: usize> IntoIterator for Neighbours<D> {
    type Item = Neighbour<D>;
    type IntoIter = std::iter::Flatten<
        std::iter::Take<std::array::IntoIter<Option<Neighbour<D>>, MAX_NEIGHBOURS>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().take(self.len).flatten()
    }
}

/// The cells of a grid with `vertices[a]` vertices along each axis `a`, numbered as in the [module docs](self).
#[derive(Clone, Debug)]
pub struct Grid<const D: usize> {
    vertices: [u32; D],
    /// The cell types in numbering order (by dimension, then bitmask); only the first `2^D` are used
    types: [CellType<D>; MAX_TYPES],
    /// `rank[mask]`: the position of the type `mask` in `types`
    rank: [u8; MAX_TYPES],
    /// `masks[i]`: the bitmask of `types[i]`
    masks: [u32; MAX_TYPES],
    /// `dimension_start[k]`: the number of the first cell of dimension `k`, for `0 <= k <= D + 1`
    dimension_start: [u32; MAX_AXES + 2],
}

/// The grid has too many cells to number with a `u32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooManyCells(pub u64);

impl<const D: usize> Grid<D> {
    /// The grid with `vertices[a]` vertices along each axis `a`. Every count must be at least one.
    ///
    /// # Errors
    /// If the grid has `u32::MAX` cells or more (the largest is about 1600 × 1600 × 1600 vertices).
    ///
    /// # Panics
    /// If `D` is zero or more than [`MAX_AXES`], or a count is zero.
    pub fn new(vertices: [u32; D]) -> Result<Self, TooManyCells> {
        assert!(
            (1..=MAX_AXES).contains(&D),
            "grids have between 1 and {MAX_AXES} axes"
        );
        assert!(vertices.iter().all(|&n| n > 0), "every axis needs a vertex");
        let n_types = 1usize << D;
        // The types in numbering order: by dimension (number of odd axes), then by bitmask
        let mut masks: Vec<u32> = (0..n_types as u32).collect();
        masks.sort_by_key(|&mask| (mask.count_ones(), mask));
        let mut types = [CellType {
            offset: 0,
            strides: [0; D],
        }; MAX_TYPES];
        let mut rank = [0u8; MAX_TYPES];
        let mut mask_array = [0u32; MAX_TYPES];
        let mut dimension_start = [0u32; MAX_AXES + 2];
        let mut offset: u64 = 0;
        for (i, &mask) in masks.iter().enumerate() {
            let dimension = mask.count_ones() as usize;
            // The first type of each dimension marks its start (types of a dimension are contiguous)
            if i == 0 || masks[i - 1].count_ones() as usize != dimension {
                dimension_start[dimension] = offset as u32;
            }
            // How many cells of this type there are along each axis: one fewer than vertices along an odd axis
            let extents: [u32; D] =
                std::array::from_fn(|a| vertices[a] - u32::from(mask >> a & 1 == 1));
            let mut strides = [0u32; D];
            let mut stride: u64 = 1;
            for a in (0..D).rev() {
                strides[a] = stride as u32;
                stride *= u64::from(extents[a]);
            }
            types[i] = CellType {
                offset: offset as u32,
                strides,
            };
            rank[mask as usize] = i as u8;
            mask_array[i] = mask;
            // `stride` is now the number of cells of this type
            offset += stride;
            if offset >= u64::from(u32::MAX) {
                let total = vertices
                    .iter()
                    .map(|&n| 2 * u64::from(n) - 1)
                    .product::<u64>();
                return Err(TooManyCells(total));
            }
        }
        dimension_start[D + 1] = offset as u32;
        Ok(Self {
            vertices,
            types,
            rank,
            masks: mask_array,
            dimension_start,
        })
    }

    /// The number of vertices along each axis.
    pub fn vertices(&self) -> [u32; D] {
        self.vertices
    }

    /// The number of cells.
    pub fn n_cells(&self) -> u32 {
        self.dimension_start[D + 1]
    }

    /// The numbers of the cells of dimension `k` (empty if `k > D`).
    pub fn cells_of_dimension(&self, k: usize) -> std::ops::Range<u32> {
        if k > D {
            let n = self.n_cells();
            return n..n;
        }
        self.dimension_start[k]..self.dimension_start[k + 1]
    }

    /// The dimension of `cube`.
    pub fn dimension(&self, cube: Cube) -> usize {
        // The dimensions' cells are contiguous: count the dimensions starting at or before `cube`
        (1..=D)
            .take_while(|&k| self.dimension_start[k] <= cube.0)
            .count()
    }

    /// The type and reduced coordinates of `cube`.
    #[inline]
    pub(crate) fn decode(&self, cube: Cube) -> Decoded<D> {
        // The type is the last whose first cell is at or before `cube` (an empty type shares its offset with the next,
        // so searching from the top finds the right one)
        let mut rank = (1 << D) - 1;
        while self.types[rank].offset > cube.0 {
            rank -= 1;
        }
        let cell_type = &self.types[rank];
        let mut index = cube.0 - cell_type.offset;
        let reduced = std::array::from_fn(|a| {
            let r = index / cell_type.strides[a];
            index -= r * cell_type.strides[a];
            r
        });
        Decoded {
            mask: self.masks[rank],
            reduced,
        }
    }

    /// The number of the cell with type `mask` and reduced coordinates `reduced` (which must be in range).
    #[inline]
    pub(crate) fn encode(&self, mask: u32, reduced: &[u32; D]) -> Cube {
        let cell_type = &self.types[self.rank[mask as usize] as usize];
        let mut number = cell_type.offset;
        for (r, stride) in reduced.iter().zip(&cell_type.strides) {
            number += r * stride;
        }
        Cube(number)
    }

    /// The cell at `coordinates` on the doubled grid, or `None` if they are outside it.
    pub fn cube_at(&self, coordinates: [u32; D]) -> Option<Cube> {
        let mut mask = 0;
        for (a, (&c, &n)) in coordinates.iter().zip(&self.vertices).enumerate() {
            if c > 2 * (n - 1) {
                return None;
            }
            mask |= (c & 1) << a;
        }
        Some(self.encode(mask, &coordinates.map(|c| c / 2)))
    }

    /// The coordinates of `cube` on the doubled grid.
    pub fn coordinates(&self, cube: Cube) -> [u32; D] {
        self.decode(cube).coordinates()
    }

    /// The cofaces of `cell`, in increasing order of number: along each even axis in increasing order, one step down
    /// then one step up (where the grid allows).
    #[inline]
    pub(crate) fn cofaces(&self, cell: &Decoded<D>) -> Neighbours<D> {
        let mut out = Neighbours::new();
        for a in 0..D {
            if cell.is_odd(a) {
                continue;
            }
            let mask = cell.mask | 1 << a;
            let r = cell.reduced[a];
            // Down: the coface spans [r - 1, r] along a, so has reduced coordinate r - 1, and this cell is its upper face
            if r > 0 {
                let mut reduced = cell.reduced;
                reduced[a] = r - 1;
                out.push(Neighbour {
                    cube: self.encode(mask, &reduced),
                    decoded: Decoded { mask, reduced },
                    negative: cell.negative(a, true),
                });
            }
            // Up: the coface spans [r, r + 1], and this cell is its lower face
            if r + 1 < self.vertices[a] {
                out.push(Neighbour {
                    cube: self.encode(mask, &cell.reduced),
                    decoded: Decoded {
                        mask,
                        reduced: cell.reduced,
                    },
                    negative: cell.negative(a, false),
                });
            }
        }
        out
    }

    /// The faces of `cell`, in decreasing order of number: along each odd axis in increasing order, one step up then one
    /// step down.
    #[inline]
    pub(crate) fn faces(&self, cell: &Decoded<D>) -> Neighbours<D> {
        let mut out = Neighbours::new();
        for a in 0..D {
            if !cell.is_odd(a) {
                continue;
            }
            let mask = cell.mask & !(1 << a);
            let mut upper = cell.reduced;
            upper[a] += 1;
            out.push(Neighbour {
                cube: self.encode(mask, &upper),
                decoded: Decoded {
                    mask,
                    reduced: upper,
                },
                negative: cell.negative(a, true),
            });
            out.push(Neighbour {
                cube: self.encode(mask, &cell.reduced),
                decoded: Decoded {
                    mask,
                    reduced: cell.reduced,
                },
                negative: cell.negative(a, false),
            });
        }
        out
    }
}
