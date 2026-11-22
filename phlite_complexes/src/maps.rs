//! Six-packs of persistence barcodes ([`phlite::six_pack`]) for explicit complexes: of the inclusion of a subcomplex
//! ([`Complex::six_pack`]), and of a filtration-preserving chain map between two complexes ([`ComplexMap`]).

use ordered_float::NotNan;
use phlite::columns::MergedColumns;
use phlite::matrices::{HasRowFiltration, MatrixOracle};
use phlite::six_pack::{FilteredChainMap, Inclusion, SixPack};

use crate::complex::{Cell, Complex, View};
use crate::field::IntField;

/// The six-pack of a map of explicit complexes, on their input cells, with filtration values.
pub type ComplexSixPack<CF> = SixPack<Cell, Cell, f64, CF>;

/// The cells of `complex` (as ranks) by dimension, in filtration order.
fn cells_by_dimension<CF: IntField>(complex: &Complex<CF>) -> Vec<Vec<Cell>> {
    let mut cells: Vec<Vec<Cell>> = vec![vec![]; complex.top_dimension().map_or(0, |d| d + 1)];
    for r in 0..complex.n_cells() as u32 {
        cells[complex.at_rank(Cell(r)).2].push(Cell(r));
    }
    cells
}

impl<CF: IntField> Complex<CF> {
    /// The six-pack of the inclusion of the subcomplex of the cells `c` with `in_subcomplex[c]` (see
    /// [`phlite::six_pack`]) in dimensions up to `max_dim` (all if `None`), with a representative for every bar if
    /// `representatives`.
    ///
    /// # Errors
    /// If the cells given aren't a subcomplex, with a face of one of them that isn't.
    pub fn six_pack(
        &self,
        in_subcomplex: &[bool],
        representatives: bool,
        max_dim: Option<usize>,
    ) -> Result<ComplexSixPack<CF>, (Cell, Cell)> {
        let n = self.n_cells();
        let by_rank: Vec<bool> = (0..n as u32)
            .map(|r| in_subcomplex[self.cell(Cell(r)).idx()])
            .collect();
        for r in 0..n as u32 {
            if by_rank[r as usize] {
                for (_, face) in self.boundary().column(Cell(r)) {
                    if !by_rank[face.idx()] {
                        return Err((self.cell(Cell(r)), self.cell(face)));
                    }
                }
            }
        }
        let inclusion = Inclusion {
            boundary: self.boundary(),
            cells: cells_by_dimension(self),
            in_subcomplex: |c: &Cell| by_rank[c.idx()],
        };
        let pack = inclusion.six_pack(MergedColumns, representatives, max_dim);
        let cell = |rank: Cell| self.cell(rank);
        Ok(pack.map(|rank: u32| self.at_rank(Cell(rank)).1, cell, cell))
    }
}

/// A filtration value and a rank within one complex: values compare across complexes, ranks break ties within one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueRank(pub NotNan<f64>, pub u32);

impl phlite::matrices::FiltrationValue for ValueRank {}

/// A complex's boundary with rows valued by `(filtration value, rank)`, so that two complexes' values compare.
#[derive(Clone, Copy)]
pub struct Valued<'a, CF> {
    view: View<'a, CF>,
    complex: &'a Complex<CF>,
}

impl<CF: IntField> MatrixOracle for Valued<'_, CF> {
    type CoefficientField = CF;
    type ColT = Cell;
    type RowT = Cell;

    fn column(&self, col: Cell) -> impl Iterator<Item = (CF, Cell)> {
        self.view.column(col)
    }
}

impl<CF: IntField> HasRowFiltration for Valued<'_, CF> {
    type FiltrationT = ValueRank;

    fn filtration_value(&self, row: Cell) -> ValueRank {
        ValueRank(
            NotNan::new(self.complex.at_rank(row).1).expect("filtration values aren't NaN"),
            row.0,
        )
    }
}

/// What can be wrong with a chain map between complexes (cells are input cells).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// The image of `cell` has `image`, of another dimension
    Dimension { cell: Cell, image: Cell },
    /// The image of `cell` has `image`, which enters later
    Later { cell: Cell, image: Cell },
    /// `f ∂ cell != ∂ f cell`
    NotChainMap { cell: Cell },
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dimension { cell, image } => write!(
                f,
                "cell {} maps to cell {} of another dimension",
                cell.0, image.0
            ),
            Self::Later { cell, image } => write!(
                f,
                "cell {} maps to cell {}, which enters later",
                cell.0, image.0
            ),
            Self::NotChainMap { cell } => write!(
                f,
                "the map doesn't commute with the boundary on cell {}",
                cell.0
            ),
        }
    }
}

impl std::error::Error for MapError {}

/// A filtration-preserving chain map `f: L -> K` between explicit complexes over the same field: the image of every
/// cell of `L` (by input cell) as a chain of cells of `K`. A [`FilteredChainMap`], whose six-pack is by its mapping
/// cylinder; [`six_pack`](Self::six_pack) gives it on input cells.
pub struct ComplexMap<'a, CF> {
    domain: &'a Complex<CF>,
    codomain: &'a Complex<CF>,
    /// By rank of `L`, chains of ranks of `K`
    images: Vec<Vec<(CF, Cell)>>,
}

impl<'a, CF: IntField> ComplexMap<'a, CF> {
    /// The map sending each input cell `c` of `domain` to the chain `images[c]` (pairs `(coefficient, input cell)` of
    /// `codomain`; zero coefficients are dropped and repeats summed).
    ///
    /// # Errors
    /// If it doesn't preserve dimensions, makes a cell earlier, or isn't a chain map.
    ///
    /// # Panics
    /// If `images` doesn't have an entry for every cell of `domain`.
    pub fn new(
        domain: &'a Complex<CF>,
        codomain: &'a Complex<CF>,
        images: Vec<Vec<(CF, Cell)>>,
    ) -> Result<Self, MapError> {
        assert_eq!(images.len(), domain.n_cells(), "an image for every cell");
        let mut by_rank = vec![vec![]; domain.n_cells()];
        for (c, image) in images.into_iter().enumerate() {
            let cell = Cell(c as u32);
            for &(_, k) in &image {
                if codomain.dimension(k) != domain.dimension(cell) {
                    return Err(MapError::Dimension { cell, image: k });
                }
                if codomain.value(k) > domain.value(cell) {
                    return Err(MapError::Later { cell, image: k });
                }
            }
            let image = image.into_iter().map(|(x, k)| (x, codomain.rank(k)));
            by_rank[domain.rank(cell).idx()] = phlite::image_kernel::consolidated(image);
        }
        let map = Self {
            domain,
            codomain,
            images: by_rank,
        };
        // f ∂ = ∂ f, cell by cell
        let sorted = |mut chain: Vec<(CF, Cell)>| {
            chain.sort_unstable_by_key(|&(_, c)| c);
            chain
        };
        let (domain_boundary, codomain_boundary) = (domain.boundary(), codomain.boundary());
        for r in 0..domain.n_cells() as u32 {
            let f_boundary = phlite::image_kernel::consolidated(
                domain_boundary
                    .column(Cell(r))
                    .flat_map(|(x, f)| map.images[f.idx()].iter().map(move |&(y, k)| (x * y, k))),
            );
            let boundary_f = phlite::image_kernel::consolidated(
                map.images[r as usize]
                    .iter()
                    .flat_map(|&(x, k)| codomain_boundary.column(k).map(move |(y, j)| (x * y, j))),
            );
            if sorted(f_boundary) != sorted(boundary_f) {
                return Err(MapError::NotChainMap {
                    cell: domain.cell(Cell(r)),
                });
            }
        }
        Ok(map)
    }

    /// The six-pack of the map in dimensions up to `max_dim` (all if `None`), on input cells (see
    /// [`phlite::six_pack`]).
    pub fn six_pack(&self, representatives: bool, max_dim: Option<usize>) -> ComplexSixPack<CF> {
        let pack = FilteredChainMap::six_pack(self, MergedColumns, representatives, max_dim);
        pack.map(
            |ValueRank(value, _)| value.into_inner(),
            |r| self.domain.cell(r),
            |r| self.codomain.cell(r),
        )
    }
}

impl<'a, CF: IntField> FilteredChainMap for ComplexMap<'a, CF> {
    type DomainCell = Cell;
    type CodomainCell = Cell;
    type Field = CF;
    type Value = ValueRank;
    type Domain = Valued<'a, CF>;
    type Codomain = Valued<'a, CF>;

    fn domain(&self) -> Valued<'a, CF> {
        Valued {
            view: self.domain.boundary(),
            complex: self.domain,
        }
    }

    fn codomain(&self) -> Valued<'a, CF> {
        Valued {
            view: self.codomain.boundary(),
            complex: self.codomain,
        }
    }

    fn domain_cells(&self) -> Vec<Vec<Cell>> {
        cells_by_dimension(self.domain)
    }

    fn codomain_cells(&self) -> Vec<Vec<Cell>> {
        cells_by_dimension(self.codomain)
    }

    fn apply(&self, cell: Cell) -> Vec<(CF, Cell)> {
        self.images[cell.idx()].clone()
    }
}
