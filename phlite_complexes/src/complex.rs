//! An explicit sparse chain complex, and its filtrations, presented to `phlite` as a pair of matrix oracles.
//!
//! There are two stages, so that one chain complex can be given many filtrations cheaply.
//!
//! 1. A [`Chain`] is the boundary matrix in compressed sparse column (CSC) format, with the dimension of every cell.
//!    On construction it is validated (every face has dimension one lower than its coface), and repeated entries in a
//!    column are summed.
//! 2. [`Chain::filter`] attaches a filtration value to every cell, giving a [`Complex`]. This checks that every face
//!    enters before its cofaces, then
//!    * sorts every column in *decreasing* filtration order, so that the pivot of a column is its first entry and its
//!      smallest entry is its last ([`column_pivot`](HasRowFiltration::column_pivot) and
//!      [`column_min`](HasRowFiltration::column_min) are then O(1), and most columns need nothing more);
//!    * builds the transpose (the coboundary matrix), sorted the same way, so that the reduction can recognise
//!      apparent pairs in O(1) via [`first_column_in_row`](HasRowFiltration::first_column_in_row)
//!      and cohomology representatives can be computed by involution.
//!
//! Inside a [`Complex`] cells are renumbered by their position in the filtration, so that the reduction, which visits
//! cells in filtration order, reads the matrices sequentially rather than all over memory
//! (this is what the [`boundary`](Complex::boundary) and [`coboundary`](Complex::coboundary) views index by).
//! A cell's rank then orders cells exactly as (value, index) does, so the views use the rank itself as the filtration
//! value, and the reduction never reads the real values (one random memory access per entry saved, and entries half
//! the size). Bars are translated back to the input's cells and values, dropping those of length zero.
//!
//! Cells are indexed by `u32`, so each matrix costs 4 bytes per non-zero entry
//! (coefficients in [`Z2`](phlite::fields::Z2) are zero-sized), plus 4 bytes per cell (with fewer than 2^32 entries); a
//! [`Complex`] also keeps about 28 bytes per cell to number cells back and look up their values and dimensions.

use std::cmp::Reverse;
use std::ops::Range;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar, Method},
    columns::{ColumnEntry, ColumnRepresentation},
    matrices::{BasisElement, HasRowFiltration, MatrixOracle, RowRanks, SplitBasis},
};

use crate::field::IntField;

/// The largest dimension we accept, so that a stray huge dimension can't make us allocate a huge basis.
pub const MAX_DIMENSION: u32 = u16::MAX as u32;

/// A cell of the complex, given by its index in the input.
/// Ordered by index, which `phlite` uses to break ties between cells with the same filtration value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell(pub u32);

// Cells are numbered 0, ..., n - 1, so the reductions can keep their state in vectors rather than hash maps
impl BasisElement for Cell {
    const DENSE_INDEX: bool = true;

    #[inline]
    fn dense_index(&self) -> usize {
        self.idx()
    }

    #[inline]
    fn from_dense_index(index: usize) -> Self {
        Cell(index as u32)
    }
}

impl Cell {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A chain complex as it arrives from the user, before any validation.
#[derive(Debug, Clone, Default)]
pub struct RawChain {
    /// The dimension of every cell.
    pub dimensions: Vec<i64>,
    /// Column `j` of the boundary matrix is `indices[indptr[j]..indptr[j + 1]]`.
    pub indptr: Vec<i64>,
    /// The row (face) of every non-zero entry.
    pub indices: Vec<i64>,
    /// The coefficient of every entry, or `None` for all ones.
    pub coefficients: Option<Vec<i64>>,
}

/// A sparse matrix in CSC format with no repeated rows in a column.
/// In a [`Chain`] columns are sorted by row index; in a [`Complex`], in decreasing filtration order.
#[derive(Debug, Clone)]
struct Csc<CF> {
    /// `u32`, half the size of `usize`, since the reduction reads it at random (so a complex has fewer than 2^32
    /// entries, checked by [`Chain::from_raw`]).
    indptr: Vec<u32>,
    rows: Vec<Cell>,
    /// Zero-sized (so free) for `Z2`.
    coeffs: Vec<CF>,
}

impl<CF: Copy> Csc<CF> {
    #[inline]
    fn range(&self, col: Cell) -> Range<usize> {
        self.indptr[col.idx()] as usize..self.indptr[col.idx() + 1] as usize
    }
}

/// A filtered chain complex with coefficients in `CF`, ready to be reduced.
///
/// Internally cell `r` is the cell at position `r` in the filtration (its *rank*); the public methods take and give the
/// input's cells, except for the [`boundary`](Self::boundary) and [`coboundary`](Self::coboundary) views.
#[derive(Debug, Clone)]
pub struct Complex<CF> {
    /// The dimension of every input cell.
    dimensions: Vec<u32>,
    /// The input's cells in filtration order (by rank), together, so that translating a bar back reads one place.
    ranked: Vec<Ranked>,
    /// The inverse of `ranked`: the rank of every input cell.
    ranks: Vec<u32>,
    /// The number of cells of each dimension.
    dimension_counts: Vec<usize>,
    /// `tie_start[r]` is the first rank with the same filtration value as rank `r`.
    tie_start: Vec<u32>,
    /// Columns and rows by rank.
    boundary: Csc<CF>,
    coboundary: Csc<CF>,
}

/// The cell at some rank, with its value and dimension.
#[derive(Debug, Clone, Copy)]
struct Ranked {
    value: NotNan<f64>,
    cell: Cell,
    dimension: u32,
}

/// The (co)boundary matrix of a [`Complex`], with an oracle for its transpose.
///
/// Implements [`MatrixOracle`] and [`HasRowFiltration`], with O(1) pivots and apparent pair detection.
pub struct View<'a, CF> {
    matrix: &'a Csc<CF>,
    transpose: &'a Csc<CF>,
    /// The first rank with the same filtration value as each rank, see [`Complex`] (rows are ranks, so their own
    /// ranks).
    tie_start: &'a [u32],
}

impl<CF> Clone for View<'_, CF> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<CF> Copy for View<'_, CF> {}

impl<'a, CF: IntField> View<'a, CF> {
    #[inline]
    fn entry(&self, matrix: &Csc<CF>, i: usize) -> ColumnEntry<u32, Cell, CF> {
        let row = matrix.rows[i];
        ColumnEntry {
            filtration_value: row.0,
            row_index: row,
            coeff: matrix.coeffs[i],
        }
    }
}

impl<'a, CF: IntField> MatrixOracle for View<'a, CF> {
    type CoefficientField = CF;
    type ColT = Cell;
    type RowT = Cell;

    fn column(&self, col: Cell) -> impl Iterator<Item = (CF, Cell)> {
        let matrix: &'a Csc<CF> = self.matrix;
        let range = matrix.range(col);
        matrix.coeffs[range.clone()]
            .iter()
            .copied()
            .zip(matrix.rows[range].iter().copied())
    }
}

impl<'a, CF: IntField> HasRowFiltration for View<'a, CF> {
    /// The rank of the cell, which orders cells as their (value, index) does
    type FiltrationT = u32;

    #[inline]
    fn filtration_value(&self, row: Cell) -> u32 {
        row.0
    }

    // Ranks are finer than the filtration: compare the real values, so that zero-length pairs can be skipped.
    // Ranks with the same value are consecutive, so that is whether the earlier rank is in the later one's run of ties,
    // which reads the later one (in a reduction, the column being reduced, so in order) rather than both.
    #[inline]
    fn same_filtration_value(&self, a: Cell, b: Cell) -> bool {
        let (early, late) = if a <= b { (a, b) } else { (b, a) };
        self.tie_start[late.idx()] <= early.0
    }

    // Columns are sorted in decreasing filtration order without repeats, so the pivot is the first entry
    fn column_pivot(&self, col: Cell) -> Option<ColumnEntry<u32, Cell, CF>> {
        let range = self.matrix.range(col);
        (!range.is_empty()).then(|| self.entry(self.matrix, range.start))
    }

    // ... and the smallest entry is the last
    fn column_min(&self, col: Cell) -> Option<ColumnEntry<u32, Cell, CF>> {
        let range = self.matrix.range(col);
        (!range.is_empty()).then(|| self.entry(self.matrix, range.end - 1))
    }

    // Columns are stored sorted, so the reductions can add them without sorting
    fn sorted_column(&self, col: Cell) -> Option<impl Iterator<Item = ColumnEntry<u32, Cell, CF>>> {
        let view = *self;
        Some(
            self.matrix
                .range(col)
                .map(move |i| view.entry(view.matrix, i)),
        )
    }

    fn sorted_column_ascending(
        &self,
        col: Cell,
    ) -> Option<impl Iterator<Item = ColumnEntry<u32, Cell, CF>>> {
        let view = *self;
        Some(
            self.matrix
                .range(col)
                .rev()
                .map(move |i| view.entry(view.matrix, i)),
        )
    }

    // Column `row` of the transpose lists the columns with an entry in `row`, also in decreasing filtration order
    fn first_column_in_row(&self, row: Cell) -> Option<Cell> {
        let range = self.transpose.range(row);
        (!range.is_empty()).then(|| self.transpose.rows[range.end - 1])
    }

    fn last_column_in_row(&self, row: Cell) -> Option<Cell> {
        let range = self.transpose.range(row);
        (!range.is_empty()).then(|| self.transpose.rows[range.start])
    }

    // Cells are numbered by their rank in the order on entries (value, then index)
    fn row_ranks(&self) -> Option<RowRanks> {
        Some(RowRanks::identity(self.tie_start.len()))
    }
}

/// Why a [`RawChain`] is not a valid chain complex, or a filtration is not valid for it.
///
/// Errors about particular cells record their indices, so that callers can name the cells however they like
/// (see [`describe`](Self::describe)).
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexError {
    /// A problem with the shape of the input, not tied to particular cells.
    Input(String),
    /// A dimension is negative or too large.
    BadDimension { cell: u32, dimension: i64 },
    /// A filtration value is NaN.
    NanValue { cell: u32 },
    /// A cell has no filtration value and no faces to extend one from.
    MissingValue { cell: u32 },
    /// A boundary entry is not a cell.
    FaceOutOfRange {
        cell: u32,
        face: i64,
        n_cells: usize,
    },
    /// A face does not have dimension one lower than its coface.
    WrongDimension {
        cell: u32,
        cell_dimension: u32,
        face: u32,
        face_dimension: u32,
    },
    /// A face does not enter the filtration before its coface.
    FaceAfterCell {
        cell: u32,
        cell_value: f64,
        face: u32,
        face_value: f64,
    },
}

impl ComplexError {
    /// A message describing the error, where `name(i)` describes cell `i` (e.g. `"cell 3"`).
    pub fn describe(&self, name: impl Fn(u32) -> String) -> String {
        match self {
            Self::Input(message) => message.clone(),
            Self::BadDimension { cell, dimension } => format!(
                "The {} has dimension {dimension}, but dimensions must be between 0 and {MAX_DIMENSION}",
                name(*cell)
            ),
            Self::NanValue { cell } => format!("The {} has filtration value NaN", name(*cell)),
            Self::MissingValue { cell } => format!(
                "The {} has no filtration value, and no faces to take one from",
                name(*cell)
            ),
            Self::FaceOutOfRange {
                cell,
                face,
                n_cells,
            } => format!(
                "The boundary of {} contains cell {face}, which is out of range (there are only {n_cells} cells)",
                name(*cell)
            ),
            Self::WrongDimension {
                cell,
                cell_dimension,
                face,
                face_dimension,
            } => format!(
                "The boundary of {} (dimension {cell_dimension}) contains {} of dimension {face_dimension}, \
                 but faces must have dimension one lower",
                name(*cell),
                name(*face)
            ),
            Self::FaceAfterCell {
                cell,
                cell_value,
                face,
                face_value,
            } => format!(
                "The boundary of {} (value {cell_value}) contains {} (value {face_value}), \
                 but faces must enter the filtration first \
                 (with a smaller value, or an equal value and coming earlier)",
                name(*cell),
                name(*face)
            ),
        }
    }
}

impl std::fmt::Display for ComplexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe(|i| format!("cell {i}")))
    }
}

impl std::error::Error for ComplexError {}

/// A chain complex with coefficients in `CF`: cells, their dimensions and boundaries, but no filtration.
#[derive(Debug, Clone)]
pub struct Chain<CF> {
    dimensions: Vec<u32>,
    /// Columns sorted by row index.
    boundary: Csc<CF>,
}

impl<CF: IntField> Chain<CF> {
    /// Validate `raw`, and sum repeated entries in each column.
    pub fn from_raw(raw: RawChain) -> Result<Self, ComplexError> {
        use ComplexError::Input;
        let RawChain {
            dimensions,
            indptr,
            indices,
            coefficients,
        } = raw;
        let n = dimensions.len();
        if n > u32::MAX as usize {
            return Err(Input(format!(
                "At most {} cells are supported, got {n}",
                u32::MAX
            )));
        }
        let dimensions: Vec<u32> = dimensions
            .into_iter()
            .enumerate()
            .map(|(i, d)| match u32::try_from(d) {
                Ok(d) if d <= MAX_DIMENSION => Ok(d),
                _ => Err(ComplexError::BadDimension {
                    cell: i as u32,
                    dimension: d,
                }),
            })
            .collect::<Result<_, _>>()?;

        // ---- The shape of the CSC arrays ----
        if indptr.len() != n + 1 {
            return Err(Input(format!(
                "indptr has length {} but there are {n} cells, so it should have length {}",
                indptr.len(),
                n + 1
            )));
        }
        if indptr[0] != 0 || indptr[n] != indices.len() as i64 {
            return Err(Input(format!(
                "indptr should start at 0 and end at the number of entries ({}), but runs from {} to {}",
                indices.len(),
                indptr[0],
                indptr[n]
            )));
        }
        // Column pointers are kept as u32 (half the memory, which the reduction reads at random)
        if indices.len() >= u32::MAX as usize {
            return Err(Input(format!(
                "There are {} entries, but at most {} are supported",
                indices.len(),
                u32::MAX - 1
            )));
        }
        if let Some(j) = (0..n).find(|&j| indptr[j] > indptr[j + 1]) {
            return Err(Input(format!(
                "indptr must be non-decreasing, but indptr[{j}] > indptr[{}]",
                j + 1
            )));
        }
        if let Some(coefficients) = &coefficients {
            if coefficients.len() != indices.len() {
                return Err(Input(format!(
                    "There are {} coefficients but {} boundary entries",
                    coefficients.len(),
                    indices.len()
                )));
            }
        }

        // ---- Columns: validate, sort by row, sum repeats ----
        let mut out_indptr = Vec::with_capacity(n + 1);
        out_indptr.push(0u32);
        let mut out_rows: Vec<Cell> = Vec::with_capacity(indices.len());
        let mut out_coeffs: Vec<CF> = Vec::with_capacity(indices.len());
        let mut scratch: Vec<(u32, i64)> = Vec::new();
        for col in 0..n {
            scratch.clear();
            for i in indptr[col] as usize..indptr[col + 1] as usize {
                let row = match usize::try_from(indices[i]) {
                    Ok(row) if row < n => row,
                    _ => {
                        return Err(ComplexError::FaceOutOfRange {
                            cell: col as u32,
                            face: indices[i],
                            n_cells: n,
                        })
                    }
                };
                if dimensions[row] + 1 != dimensions[col] {
                    return Err(ComplexError::WrongDimension {
                        cell: col as u32,
                        cell_dimension: dimensions[col],
                        face: row as u32,
                        face_dimension: dimensions[row],
                    });
                }
                let coeff = coefficients.as_ref().map_or(1, |c| c[i]);
                scratch.push((row as u32, coeff));
            }
            scratch.sort_unstable_by_key(|&(row, _)| row);
            let mut k = 0;
            while k < scratch.len() {
                // Sum the coefficients of a run of repeated rows (in i128, so that the sum can't overflow)
                let row = scratch[k].0;
                let mut total: i128 = 0;
                while k < scratch.len() && scratch[k].0 == row {
                    total += scratch[k].1 as i128;
                    k += 1;
                }
                let residue = total.rem_euclid(CF::MODULUS as i128) as i64;
                if let Some(coeff) = CF::from_int(residue) {
                    out_rows.push(Cell(row));
                    out_coeffs.push(coeff);
                }
            }
            out_indptr.push(out_rows.len() as u32);
        }
        out_rows.shrink_to_fit();
        out_coeffs.shrink_to_fit();
        Ok(Self {
            dimensions,
            boundary: Csc {
                indptr: out_indptr,
                rows: out_rows,
                coeffs: out_coeffs,
            },
        })
    }

    /// The number of cells.
    pub fn n_cells(&self) -> usize {
        self.dimensions.len()
    }

    /// The number of non-zero entries in the boundary matrix (after summing repeated entries).
    pub fn n_entries(&self) -> usize {
        self.boundary.rows.len()
    }

    /// The largest dimension of a cell, or `None` if there are no cells.
    pub fn top_dimension(&self) -> Option<usize> {
        self.dimensions.iter().max().map(|&d| d as usize)
    }

    /// Fill in missing filtration values: a cell without one gets the largest value of its faces.
    ///
    /// For example, giving values only to vertices of a simplicial complex produces the lower-star filtration.
    /// Given values are kept as they are (even if smaller than a face's, which [`filter`](Self::filter) will reject).
    pub fn extend(&self, partial: Vec<Option<f64>>) -> Result<Vec<f64>, ComplexError> {
        let n = self.n_cells();
        if partial.len() != n {
            return Err(ComplexError::Input(format!(
                "There are {} values but {n} cells",
                partial.len()
            )));
        }
        // Faces have lower dimension, so visiting cells by dimension sees every face before its cofaces
        let mut by_dimension: Vec<u32> = (0..n as u32).collect();
        by_dimension.sort_by_key(|&c| self.dimensions[c as usize]);
        let mut values = partial;
        for cell in by_dimension {
            if values[cell as usize].is_some() {
                continue;
            }
            let faces = &self.boundary.rows[self.boundary.range(Cell(cell))];
            let max = faces
                .iter()
                .map(|face| values[face.idx()].expect("faces were visited first"))
                .max_by(f64::total_cmp);
            match max {
                Some(max) => values[cell as usize] = Some(max),
                None => return Err(ComplexError::MissingValue { cell }),
            }
        }
        Ok(values.into_iter().map(|v| v.expect("all filled")).collect())
    }

    /// Attach a filtration: a value for every cell, or `None` to use the order of the cells.
    ///
    /// Cells enter in order of value, breaking ties by index; every face must enter before its cofaces.
    pub fn filter(&self, values: Option<Vec<f64>>) -> Result<Complex<CF>, ComplexError> {
        let n = self.n_cells();
        let values: Vec<NotNan<f64>> = match values {
            None => (0..n).map(|i| NotNan::new(i as f64).unwrap()).collect(),
            Some(values) => {
                if values.len() != n {
                    return Err(ComplexError::Input(format!(
                        "There are {} values but {n} cells",
                        values.len()
                    )));
                }
                values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        NotNan::new(v).map_err(|_| ComplexError::NanValue { cell: i as u32 })
                    })
                    .collect::<Result<_, _>>()?
            }
        };
        let key = |cell: usize| (values[cell], cell);
        let mut order: Vec<Cell> = (0..n as u32).map(Cell).collect();
        order.sort_unstable_by_key(|cell| key(cell.idx()));
        let mut ranks = vec![0u32; n];
        for (rank, cell) in order.iter().enumerate() {
            ranks[cell.idx()] = rank as u32;
        }

        // Check the order, and renumber by rank, sorting each column in decreasing filtration (so rank) order
        let source = &self.boundary;
        let mut indptr: Vec<u32> = Vec::with_capacity(n + 1);
        indptr.push(0);
        let mut rows: Vec<Cell> = Vec::with_capacity(source.rows.len());
        let mut coeffs: Vec<CF> = Vec::with_capacity(source.rows.len());
        let mut scratch: Vec<(Reverse<u32>, CF)> = Vec::new();
        for &col in &order {
            let col_rank = ranks[col.idx()];
            scratch.clear();
            for i in source.range(col) {
                let row = source.rows[i];
                let row_rank = ranks[row.idx()];
                if row_rank >= col_rank {
                    return Err(ComplexError::FaceAfterCell {
                        cell: col.0,
                        cell_value: values[col.idx()].into_inner(),
                        face: row.0,
                        face_value: values[row.idx()].into_inner(),
                    });
                }
                scratch.push((Reverse(row_rank), source.coeffs[i]));
            }
            scratch.sort_unstable_by_key(|(r, _)| *r);
            for &(Reverse(row), coeff) in &scratch {
                rows.push(Cell(row));
                coeffs.push(coeff);
            }
            indptr.push(rows.len() as u32);
        }
        let boundary = Csc {
            indptr,
            rows,
            coeffs,
        };
        let coboundary = transpose(&boundary, n);
        let mut dimension_counts = vec![0; self.top_dimension().map_or(0, |top| top + 1)];
        for &dim in &self.dimensions {
            dimension_counts[dim as usize] += 1;
        }
        let ranked: Vec<Ranked> = order
            .iter()
            .map(|&cell| Ranked {
                value: values[cell.idx()],
                cell,
                dimension: self.dimensions[cell.idx()],
            })
            .collect();
        let mut tie_start = Vec::with_capacity(n);
        for (rank, ranked_cell) in ranked.iter().enumerate() {
            let start = match rank {
                0 => 0,
                _ if ranked[rank - 1].value == ranked_cell.value => tie_start[rank - 1],
                _ => rank as u32,
            };
            tie_start.push(start);
        }

        Ok(Complex {
            dimensions: self.dimensions.clone(),
            ranked,
            ranks,
            dimension_counts,
            tie_start,
            boundary,
            coboundary,
        })
    }
}

impl<CF: IntField> Complex<CF> {
    /// The number of cells.
    pub fn n_cells(&self) -> usize {
        self.dimensions.len()
    }

    /// The number of non-zero entries in the boundary matrix (after summing repeated entries).
    pub fn n_entries(&self) -> usize {
        self.boundary.rows.len()
    }

    /// The largest dimension of a cell, or `None` if there are no cells.
    pub fn top_dimension(&self) -> Option<usize> {
        self.dimensions.iter().max().map(|&d| d as usize)
    }

    /// The dimension of `cell`.
    pub fn dimension(&self, cell: Cell) -> usize {
        self.dimensions[cell.idx()] as usize
    }

    /// The filtration value of `cell`.
    pub fn value(&self, cell: Cell) -> f64 {
        self.ranked[self.rank(cell).idx()].value.into_inner()
    }

    /// The position of `cell` in the filtration, which is how the views number it.
    pub fn rank(&self, cell: Cell) -> Cell {
        Cell(self.ranks[cell.idx()])
    }

    /// The input cell at position `rank` in the filtration.
    pub fn cell(&self, rank: Cell) -> Cell {
        self.ranked[rank.idx()].cell
    }

    /// The input cell at position `rank` in the filtration, with its filtration value and dimension.
    pub fn at_rank(&self, rank: Cell) -> (Cell, f64, usize) {
        let ranked = self.ranked[rank.idx()];
        (
            ranked.cell,
            ranked.value.into_inner(),
            ranked.dimension as usize,
        )
    }

    /// The boundary matrix, with cells numbered by [`rank`](Self::rank); its transpose is
    /// [`coboundary`](Self::coboundary).
    pub fn boundary(&self) -> View<'_, CF> {
        View {
            matrix: &self.boundary,
            transpose: &self.coboundary,
            tie_start: &self.tie_start,
        }
    }

    /// The coboundary matrix, with cells numbered by [`rank`](Self::rank); its transpose is [`boundary`](Self::boundary).
    pub fn coboundary(&self) -> View<'_, CF> {
        View {
            matrix: &self.coboundary,
            transpose: &self.boundary,
            tie_start: &self.tie_start,
        }
    }

    /// The cells of dimension at most `max_dim`, by [`rank`](Self::rank), split by dimension and in filtration order.
    pub fn basis(&self, max_dim: usize) -> SplitBasis<Cell> {
        let top = self.top_dimension().map_or(0, |top| top.min(max_dim));
        // Fill each dimension's range at once, from the counts
        let mut offsets = vec![0];
        for dim in 0..=top {
            let count = self.dimension_counts.get(dim).copied().unwrap_or(0);
            offsets.push(offsets[dim] + count);
        }
        let mut next = offsets.clone();
        let mut elems = vec![Cell(0); offsets[top + 1]];
        for (rank, ranked) in self.ranked.iter().enumerate() {
            let dim = ranked.dimension as usize;
            if dim <= top {
                elems[next[dim]] = Cell(rank as u32);
                next[dim] += 1;
            }
        }
        SplitBasis::from_parts(elems, offsets)
    }

    /// The barcode in dimensions `0..=max_dim` (by default, every dimension of the complex).
    ///
    /// Zero-length bars are omitted, and bars are returned in no particular order.
    pub fn barcode<R: ColumnRepresentation>(
        &self,
        max_dim: Option<usize>,
        method: Method,
        algorithm: Algorithm<'_, R>,
        representatives: bool,
    ) -> Vec<Bar<Cell, NotNan<f64>, CF>> {
        let bars = self.barcode_by_rank(max_dim, method, algorithm, representatives);
        // Back to the input's cells and values, dropping bars of length zero
        bars.into_iter()
            .filter_map(|bar| {
                let birth = self.ranked[bar.birth.idx()];
                let death = bar.death.map(|(death, _)| self.ranked[death.idx()]);
                if death.is_some_and(|death| death.value == birth.value) {
                    return None;
                }
                Some(Bar {
                    birth: birth.cell,
                    birth_value: birth.value,
                    death: death.map(|death| (death.cell, death.value)),
                    representative: bar.representative.map(|rep| {
                        rep.into_iter()
                            .map(|(coeff, cell)| (coeff, self.cell(cell)))
                            .collect()
                    }),
                })
            })
            .collect()
    }

    /// As [`barcode`](Self::barcode), but with cells numbered by [`rank`](Self::rank) and their ranks as filtration
    /// values (see [`at_rank`](Self::at_rank) for the input's cells and values), and including bars of length zero.
    /// For translating the bars straight into another form.
    pub fn barcode_by_rank<R: ColumnRepresentation>(
        &self,
        max_dim: Option<usize>,
        method: Method,
        algorithm: Algorithm<'_, R>,
        representatives: bool,
    ) -> Vec<Bar<Cell, u32, CF>> {
        let Some(top) = self.top_dimension() else {
            return vec![];
        };
        let max_dim = max_dim.unwrap_or(top);
        // The basis is every cell up to some dimension, so the pivot table can hold apparent pairs too
        let algorithm = match algorithm {
            Algorithm::Clearing(clearing) => Algorithm::Clearing(clearing.full_basis(true)),
            other => other,
        };
        // Each matrix is given its transpose, so that the reduction can find apparent pairs,
        // and (for cohomology) so that representatives can be computed by involution.
        match method {
            Method::Cohomology => {
                // Columns in dimensions 0..=max_dim, rows up to max_dim + 1
                let coboundary = self.coboundary().with_basis(self.basis(max_dim));
                let coboundary = (&coboundary).with_transpose(self.boundary());
                if representatives {
                    barcode::cohomology_with_involution(coboundary, algorithm)
                } else {
                    barcode::cohomology(coboundary, algorithm)
                }
            }
            Method::Homology => {
                // Classes in dimension max_dim are killed by cells in dimension max_dim + 1
                let boundary = self.boundary().with_basis(self.basis(max_dim + 1));
                let boundary = (&boundary).with_transpose(self.coboundary());
                let mut bars = barcode::homology(boundary, algorithm, representatives);
                bars.retain(|bar| self.ranked[bar.birth.idx()].dimension as usize <= max_dim);
                bars
            }
        }
    }
}

/// The transpose of `matrix`, an `n × n` matrix of cells numbered by rank, with each column in decreasing rank order.
fn transpose<CF: Copy>(matrix: &Csc<CF>, n: usize) -> Csc<CF> {
    let mut indptr = vec![0u32; n + 1];
    for row in &matrix.rows {
        indptr[row.idx() + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let nnz = matrix.rows.len();
    let mut rows = vec![Cell(0); nnz];
    let mut coeffs: Vec<CF> = Vec::with_capacity(nnz);
    // Coefficients are filled in the same (scattered) positions as the rows; for Z2 this is all free
    let mut slots: Vec<Option<CF>> = Vec::new();
    let zero_sized = std::mem::size_of::<CF>() == 0;
    if !zero_sized {
        slots = vec![None; nnz];
    }
    let mut next = indptr.clone();
    // Visit columns in decreasing rank order, so that each column of the transpose comes out sorted that way
    for col in (0..n as u32).rev().map(Cell) {
        for i in matrix.range(col) {
            let row = matrix.rows[i];
            let slot = next[row.idx()] as usize;
            next[row.idx()] += 1;
            rows[slot] = col;
            if !zero_sized {
                slots[slot] = Some(matrix.coeffs[i]);
            }
        }
    }
    if zero_sized {
        // Every value of a zero-sized type is the same, so any coefficient will do
        if let Some(&c) = matrix.coeffs.first() {
            coeffs.resize(nnz, c);
        }
    } else {
        coeffs.extend(slots.into_iter().map(|c| c.expect("every slot is filled")));
    }
    Csc {
        indptr,
        rows,
        coeffs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::fields::{Z2, Z3};
    use phlite::matrices::{ColBasis, HasSplitColBasis};
    use phlite::reduction::{Clearing, Standard};

    /// The boundary of a hollow triangle, filled in last, with a lone extra vertex.
    /// 0,1,2,3: vertices; 4: {0,1}; 5: {0,2}; 6: {1,2}; 7: {0,1,2}
    fn triangle() -> RawChain {
        let columns: Vec<Vec<i64>> = vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![4, 5, 6],
        ];
        let mut indptr = vec![0];
        let mut indices = vec![];
        for col in columns {
            indices.extend(col);
            indptr.push(indices.len() as i64);
        }
        RawChain {
            dimensions: vec![0, 0, 0, 0, 1, 1, 1, 2],
            indptr,
            indices,
            coefficients: None,
        }
    }

    fn build<CF: IntField>(
        raw: RawChain,
        values: Option<Vec<f64>>,
    ) -> Result<Complex<CF>, ComplexError> {
        Chain::<CF>::from_raw(raw)?.filter(values)
    }

    fn summary<CF>(mut bars: Vec<Bar<Cell, NotNan<f64>, CF>>) -> Vec<(u32, Option<u32>)> {
        let mut out: Vec<_> = bars
            .drain(..)
            .map(|bar| (bar.birth.0, bar.death.map(|(d, _)| d.0)))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn triangle_pairs() {
        let complex = build::<Z2>(triangle(), None).unwrap();
        let expected = vec![
            (0, None),
            (1, Some(4)),
            (2, Some(5)),
            (3, None),
            (6, Some(7)),
        ];
        for method in [Method::Cohomology, Method::Homology] {
            for algorithm in [
                Algorithm::Standard(Standard::new()),
                Algorithm::Clearing(Clearing::new()),
                Algorithm::ParallelClearing(Clearing::new().parallel()),
            ] {
                for reps in [false, true] {
                    let bars = complex.barcode(None, method, algorithm.clone(), reps);
                    assert_eq!(summary(bars), expected, "{method:?} {algorithm:?} {reps}");
                }
            }
        }
    }

    #[test]
    fn max_dim_restricts() {
        let complex = build::<Z2>(triangle(), None).unwrap();
        for method in [Method::Cohomology, Method::Homology] {
            let bars = complex.barcode(Some(0), method, Algorithm::default(), false);
            assert_eq!(
                summary(bars),
                vec![(0, None), (1, Some(4)), (2, Some(5)), (3, None)]
            );
        }
    }

    #[test]
    fn zero_length_bars_dropped() {
        // Everything at time 0 except the triangle at time 1
        let values = Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let complex = build::<Z2>(triangle(), values).unwrap();
        let bars = complex.barcode(None, Method::Cohomology, Algorithm::default(), false);
        assert_eq!(summary(bars), vec![(0, None), (3, None), (6, Some(7))]);
    }

    /// `sorted_column` (and its ascending version) give each column in exactly the order the reductions rely on,
    /// including through `reverse()`, which is how cohomology is computed
    #[test]
    fn sorted_columns_are_sorted() {
        fn check<M: HasRowFiltration>(matrix: M, cols: impl Iterator<Item = M::ColT>)
        where
            M::ColT: Clone,
            M::RowT: std::fmt::Debug,
        {
            for col in cols {
                let mut expected: Vec<_> = matrix.column_with_filtration(col.clone()).collect();
                expected.sort_by(|a, b| b.cmp(a));
                let rows = |entries: Vec<ColumnEntry<_, M::RowT, _>>| {
                    entries.into_iter().map(|e| e.row_index).collect::<Vec<_>>()
                };
                let descending: Vec<_> = matrix.sorted_column(col.clone()).unwrap().collect();
                let mut ascending: Vec<_> = matrix.sorted_column_ascending(col).unwrap().collect();
                ascending.reverse();
                assert_eq!(rows(descending), rows(expected.clone()));
                assert_eq!(rows(ascending), rows(expected));
            }
        }
        // Values out of index order, so that filtration order and index order differ
        let values = vec![0.0, 2.0, 1.0, 5.0, 2.0, 3.0, 2.5, 4.0];
        let complex = build::<Z2>(triangle(), Some(values)).unwrap();
        let cells = || (0..8).map(Cell);
        check(complex.boundary(), cells());
        check(complex.coboundary(), cells());
        check(complex.boundary().reverse(), cells().map(Reverse));
        check(complex.coboundary().reverse(), cells().map(Reverse));
    }

    /// `row_ranks` numbers the rows in the order on entries (filtration value, then row), including through
    /// `reverse()`, as `PivotColumns` relies on
    #[test]
    fn row_ranks_follow_entry_order() {
        fn check<M: HasRowFiltration>(matrix: M, rows: impl Iterator<Item = M::RowT>)
        where
            M::RowT: std::fmt::Debug,
        {
            let ranks = matrix.row_ranks().unwrap();
            let mut entries: Vec<_> = rows
                .map(|row| ColumnEntry {
                    filtration_value: matrix.filtration_value(row.clone()),
                    row_index: row,
                    coeff: (),
                })
                .collect();
            entries.sort();
            let by_rank: Vec<_> = entries.iter().map(|e| ranks.rank(&e.row_index)).collect();
            assert_eq!(by_rank, (0..entries.len()).collect::<Vec<_>>());
            assert_eq!(ranks.bound(), entries.len());
        }
        // Values out of index order, with ties broken by index
        let values = vec![0.0, 2.0, 1.0, 5.0, 2.0, 3.0, 2.5, 4.0];
        let complex = build::<Z2>(triangle(), Some(values)).unwrap();
        let cells = || (0..8).map(Cell);
        check(complex.boundary(), cells());
        check(complex.coboundary(), cells());
        check(complex.boundary().reverse(), cells().map(Reverse));
        check(complex.coboundary().reverse(), cells().map(Reverse));
    }

    #[test]
    fn columns_sorted_and_transposed() {
        let complex = build::<Z2>(triangle(), None).unwrap();
        let b = complex.boundary();
        let col: Vec<_> = b.column(Cell(7)).map(|(_, r)| r.0).collect();
        assert_eq!(col, vec![6, 5, 4]);
        assert_eq!(b.column_pivot(Cell(7)).unwrap().row_index, Cell(6));
        assert_eq!(b.column_min(Cell(7)).unwrap().row_index, Cell(4));
        // Cell 0 is in the boundary of 4 and 5; the first is 4
        assert_eq!(b.first_column_in_row(Cell(0)), Some(Cell(4)));
        assert_eq!(b.last_column_in_row(Cell(0)), Some(Cell(5)));
        let cob: Vec<_> = complex
            .coboundary()
            .column(Cell(0))
            .map(|(_, r)| r.0)
            .collect();
        assert_eq!(cob, vec![5, 4]);
    }

    #[test]
    fn columns_follow_the_filtration() {
        // Edge 6 enters first, so is the smallest entry of the triangle's boundary
        let values = Some(vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 1.0, 4.0]);
        let complex = build::<Z2>(triangle(), values).unwrap();
        // The views number cells by rank: 0-3 are the vertices, then edges 6, 4, 5, then the triangle
        assert_eq!([4, 5, 6, 7].map(|c| complex.rank(Cell(c)).0), [5, 6, 4, 7]);
        let col: Vec<_> = complex
            .boundary()
            .column(complex.rank(Cell(7)))
            .map(|(_, r)| complex.cell(r).0)
            .collect();
        assert_eq!(col, vec![5, 4, 6]);
    }

    #[test]
    fn repeated_entries_sum() {
        let mut raw = triangle();
        // Edge 4 = {0, 1}: add vertex 2 twice, which cancels mod 2
        raw.indices.splice(0..0, [2, 2]);
        for p in raw.indptr.iter_mut().skip(5) {
            *p += 2;
        }
        let complex = build::<Z2>(raw.clone(), None).unwrap();
        assert_eq!(complex.n_entries(), 9);
        // But not mod 3, so with coefficients the edge has boundary 2*v2 + v0 + v1, which fails d^2 = 0 but is a valid matrix
        let complex = build::<Z3>(raw, None).unwrap();
        let col: Vec<_> = complex
            .boundary()
            .column(Cell(4))
            .map(|(c, r)| (c.to_int(), r.0))
            .collect();
        assert_eq!(col, vec![(2, 2), (1, 1), (1, 0)]);
    }

    #[test]
    fn coefficients_over_z3() {
        // An oriented triangle: d{01} = v1 - v0, etc, d{012} = {12} - {02} + {01}
        let mut raw = triangle();
        raw.coefficients = Some(vec![-1, 1, -1, 1, -1, 1, 1, -1, 1]);
        let complex = build::<Z3>(raw, None).unwrap();
        let bars = complex.barcode(None, Method::Cohomology, Algorithm::default(), true);
        assert_eq!(
            summary(bars),
            vec![
                (0, None),
                (1, Some(4)),
                (2, Some(5)),
                (3, None),
                (6, Some(7))
            ]
        );
    }

    #[test]
    fn rejects_bad_input() {
        let mut raw = triangle();
        raw.dimensions[4] = 2;
        assert!(build::<Z2>(raw, None)
            .unwrap_err()
            .to_string()
            .contains("dimension one lower"));

        let values = Some(vec![0.0, 0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        assert!(build::<Z2>(triangle(), values)
            .unwrap_err()
            .to_string()
            .contains("enter the filtration first"));

        let mut raw = triangle();
        raw.indices[0] = 99;
        assert!(build::<Z2>(raw, None)
            .unwrap_err()
            .to_string()
            .contains("only 8 cells"));

        let mut values = vec![0.0; 8];
        values[3] = f64::NAN;
        assert!(build::<Z2>(triangle(), Some(values))
            .unwrap_err()
            .to_string()
            .contains("NaN"));
    }

    #[test]
    fn values_need_not_follow_index_order() {
        // Edge 6 enters before edges 4, 5; the filtration is still valid
        let values = Some(vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 1.0, 4.0]);
        let complex = build::<Z2>(triangle(), values).unwrap();
        let bars = complex.barcode(None, Method::Homology, Algorithm::default(), false);
        assert_eq!(
            summary(bars),
            vec![
                (0, None),
                (1, Some(4)),
                (2, Some(6)),
                (3, None),
                (5, Some(7))
            ]
        );
    }

    #[test]
    fn one_chain_many_filtrations() {
        let chain = Chain::<Z2>::from_raw(triangle()).unwrap();
        let a = chain.filter(None).unwrap();
        let b = chain
            .filter(Some(vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 1.0, 4.0]))
            .unwrap();
        let bars = |c: &Complex<Z2>| {
            summary(c.barcode(None, Method::Cohomology, Algorithm::default(), false))
        };
        assert_eq!(bars(&a)[4], (6, Some(7)));
        assert_eq!(bars(&b)[4], (5, Some(7)));
    }

    #[test]
    fn extend_takes_max_of_faces() {
        let chain = Chain::<Z2>::from_raw(triangle()).unwrap();
        // Lower-star: values on vertices only
        let mut partial = vec![None; 8];
        for (v, x) in [3.0, 1.0, 2.0, 0.5].into_iter().enumerate() {
            partial[v] = Some(x);
        }
        // ... and one edge given explicitly
        partial[6] = Some(5.0);
        let values = chain.extend(partial).unwrap();
        assert_eq!(values, vec![3.0, 1.0, 2.0, 0.5, 3.0, 3.0, 5.0, 5.0]);
        assert!(chain.filter(Some(values)).is_ok());

        let mut partial = vec![None; 8];
        partial[0] = Some(0.0);
        assert_eq!(
            chain.extend(partial),
            Err(ComplexError::MissingValue { cell: 1 })
        );
    }

    // ====== Random complexes ===================================================

    /// A small deterministic random generator, so that these tests need no dependencies.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    /// A random simplicial complex (random triangles and edges, closed under faces) with oriented boundaries, values
    /// that grow with faces and have many ties, and its cells listed by dimension, at random within a dimension: so a
    /// cell's index is not its position in the filtration, but faces still come before cofaces of equal value.
    fn random_simplicial(seed: u64) -> (RawChain, Vec<f64>) {
        let mut rng = Lcg(seed);
        let n = 1 + rng.below(9);
        let max_value = [1, 2, 5, 100][rng.below(4)];
        let mut simplices = std::collections::BTreeSet::new();
        for v in 0..n {
            simplices.insert(vec![v]);
        }
        for _ in 0..rng.below(12) {
            let mut s: Vec<usize> = (0..2 + rng.below(2)).map(|_| rng.below(n)).collect();
            s.sort_unstable();
            s.dedup();
            for mask in 1..(1u32 << s.len()) {
                let face: Vec<usize> = (0..s.len())
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| s[i])
                    .collect();
                simplices.insert(face);
            }
        }
        // By dimension, and at random within one
        let mut ordered: Vec<(usize, usize, Vec<usize>)> = simplices
            .into_iter()
            .map(|s| (s.len(), rng.below(1000), s))
            .collect();
        ordered.sort();
        let index: std::collections::HashMap<Vec<usize>, usize> = ordered
            .iter()
            .enumerate()
            .map(|(i, (_, _, s))| (s.clone(), i))
            .collect();
        let (mut indptr, mut indices, mut coefficients, mut dimensions) =
            (vec![0], vec![], vec![], vec![]);
        let mut values: Vec<f64> = vec![];
        for (_, _, s) in &ordered {
            let mut value = rng.below(max_value + 1) as f64;
            if s.len() > 1 {
                for i in 0..s.len() {
                    let mut face = s.clone();
                    face.remove(i);
                    let f = index[&face];
                    indices.push(f as i64);
                    coefficients.push(if i % 2 == 0 { 1 } else { -1 });
                    value = value.max(values[f]);
                }
            }
            values.push(value);
            dimensions.push(s.len() as i64 - 1);
            indptr.push(indices.len() as i64);
        }
        let raw = RawChain {
            dimensions,
            indptr,
            indices,
            coefficients: Some(coefficients),
        };
        (raw, values)
    }

    proptest::proptest! {
        /// The renumbering by rank, the views and the bases on random complexes.
        #[test]
        fn rank_numbering_on_random_complexes(seed in proptest::prelude::any::<u64>()) {
            check_rank_numbering(seed);
        }

        /// Clearing with and without the promise of a full basis, storing R or not, with every column
        /// representation, agrees with the standard algorithm, over Z/2 (with the inline pivot table) and Z/3
        /// (without).
        #[test]
        fn algorithms_agree_on_random_complexes(seed in proptest::prelude::any::<u64>()) {
            algorithms_agree::<Z2>(seed);
            algorithms_agree::<Z3>(seed);
        }
    }

    fn check_rank_numbering(seed: u64) {
        {
            let (raw, values) = random_simplicial(seed);
            let complex = build::<Z3>(raw.clone(), Some(values.clone())).unwrap();
            let n = complex.n_cells();
            let key = |c: usize| (NotNan::new(values[c]).unwrap(), c);
            // `rank` and `cell` are inverse, and ranks follow (value, index)
            for c in 0..n as u32 {
                assert_eq!(complex.cell(complex.rank(Cell(c))), Cell(c));
                let (cell, value, dim) = complex.at_rank(complex.rank(Cell(c)));
                assert_eq!(
                    (cell, value, dim),
                    (
                        Cell(c),
                        values[c as usize],
                        raw.dimensions[c as usize] as usize
                    )
                );
            }
            for r in 1..n as u32 {
                assert!(key(complex.cell(Cell(r - 1)).idx()) < key(complex.cell(Cell(r)).idx()));
            }
            // Columns of the boundary are the faces, by rank, each before its coface, in decreasing order;
            // the coboundary is their transpose
            let (boundary, coboundary) = (complex.boundary(), complex.coboundary());
            let mut entries = vec![];
            for col in (0..n as u32).map(Cell) {
                let rows: Vec<Cell> = boundary.column(col).map(|(_, r)| r).collect();
                assert!(rows.windows(2).all(|w| w[0] > w[1]) && rows.iter().all(|&r| r < col));
                let cell = complex.cell(col).idx();
                let range = raw.indptr[cell] as usize..raw.indptr[cell + 1] as usize;
                let mut faces: Vec<u32> = raw.indices[range].iter().map(|&f| f as u32).collect();
                let mut found: Vec<u32> = rows.iter().map(|&r| complex.cell(r).0).collect();
                faces.sort();
                found.sort();
                assert_eq!(faces, found);
                entries.extend(boundary.column(col).map(|(c, r)| (r, col, c.to_int())));
                let corows: Vec<Cell> = coboundary.column(col).map(|(_, r)| r).collect();
                assert!(corows.windows(2).all(|w| w[0] > w[1]));
            }
            let mut transposed: Vec<_> = (0..n as u32)
                .map(Cell)
                .flat_map(|row| {
                    coboundary
                        .column(row)
                        .map(move |(c, col)| (row, col, c.to_int()))
                })
                .collect();
            entries.sort();
            transposed.sort();
            assert_eq!(entries, transposed);
            // Whether two cells share a value, directly and through `reverse`
            let reversed = boundary.reverse();
            for a in (0..n as u32).map(Cell) {
                for b in (0..n as u32).map(Cell) {
                    let same = complex.at_rank(a).1 == complex.at_rank(b).1;
                    assert_eq!(boundary.same_filtration_value(a, b), same);
                    assert_eq!(reversed.same_filtration_value(Reverse(a), Reverse(b)), same);
                }
            }
            // The basis up to each dimension: the ranks of each dimension, increasing
            let top = complex.top_dimension().unwrap();
            for max_dim in 0..=top + 1 {
                let with_basis = complex.boundary().with_basis(complex.basis(max_dim));
                assert_eq!(with_basis.n_dimensions(), top.min(max_dim) + 1);
                for dim in 0..=top.min(max_dim) {
                    let sub = with_basis.sub_basis(dim);
                    let got: Vec<Cell> = (0..sub.size()).map(|i| sub.element(i)).collect();
                    let want: Vec<Cell> = (0..n as u32)
                        .map(Cell)
                        .filter(|&r| complex.at_rank(r).2 == dim)
                        .collect();
                    assert_eq!(got, want);
                }
            }
        }
    }

    /// The bars of every algorithm, straight from the views (so without `barcode`'s promise of a full basis), as
    /// sorted `(birth, death)` input cells without zero-length bars.
    fn bars_from_views<CF: IntField, R: ColumnRepresentation>(
        complex: &Complex<CF>,
        method: Method,
        algorithm: Algorithm<'_, R>,
    ) -> Vec<(u32, Option<u32>)> {
        let top = complex.top_dimension().unwrap();
        let bars: Vec<Bar<Cell, u32, CF>> = match method {
            Method::Cohomology => {
                let coboundary = complex.coboundary().with_basis(complex.basis(top));
                barcode::cohomology((&coboundary).with_transpose(complex.boundary()), algorithm)
            }
            Method::Homology => {
                let boundary = complex.boundary().with_basis(complex.basis(top));
                barcode::homology(
                    (&boundary).with_transpose(complex.coboundary()),
                    algorithm,
                    false,
                )
            }
        };
        let mut out: Vec<_> = bars
            .into_iter()
            .filter(|bar| {
                bar.death
                    .as_ref()
                    .is_none_or(|(d, _)| complex.at_rank(*d).1 != complex.at_rank(bar.birth).1)
            })
            .map(|bar| {
                (
                    complex.cell(bar.birth).0,
                    bar.death.map(|(d, _)| complex.cell(d).0),
                )
            })
            .collect();
        out.sort();
        out
    }

    fn algorithms_agree<CF: IntField>(seed: u64) {
        use phlite::columns::{HeapColumns, MergedColumns, PivotColumns, SortedColumns};
        {
            let (raw, values) = random_simplicial(seed);
            let complex = build::<CF>(raw, Some(values)).unwrap();
            for method in [Method::Cohomology, Method::Homology] {
                let reference =
                    bars_from_views(&complex, method, Algorithm::Standard(Standard::new()));
                let via_barcode =
                    summary(complex.barcode(None, method, Algorithm::default(), false));
                assert_eq!(via_barcode, reference, "seed {seed}");
                for full_basis in [false, true] {
                    for store_r in [false, true] {
                        let clearing = Clearing::new().full_basis(full_basis).store_r(store_r);
                        let check = |bars: Vec<(u32, Option<u32>)>, columns: &str| {
                            assert_eq!(bars, reference, "seed {seed} {method:?} {columns} full_basis={full_basis} store_r={store_r}");
                        };
                        let c = clearing.clone();
                        check(
                            bars_from_views(
                                &complex,
                                method,
                                Algorithm::Clearing(c.with_columns(PivotColumns)),
                            ),
                            "pivot",
                        );
                        let c = clearing.clone();
                        check(
                            bars_from_views(
                                &complex,
                                method,
                                Algorithm::Clearing(c.with_columns(HeapColumns)),
                            ),
                            "heap",
                        );
                        let c = clearing.clone();
                        check(
                            bars_from_views(
                                &complex,
                                method,
                                Algorithm::Clearing(c.with_columns(SortedColumns)),
                            ),
                            "sorted",
                        );
                        check(
                            bars_from_views(
                                &complex,
                                method,
                                Algorithm::Clearing(clearing.with_columns(MergedColumns)),
                            ),
                            "merged",
                        );
                    }
                }
            }
        }
    }
}
