//! An explicit sparse filtered chain complex, presented to `phlite` as a pair of matrix oracles.
//!
//! The complex is given as a boundary matrix in compressed sparse column (CSC) format, together with the dimension and
//! filtration value of every cell. On construction we
//! * validate it (faces have dimension one lower, and come earlier in the filtration);
//! * sort every column in *decreasing* filtration order and sum repeated entries, so that the pivot of a column is its
//!   first entry and its smallest entry is its last ([`column_pivot`](HasRowFiltration::column_pivot) and
//!   [`column_min`](HasRowFiltration::column_min) are then O(1), and most columns need nothing more);
//! * build the transpose (the coboundary matrix), sorted the same way, so that the reduction can recognise
//!   apparent pairs in O(1) via [`first_column_in_row`](HasRowFiltration::first_column_in_row)
//!   and cohomology representatives can be computed by involution.
//!
//! Cells are indexed by `u32`, so storage costs 4 bytes per non-zero entry in each of the two matrices
//! (coefficients in [`Z2`](phlite::fields::Z2) are zero-sized), plus 16 bytes per cell.

use std::cmp::Reverse;
use std::ops::Range;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar, Method},
    columns::{ColumnEntry, ColumnRepresentation},
    matrices::{BasisElement, HasRowFiltration, MatrixOracle, SplitBasis},
};

use crate::field::IntField;

/// The largest dimension we accept, so that a stray huge dimension can't make us allocate a huge basis.
pub const MAX_DIMENSION: u32 = u16::MAX as u32;

/// A cell of the complex, given by its index in the input.
/// Ordered by index, which `phlite` uses to break ties between cells with the same filtration value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell(pub u32);

impl BasisElement for Cell {}

impl Cell {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

/// Orders cells by decreasing (filtration value, index).
type SortKey = Reverse<(NotNan<f64>, usize)>;

/// The complex as it arrives from the user, before any validation.
#[derive(Debug, Clone, Default)]
pub struct RawComplex {
    /// The dimension of every cell.
    pub dimensions: Vec<i64>,
    /// The filtration value of every cell, or `None` to use the cell indices.
    pub values: Option<Vec<f64>>,
    /// Column `j` of the boundary matrix is `indices[indptr[j]..indptr[j + 1]]`.
    pub indptr: Vec<i64>,
    /// The row (face) of every non-zero entry.
    pub indices: Vec<i64>,
    /// The coefficient of every entry, or `None` for all ones.
    pub coefficients: Option<Vec<i64>>,
}

/// A sparse matrix in CSC format whose columns are sorted in decreasing filtration order, with no repeated rows.
#[derive(Debug, Clone)]
struct Csc<CF> {
    indptr: Vec<usize>,
    rows: Vec<Cell>,
    /// Zero-sized (so free) for `Z2`.
    coeffs: Vec<CF>,
}

impl<CF: Copy> Csc<CF> {
    #[inline]
    fn range(&self, col: Cell) -> Range<usize> {
        self.indptr[col.idx()]..self.indptr[col.idx() + 1]
    }
}

/// A filtered chain complex with coefficients in `CF`, ready to be reduced.
#[derive(Debug, Clone)]
pub struct Complex<CF> {
    dimensions: Vec<u32>,
    values: Vec<NotNan<f64>>,
    /// The cells in filtration order.
    order: Vec<Cell>,
    boundary: Csc<CF>,
    coboundary: Csc<CF>,
}

/// The (co)boundary matrix of a [`Complex`], with an oracle for its transpose.
///
/// Implements [`MatrixOracle`] and [`HasRowFiltration`], with O(1) pivots and apparent pair detection.
pub struct View<'a, CF> {
    matrix: &'a Csc<CF>,
    transpose: &'a Csc<CF>,
    values: &'a [NotNan<f64>],
}

impl<CF> Clone for View<'_, CF> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<CF> Copy for View<'_, CF> {}

impl<'a, CF: IntField> View<'a, CF> {
    #[inline]
    fn entry(&self, matrix: &Csc<CF>, i: usize) -> ColumnEntry<NotNan<f64>, Cell, CF> {
        let row = matrix.rows[i];
        ColumnEntry {
            filtration_value: self.values[row.idx()],
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
    type FiltrationT = NotNan<f64>;

    #[inline]
    fn filtration_value(&self, row: Cell) -> NotNan<f64> {
        self.values[row.idx()]
    }

    // Columns are sorted in decreasing filtration order without repeats, so the pivot is the first entry
    fn column_pivot(&self, col: Cell) -> Option<ColumnEntry<NotNan<f64>, Cell, CF>> {
        let range = self.matrix.range(col);
        (!range.is_empty()).then(|| self.entry(self.matrix, range.start))
    }

    // ... and the smallest entry is the last
    fn column_min(&self, col: Cell) -> Option<ColumnEntry<NotNan<f64>, Cell, CF>> {
        let range = self.matrix.range(col);
        (!range.is_empty()).then(|| self.entry(self.matrix, range.end - 1))
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
}

fn int_to<T: TryFrom<i64>>(value: i64, what: impl FnOnce() -> String) -> Result<T, String> {
    T::try_from(value).map_err(|_| format!("{} is {value}, which is out of range", what()))
}

impl<CF: IntField> Complex<CF> {
    /// Validate `raw` and build the boundary and coboundary matrices.
    pub fn from_raw(raw: RawComplex) -> Result<Self, String> {
        let RawComplex {
            dimensions,
            values,
            indptr,
            indices,
            coefficients,
        } = raw;
        let n = dimensions.len();
        if n > u32::MAX as usize {
            return Err(format!("At most {} cells are supported, got {n}", u32::MAX));
        }

        // ---- Per-cell data ----
        let dimensions: Vec<u32> = dimensions
            .into_iter()
            .enumerate()
            .map(|(i, d)| {
                let d: u32 = int_to(d, || format!("dimensions[{i}]"))?;
                if d > MAX_DIMENSION {
                    return Err(format!(
                        "dimensions[{i}] is {d}, but at most {MAX_DIMENSION} is supported"
                    ));
                }
                Ok(d)
            })
            .collect::<Result<_, String>>()?;
        let values: Vec<NotNan<f64>> = match values {
            None => (0..n).map(|i| NotNan::new(i as f64).unwrap()).collect(),
            Some(values) => {
                if values.len() != n {
                    return Err(format!(
                        "values has length {} but there are {n} cells",
                        values.len()
                    ));
                }
                values
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| NotNan::new(v).map_err(|_| format!("values[{i}] is NaN")))
                    .collect::<Result<_, String>>()?
            }
        };
        let key = |cell: usize| (values[cell], cell);

        // ---- The shape of the CSC arrays ----
        if indptr.len() != n + 1 {
            return Err(format!(
                "indptr has length {} but there are {n} cells, so it should have length {}",
                indptr.len(),
                n + 1
            ));
        }
        if indptr[0] != 0 || indptr[n] != indices.len() as i64 {
            return Err(format!(
                "indptr should start at 0 and end at the number of entries ({}), but runs from {} to {}",
                indices.len(),
                indptr[0],
                indptr[n]
            ));
        }
        if let Some(j) = (0..n).find(|&j| indptr[j] > indptr[j + 1]) {
            return Err(format!(
                "indptr must be non-decreasing, but indptr[{j}] > indptr[{}]",
                j + 1
            ));
        }
        if let Some(coefficients) = &coefficients {
            if coefficients.len() != indices.len() {
                return Err(format!(
                    "There are {} coefficients but {} boundary entries",
                    coefficients.len(),
                    indices.len()
                ));
            }
        }

        // ---- Columns of the boundary: validate, sort decreasing, sum repeats ----
        let mut out_indptr = Vec::with_capacity(n + 1);
        out_indptr.push(0usize);
        let mut out_rows: Vec<Cell> = Vec::with_capacity(indices.len());
        let mut out_coeffs: Vec<CF> = Vec::with_capacity(indices.len());
        // (reversed (value, row) key, coefficient) for each entry in the current column
        let mut scratch: Vec<(SortKey, i64)> = Vec::new();
        for col in 0..n {
            scratch.clear();
            for i in indptr[col] as usize..indptr[col + 1] as usize {
                let row: usize = int_to(indices[i], || format!("Boundary entry {i}"))?;
                if row >= n {
                    return Err(format!(
                        "The boundary of cell {col} contains cell {row}, but there are only {n} cells"
                    ));
                }
                if dimensions[row] + 1 != dimensions[col] {
                    return Err(format!(
                        "The boundary of cell {col} (dimension {}) contains cell {row} of dimension {}, \
                         but faces must have dimension one lower",
                        dimensions[col], dimensions[row]
                    ));
                }
                if key(row) >= key(col) {
                    return Err(format!(
                        "The boundary of cell {col} (value {}) contains cell {row} (value {}), \
                         but faces must enter the filtration first \
                         (with a smaller value, or an equal value and a smaller index)",
                        values[col], values[row]
                    ));
                }
                let coeff = coefficients.as_ref().map_or(1, |c| c[i]);
                scratch.push((Reverse(key(row)), coeff));
            }
            scratch.sort_unstable_by_key(|(k, _)| *k);
            let mut k = 0;
            while k < scratch.len() {
                // Sum the coefficients of a run of repeated rows (in i128, so that the sum can't overflow)
                let (row_key, _) = scratch[k];
                let mut total: i128 = 0;
                while k < scratch.len() && scratch[k].0 == row_key {
                    total += scratch[k].1 as i128;
                    k += 1;
                }
                let residue = total.rem_euclid(CF::MODULUS as i128) as i64;
                if let Some(coeff) = CF::from_int(residue) {
                    out_rows.push(Cell(row_key.0 .1 as u32));
                    out_coeffs.push(coeff);
                }
            }
            out_indptr.push(out_rows.len());
        }
        drop(indices);
        drop(coefficients);
        out_rows.shrink_to_fit();
        out_coeffs.shrink_to_fit();
        let boundary = Csc {
            indptr: out_indptr,
            rows: out_rows,
            coeffs: out_coeffs,
        };

        // ---- The filtration order, and the transpose ----
        let mut order: Vec<Cell> = (0..n as u32).map(Cell).collect();
        order.sort_unstable_by_key(|cell| key(cell.idx()));
        let coboundary = transpose(&boundary, &order, n);

        Ok(Self {
            dimensions,
            values,
            order,
            boundary,
            coboundary,
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

    /// The dimension of `cell`.
    pub fn dimension(&self, cell: Cell) -> usize {
        self.dimensions[cell.idx()] as usize
    }

    /// The filtration value of `cell`.
    pub fn value(&self, cell: Cell) -> f64 {
        self.values[cell.idx()].into_inner()
    }

    /// The boundary matrix, whose transpose is [`coboundary`](Self::coboundary).
    pub fn boundary(&self) -> View<'_, CF> {
        View {
            matrix: &self.boundary,
            transpose: &self.coboundary,
            values: &self.values,
        }
    }

    /// The coboundary matrix, whose transpose is [`boundary`](Self::boundary).
    pub fn coboundary(&self) -> View<'_, CF> {
        View {
            matrix: &self.coboundary,
            transpose: &self.boundary,
            values: &self.values,
        }
    }

    /// The cells of dimension at most `max_dim`, split by dimension and in filtration order.
    pub fn basis(&self, max_dim: usize) -> SplitBasis<Cell> {
        let top = self.top_dimension().map_or(0, |top| top.min(max_dim));
        let mut dims: Vec<Vec<Cell>> = vec![vec![]; top + 1];
        for &cell in &self.order {
            let dim = self.dimension(cell);
            if dim <= max_dim {
                dims[dim].push(cell);
            }
        }
        SplitBasis::from_dimensions(dims)
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
        let Some(top) = self.top_dimension() else {
            return vec![];
        };
        let max_dim = max_dim.unwrap_or(top);
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
                bars.retain(|bar| self.dimension(bar.birth) <= max_dim);
                bars
            }
        }
    }
}

/// The transpose of `matrix`, with each column sorted in decreasing filtration order (`order` lists the cells increasing).
fn transpose<CF: Copy>(matrix: &Csc<CF>, order: &[Cell], n: usize) -> Csc<CF> {
    let mut indptr = vec![0usize; n + 1];
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
    // Visit columns in decreasing filtration order, so that each column of the transpose comes out sorted that way
    for &col in order.iter().rev() {
        for i in matrix.range(col) {
            let row = matrix.rows[i];
            let slot = next[row.idx()];
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
    use phlite::reduction::{Clearing, Standard};

    /// The boundary of a hollow triangle, filled in last, with a lone extra vertex.
    /// 0,1,2,3: vertices; 4: {0,1}; 5: {0,2}; 6: {1,2}; 7: {0,1,2}
    fn triangle(values: Option<Vec<f64>>) -> RawComplex {
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
        RawComplex {
            dimensions: vec![0, 0, 0, 0, 1, 1, 1, 2],
            values,
            indptr,
            indices,
            coefficients: None,
        }
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
        let complex = Complex::<Z2>::from_raw(triangle(None)).unwrap();
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
        let complex = Complex::<Z2>::from_raw(triangle(None)).unwrap();
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
        let complex = Complex::<Z2>::from_raw(triangle(values)).unwrap();
        let bars = complex.barcode(None, Method::Cohomology, Algorithm::default(), false);
        assert_eq!(summary(bars), vec![(0, None), (3, None), (6, Some(7))]);
    }

    #[test]
    fn columns_sorted_and_transposed() {
        let complex = Complex::<Z2>::from_raw(triangle(None)).unwrap();
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
    fn repeated_entries_sum() {
        let mut raw = triangle(None);
        // Edge 4 = {0, 1}: add vertex 2 twice, which cancels mod 2
        raw.indices.splice(0..0, [2, 2]);
        for p in raw.indptr.iter_mut().skip(5) {
            *p += 2;
        }
        let complex = Complex::<Z2>::from_raw(raw.clone()).unwrap();
        assert_eq!(complex.n_entries(), 9);
        // But not mod 3, so with coefficients the edge has boundary 2*v2 + v0 + v1, which fails d^2 = 0 but is a valid matrix
        let complex = Complex::<Z3>::from_raw(raw).unwrap();
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
        let mut raw = triangle(None);
        raw.coefficients = Some(vec![-1, 1, -1, 1, -1, 1, 1, -1, 1]);
        let complex = Complex::<Z3>::from_raw(raw).unwrap();
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
        let mut raw = triangle(None);
        raw.dimensions[4] = 2;
        assert!(Complex::<Z2>::from_raw(raw)
            .unwrap_err()
            .contains("dimension one lower"));

        let raw = triangle(Some(vec![0.0, 0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 2.0]));
        assert!(Complex::<Z2>::from_raw(raw)
            .unwrap_err()
            .contains("enter the filtration first"));

        let mut raw = triangle(None);
        raw.indices[0] = 99;
        assert!(Complex::<Z2>::from_raw(raw)
            .unwrap_err()
            .contains("only 8 cells"));

        let mut raw = triangle(Some(vec![0.0; 8]));
        raw.values.as_mut().unwrap()[3] = f64::NAN;
        assert!(Complex::<Z2>::from_raw(raw).unwrap_err().contains("NaN"));
    }

    #[test]
    fn values_need_not_follow_index_order() {
        // Edge 6 enters before edges 4, 5; the filtration is still valid
        let values = Some(vec![0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 1.0, 4.0]);
        let complex = Complex::<Z2>::from_raw(triangle(values)).unwrap();
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
}
