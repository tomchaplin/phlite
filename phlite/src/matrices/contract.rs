//! Check that a matrix meets phlite's contract (see the book's "The contract" chapter): the requirements every
//! reduction relies on but never checks, and which give wrong barcodes, not errors, when they fail.
//!
//! These checks look at every column, entry and hook, and are at least quadratic in the number of cells: run them in
//! your tests, on small examples of your filtration.
//!
//! ```
//! use phlite::fields::Z2;
//! use phlite::matrices::{bases::SplitBasis, contract, implementors::VecVecMatrix, MatrixOracle};
//!
//! // The boundary of a filled triangle: vertices 0, 1, 2, edges 3 = 01, 4 = 12, 5 = 02, and the triangle 6
//! let columns = vec![
//!     vec![], vec![], vec![],
//!     vec![(Z2, 0), (Z2, 1)], vec![(Z2, 1), (Z2, 2)], vec![(Z2, 0), (Z2, 2)],
//!     vec![(Z2, 3), (Z2, 4), (Z2, 5)],
//! ];
//! let boundary = VecVecMatrix::from(columns)
//!     .with_filtration(|cell: usize| cell) // each cell enters at its own index
//!     .with_basis(SplitBasis::from_dimensions(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]));
//! contract::check_boundary(&boundary).unwrap();
//! contract::check_split_basis(&boundary, contract::Direction::Boundary).unwrap();
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::hash::Hash;

use crate::fields::NonZeroCoefficient;
use crate::matrices::{BasisElement, ColBasis, HasColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle};

/// Whether a matrix is a boundary matrix (each column's rows are its faces, which come first in the filtration order)
/// or a coboundary matrix (its cofaces, which come after).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Boundary,
    Coboundary,
}

/// The first way a matrix was found to break the contract. Cells are named by your cell type `C`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation<C> {
    /// Two calls to `column(column)` gave different columns.
    ColumnChanges { column: C },
    /// The basis lists `cell` twice.
    BasisRepeats { cell: C },
    /// Basis element `index` comes after an earlier one it should come before in the filtration order (filtration
    /// value, then `C`): the one before it in its sub-basis, or an earlier column sharing a row with it.
    BasisOutOfOrder { index: usize },
    /// `row` is an entry of `column` but on the wrong side of it in the filtration order (after it for a boundary
    /// matrix, before it for a coboundary): a cell enters before one of its faces.
    NotAFiltration { column: C, row: C },
    /// The square of the matrix is not zero: its entry in `row` of `column` isn't.
    NotAChainComplex { column: C, row: C },
    /// `row` is an entry of `column`, which is in sub-basis `dimension`, but `row` is not in the sub-basis one below
    /// (boundary) or above (coboundary).
    WrongDimension { column: C, row: C, dimension: usize },
    /// The hook `hook` (a method you implemented) disagrees with the matrix at `cell` (a column, or for
    /// `first_column_in_row`, `last_column_in_row` and `row_ranks`, a row).
    Hook { hook: &'static str, cell: C },
    /// The two matrices are not transposes of each other: the entry in `row` of `column` of the first is not the entry
    /// in `column` of `row` of the second.
    NotTransposes { column: C, row: C },
}

impl<C: Debug> fmt::Display for Violation<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Violation::ColumnChanges { column } => write!(f, "column {column:?} differs between two calls"),
            Violation::BasisRepeats { cell } => write!(f, "the basis lists {cell:?} twice"),
            Violation::BasisOutOfOrder { index } => {
                write!(f, "basis element {index} is out of filtration order")
            }
            Violation::NotAFiltration { column, row } => write!(
                f,
                "row {row:?} of column {column:?} is on the wrong side of it in the filtration order"
            ),
            Violation::NotAChainComplex { column, row } => {
                write!(f, "the matrix squared is non-zero in row {row:?} of column {column:?}")
            }
            Violation::WrongDimension { column, row, dimension } => write!(
                f,
                "row {row:?} of column {column:?} (dimension {dimension}) is not in an adjacent dimension"
            ),
            Violation::Hook { hook, cell } => write!(f, "{hook} disagrees with the matrix at {cell:?}"),
            Violation::NotTransposes { column, row } => write!(
                f,
                "row {row:?} of column {column:?} differs from row {column:?} of column {row:?} of the transpose"
            ),
        }
    }
}

impl<C: Debug> std::error::Error for Violation<C> {}

/// A column with repeated rows summed and zero sums dropped.
fn summed<C, CF, M>(matrix: &M, column: C) -> HashMap<C, CF>
where
    C: BasisElement + Hash,
    CF: NonZeroCoefficient,
    M: MatrixOracle<ColT = C, RowT = C, CoefficientField = CF>,
{
    let mut sums: HashMap<C, CF> = HashMap::new();
    for (coeff, row) in matrix.column(column) {
        let next = match sums.remove(&row) {
            None => Some(coeff),
            Some(current) => coeff + current,
        };
        if let Some(next) = next {
            sums.insert(row, next);
        }
    }
    sums
}

/// The basis, as a vector.
fn basis_of<M: HasColBasis>(matrix: &M) -> Vec<M::ColT> {
    let basis = matrix.basis();
    (0..basis.size()).map(|i| basis.element(i)).collect()
}

/// Check a boundary matrix: requirements 1 to 4 of the contract (well-defined columns, a filtration, `D² = 0`, the
/// basis in filtration order) and every hook it implements.
pub fn check_boundary<C, M>(matrix: &M) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration + HasColBasis,
{
    check(matrix, Direction::Boundary)
}

/// As [`check_boundary`], for a coboundary matrix (each column's rows come after it in the filtration order).
pub fn check_coboundary<C, M>(matrix: &M) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration + HasColBasis,
{
    check(matrix, Direction::Coboundary)
}

fn check<C, M>(matrix: &M, direction: Direction) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration + HasColBasis,
{
    let basis = basis_of(matrix);
    let key = |c: &C| (matrix.filtration_value(c.clone()), c.clone());

    // The basis: no repeats (its order is checked below, for columns that can meet in a reduction)
    let mut seen = HashSet::new();
    for cell in &basis {
        if !seen.insert(cell.clone()) {
            return Err(Violation::BasisRepeats { cell: cell.clone() });
        }
    }

    // Columns: the same each time, and rows on the right side. Rows may be cells outside the basis (a coboundary
    // computed up to some dimension has rows one dimension up)
    let mut columns: HashMap<C, HashMap<C, M::CoefficientField>> = HashMap::new();
    for column in &basis {
        let entries = summed(matrix, column.clone());
        if summed(matrix, column.clone()) != entries {
            return Err(Violation::ColumnChanges { column: column.clone() });
        }
        for row in entries.keys() {
            let ok = match direction {
                Direction::Boundary => key(row) < key(column),
                Direction::Coboundary => key(row) > key(column),
            };
            if !ok {
                return Err(Violation::NotAFiltration { column: column.clone(), row: row.clone() });
            }
        }
        columns.insert(column.clone(), entries);
    }

    // Columns that share a row (so can be added to each other in a reduction) come in increasing filtration order.
    // A basis split by dimension isn't sorted overall (see `check_split_basis` for the order in each dimension)
    let mut last_in_row: HashMap<&C, &C> = HashMap::new();
    for (index, column) in basis.iter().enumerate() {
        for row in columns[column].keys() {
            if let Some(previous) = last_in_row.insert(row, column) {
                if key(previous) >= key(column) {
                    return Err(Violation::BasisOutOfOrder { index });
                }
            }
        }
    }

    // D² = 0 (looking up the columns of rows outside the basis too)
    let mut outside: HashMap<C, HashMap<C, M::CoefficientField>> = HashMap::new();
    for column in &basis {
        let mut square: HashMap<C, M::CoefficientField> = HashMap::new();
        for (row, &a) in &columns[column] {
            let of_row = match columns.get(row) {
                Some(entries) => entries,
                None => outside.entry(row.clone()).or_insert_with(|| summed(matrix, row.clone())),
            };
            for (row2, &b) in of_row {
                let next = match square.remove(row2) {
                    None => Some(b * a),
                    Some(current) => (b * a) + current,
                };
                if let Some(next) = next {
                    square.insert(row2.clone(), next);
                }
            }
        }
        if let Some(row) = square.keys().min() {
            return Err(Violation::NotAChainComplex { column: column.clone(), row: row.clone() });
        }
    }

    check_hooks(matrix, &basis, &columns)
}

/// Every hook of [`HasRowFiltration`] (and the dense index) against the columns themselves, but
/// `same_filtration_value`.
fn check_hooks<C, M>(
    matrix: &M,
    basis: &[C],
    columns: &HashMap<C, HashMap<C, M::CoefficientField>>,
) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration + HasColBasis,
{
    let key = |c: &C| (matrix.filtration_value(c.clone()), c.clone());
    let hook = |hook: &'static str, cell: &C| Violation::Hook { hook, cell: cell.clone() };
    // Each column's entries in increasing filtration order
    let sorted = |column: &C| {
        let mut entries: Vec<(C, M::CoefficientField)> =
            columns[column].iter().map(|(r, &a)| (r.clone(), a)).collect();
        entries.sort_by_key(|(r, _)| key(r));
        entries
    };

    for column in basis {
        let ascending = sorted(column);
        let as_pair = |e: crate::columns::ColumnEntry<M::FiltrationT, C, M::CoefficientField>| (e.row_index, e.coeff);
        if matrix.column_pivot(column.clone()).map(as_pair) != ascending.last().cloned() {
            return Err(hook("column_pivot", column));
        }
        if matrix.column_min(column.clone()).map(as_pair) != ascending.first().cloned() {
            return Err(hook("column_min", column));
        }
        if let Some(entries) = matrix.sorted_column(column.clone()) {
            let got: Vec<_> = entries.map(as_pair).collect();
            if got.iter().rev().cloned().collect::<Vec<_>>() != ascending {
                return Err(hook("sorted_column", column));
            }
        }
        if let Some(entries) = matrix.sorted_column_ascending(column.clone()) {
            if entries.map(as_pair).collect::<Vec<_>>() != ascending {
                return Err(hook("sorted_column_ascending", column));
            }
        }
        // (`same_filtration_value` compares the values of the filtration whose bars are reported, which may be coarser
        // than `FiltrationT`, so can't be checked from the matrix alone)
    }

    // First and last columns in each row (columns are in filtration order in the basis)
    let mut in_row: HashMap<C, Vec<C>> = HashMap::new();
    for column in basis {
        for row in columns[column].keys() {
            in_row.entry(row.clone()).or_default().push(column.clone());
        }
    }
    for row in basis {
        let these = in_row.get(row);
        if let Some(first) = matrix.first_column_in_row(row.clone()) {
            if these.and_then(|c| c.first()) != Some(&first) {
                return Err(hook("first_column_in_row", row));
            }
        }
        if let Some(last) = matrix.last_column_in_row(row.clone()) {
            if these.and_then(|c| c.last()) != Some(&last) {
                return Err(hook("last_column_in_row", row));
            }
        }
    }

    // Row ranks increase along the filtration order
    if let Some(ranks) = matrix.row_ranks() {
        let mut in_order: Vec<C> = basis.to_vec();
        in_order.sort_by_key(|c| key(c));
        for pair in in_order.windows(2) {
            if ranks.rank(&pair[0]) >= ranks.rank(&pair[1]) || ranks.rank(&pair[1]) >= ranks.bound() {
                return Err(hook("row_ranks", &pair[1]));
            }
        }
    }

    // The dense index: injective, and inverted by from_dense_index
    if C::DENSE_INDEX {
        let mut indices = HashSet::new();
        for cell in basis {
            let index = cell.dense_index();
            if !indices.insert(index) || C::from_dense_index(index) != *cell {
                return Err(hook("dense_index", cell));
            }
        }
    }
    Ok(())
}

/// Check a matrix's split basis (requirement 4, for clearing): each sub-basis in increasing filtration order, and each
/// column's rows that are in the basis in the sub-basis one dimension below (`Boundary`) or above (`Coboundary`).
pub fn check_split_basis<C, M>(matrix: &M, direction: Direction) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration + HasSplitColBasis,
{
    let key = |c: &C| (matrix.filtration_value(c.clone()), c.clone());
    let mut dimension_of: HashMap<C, usize> = HashMap::new();
    for dimension in 0..matrix.n_dimensions() {
        let sub = matrix.sub_basis(dimension);
        let cells: Vec<C> = (0..sub.size()).map(|i| sub.element(i)).collect();
        for (index, pair) in cells.windows(2).enumerate() {
            if key(&pair[0]) >= key(&pair[1]) {
                return Err(Violation::BasisOutOfOrder { index: index + 1 });
            }
        }
        for cell in cells {
            if dimension_of.insert(cell.clone(), dimension).is_some() {
                return Err(Violation::BasisRepeats { cell });
            }
        }
    }
    for (column, &dimension) in &dimension_of {
        for row in summed(matrix, column.clone()).keys() {
            let expected = match direction {
                Direction::Boundary => dimension.checked_sub(1),
                Direction::Coboundary => Some(dimension + 1),
            };
            // (A row outside the basis, one dimension above its top, is fine)
            if dimension_of.get(row).is_some_and(|&d| Some(d) != expected) {
                return Err(Violation::WrongDimension { column: column.clone(), row: row.clone(), dimension });
            }
        }
    }
    Ok(())
}

/// Check that `transpose` is the transpose of `matrix` (requirements 6 and 7): every entry in row `r` of column `c` of
/// `matrix` is the entry in row `c` of column `r` of `transpose`, and `transpose` has no others, on the cells of
/// `matrix`'s basis. For a matrix with an attached transpose, call `check_transposes(&matrix, matrix.transpose())`.
pub fn check_transposes<C, CF, M, T>(matrix: &M, transpose: T) -> Result<(), Violation<C>>
where
    C: BasisElement + Hash + Debug,
    CF: NonZeroCoefficient,
    M: MatrixOracle<ColT = C, RowT = C, CoefficientField = CF> + HasColBasis,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = CF>,
{
    let basis = basis_of(matrix);
    let cells: HashSet<C> = basis.iter().cloned().collect();
    let mine: HashMap<C, HashMap<C, CF>> = basis.iter().map(|c| (c.clone(), summed(matrix, c.clone()))).collect();
    // The transpose's columns at every cell of the basis, and at every row of `matrix` outside it
    let mut theirs: HashMap<C, HashMap<C, CF>> =
        basis.iter().map(|c| (c.clone(), summed(&transpose, c.clone()))).collect();
    for entries in mine.values() {
        for row in entries.keys() {
            if !theirs.contains_key(row) {
                theirs.insert(row.clone(), summed(&transpose, row.clone()));
            }
        }
    }
    for (column, entries) in &mine {
        for (row, a) in entries {
            if theirs.get(row).and_then(|t| t.get(column)) != Some(a) {
                return Err(Violation::NotTransposes { column: column.clone(), row: row.clone() });
            }
        }
    }
    for (row, entries) in &theirs {
        for (column, a) in entries {
            if !cells.contains(column) || mine.get(column).and_then(|m| m.get(row)) != Some(a) {
                return Err(Violation::NotTransposes { column: column.clone(), row: row.clone() });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{NonZeroCoefficient, Z2, Z3};
    use crate::matrices::{bases::SplitBasis, implementors::VecVecMatrix};

    /// The filled triangle over Z3, with signs: vertices 0-2, edges 3 = 01, 4 = 12, 5 = 02, triangle 6.
    fn triangle() -> Vec<Vec<(Z3, usize)>> {
        let (one, minus) = (Z3::one(), Z3::one().additive_inverse());
        vec![
            vec![],
            vec![],
            vec![],
            vec![(minus, 0), (one, 1)],
            vec![(minus, 1), (one, 2)],
            vec![(minus, 0), (one, 2)],
            vec![(one, 3), (one, 4), (minus, 5)],
        ]
    }

    fn split() -> SplitBasis<usize> {
        SplitBasis::from_dimensions(vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]])
    }

    #[test]
    fn a_good_boundary_passes() {
        let d = VecVecMatrix::from(triangle()).with_filtration(|c: usize| c).with_basis(split());
        assert_eq!(check_boundary(&d), Ok(()));
        assert_eq!(check_split_basis(&d, Direction::Boundary), Ok(()));
    }

    #[test]
    fn violations_are_caught() {
        // D² ≠ 0: the triangle's boundary with the wrong sign on 5
        let mut cols = triangle();
        cols[6][2].0 = Z3::one();
        let d = VecVecMatrix::from(cols).with_filtration(|c: usize| c).with_basis(split());
        assert_eq!(check_boundary(&d), Err(Violation::NotAChainComplex { column: 6, row: 0 }));

        // Not a filtration: the triangle enters before its edge 5
        let d = VecVecMatrix::from(triangle())
            .with_filtration(|c: usize| if c == 6 { 4 } else { c })
            .with_basis(split());
        assert!(matches!(check_boundary(&d), Err(Violation::BasisOutOfOrder { .. } | Violation::NotAFiltration { .. })));

        // A coboundary checked as a boundary
        let d = VecVecMatrix::from(triangle()).with_filtration(|c: usize| c).with_basis(split());
        assert!(matches!(check_coboundary(&d), Err(Violation::NotAFiltration { .. })));

        // Rows in the wrong dimension: a vertex as an edge
        let bad = SplitBasis::from_dimensions(vec![vec![0, 1], vec![2, 3, 4, 5], vec![6]]);
        let d = VecVecMatrix::from(triangle()).with_filtration(|c: usize| c).with_basis(bad);
        assert!(matches!(check_split_basis(&d, Direction::Boundary), Err(Violation::WrongDimension { .. })));
    }

    /// The triangle, with an implementation of every hook that lies when told to.
    struct Lying {
        columns: Vec<Vec<(Z3, usize)>>,
        lie: &'static str,
    }

    impl MatrixOracle for Lying {
        type CoefficientField = Z3;
        type ColT = usize;
        type RowT = usize;
        fn column(&self, col: usize) -> impl Iterator<Item = (Z3, usize)> {
            self.columns[col].clone().into_iter()
        }
    }

    impl HasRowFiltration for Lying {
        type FiltrationT = usize;
        fn filtration_value(&self, row: usize) -> usize {
            row
        }
        fn first_column_in_row(&self, row: usize) -> Option<usize> {
            // The truth for the triangle's edges is 6; for vertex 0 it is 3
            match (self.lie, row) {
                ("first", 0) => Some(5),
                (_, 0) => Some(3),
                (_, 3..=5) => Some(6),
                _ => None,
            }
        }
        fn column_pivot(&self, col: usize) -> Option<crate::columns::ColumnEntry<usize, usize, Z3>> {
            let mut entries: Vec<_> = self.columns[col].clone();
            entries.sort_by_key(|&(_, r)| r);
            let pick = if self.lie == "pivot" { entries.first() } else { entries.last() };
            pick.map(|&(coeff, r)| (coeff, r, r).into())
        }
        fn sorted_column(
            &self,
            col: usize,
        ) -> Option<impl Iterator<Item = crate::columns::ColumnEntry<usize, usize, Z3>>> {
            let mut entries: Vec<_> = self.columns[col].clone();
            entries.sort_by_key(|&(_, r)| std::cmp::Reverse(r));
            if self.lie == "sorted" {
                entries.reverse();
            }
            Some(entries.into_iter().map(|(coeff, r)| (coeff, r, r).into()))
        }
    }

    #[test]
    fn lying_hooks_are_caught() {
        let check = |lie| check_boundary(&Lying { columns: triangle(), lie }.with_basis(split()));
        assert_eq!(check("none"), Ok(()));
        assert_eq!(check("first"), Err(Violation::Hook { hook: "first_column_in_row", cell: 0 }));
        assert!(matches!(check("pivot"), Err(Violation::Hook { hook: "column_pivot", .. })));
        assert!(matches!(check("sorted"), Err(Violation::Hook { hook: "sorted_column", .. })));
    }

    #[test]
    fn transposes() {
        let cols = triangle();
        let mut rows: Vec<Vec<(Z3, usize)>> = vec![vec![]; 7];
        for (c, col) in cols.iter().enumerate() {
            for &(a, r) in col {
                rows[r].push((a, c));
            }
        }
        let d = VecVecMatrix::from(cols.clone()).with_filtration(|c: usize| c).with_basis(split());
        let t = VecVecMatrix::from(rows.clone());
        assert_eq!(check_transposes(&d, &t), Ok(()));
        rows[0].pop();
        let t = VecVecMatrix::from(rows);
        assert!(matches!(check_transposes(&d, &t), Err(Violation::NotTransposes { .. })));
        // Over Z2, the coboundary of the unsigned boundary is its plain transpose
        let z2: Vec<Vec<(Z2, usize)>> = cols.iter().map(|c| c.iter().map(|&(_, r)| (Z2, r)).collect()).collect();
        let d2 = VecVecMatrix::from(z2).with_filtration(|c: usize| c).with_basis(split());
        assert_eq!(check_boundary(&d2), Ok(()));
    }
}
