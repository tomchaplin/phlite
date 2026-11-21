//! Persistent homology of images, kernels and cokernels: for a filtered complex `K` and a subcomplex `L` (with the
//! restricted filtration), the barcodes of the image, kernel and cokernel of `H(L_t) -> H(K_t)` (Cohen-Steiner,
//! Edelsbrunner, Harer and Morozov, *Persistent homology for kernels, images, and cokernels*, 2009), as phimaker
//! computes them, for any boundary oracle.
//!
//! Like phimaker, it reads the three barcodes off a few reductions (and the relative barcode of `(K, L)` off one more):
//!
//! * `D_f`, the boundary of `K`, and `D_g`, that of `L`: which cells are negative (kill a class), and cycles `z_j`
//!   (with youngest cell `j`) for every positive cell, from the reductions' `V`;
//! * `D_im`: `D_f` with the rows of `L` first. Only the columns negative in `D_f` need reducing (a column reduces to
//!   zero whatever the order of the rows), so the others are skipped, as clearing would;
//! * `D_ker`: the cycles `z_j` of `K` (any basis of cycles compatible with the filtration has the same pivots), rows
//!   of `L` first;
//! * `D_cok`: `D_f` with the columns of cells positive in `L` replaced by `L`'s cycles.
//!
//! Then, with `low_im(j)` the pivot of column `j` of the reduced `D_im`:
//!
//! * **image**: a class is born at each cell of `L` positive in `L`, and dies at a cell `j` negative in `K` whose
//!   `low_im(j)` is in `L`, killing the class born there;
//! * **kernel**: a class is born at each cell `j` not in `L`, negative in `K`, with `low_im(j)` in `L`, and dies at
//!   each cell `j` of `L` negative in `L` but positive in `K`, killing the class born at the pivot of `z_j` in the
//!   reduced `D_ker`;
//! * **cokernel**: a class is born at each cell positive in `K` and not positive in `L`, and dies at each cell `j`
//!   negative in `K` with `low_im(j)` not in `L`, killing the class born at the pivot of column `j` of the reduced
//!   `D_cok`;
//! * **relative**: the barcode of the quotient `K / L`.
//!
//! Over any field. The order on cells, (filtration value, cell), must be a filtration order (faces first).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::columns::{ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow};
use crate::fields::Invertible;
use crate::matrices::{
    BasisElement, FiltrationValue, HasRowFiltration, MatrixOracle, SplitBasis,
};
use crate::reduction::{Clearing, Diagram, ReductionAlgorithm, Standard};

/// A filtration value with the rows of the subcomplex first: `(false, value)` for a cell of `L`, `(true, value)` for
/// one of `K` outside it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubcomplexFirst<F>(pub bool, pub F);

impl<F: FiltrationValue> FiltrationValue for SubcomplexFirst<F> {}

/// The barcodes of the image, kernel and cokernel of `H(L) -> H(K)`, and of `(K, L)` (bars as pairs of cells: `(birth,
/// Some(death))`, or `(birth, None)` if essential), with those of `K` and `L` themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageKernelCokernel<C> {
    pub image: Diagram<C>,
    pub kernel: Diagram<C>,
    pub cokernel: Diagram<C>,
    pub relative: Diagram<C>,
    pub complex: Diagram<C>,
    pub subcomplex: Diagram<C>,
}

/// A boundary oracle restricted to some of its cells (a subcomplex, or the cells of a quotient): its columns and row
/// values, and the shortcuts that only depend on a column (its pivot), but not those that look along a row (the oldest
/// or youngest coface of a cell, or row ranks), which would see cofaces outside the restricted basis.
#[derive(Clone)]
struct Restricted<M>(M);

impl<M: MatrixOracle> MatrixOracle for Restricted<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(&self, col: M::ColT) -> impl Iterator<Item = (M::CoefficientField, M::RowT)> {
        self.0.column(col)
    }
}

impl<M: HasRowFiltration> HasRowFiltration for Restricted<M> {
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: M::RowT) -> M::FiltrationT {
        self.0.filtration_value(row)
    }

    fn same_filtration_value(&self, a: M::RowT, b: M::RowT) -> bool {
        self.0.same_filtration_value(a, b)
    }

    fn column_pivot(
        &self,
        col: M::ColT,
    ) -> Option<crate::columns::ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>> {
        self.0.column_pivot(col)
    }
}

/// A matrix with its columns given by a lookup (a column of another matrix, or a stored chain), for `D_ker` and
/// `D_cok`.
struct Columns<'a, C, CF, F> {
    columns: HashMap<C, Vec<(CF, C)>>,
    filtration: &'a F,
}

impl<C: BasisElement + Hash, CF: crate::fields::NonZeroCoefficient, F> MatrixOracle for Columns<'_, C, CF, F> {
    type CoefficientField = CF;
    type ColT = C;
    type RowT = C;

    fn column(&self, col: C) -> impl Iterator<Item = (CF, C)> {
        self.columns.get(&col).into_iter().flatten().cloned()
    }
}

impl<C: BasisElement + Hash, CF: crate::fields::NonZeroCoefficient, FT: FiltrationValue, F: Fn(C) -> FT>
    HasRowFiltration for Columns<'_, C, CF, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: C) -> FT {
        (self.filtration)(row)
    }
}

/// A column with repeated rows summed (and those that cancel dropped): the columns of `V` for cleared cells are
/// products `D V_d`, sums of boundaries full of repeats.
fn consolidated<C: BasisElement + Hash, CF: crate::fields::NonZeroCoefficient>(
    column: impl Iterator<Item = (CF, C)>,
) -> Vec<(CF, C)> {
    let mut sums: HashMap<C, Option<CF>> = HashMap::new();
    for (coefficient, row) in column {
        let entry = sums.entry(row).or_insert(None);
        *entry = match *entry {
            None => Some(coefficient),
            Some(c) => c + coefficient,
        };
    }
    sums.into_iter().filter_map(|(row, c)| c.map(|c| (c, row))).collect()
}

/// Run both, in parallel on the current rayon thread pool when phlite is built with it.
fn join<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}

/// The pairs `(low, column)` of the reduced matrix, by the standard algorithm over the columns in `basis` (in order).
fn pivots<M, R>(matrix: M, basis: Vec<M::ColT>, columns: R) -> HashMap<M::ColT, M::RowT>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow + Hash + 'static,
    R: ColumnRepresentation,
{
    let reduction = Standard::new().with_columns(columns).reduce(matrix.with_basis(basis));
    reduction.diagram.pairings.into_iter().map(|(low, col)| (col, low)).collect()
}

/// The barcodes of the image, kernel and cokernel of `H(L) -> H(K)` and of `(K, L)` (see the [module docs](self)):
/// `boundary` is the boundary of `K` (rows and columns its cells), `cells` its cells by dimension (each in filtration
/// order), and `in_subcomplex` says which are in `L` (a subcomplex: every face of a cell of `L` is in `L`).
pub fn image_kernel_cokernel<M, R>(
    boundary: M,
    cells: Vec<Vec<M::ColT>>,
    in_subcomplex: impl Fn(&M::ColT) -> bool + Sync,
    columns: R,
) -> ImageKernelCokernel<M::ColT>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration + Clone + Send + Sync,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::ColT: ColumnRow + Hash + Debug + 'static,
    R: ColumnRepresentation,
{
    type Cell<M> = <M as MatrixOracle>::ColT;
    let key = |c: &Cell<M>| (boundary.filtration_value(c.clone()), c.clone());
    let mut all: Vec<Cell<M>> = cells.iter().flatten().cloned().collect();
    all.sort_by_cached_key(key);
    let l_cells: Vec<Vec<Cell<M>>> =
        cells.iter().map(|dim| dim.iter().filter(|c| in_subcomplex(c)).cloned().collect()).collect();
    let quotient_cells: Vec<Vec<Cell<M>>> =
        cells.iter().map(|dim| dim.iter().filter(|c| !in_subcomplex(c)).cloned().collect()).collect();

    // D_f and D_g: positive and negative cells, and cycles for the positive ones
    let full = || Clearing::new().with_columns(columns.clone()).full_basis(true).skip_zero_apparent_pairs(false);
    let (k_basis, l_basis) = (SplitBasis::from_dimensions(cells.clone()), SplitBasis::from_dimensions(l_cells));
    let (k, l) = join(
        || full().reduce(boundary.clone().with_basis(k_basis)),
        || full().reduce(Restricted(boundary.clone()).with_basis(l_basis)),
    );
    let negative_k: HashSet<Cell<M>> = k.diagram.pairings.iter().map(|(_, d)| d.clone()).collect();
    let negative_l: HashSet<Cell<M>> = l.diagram.pairings.iter().map(|(_, d)| d.clone()).collect();
    let positive_l = |c: &Cell<M>| in_subcomplex(c) && !negative_l.contains(c);
    let subcomplex_first = |row: Cell<M>| SubcomplexFirst(!in_subcomplex(&row), boundary.filtration_value(row));

    // D_im, on the columns negative in K
    let negatives: Vec<Cell<M>> = all.iter().filter(|c| negative_k.contains(c)).cloned().collect();
    let image_matrix = || pivots(boundary.clone().with_filtration(&subcomplex_first), negatives.clone(), columns.clone());

    // D_ker: the cycles of K (columns of V for the positive cells), rows of L first. Only the pivots of kernel deaths
    // (cells of L negative in L, positive in K) are read, and a column only meets those of its own dimension, so only
    // the positive cells of dimensions with a kernel death, up to the last one, are reduced
    let dimension_of: HashMap<Cell<M>, usize> =
        cells.iter().enumerate().flat_map(|(d, dim)| dim.iter().map(move |c| (c.clone(), d))).collect();
    let mut last_death: HashMap<usize, usize> = HashMap::new();
    for (i, c) in all.iter().enumerate() {
        if in_subcomplex(c) && negative_l.contains(c) && !negative_k.contains(c) {
            last_death.insert(dimension_of[c], i);
        }
    }
    let positives: Vec<Cell<M>> = all
        .iter()
        .enumerate()
        .filter(|(i, c)| !negative_k.contains(*c) && last_death.get(&dimension_of[*c]).is_some_and(|&last| *i <= last))
        .map(|(_, c)| c.clone())
        .collect();
    let kernel_matrix = || {
        let kernel_columns = Columns {
            columns: positives.iter().map(|c| (c.clone(), consolidated(k.v.column(c.clone())))).collect(),
            filtration: &subcomplex_first,
        };
        pivots(kernel_columns, positives.clone(), columns.clone())
    };

    // D_cok: D_f with the cells positive in L replaced by L's cycles. A column positive in K and not positive in L is
    // its boundary, in the span of the earlier boundaries, which the replaced ones (positive in L) are in the span of
    // L's earlier columns: so it reduces to zero, and is left out
    let row_value = |row: Cell<M>| boundary.filtration_value(row);
    let cokernel_matrix = || {
        let needed: Vec<Cell<M>> =
            all.iter().filter(|c| negative_k.contains(c) || positive_l(c)).cloned().collect();
        let cokernel_columns = Columns {
            columns: needed
                .iter()
                .map(|c| {
                    let column: Vec<_> = if positive_l(c) {
                        consolidated(l.v.column(c.clone()))
                    } else {
                        boundary.column(c.clone()).collect()
                    };
                    (c.clone(), column)
                })
                .collect(),
            filtration: &row_value,
        };
        pivots(cokernel_columns, needed, columns.clone())
    };

    // (K, L): the quotient's boundary, dropping faces in L, reduced by cohomology (its transpose, with clearing)
    let relative_matrix = || {
        let mut coboundary: HashMap<Cell<M>, Vec<(M::CoefficientField, Cell<M>)>> = HashMap::new();
        for c in all.iter().filter(|c| !in_subcomplex(c)) {
            coboundary.entry(c.clone()).or_default();
            for (x, f) in boundary.column(c.clone()) {
                if !in_subcomplex(&f) {
                    coboundary.entry(f).or_default().push((x, c.clone()));
                }
            }
        }
        let quotient = Columns { columns: coboundary, filtration: &row_value }
            .with_basis(SplitBasis::from_dimensions(quotient_cells));
        let algorithm = crate::barcode::Algorithm::Clearing(Clearing::new().with_columns(columns.clone()));
        let bars = crate::barcode::cohomology(&quotient, algorithm);
        let mut diagram = Diagram { essential: vec![], pairings: vec![] };
        for bar in bars {
            match bar.death {
                Some((death, _)) => diagram.pairings.push((bar.birth, death)),
                None => diagram.essential.push(bar.birth),
            }
        }
        diagram
    };
    let ((low_im, low_ker), (low_cok, relative)) =
        join(|| join(image_matrix, kernel_matrix), || join(cokernel_matrix, relative_matrix));

    // The barcodes
    let mut image = Diagram { essential: vec![], pairings: vec![] };
    let mut kernel = Diagram { essential: vec![], pairings: vec![] };
    let mut cokernel = Diagram { essential: vec![], pairings: vec![] };
    let (mut image_dies, mut kernel_dies, mut cokernel_dies) = (HashSet::new(), HashSet::new(), HashSet::new());
    for j in &all {
        if negative_k.contains(j) {
            let low = low_im.get(j).expect("a negative column has a pivot, whatever the row order");
            if in_subcomplex(low) {
                image.pairings.push((low.clone(), j.clone()));
                image_dies.insert(low.clone());
            } else {
                let birth = low_cok.get(j).expect("a cokernel death has a pivot").clone();
                cokernel.pairings.push((birth.clone(), j.clone()));
                cokernel_dies.insert(birth);
            }
        }
        if in_subcomplex(j) && negative_l.contains(j) && !negative_k.contains(j) {
            let birth = low_ker.get(j).expect("a kernel death has a pivot").clone();
            kernel.pairings.push((birth.clone(), j.clone()));
            kernel_dies.insert(birth);
        }
    }
    for j in &all {
        let negative = negative_k.contains(j);
        if positive_l(j) && !image_dies.contains(j) {
            image.essential.push(j.clone());
        }
        if !in_subcomplex(j) && negative && low_im.get(j).is_some_and(&in_subcomplex) && !kernel_dies.contains(j) {
            kernel.essential.push(j.clone());
        }
        if !negative && !positive_l(j) && !cokernel_dies.contains(j) {
            cokernel.essential.push(j.clone());
        }
    }

    ImageKernelCokernel { image, kernel, cokernel, relative, complex: k.diagram, subcomplex: l.diagram }
}
