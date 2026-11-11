//! An implementation of the standard algorithm with the clearing optimisation.
//!
//! Run it with `Clearing::new().reduce(matrix)`, see [`ReductionAlgorithm`].

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::columns::{
    BHCol, ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow, MergedColumns,
    WorkingColumn,
};
use crate::matrices::combinators::product;
use crate::matrices::SquareMatrix;
use crate::matrix_col_product;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle},
};

use crate::reduction::{Diagram, Reduction, ReductionAlgorithm};

use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone)]
#[allow(clippy::module_name_repetitions)]
///
/// Columns with `V_i = e_i` (the vast majority, typically) are not stored at all.
pub(crate) enum ReductionColumn<CF, ColT> {
    /// This variant is used when `(i, j)` is found as a pair, in which case column `i` can be reduced by setting the column `V_i = R_j`, we store j here.
    Cleared(ColT),
    /// This variant is used when the column is not cleared and hence must be reduced by the main loop.
    /// The inner `Vec` represents the sum of columns used in the reduction, minus the starting column (which corresponds to the +1 on the diagonal of `V`).
    Reduced(Vec<(CF, ColT)>),
}

type ReductionCols<'a, ColT, CoefficientField> =
    Cow<'a, FxHashMap<ColT, ReductionColumn<CoefficientField, ColT>>>;

#[derive(Clone)]
/// The reduction matrix produced by the clearing algorithm.
///
/// Since some columns in this matrix are just copies of columns in the original boundary matrix (D) this must hold onto a reference to your matrix.
/// When you request a column of this matrix, if the column was reduced by clearing then the corresponding column from D is returned, otherwise the column was reduced conventially so the stored column of V is returned.
pub struct ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    boundary: M,
    reduction_columns: ReductionCols<'a, M::ColT, M::CoefficientField>,
}

impl<'a, M> ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    fn build_from_ref(
        boundary: M,
        reduction_columns: &'a FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
    ) -> Self {
        Self {
            boundary,
            reduction_columns: Cow::Borrowed(reduction_columns),
        }
    }

    /// Provides a more efficient check for whether a given column represents a cycle in the reduced matrix R.
    ///
    /// A column that was cleared by the clearing optimisation is necessarily a cycle.
    /// Otherwise, if the column was conventionally reduced then we have to build that column in R.
    pub fn col_is_cycle(&self, col: M::ColT) -> bool {
        match self.reduction_columns.get(&col) {
            Some(ReductionColumn::Cleared(_)) => true,
            Some(ReductionColumn::Reduced(_)) | None => {
                let r_matrix = product(&self.boundary, &self);
                let mut r_col = r_matrix.build_bhcol(col);
                r_col.pop_pivot().is_none()
            }
        }
    }
}

impl<M> ClearedReductionMatrix<'static, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    pub(crate) fn build_from_owned(
        boundary: M,
        reduction_columns: FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
    ) -> Self {
        Self {
            boundary,
            reduction_columns: Cow::Owned(reduction_columns),
        }
    }
}

impl<'a, M> MatrixOracle for ClearedReductionMatrix<'a, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::ColT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // TODO: Is there a way to do this without Box?
        let output_iter: Box<dyn Iterator<Item = (M::CoefficientField, M::RowT)>> =
            match self.reduction_columns.get(&col) {
                // Columns that needed no reduction are not stored: V_i = e_i
                None => Box::new(iter::once((M::CoefficientField::one(), col))),
                Some(ReductionColumn::Cleared(death_idx)) => {
                    // This returns the death_idx column of R = D V
                    let v_j = self.column(death_idx.clone());
                    // v_j should be of the Reduced variant
                    Box::new(matrix_col_product!(self.boundary, v_j))
                    //Box::new(vec.iter().copied())
                }
                Some(ReductionColumn::Reduced(vec)) => Box::new(
                    // We don't store the diagonal so we have to chain +1 on the diagonal to the output
                    vec.iter()
                        .cloned()
                        .chain(iter::once((M::CoefficientField::one(), col))),
                ),
            };

        output_iter
    }
}

impl<M> HasColBasis for ClearedReductionMatrix<'_, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type BasisT<'b>
        = M::BasisT<'b>
    where
        Self: 'b;

    fn basis(&self) -> Self::BasisT<'_> {
        self.boundary.basis()
    }
}

impl<M> HasSplitColBasis for ClearedReductionMatrix<'_, M>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    type SubBasisT<'b>
        = M::SubBasisT<'b>
    where
        Self: 'b;

    fn n_dimensions(&self) -> usize {
        self.boundary.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        self.boundary.sub_basis(dimension)
    }
}

/// The pivot table of a dimension: `pivot row -> (column, coefficient of the pivot)`.
/// Apparent pairs are not stored here, see [`CRMBuilder::owner`].
type LowInverse<RowT, CF> = FxHashMap<RowT, (RowT, CF)>;

struct CRMBuilder<M, DimIter, C, FilT, RowT, CF>
where
    DimIter: Iterator,
    C: ColumnRepresentation,
    FilT: ColumnFiltration,
    RowT: ColumnRow,
    CF: ColumnCoefficient,
{
    boundary: M,
    dimension_order: std::iter::Peekable<DimIter>,
    diagram: Diagram<RowT>,
    reduction_columns: FxHashMap<RowT, ReductionColumn<CF, RowT>>,
    /// The pairs `pivot row -> column` found in the previous dimension, whose rows are the columns to clear in this one
    previous_pairs: FxHashMap<RowT, RowT>,
    skip_zero_apparent_pairs: bool,
    /// Births of skipped (zero-persistence apparent) pairs that weren't cleared, and so may be wrongly marked essential.
    /// Only ever non-empty when dimensions are reduced against the direction of the matrix.
    skipped_births: FxHashSet<RowT>,
    columns: C,
    /// Shared by the columns reduced in the current dimension
    cache: C::Cache<FilT, RowT, CF>,
}

impl<M, DimIter, C> CRMBuilder<M, DimIter, C, M::FiltrationT, M::RowT, M::CoefficientField>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
    M::CoefficientField: Invertible + ColumnCoefficient,
    M::FiltrationT: ColumnFiltration,
    M::ColT: ColumnRow + Debug,
    DimIter: Iterator<Item = usize>,
    C: ColumnRepresentation,
{
    fn init(
        boundary: M,
        dimension_order: DimIter,
        skip_zero_apparent_pairs: bool,
        columns: C,
    ) -> Self {
        Self {
            skip_zero_apparent_pairs,
            columns,
            cache: Default::default(),
            boundary,
            dimension_order: dimension_order.peekable(),
            diagram: Diagram {
                essential: FxHashSet::default(),
                pairings: FxHashSet::default(),
            },
            reduction_columns: FxHashMap::default(),
            previous_pairs: FxHashMap::default(),
            skipped_births: FxHashSet::default(),
        }
    }

    /// The column that owns `row` as its pivot (in the current dimension), and the coefficient of that pivot.
    ///
    /// Apparent pairs are never stored, so if `row` is not in `low_inverse` we check whether it is the pivot of the first column in its row.
    fn owner(
        &self,
        low_inverse: &LowInverse<M::RowT, M::CoefficientField>,
        row: &M::RowT,
    ) -> Option<(M::ColT, M::CoefficientField)> {
        if let Some(owner) = low_inverse.get(row) {
            return Some(owner.clone());
        }
        let first = self.boundary.first_column_in_row(row.clone())?;
        // A cleared column has R = 0 so owns nothing (this can't happen for a chain complex, but costs nothing to check)
        if matches!(
            self.reduction_columns.get(&first),
            Some(ReductionColumn::Cleared(_))
        ) {
            return None;
        }
        let pivot = self.boundary.column_pivot(first.clone())?;
        (pivot.row_index == *row).then_some((first, pivot.coeff))
    }

    /// Reduce the column `r_i` (and keep track of `v_i`), returning the pivot of `R_i` if it is non-zero.
    fn reduce_column(
        &self,
        low_inverse: &LowInverse<M::RowT, M::CoefficientField>,
        r_i: &mut C::Column<M::FiltrationT, M::RowT, M::CoefficientField>,
        v_i: &mut BHCol<(), M::RowT, M::CoefficientField>,
    ) -> Option<(M::RowT, M::CoefficientField)> {
        loop {
            // Column reduced to 0 -> found cycle
            let pivot_entry = r_i.pop_pivot()?;

            // Check if there is a column with the same pivot, otherwise we found a boundary
            let Some((j_basis_element, j_coeff)) = self.owner(low_inverse, &pivot_entry.row_index)
            else {
                return Some((pivot_entry.row_index, pivot_entry.coeff));
            };

            // TODO: Over "fields" such as f64 this might not cancel to 0
            //       We need some way to enforce that the pivot is deleted
            // We add a multiple of R_j = D V_j to cancel out the pivot, which we put back to be cancelled
            let col_multiple = pivot_entry.coeff.additive_inverse() * (j_coeff.mult_inverse());
            r_i.add_entry(pivot_entry);

            let v_matrix =
                ClearedReductionMatrix::build_from_ref(&self.boundary, &self.reduction_columns);
            for (coeff, k) in v_matrix.column(j_basis_element) {
                let multiple = coeff * col_multiple;
                self.columns.add_matrix_column(
                    &self.cache,
                    &self.boundary,
                    r_i,
                    k.clone(),
                    multiple,
                );
                v_i.add_tuple((multiple, k, ()));
            }
        }
    }

    /// Record that the column `basis_element` has been reduced, with the given pivot (if any).
    /// Apparent pairs are not added to `low_inverse`, since [`owner`](Self::owner) recomputes them.
    fn record_reduced(
        &mut self,
        basis_element: M::ColT,
        pivot: Option<(M::RowT, M::CoefficientField)>,
        apparent: bool,
        low_inverse: &mut LowInverse<M::RowT, M::CoefficientField>,
        new_pairs: &mut Option<FxHashMap<M::RowT, M::ColT>>,
    ) {
        let Some((pivot_row, pivot_coeff)) = pivot else {
            self.diagram.essential.insert(basis_element);
            return;
        };
        if !apparent {
            low_inverse.insert(pivot_row.clone(), (basis_element.clone(), pivot_coeff));
        }
        // Remember the pair so that the pivot row's column can be cleared in the next dimension
        if let Some(new_pairs) = new_pairs {
            new_pairs.insert(pivot_row.clone(), basis_element.clone());
        }
        // If dimensions are visited in a sensible order then we see pairings before their birth columns are reduced.
        // Otherwise the birth may already be marked essential, which is fixed up once at the end.
        let skip = apparent
            && self.skip_zero_apparent_pairs
            && self.boundary.filtration_value(pivot_row.clone())
                == self.boundary.filtration_value(basis_element.clone());
        if skip {
            // The birth's column may already have been reduced (if dimensions are visited against the direction of the
            // matrix), and then marked essential since nothing had claimed it yet
            self.diagram.essential.remove(&pivot_row);
        } else {
            self.diagram.pairings.insert((pivot_row, basis_element));
        }
    }

    fn save_column(
        &mut self,
        basis_element: M::ColT,
        low_inverse: &mut LowInverse<M::RowT, M::CoefficientField>,
        new_pairs: &mut Option<FxHashMap<M::RowT, M::ColT>>,
        pivot: Option<(M::RowT, M::CoefficientField)>,
        mut v_i: BHCol<(), M::RowT, M::CoefficientField>,
    ) {
        self.record_reduced(basis_element.clone(), pivot, false, low_inverse, new_pairs);

        // Then save v_i (minus the diagonal) to the reduction matrix, unless it is trivial
        // TODO: Add option to only store this column when pivot is Some
        // Because otherwise we will never need the column again during reduction
        let v_i: Vec<_> = v_i
            .drain_sorted()
            .map(|entry| (entry.coeff, entry.row_index))
            .collect();
        if !v_i.is_empty() {
            self.reduction_columns
                .insert(basis_element, ReductionColumn::Reduced(v_i));
        }
    }

    fn reduce_dimension(&mut self, dim: usize, is_last: bool) {
        // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
        // We build a new one for each dimension
        let mut low_inverse: LowInverse<M::RowT, M::CoefficientField> = FxHashMap::default();
        // The pairs found in this dimension, for clearing the next one (if there is one)
        let mut new_pairs = (!is_last).then(FxHashMap::default);

        let basis_size = self.boundary.sub_basis(dim).size();

        'column_loop: for i in 0..basis_size {
            // Reduce column i
            let basis_element = self.boundary.sub_basis(dim).element(i);

            // First check whether it can be cleared: V_i = R_j where (i, j) was a pair in the previous dimension
            if let Some(birth) = self.previous_pairs.remove(&basis_element) {
                self.reduction_columns
                    .insert(basis_element, ReductionColumn::Cleared(birth));
                continue 'column_loop;
            }

            // Most columns need no reduction: their pivot is not yet claimed (or they are zero).
            // Find the pivot without building the whole column, and only build it if there is work to do.
            let pivot = self.boundary.column_pivot(basis_element.clone());
            let apparent = pivot.as_ref().is_some_and(|entry| {
                self.boundary
                    .first_column_in_row(entry.row_index.clone())
                    .as_ref()
                    == Some(&basis_element)
            });
            match pivot {
                Some(entry)
                    if !apparent && self.owner(&low_inverse, &entry.row_index).is_some() => {}
                pivot => {
                    let pivot = pivot.map(|entry| (entry.row_index, entry.coeff));
                    self.record_reduced(
                        basis_element,
                        pivot,
                        apparent,
                        &mut low_inverse,
                        &mut new_pairs,
                    );
                    continue 'column_loop;
                }
            }

            // Otherwise reduce the column and update the builder accordingly
            let mut v_i = {
                let self_borrow = &self.boundary;
                self_borrow.with_trivial_filtration().empty_bhcol()
            };
            let mut r_i = C::Column::default();
            self.columns.add_matrix_column(
                &self.cache,
                &self.boundary,
                &mut r_i,
                basis_element.clone(),
                M::CoefficientField::one(),
            );
            let pivot = self.reduce_column(&low_inverse, &mut r_i, &mut v_i);
            self.save_column(basis_element, &mut low_inverse, &mut new_pairs, pivot, v_i);
        }
        // Columns of D are only ever added to columns of the same dimension
        self.cache = Default::default();
        // Rows paired in the previous dimension that weren't columns of this one aren't needed for clearing.
        // But if their pair was skipped, the row's column may be reduced in a later dimension and marked essential.
        // (When dimensions are visited in the direction of the matrix there are no such rows.)
        let leftover = std::mem::replace(&mut self.previous_pairs, new_pairs.unwrap_or_default());
        if self.skip_zero_apparent_pairs {
            for (row, col) in leftover {
                if self.boundary.filtration_value(row.clone())
                    == self.boundary.filtration_value(col)
                {
                    self.skipped_births.insert(row);
                }
            }
        }
    }

    fn reduce_all_dimension(&mut self) {
        while let Some(dim) = self.dimension_order.next() {
            let is_last = self.dimension_order.peek().is_none();
            self.reduce_dimension(dim, is_last);
        }
    }
}

// TODO: Experiment with different Hashers, maybe nohash_hash? Make generic over ColT hashser?
// TODO: Experiment with occasionally consolidating v_i and r_i
//       This will take some time but reduce memory usage - maybe make configurable?
/// The standard algorithm with the clearing optimisation.
///
/// Dimensions are reduced one at a time, and the pivots found in one dimension are the columns that can be skipped (cleared) in the next.
/// For this to save work the dimensions must be visited in the direction the matrix maps them:
/// ascending for a coboundary matrix, descending for a boundary matrix.
/// By default we work this out from the matrix itself (see [`Clearing::new`]), but you can set the order yourself with [`with_dimension_order`](Clearing::with_dimension_order).
/// The diagram is correct whatever the order, only the amount of work changes.
///
/// Columns whose pivot is unclaimed are recorded without building them, and apparent pairs are recognised if the matrix
/// knows the first column in each row; see the [module docs](crate::reduction#doing-less-work).
/// Neither stores a reduction column, and apparent pairs needn't be in the column basis at all.
///
/// The `boundary` matrix needs a [`HasSplitColBasis`].
/// The reduction matrix V is returned as a [`ClearedReductionMatrix`], which holds onto your matrix.
/// For a parallel version, see `Clearing::parallel` (feature `parallel`).
///
/// The algorithm is generic over how it represents the column being reduced, see [`ColumnRepresentation`];
/// the default is [`MergedColumns`]. Choose another with [`with_columns`](Clearing::with_columns).
#[derive(Debug, Clone)]
pub struct Clearing<C = MergedColumns> {
    dimension_order: Option<Vec<usize>>,
    pub(crate) skip_zero_apparent_pairs: bool,
    pub(crate) columns: C,
}

impl<C: Default> Default for Clearing<C> {
    fn default() -> Self {
        Self {
            dimension_order: None,
            skip_zero_apparent_pairs: true,
            columns: C::default(),
        }
    }
}

impl Clearing {
    /// Reduce every dimension of the matrix, in whichever direction the matrix maps between them.
    /// Zero-persistence apparent pairs are left out of the diagram, see
    /// [`skip_zero_apparent_pairs`](Self::skip_zero_apparent_pairs).
    ///
    /// The direction is detected by looking at the first non-zero column we find (in dimension `d`) and checking
    /// whether its pivot row lies in dimension `d - 1` (descending, like a boundary matrix) or not (ascending, like a coboundary matrix).
    /// This costs at most one pass over a single sub-basis.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<C: ColumnRepresentation> Clearing<C> {
    /// Represent the columns being reduced with `columns` instead, see [`ColumnRepresentation`].
    pub fn with_columns<C2: ColumnRepresentation>(self, columns: C2) -> Clearing<C2> {
        Clearing {
            dimension_order: self.dimension_order,
            skip_zero_apparent_pairs: self.skip_zero_apparent_pairs,
            columns,
        }
    }

    /// Reduce only the given dimensions, in the order given, instead of detecting them.
    ///
    /// For example to only reduce up to dimension `max_dim` of a coboundary matrix use `0..=max_dim`, and for a boundary matrix `(0..=max_dim).rev()`.
    /// Dimensions the matrix doesn't have (at least its [`n_dimensions`](HasSplitColBasis::n_dimensions)) are empty, so are ignored.
    pub fn with_dimension_order(
        mut self,
        dimension_order: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.dimension_order = Some(dimension_order.into_iter().collect());
        self
    }

    /// Whether to leave apparent pairs of zero persistence (whose row and column have the same filtration value) out of
    /// the diagram. On by default.
    ///
    /// Apparent pairs need no reduction, but in many filtrations nearly every pair is a zero-persistence apparent pair,
    /// so recording them dominates the time and memory of the reduction.
    /// They are still used for clearing, and the reduction matrix `V` is unaffected. Their births are never reported
    /// as essential, whatever the dimension order.
    /// Only matrices with [`first_column_in_row`](crate::matrices::HasRowFiltration::first_column_in_row) have apparent pairs.
    pub fn skip_zero_apparent_pairs(mut self, skip: bool) -> Self {
        self.skip_zero_apparent_pairs = skip;
        self
    }

    /// The order in which dimensions of `boundary` will be reduced: the one given to [`with_dimension_order`](Self::with_dimension_order),
    /// or else every dimension in the detected direction (see [`new`](Self::new)).
    pub fn dimension_order<M>(&self, boundary: &M) -> Vec<usize>
    where
        M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
        M::CoefficientField: Invertible,
    {
        let n_dimensions = boundary.n_dimensions();
        if let Some(order) = &self.dimension_order {
            // Dimensions beyond the matrix are empty, and would index past its sub-bases
            return order
                .iter()
                .copied()
                .filter(|&dim| dim < n_dimensions)
                .collect();
        }
        if maps_upwards(boundary) {
            (0..n_dimensions).collect()
        } else {
            (0..n_dimensions).rev().collect()
        }
    }
}

/// Does `boundary` map dimension `d` to dimension `d + 1` (as opposed to `d - 1`)?
/// Looks at the pivot of the first non-zero column whose pivot we can find in a neighbouring sub-basis.
/// If every column is zero then the order is irrelevant.
fn maps_upwards<M>(boundary: &M) -> bool
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
    M::CoefficientField: Invertible,
{
    for dim in 0..boundary.n_dimensions() {
        let basis = boundary.sub_basis(dim);
        for i in 0..basis.size() {
            let Some(pivot) = boundary
                .column_pivot(basis.element(i))
                .map(|entry| entry.row_index)
            else {
                continue;
            };
            if dim == 0 {
                return true;
            }
            let below = boundary.sub_basis(dim - 1);
            if (0..below.size()).any(|k| below.element(k) == pivot) {
                return false;
            }
            // The basis needn't contain every cell (e.g. it may omit apparent pairs), so only conclude
            // the matrix maps upwards if we can see the pivot in the dimension above; otherwise try another column
            if dim + 1 < boundary.n_dimensions() {
                let above = boundary.sub_basis(dim + 1);
                if (0..above.size()).any(|k| above.element(k) == pivot) {
                    return true;
                }
            } else {
                return true;
            }
        }
    }
    true
}

impl<M, C> ReductionAlgorithm<M> for Clearing<C>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow + 'static,
    M::ColT: Debug,
    C: ColumnRepresentation,
{
    type V = ClearedReductionMatrix<'static, M>;

    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT> {
        let dimension_order = self.dimension_order(&boundary);
        let mut builder = CRMBuilder::init(
            boundary,
            dimension_order.into_iter(),
            self.skip_zero_apparent_pairs,
            self.columns,
        );
        builder.reduce_all_dimension();
        let v =
            ClearedReductionMatrix::build_from_owned(builder.boundary, builder.reduction_columns);
        let mut diagram = builder.diagram;
        diagram.remove_births_from_essential();
        diagram
            .essential
            .retain(|elem| !builder.skipped_births.contains(elem));
        Reduction { v, diagram }
    }
}
