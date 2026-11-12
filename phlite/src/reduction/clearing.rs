//! An implementation of the standard algorithm with the clearing optimisation.
//!
//! Run it with `Clearing::new().reduce(matrix)`, see [`ReductionAlgorithm`].

use std::borrow::Cow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::columns::{
    BHCol, ColumnCache, ColumnCoefficient, ColumnEntry, ColumnFiltration, ColumnRepresentation,
    ColumnRow, MergedColumns, SortedRun, WorkingColumn,
};
use crate::matrices::combinators::product;
use crate::matrices::SquareMatrix;
use crate::matrix_col_product;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle},
};

use crate::reduction::{Diagram, DiagramBuilder, Reduction, ReductionAlgorithm};

use crate::bookkeeping::{ElementMap, ElementSet};

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
    Cow<'a, ElementMap<ColT, ReductionColumn<CoefficientField, ColT>>>;

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
        reduction_columns: &'a ElementMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
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
        reduction_columns: ElementMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
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

/// The pivot table of a dimension: `pivot row -> (owner, coefficient of the pivot)`.
/// Apparent pairs are only stored here if it is complete, see [`CRMBuilder::complete`].
type LowInverse<RowT, CF> = ElementMap<RowT, (Owner<RowT>, CF)>;

/// The column owning a pivot.
///
/// When R is stored, a reduced column is found by its position in [`CRMBuilder::r_columns`], so that adding it costs
/// one lookup (in the pivot table) rather than another in a map of stored columns.
/// (For explicit complexes, an `Option` of this and a [`Z2`](crate::fields::Z2) coefficient fits in 8 bytes, so that the
/// pivot table is [inline](ElementMap::is_inline).)
#[derive(Debug, Clone)]
enum Owner<ColT> {
    /// A column whose `R` is not stored, either because it needed no reduction (`R = D`) or R isn't being stored
    Column(ColT),
    /// The column `r_columns[k].0`, with `R = r_columns[k].1`
    Stored(u32),
}

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
    diagram: DiagramBuilder<RowT>,
    reduction_columns: ElementMap<RowT, ReductionColumn<CF, RowT>>,
    /// The pairs `pivot row -> column` found in the previous dimension, whose rows are the columns to clear in this one
    previous_pairs: ElementMap<RowT, RowT>,
    skip_zero_apparent_pairs: bool,
    /// Whether the basis holds every column, see [`Clearing::full_basis`]
    full_basis: bool,
    /// Births of skipped (zero-persistence apparent) pairs that weren't cleared, and so may be wrongly marked essential.
    /// Only ever non-empty when dimensions are reduced against the direction of the matrix.
    skipped_births: ElementSet<RowT>,
    columns: C,
    cache_columns: bool,
    /// Shared by the columns reduced in the current dimension
    cache: ColumnCache<FilT, RowT, CF>,
    /// Whether to keep the reduced column `R_i` of every column reduced (see [`Clearing::store_r`])
    store_r: bool,
    /// Whether to keep track of `V` (only optional when `store_r`, since otherwise `R` is recomputed from `V`)
    track_v: bool,
    /// The reduced columns `(i, R_i)` of the current dimension that have a pivot, if `store_r`
    r_columns: Vec<(RowT, SortedRun<FilT, RowT, CF>)>,
    /// The working columns, reused for every column reduced so that their memory is only allocated once
    r_i: C::Column<FilT, RowT, CF>,
    v_i: BHCol<(), RowT, CF>,
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
        full_basis: bool,
        columns: C,
        cache_columns: bool,
        store_r: bool,
        track_v: bool,
    ) -> Self {
        debug_assert!(
            track_v || store_r,
            "R is recomputed from V unless it is stored"
        );
        Self {
            track_v,
            r_i: columns.new_column(&boundary),
            v_i: BHCol::default(),
            skip_zero_apparent_pairs,
            full_basis,
            columns,
            cache_columns,
            cache: ColumnCache::new(cache_columns),
            store_r,
            r_columns: Vec::new(),
            boundary,
            dimension_order: dimension_order.peekable(),
            diagram: DiagramBuilder::default(),
            reduction_columns: ElementMap::new(),
            previous_pairs: ElementMap::new(),
            skipped_births: ElementSet::default(),
        }
    }

    /// Whether `low_inverse` holds every pivot, apparent pairs included, so that it alone says who owns a row.
    ///
    /// Apparent pairs are stored when that is free (`low_inverse` is [inline](ElementMap::is_inline), so has a place for
    /// every row anyway) and saves looking them up. But only when the basis holds every column
    /// ([`Clearing::full_basis`]): otherwise the column of an apparent pair may never be visited, so never recorded.
    #[inline]
    fn complete(&self, low_inverse: &LowInverse<M::RowT, M::CoefficientField>) -> bool {
        self.full_basis && low_inverse.is_inline()
    }

    /// The column that owns `row` as its pivot (in the current dimension), and the coefficient of that pivot.
    ///
    /// Unless `low_inverse` is [complete](Self::complete), apparent pairs aren't stored, so if `row` is not in
    /// `low_inverse` we check whether it is the pivot of the first column in its row.
    #[inline]
    fn owner(
        &self,
        low_inverse: &LowInverse<M::RowT, M::CoefficientField>,
        row: &M::RowT,
    ) -> Option<(Owner<M::ColT>, M::CoefficientField)> {
        if let Some(owner) = low_inverse.get(row) {
            return Some(owner.clone());
        }
        if self.complete(low_inverse) {
            return None;
        }
        let first = self.boundary.first_column_in_row(row.clone())?;
        // A cleared column has R = 0 so owns nothing. This can't happen: if the first column in a row had its pivot
        // there it would be an apparent pair, with R = D != 0. So we only check when tracking V (which records
        // cleared columns anyway).
        if self.track_v
            && matches!(
                self.reduction_columns.get(&first),
                Some(ReductionColumn::Cleared(_))
            )
        {
            return None;
        }
        let pivot = self.boundary.column_pivot(first.clone())?;
        (pivot.row_index == *row).then_some((Owner::Column(first), pivot.coeff))
    }

    /// Call `f` on every term `(coefficient, column)` of `V_j`, without boxing the iterator for the usual cases.
    #[inline]
    fn for_each_v_term(&self, j: M::ColT, mut f: impl FnMut(M::CoefficientField, M::ColT)) {
        match self.reduction_columns.get(&j) {
            // Columns that needed no reduction are not stored: V_j = e_j
            None => f(M::CoefficientField::one(), j),
            Some(ReductionColumn::Reduced(terms)) => {
                for (coeff, k) in terms {
                    f(*coeff, k.clone());
                }
                f(M::CoefficientField::one(), j);
            }
            // Cleared columns have R_j = 0 so never own a pivot, but handle them anyway
            Some(ReductionColumn::Cleared(_)) => {
                let v_matrix =
                    ClearedReductionMatrix::build_from_ref(&self.boundary, &self.reduction_columns);
                for (coeff, k) in v_matrix.column(j) {
                    f(coeff, k);
                }
            }
        }
    }

    /// Reduce the column `r_i` (and keep track of `v_i`), returning the pivot of `R_i` if it is non-zero
    /// (`r_i` then holds the rest of `R_i`).
    fn reduce_column(
        &self,
        low_inverse: &LowInverse<M::RowT, M::CoefficientField>,
        r_i: &mut C::Column<M::FiltrationT, M::RowT, M::CoefficientField>,
        v_i: &mut BHCol<(), M::RowT, M::CoefficientField>,
    ) -> Option<ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>> {
        loop {
            // Column reduced to 0 -> found cycle
            let pivot_entry = r_i.pop_pivot()?;

            // Check if there is a column with the same pivot, otherwise we found a boundary
            let Some((owner, j_coeff)) = self.owner(low_inverse, &pivot_entry.row_index) else {
                return Some(pivot_entry);
            };

            // TODO: Over "fields" such as f64 this might not cancel to 0
            //       We need some way to enforce that the pivot is deleted
            // We add a multiple of R_j = D V_j to cancel out the pivot, which we put back to be cancelled
            let col_multiple = pivot_entry.coeff.additive_inverse() * (j_coeff.mult_inverse());
            r_i.add_entry(pivot_entry);

            if self.store_r {
                // Add R_j itself: stored if column j was reduced, or else D_j (it needed no reduction, so V_j = e_j)
                let j_basis_element = match owner {
                    Owner::Stored(k) => {
                        let (j, run) = &self.r_columns[k as usize];
                        self.columns.add_run(r_i, run, col_multiple);
                        j.clone()
                    }
                    Owner::Column(j) => {
                        self.columns.add_matrix_column(
                            &self.cache,
                            &self.boundary,
                            r_i,
                            j.clone(),
                            col_multiple,
                        );
                        j
                    }
                };
                // and V_j to V_i
                if self.track_v {
                    self.for_each_v_term(j_basis_element, |coeff, k| {
                        v_i.add_tuple((coeff * col_multiple, k, ()));
                    });
                }
            } else {
                // Add R_j = D V_j by adding the column of D for every term of V_j
                let Owner::Column(j_basis_element) = owner else {
                    unreachable!("R is only stored if store_r")
                };
                self.for_each_v_term(j_basis_element, |coeff, k| {
                    let multiple = coeff * col_multiple;
                    self.columns.add_matrix_column(
                        &self.cache,
                        &self.boundary,
                        r_i,
                        k.clone(),
                        multiple,
                    );
                    v_i.add_tuple((multiple, k, ()));
                });
            }
        }
    }

    /// Record that the column `basis_element` has been reduced, with the given pivot (if any).
    /// Apparent pairs are only added to `low_inverse` if it is [complete](Self::complete).
    fn record_reduced(
        &mut self,
        basis_element: M::ColT,
        pivot: Option<(M::RowT, M::CoefficientField)>,
        apparent: bool,
        stored: Option<u32>,
        low_inverse: &mut LowInverse<M::RowT, M::CoefficientField>,
        new_pairs: &mut Option<ElementMap<M::RowT, M::ColT>>,
    ) {
        let Some((pivot_row, pivot_coeff)) = pivot else {
            self.diagram.essential.insert(basis_element);
            return;
        };
        if !apparent || self.complete(low_inverse) {
            let owner = match stored {
                Some(k) => Owner::Stored(k),
                None => Owner::Column(basis_element.clone()),
            };
            low_inverse.insert(pivot_row.clone(), (owner, pivot_coeff));
        }
        // Remember the pair so that the pivot row's column can be cleared in the next dimension
        if let Some(new_pairs) = new_pairs {
            new_pairs.insert(pivot_row.clone(), basis_element.clone());
        }
        // If dimensions are visited in a sensible order then we see pairings before their birth columns are reduced.
        // Otherwise the birth may already be marked essential, which is fixed up once at the end.
        let skip = apparent
            && self.skip_zero_apparent_pairs
            && self
                .boundary
                .same_filtration_value(pivot_row.clone(), basis_element.clone());
        if skip {
            // The birth's column may already have been reduced (if dimensions are visited against the direction of the
            // matrix), and then marked essential since nothing had claimed it yet
            self.diagram.essential.remove(&pivot_row);
        } else {
            self.diagram.pairings.push((pivot_row, basis_element));
        }
    }

    fn save_column(
        &mut self,
        basis_element: M::ColT,
        low_inverse: &mut LowInverse<M::RowT, M::CoefficientField>,
        new_pairs: &mut Option<ElementMap<M::RowT, M::ColT>>,
        pivot: Option<(M::RowT, M::CoefficientField)>,
        stored: Option<u32>,
        v_i: &mut BHCol<(), M::RowT, M::CoefficientField>,
        scratch: &mut Vec<(M::CoefficientField, M::RowT)>,
    ) {
        self.record_reduced(
            basis_element.clone(),
            pivot,
            false,
            stored,
            low_inverse,
            new_pairs,
        );

        // Then save v_i (minus the diagonal) to the reduction matrix, unless it is trivial.
        // Drained into a reused buffer first, so that the stored column is allocated once at its exact size.
        // TODO: Add option to only store this column when pivot is Some
        // Because otherwise we will never need the column again during reduction
        if !self.track_v {
            return;
        }
        scratch.clear();
        scratch.extend(
            v_i.drain_sorted()
                .map(|entry| (entry.coeff, entry.row_index)),
        );
        if !scratch.is_empty() {
            self.reduction_columns
                .insert(basis_element, ReductionColumn::Reduced(scratch.to_vec()));
        }
    }

    fn reduce_dimension(&mut self, dim: usize, is_last: bool) {
        // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
        // We build a new one for each dimension
        let mut low_inverse: LowInverse<M::RowT, M::CoefficientField> = ElementMap::new();
        // The pairs found in this dimension, for clearing the next one (if there is one)
        let mut new_pairs = (!is_last).then(ElementMap::new);

        let basis_size = self.boundary.sub_basis(dim).size();

        // Reused by every column reduced in this dimension, so that their capacity is only allocated once
        let mut r_i = std::mem::take(&mut self.r_i);
        let mut v_i = std::mem::take(&mut self.v_i);
        let mut r_scratch = Vec::new();
        let mut v_scratch = Vec::new();

        'column_loop: for i in 0..basis_size {
            // Reduce column i
            let basis_element = self.boundary.sub_basis(dim).element(i);

            // First check whether it can be cleared: V_i = R_j where (i, j) was a pair in the previous dimension
            if let Some(birth) = self.previous_pairs.remove(&basis_element) {
                // Only `V` needs to know (`R_i = 0`, so column i never owns a pivot)
                if self.track_v {
                    self.reduction_columns
                        .insert(basis_element, ReductionColumn::Cleared(birth));
                }
                continue 'column_loop;
            }

            // Most columns need no reduction: their pivot is not yet claimed (or they are zero).
            // Find the pivot without building the whole column, and only build it if there is work to do.
            let pivot = self.boundary.column_pivot(basis_element.clone());
            // When `low_inverse` holds every pivot (apparent or not), it alone says whether the pivot is claimed, and
            // whether the column is apparent only matters for skipping zero-length pairs, so is only checked then
            let complete = self.complete(&low_inverse);
            let claimed = complete
                && pivot
                    .as_ref()
                    .is_some_and(|entry| low_inverse.contains_key(&entry.row_index));
            let apparent = !claimed
                && pivot.as_ref().is_some_and(|entry| {
                    (!complete
                        || (self.skip_zero_apparent_pairs
                            && self.boundary.same_filtration_value(
                                entry.row_index.clone(),
                                basis_element.clone(),
                            )))
                        && self
                            .boundary
                            .first_column_in_row(entry.row_index.clone())
                            .as_ref()
                            == Some(&basis_element)
                });
            match pivot {
                Some(_) if claimed => {}
                Some(entry)
                    if !complete
                        && !apparent
                        && self.owner(&low_inverse, &entry.row_index).is_some() => {}
                pivot => {
                    let pivot = pivot.map(|entry| (entry.row_index, entry.coeff));
                    self.record_reduced(
                        basis_element,
                        pivot,
                        apparent,
                        None,
                        &mut low_inverse,
                        &mut new_pairs,
                    );
                    continue 'column_loop;
                }
            }

            // Otherwise reduce the column and update the builder accordingly
            self.columns.add_matrix_column(
                &self.cache,
                &self.boundary,
                &mut r_i,
                basis_element.clone(),
                M::CoefficientField::one(),
            );
            let pivot = self.reduce_column(&low_inverse, &mut r_i, &mut v_i);
            let mut stored = None;
            if self.store_r {
                if let Some(pivot) = &pivot {
                    // Keep R_i: its pivot, then the rest of the column (drained into a reused buffer, so that the
                    // stored column is allocated once at its exact size)
                    r_scratch.clear();
                    r_scratch.push(pivot.clone());
                    r_i.drain_into(&mut r_scratch);
                    let k = u32::try_from(self.r_columns.len())
                        .expect("fewer than 2^32 columns reduced in a dimension");
                    stored = Some(k);
                    self.r_columns
                        .push((basis_element.clone(), SortedRun::from(&r_scratch[..])));
                }
            }
            // Whatever is left below the pivot isn't needed
            r_i.clear();
            let pivot = pivot.map(|entry| (entry.row_index, entry.coeff));
            self.save_column(
                basis_element,
                &mut low_inverse,
                &mut new_pairs,
                pivot,
                stored,
                &mut v_i,
                &mut v_scratch,
            );
        }
        self.r_i = r_i;
        self.v_i = v_i;
        // Columns of D are only ever added to columns of the same dimension
        self.cache = ColumnCache::new(self.cache_columns);
        // Stored columns of R are only ever added to columns of the same dimension
        self.r_columns = Vec::new();
        // Rows paired in the previous dimension that weren't columns of this one aren't needed for clearing.
        // But if their pair was skipped, the row's column may be reduced in a later dimension and marked essential.
        // (When dimensions are visited in the direction of the matrix there are no such rows.)
        let leftover = std::mem::replace(&mut self.previous_pairs, new_pairs.unwrap_or_default());
        if self.skip_zero_apparent_pairs {
            for (row, col) in leftover.into_entries() {
                if self.boundary.same_filtration_value(row.clone(), col) {
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
    pub(crate) full_basis: bool,
    pub(crate) columns: C,
    pub(crate) cache_columns: bool,
    pub(crate) store_r: bool,
}

impl<C: Default> Default for Clearing<C> {
    fn default() -> Self {
        Self {
            dimension_order: None,
            skip_zero_apparent_pairs: true,
            full_basis: false,
            columns: C::default(),
            cache_columns: true,
            store_r: false,
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
            full_basis: self.full_basis,
            columns,
            cache_columns: self.cache_columns,
            store_r: self.store_r,
        }
    }

    /// Promise that, in each dimension reduced, the basis holds every column of the matrix with an entry in that
    /// dimension's rows (as when the basis is the whole complex, up to some dimension). Off by default.
    ///
    /// Then, for matrices whose rows have [dense indices](crate::matrices::BasisElement::DENSE_INDEX), the pivot table
    /// also records apparent pairs (which costs no memory there), so that finding the owner of a pivot is one lookup.
    /// Otherwise apparent pairs are recognised again whenever they are needed, which reads the matrix and its
    /// transpose. A false promise gives wrong results.
    pub fn full_basis(mut self, full: bool) -> Self {
        self.full_basis = full;
        self
    }

    /// Whether to keep each reduced column of `R = D V` and add it directly, rather than recompute it from `V`
    /// (off by default). Serial only: the parallel algorithm ([`parallel`](Self::parallel)) ignores this.
    ///
    /// To cancel a pivot owned by an earlier column `j`, the reduction adds `R_j` to the column being reduced. By
    /// default only `V` is stored, so `R_j = D V_j` is recomputed by adding one column of `D` for every term of `V_j`.
    /// With this on, `R_j` is kept once reduced (for the rest of its dimension) and added as one column, as in PHAT.
    /// That costs memory for the columns of `R` that have a pivot, and pays off when `V_j` gets long, as in dense or
    /// cubical complexes. Apparent pairs are still never stored (`R_j = D_j` for them), so Vietoris–Rips cohomology,
    /// where almost every pair is apparent, gains little. `V` is still tracked for [`reduce`](ReductionAlgorithm::reduce),
    /// but not at all for [`diagram`](ReductionAlgorithm::diagram) (as when no representatives are wanted), since
    /// nothing needs it: that saves the time and memory of building `V`, which is most of the work left on dense
    /// matrices.
    pub fn store_r(mut self, store: bool) -> Self {
        self.store_r = store;
        self
    }

    /// Whether to cache the columns of `D` added during the reduction, see [`ColumnCache`]. On by default.
    ///
    /// Caching saves recomputing (and, for [`MergedColumns`], re-sorting) a column of `D` every time it is added,
    /// which pays off when columns are expensive to compute, but holds every column needed in a dimension in memory
    /// until that dimension is done. Without it, columns are recomputed from the matrix every time.
    pub fn cache_columns(mut self, cache: bool) -> Self {
        self.cache_columns = cache;
        self
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
        let builder = self.run(boundary, true);
        let v =
            ClearedReductionMatrix::build_from_owned(builder.boundary, builder.reduction_columns);
        let diagram = builder.diagram.finish(Some(&builder.skipped_births));
        Reduction { v, diagram }
    }

    /// With [`store_r`](Clearing::store_r), `V` isn't needed to reduce, so it isn't tracked at all.
    fn diagram(self, boundary: M) -> Diagram<M::ColT> {
        let track_v = !self.store_r;
        let builder = self.run(boundary, track_v);
        builder.diagram.finish(Some(&builder.skipped_births))
    }
}

impl<C: ColumnRepresentation> Clearing<C> {
    fn run<M>(
        self,
        boundary: M,
        track_v: bool,
    ) -> CRMBuilder<M, std::vec::IntoIter<usize>, C, M::FiltrationT, M::RowT, M::CoefficientField>
    where
        M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis,
        M::CoefficientField: Invertible + ColumnCoefficient + 'static,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow + 'static,
        M::ColT: Debug,
    {
        let dimension_order = self.dimension_order(&boundary);
        let mut builder = CRMBuilder::init(
            boundary,
            dimension_order.into_iter(),
            self.skip_zero_apparent_pairs,
            self.full_basis,
            self.columns,
            self.cache_columns,
            self.store_r,
            track_v,
        );
        builder.reduce_all_dimension();
        builder
    }
}

#[cfg(test)]
mod tests {
    use super::Owner;
    use crate::fields::Z2;

    /// The pivot table of 4-byte cells over Z/2 must stay small enough to be stored inline (one memory access per
    /// lookup, and room to hold apparent pairs for free): a layout change would silently lose that.
    #[test]
    fn pivot_table_of_small_cells_is_inline() {
        assert!(std::mem::size_of::<Option<(Owner<u32>, Z2)>>() <= 8);
    }
}
