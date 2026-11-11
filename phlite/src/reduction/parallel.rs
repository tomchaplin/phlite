//! A parallel, label-keyed variant of the clearing algorithm, adapted from
//! [Morozov and Nigmetov, *Towards Lockfree Persistent Homology*](https://doi.org/10.1145/3350755.3400244).
//!
//! Run it with `Clearing::new().parallel().reduce(matrix)`, see [`ParallelClearing`].
//!
//! Compared to the serial [`Clearing`], each dimension is reduced in three phases (much as in giotto-ph):
//!
//! 1. **Classify** every column of the dimension in parallel: cleared (by a pair from the previous dimension),
//!    zero, an apparent pair (if the matrix provides [`first_column_in_row`](HasRowFiltration::first_column_in_row)),
//!    or needing reduction. The results go into per-thread vectors, so this phase shares no state.
//!    Typically only a tiny fraction of columns need reduction, and only those get any per-column state below.
//! 2. **Reduce** the columns needing it concurrently, using the same out-of-order scheme as the paper:
//!    a thread reduces a private copy of a column and, whenever it finds a pivot owned by a column to its *right*,
//!    it publishes its own column, steals the pivot and carries on reducing the displaced column.
//!    A column whose pivot is not yet claimed is finished without building it in full
//!    (see [`column_pivot`](HasRowFiltration::column_pivot)).
//!    Pivots owned by apparent pairs are recomputed rather than stored, since apparent pairs never change.
//! 3. **Read off** the pairs, from the per-thread vectors and the reduced columns.
//!
//! By the pairing uniqueness lemma the resulting diagram is the same as the serial algorithm's
//! (the reduction matrix `V` that comes out may differ, but is still a valid one).
//!
//! # Differences from the integer-indexed algorithm
//!
//! * **Rows are never indexed.**
//!   The pivot table is a concurrent hash map keyed by the row *label* ([`RowT`](MatrixOracle::RowT)),
//!   which only ever holds rows that are the pivot of some reduced column (so at most one entry per column).
//!   Row order comes from the usual `(filtration value, label)` order on column entries.
//!   This means we never enumerate the row space, which is the main saving of cohomology.
//! * **Only the columns needing reduction are indexed**, densely, by their position (in basis order) in that short list.
//!   That is exactly the order the serial algorithm sweeps in, and it's all the pivot-stealing logic needs.
//!   Entries of `V` are column *labels*, so they can refer to columns outside the list (apparent pairs, with `V = e`).
//! * **We store `V`, not `R`.** The paper reduces `R` in place and never touches `V`.
//!   Here each published column is a snapshot of `V_j` together with the cached pivot (row label and coefficient) of `R_j = D V_j`.
//!   Readers check that cached pivot against the row they are looking up, which is the same
//!   "is the column I read still the one with this pivot" check as in the paper, but needs no access to `R`.
//!   When a column `R_p` is needed it is recomputed as `D V_p` from the oracle, exactly as the serial algorithm does.
//!   This keeps the memory profile of the serial algorithm (and lets us return a [`ClearedReductionMatrix`]).
//! * **Two storage backends**, sharing all of the reduction logic:
//!   * *Locking* ([`Storage::Locking`], the default):
//!     the pivot table is a [`dashmap::DashMap`] (sharded [`RwLock`]s) and published columns are [`Arc`] snapshots behind a [`RwLock`],
//!     all held only briefly.
//!   * *Lock-free* (`Storage::LockFree`, feature `lock-free`):
//!     the pivot table is a [`papaya`](https://docs.rs/papaya) concurrent hash map updated by compare-and-swap,
//!     and snapshots live in [`arc_swap::ArcSwapOption`](https://docs.rs/arc-swap) slots.
//!     No thread ever waits on another thread holding a lock.
//!
//! Enabled by the `parallel` feature (and `lock-free` for the second backend).

use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use dashmap::{mapref::entry::Entry, DashMap};
use rayon::prelude::*;

use super::clearing::{ClearedReductionMatrix, ReductionColumn};
use super::{Clearing, Reduction, ReductionAlgorithm};
use super::{Diagram, DiagramBuilder};
use crate::columns::{
    BHCol, ColumnCache, ColumnCoefficient, ColumnEntry, ColumnFiltration, ColumnRepresentation,
    ColumnRow, MergedColumns, WorkingColumn,
};
use crate::fields::{Invertible, NonZeroCoefficient};
use crate::matrices::{ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle, SquareMatrix};

/// Largest block of consecutive columns handed to a single rayon task.
/// Columns vary a lot in cost, so fine-grained tasks let rayon balance the load dynamically.
const MAX_TASK_LEN: usize = 16;
/// The fewest columns classified per task: classifying a column is cheap, so tasks must be big to amortise rayon's overhead
const MIN_CLASSIFY_LEN: usize = 1024;

use crate::bookkeeping::{ElementMap, ElementSet};
use crate::hash::FxBuild;
pub use crate::hash::MixedFx;

// ====== Published columns =======================================

/// An immutable snapshot of a column: `V_j` (including its diagonal) and the pivot of `R_j = D V_j`.
struct PublishedColumn<ColT, CF> {
    /// Entries `(coefficient, column label)`.
    /// Empty when `V_j = e_j` (the commonest case), so as not to allocate; use [`v_entries`].
    v: Vec<(CF, ColT)>,
    /// Row label and coefficient of the pivot of `R_j`, or `None` if `R_j = 0`
    pivot: Option<(ColT, CF)>,
}

type Snapshot<ColT, CF> = Arc<PublishedColumn<ColT, CF>>;

/// The entries of `V_j` from its published snapshot (expanding the empty, trivial case to `e_j`, where `label` is column `j`).
fn v_entries<'s, ColT: Clone, CF: NonZeroCoefficient>(
    snapshot: &'s PublishedColumn<ColT, CF>,
    label: &ColT,
) -> impl Iterator<Item = (CF, ColT)> + 's {
    let trivial = snapshot.v.is_empty().then(|| (CF::one(), label.clone()));
    trivial.into_iter().chain(snapshot.v.iter().cloned())
}

// ====== Storage backends ========================================

/// The shared state of a dimension's reduction: the pivot table and the slot holding each column's latest published snapshot.
/// Abstracting this is what lets the locking and lock-free variants share all of the reduction logic.
trait Backend<RowT, CF>: Sync {
    fn new(n_columns: usize) -> Self;
    /// The (dimension-local) index of the column that currently owns `row` as its pivot
    fn pivot_owner(&self, row: &RowT) -> Option<usize>;
    /// Atomically set the owner of `row` to `new` if it is currently `current`. Returns whether this happened.
    fn compare_set_pivot(&self, row: &RowT, current: Option<usize>, new: usize) -> bool;
    /// The latest snapshot published for column `j`, if any
    fn load(&self, j: usize) -> Option<Snapshot<RowT, CF>>;
    /// Replace the snapshot published for column `j`
    fn store(&self, j: usize, snapshot: Snapshot<RowT, CF>);
}

/// A [`DashMap`] (a hash map sharded behind [`RwLock`]s) for the pivots, and [`Arc`] snapshots behind a [`RwLock`] each.
struct LockingBackend<RowT, CF> {
    pivots: DashMap<RowT, usize, FxBuild>,
    published: Vec<RwLock<Option<Snapshot<RowT, CF>>>>,
}

impl<RowT: Hash + Eq + Clone + Send + Sync, CF: Send + Sync> Backend<RowT, CF>
    for LockingBackend<RowT, CF>
{
    fn new(n_columns: usize) -> Self {
        Self {
            pivots: DashMap::with_capacity_and_hasher(n_columns, FxBuild::default()),
            published: (0..n_columns).map(|_| RwLock::new(None)).collect(),
        }
    }
    fn pivot_owner(&self, row: &RowT) -> Option<usize> {
        self.pivots.get(row).map(|owner| *owner)
    }
    fn compare_set_pivot(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
        match self.pivots.entry(row.clone()) {
            Entry::Occupied(mut entry) if current == Some(*entry.get()) => {
                entry.insert(new);
                true
            }
            Entry::Vacant(entry) if current.is_none() => {
                entry.insert(new);
                true
            }
            _ => false,
        }
    }
    fn load(&self, j: usize) -> Option<Snapshot<RowT, CF>> {
        self.published[j].read().unwrap().clone()
    }
    fn store(&self, j: usize, snapshot: Snapshot<RowT, CF>) {
        *self.published[j].write().unwrap() = Some(snapshot);
    }
}

#[cfg(feature = "lock-free")]
pub use lock_free_backend::LockFreeBackend;

#[cfg(feature = "lock-free")]
mod lock_free_backend {
    use super::*;
    use arc_swap::ArcSwapOption;
    use papaya::{Compute, Operation};

    /// Lock-free storage, built from existing crates:
    ///
    /// * the pivot table is a [`papaya::HashMap`], updated with its atomic [`compute`](papaya::HashMap::compute) (a compare-and-swap loop),
    /// * each column's snapshot lives in an [`ArcSwapOption`], whose loads and stores never block.
    ///
    /// The pivot table is created with room for one entry per column, which is the most it can ever hold
    /// (a row is only in the table while it is some column's pivot),
    /// so it never needs to resize, which is the one operation `papaya` doesn't promise to be lock-free.
    pub struct LockFreeBackend<RowT, CF> {
        pivots: papaya::HashMap<RowT, usize, FxBuild>,
        published: Vec<ArcSwapOption<PublishedColumn<RowT, CF>>>,
    }

    impl<RowT: Hash + Eq + Clone + Send + Sync, CF: Send + Sync> Backend<RowT, CF>
        for LockFreeBackend<RowT, CF>
    {
        fn new(n_columns: usize) -> Self {
            Self {
                pivots: papaya::HashMap::with_capacity_and_hasher(n_columns, Default::default()),
                published: (0..n_columns).map(|_| ArcSwapOption::empty()).collect(),
            }
        }
        fn pivot_owner(&self, row: &RowT) -> Option<usize> {
            self.pivots.pin().get(row).copied()
        }
        fn compare_set_pivot(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
            let pivots = self.pivots.pin();
            let result = pivots.compute(row.clone(), |entry| {
                if entry.map(|(_, owner)| *owner) == current {
                    Operation::Insert(new)
                } else {
                    Operation::Abort(())
                }
            });
            !matches!(result, Compute::Aborted(_))
        }
        fn load(&self, j: usize) -> Option<Snapshot<RowT, CF>> {
            self.published[j].load_full()
        }
        fn store(&self, j: usize, snapshot: Snapshot<RowT, CF>) {
            self.published[j].store(Some(snapshot));
        }
    }
}

// ====== Reduction of a single dimension =========================

/// The column that owns a pivot row
enum Owner<ColT, CF> {
    /// A column of this dimension's columns to reduce, by its local index, with a consistent snapshot
    Published(usize, Snapshot<ColT, CF>),
    /// An apparent pair: column `.0` (which is not one of the columns to reduce, and has `V = e`) with pivot coefficient `.1`
    Apparent(ColT, CF),
}

struct DimensionReducer<'a, M, S, C>
where
    M: HasRowFiltration,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow,
    M::CoefficientField: ColumnCoefficient,
    C: ColumnRepresentation,
{
    boundary: &'a M,
    /// The columns of this dimension that need reducing, in basis order. Position in this vec is the local index.
    cols: Vec<M::ColT>,
    /// The columns cleared in this dimension (keys), which own no pivots
    cleared: &'a ElementMap<M::RowT, M::ColT>,
    state: S,
    columns: &'a C,
    /// Shared by all of the columns reduced in this dimension, across threads
    cache: ColumnCache<M::FiltrationT, M::RowT, M::CoefficientField>,
    _marker: PhantomData<M>,
}

type RCol<M, C> = <C as ColumnRepresentation>::Column<
    <M as HasRowFiltration>::FiltrationT,
    <M as MatrixOracle>::RowT,
    <M as MatrixOracle>::CoefficientField,
>;
type VCol<M> = BHCol<(), <M as MatrixOracle>::ColT, <M as MatrixOracle>::CoefficientField>;

impl<'a, M, S, C> DimensionReducer<'a, M, S, C>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration,
    M::CoefficientField: Invertible + ColumnCoefficient,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow,
    S: Backend<M::RowT, M::CoefficientField>,
    C: ColumnRepresentation,
{
    fn new(
        boundary: &'a M,
        cols: Vec<M::ColT>,
        cleared: &'a ElementMap<M::RowT, M::ColT>,
        columns: &'a C,
        cache_columns: bool,
    ) -> Self {
        Self {
            boundary,
            state: S::new(cols.len()),
            cols,
            cleared,
            columns,
            cache: ColumnCache::new(cache_columns),
            _marker: PhantomData,
        }
    }

    /// Add `mult * D_k` to `r`
    fn add_d_column(&self, r: &mut RCol<M, C>, k: M::ColT, mult: M::CoefficientField) {
        self.columns
            .add_matrix_column(&self.cache, self.boundary, r, k, mult);
    }

    /// State of a column that no-one has touched yet: `R_j = D_j`, `V_j = e_j`
    fn initial_state(&self, j: usize) -> (RCol<M, C>, VCol<M>) {
        let mut r = RCol::<M, C>::default();
        self.add_d_column(&mut r, self.cols[j].clone(), M::CoefficientField::one());
        let mut v = VCol::<M>::default();
        v.add_tuple((M::CoefficientField::one(), self.cols[j].clone(), ()));
        (r, v)
    }

    /// Reconstruct the working state of column `j` from a published snapshot (recomputing `R = D V`)
    fn state_from(
        &self,
        snapshot: &PublishedColumn<M::ColT, M::CoefficientField>,
        j: usize,
    ) -> (RCol<M, C>, VCol<M>) {
        let label = &self.cols[j];
        let mut r = RCol::<M, C>::default();
        for (coeff, k) in v_entries(snapshot, label) {
            self.add_d_column(&mut r, k, coeff);
        }
        let mut v = VCol::<M>::default();
        v.add_tuples(v_entries(snapshot, label).map(|(coeff, k)| (coeff, k, ())));
        (r, v)
    }

    /// Publish the current state of column `j`. Must be called *before* the pivot table is updated to point at `j`.
    fn publish(&self, j: usize, v: &mut VCol<M>, pivot: Option<(M::RowT, M::CoefficientField)>) {
        // Consolidate (cancels duplicates) and snapshot in one go
        let entries: Vec<_> = v.drain_sorted().collect();
        v.add_entries(entries.iter().cloned());
        // V_j always has 1 on the diagonal, so a single entry means V_j = e_j: store that as empty
        let v_snapshot = if entries.len() == 1 {
            vec![]
        } else {
            entries
                .into_iter()
                .map(|entry| (entry.coeff, entry.row_index))
                .collect()
        };
        self.state.store(
            j,
            Arc::new(PublishedColumn {
                v: v_snapshot,
                pivot,
            }),
        );
    }

    /// Publish column `j` with `V_j = e_j` and the given pivot of `D_j`
    fn publish_trivial(&self, j: usize, pivot: Option<(M::RowT, M::CoefficientField)>) {
        self.state
            .store(j, Arc::new(PublishedColumn { v: vec![], pivot }));
    }

    /// Find the column owning pivot row `l` (with a consistent snapshot, if it is one of ours).
    /// Retries if the column has since changed pivot (see the paper's `Algorithm 3`).
    fn get_owner(&self, l: &M::RowT) -> Option<Owner<M::ColT, M::CoefficientField>> {
        loop {
            let Some(owner) = self.state.pivot_owner(l) else {
                return self.apparent_owner(l);
            };
            let snapshot = self
                .state
                .load(owner)
                .expect("A column is always published before it is made a pivot owner");
            if snapshot.pivot.as_ref().map(|(row, _)| row) == Some(l) {
                return Some(Owner::Published(owner, snapshot));
            }
        }
    }

    /// If `l` is the pivot of the first column in its row then that column owns it (an apparent pair).
    /// Apparent pairs are never published, so we recompute them.
    fn apparent_owner(&self, l: &M::RowT) -> Option<Owner<M::ColT, M::CoefficientField>> {
        let first = self.boundary.first_column_in_row(l.clone())?;
        // A cleared column has R = 0 so owns nothing (this can't happen for a chain complex, but costs little to check)
        if self.cleared.contains_key(&first) {
            return None;
        }
        let pivot = self.boundary.column_pivot(first.clone())?;
        (pivot.row_index == *l).then_some(Owner::Apparent(first, pivot.coeff))
    }

    /// Reduce column `j` (which is not cleared, zero or an apparent pair)
    fn reduce_column(&self, j: usize) {
        // Most columns need no reduction: their pivot is not yet claimed.
        // Try to finish those without building the whole column.
        match self.boundary.column_pivot(self.cols[j].clone()) {
            None => {
                self.publish_trivial(j, None);
                return;
            }
            Some(entry) if self.get_owner(&entry.row_index).is_none() => {
                self.publish_trivial(j, Some((entry.row_index.clone(), entry.coeff)));
                if self.state.compare_set_pivot(&entry.row_index, None, j) {
                    return;
                }
                // Someone else claimed the pivot in the meantime, so reduce as usual
            }
            Some(_) => {}
        }

        let (mut r, mut v) = self.initial_state(j);
        // The column we are currently working on, which changes when we steal a pivot
        let mut w = j;
        loop {
            let Some(pivot_entry) = r.clone_pivot() else {
                // Reduced to zero: a cycle
                self.publish(w, &mut v, None);
                return;
            };
            let l = pivot_entry.row_index;
            let coeff = pivot_entry.coeff;

            match self.get_owner(&l) {
                None => {
                    self.publish(w, &mut v, Some((l.clone(), coeff)));
                    if self.state.compare_set_pivot(&l, None, w) {
                        return;
                    }
                    // Someone beat us to it, our local state is still what we published so just go round again
                }
                Some(Owner::Apparent(owner, owner_coeff)) => {
                    // The first column in row l is always to our left, and has V = e
                    let multiple = coeff.additive_inverse() * owner_coeff.mult_inverse();
                    self.add_d_column(&mut r, owner.clone(), multiple);
                    v.add_tuple((multiple, owner, ()));
                }
                Some(Owner::Published(owner, snapshot)) if owner < w => {
                    let owner_coeff = snapshot.pivot.as_ref().unwrap().1;
                    let multiple = coeff.additive_inverse() * owner_coeff.mult_inverse();
                    let label = &self.cols[owner];
                    for (c, k) in v_entries(&snapshot, label) {
                        self.add_d_column(&mut r, k, c * multiple);
                    }
                    v.add_entries(
                        v_entries(&snapshot, label)
                            .map(|(c, k)| ColumnEntry::from((c * multiple, k, ()))),
                    );
                }
                Some(Owner::Published(owner, snapshot)) if owner > w => {
                    self.publish(w, &mut v, Some((l.clone(), coeff)));
                    if self.state.compare_set_pivot(&l, Some(owner), w) {
                        // We now own the pivot and the displaced column is ours to reduce further
                        w = owner;
                        (r, v) = self.state_from(&snapshot, owner);
                    }
                }
                Some(Owner::Published(..)) => {
                    unreachable!("A column cannot own the pivot of the column being reduced")
                }
            }
        }
    }
}

/// How the columns of a dimension were dealt with before any reduction, collected per thread
struct Classified<C> {
    /// Columns in this dimension cleared by the previous one: `(column, its V = R of this column)`
    cleared: Vec<(C, C)>,
    /// Zero columns (so essential, with `V = e`)
    zero: Vec<C>,
    /// Apparent pairs `(pivot row, column, whether to record it in the diagram)`
    apparent: Vec<(C, C, bool)>,
    /// Positions in the basis of the columns that need reducing
    to_reduce: Vec<usize>,
}

impl<C> Default for Classified<C> {
    fn default() -> Self {
        Self {
            cleared: vec![],
            zero: vec![],
            apparent: vec![],
            to_reduce: vec![],
        }
    }
}

impl<C> Classified<C> {
    fn merge(mut self, mut other: Self) -> Self {
        self.cleared.append(&mut other.cleared);
        self.zero.append(&mut other.zero);
        self.apparent.append(&mut other.apparent);
        self.to_reduce.append(&mut other.to_reduce);
        self
    }
}

// ====== Public interface ========================================

/// Which concurrent data structures [`ParallelClearing`] uses to share the pivot table and published columns between threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Storage {
    /// A [`DashMap`] for pivots and a [`RwLock`] around each published column, all held only briefly.
    #[default]
    Locking,
    /// A [`papaya`](https://docs.rs/papaya) map for pivots and [`arc_swap`](https://docs.rs/arc-swap) slots for columns: no thread ever waits on a lock.
    /// In our benchmarks this was no faster than [`Locking`](Storage::Locking).
    #[cfg(feature = "lock-free")]
    LockFree,
}

/// The [`Clearing`] algorithm with the columns of each dimension reduced in parallel, see the [module docs](self).
/// Obtain one from [`Clearing::parallel`] and configure it with [`in_pool`](Self::in_pool) and [`storage`](Self::storage).
///
/// The diagram is the same as [`Clearing`]'s, but the reduction matrix `V` that comes out may differ (it is still a valid one).
#[derive(Debug, Clone, Default)]
pub struct ParallelClearing<'p, C = MergedColumns> {
    clearing: Clearing<C>,
    pool: Option<&'p rayon::ThreadPool>,
    storage: Storage,
}

impl<C: ColumnRepresentation> Clearing<C> {
    /// Reduce the columns of each dimension in parallel, on rayon's global thread pool and with [`Storage::Locking`].
    pub fn parallel(self) -> ParallelClearing<'static, C> {
        ParallelClearing {
            clearing: self,
            pool: None,
            storage: Storage::default(),
        }
    }
}

impl<'p, C: ColumnRepresentation> ParallelClearing<'p, C> {
    /// Run on the given thread pool rather than rayon's global one.
    pub fn in_pool<'q>(self, pool: &'q rayon::ThreadPool) -> ParallelClearing<'q, C> {
        ParallelClearing {
            clearing: self.clearing,
            pool: Some(pool),
            storage: self.storage,
        }
    }

    /// See [`Clearing::skip_zero_apparent_pairs`].
    pub fn skip_zero_apparent_pairs(mut self, skip: bool) -> Self {
        self.clearing = self.clearing.skip_zero_apparent_pairs(skip);
        self
    }

    /// See [`Clearing::cache_columns`].
    pub fn cache_columns(mut self, cache: bool) -> Self {
        self.clearing = self.clearing.cache_columns(cache);
        self
    }

    /// See [`Clearing::with_columns`].
    pub fn with_columns<C2: ColumnRepresentation>(self, columns: C2) -> ParallelClearing<'p, C2> {
        ParallelClearing {
            clearing: self.clearing.with_columns(columns),
            pool: self.pool,
            storage: self.storage,
        }
    }

    /// Choose the concurrent data structures to use.
    pub fn storage(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }
}

impl<M, C> ReductionAlgorithm<M> for ParallelClearing<'_, C>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync + Send,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::RowT: ColumnRow + 'static,
    M::FiltrationT: ColumnFiltration,
    C: ColumnRepresentation,
{
    type V = ClearedReductionMatrix<'static, M>;

    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT> {
        let dimension_order = self.clearing.dimension_order(&boundary);
        let skip_zero_apparent_pairs = self.clearing.skip_zero_apparent_pairs;
        let columns = self.clearing.columns;
        let cache_columns = self.clearing.cache_columns;
        let run = move || {
            let (v, diagram) = match self.storage {
                Storage::Locking => build_with_diagram_using::<M, LockingBackend<_, _>, C>(
                    boundary,
                    dimension_order,
                    skip_zero_apparent_pairs,
                    &columns,
                    cache_columns,
                ),
                #[cfg(feature = "lock-free")]
                Storage::LockFree => build_with_diagram_using::<M, LockFreeBackend<_, _>, C>(
                    boundary,
                    dimension_order,
                    skip_zero_apparent_pairs,
                    &columns,
                    cache_columns,
                ),
            };
            Reduction { v, diagram }
        };
        match self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }
}

/// Reduce all of the given dimensions, one after the other, each in parallel
fn build_with_diagram_using<M, S, C>(
    boundary: M,
    dimension_order: Vec<usize>,
    skip_zero_apparent_pairs: bool,
    columns: &C,
    cache_columns: bool,
) -> (ClearedReductionMatrix<'static, M>, Diagram<M::ColT>)
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync,
    M::CoefficientField: Invertible + ColumnCoefficient,
    M::RowT: ColumnRow,
    M::FiltrationT: ColumnFiltration,
    S: Backend<M::RowT, M::CoefficientField>,
    C: ColumnRepresentation,
{
    let mut reduction_columns: ElementMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>> =
        ElementMap::new();
    let mut diagram = DiagramBuilder::default();
    // The pairs `pivot row -> column` found in the previous dimension, whose rows are the columns to clear in this one
    let mut previous_pairs: ElementMap<M::RowT, M::ColT> = ElementMap::new();
    // Births of skipped (zero-persistence apparent) pairs that weren't cleared, and so may be wrongly marked essential.
    // Only ever non-empty when dimensions are reduced against the direction of the matrix.
    let mut skipped_births: ElementSet<M::RowT> = ElementSet::default();
    let n_dims = dimension_order.len();

    for (position, dim) in dimension_order.into_iter().enumerate() {
        let is_last = position + 1 == n_dims;
        // The pairs of the previous dimension: their rows are the columns to clear in this one
        let clearing = std::mem::take(&mut previous_pairs);

        // 1. Classify every column in parallel: cleared, zero, apparent, or needing reduction.
        //    Only the last get any per-column state, and there are usually very few of them.
        let basis_size = boundary.sub_basis(dim).size();
        let classified = (0..basis_size)
            .into_par_iter()
            .with_min_len(MIN_CLASSIFY_LEN)
            .fold(Classified::default, |mut acc, i| {
                let col = boundary.sub_basis(dim).element(i);
                if let Some(birth) = clearing.get(&col) {
                    acc.cleared.push((col, birth.clone()));
                    return acc;
                }
                match boundary.column_pivot(col.clone()) {
                    None => acc.zero.push(col),
                    Some(entry)
                        if boundary
                            .first_column_in_row(entry.row_index.clone())
                            .as_ref()
                            == Some(&col) =>
                    {
                        let record = !(skip_zero_apparent_pairs
                            && boundary.filtration_value(entry.row_index.clone())
                                == boundary.filtration_value(col.clone()));
                        // Unrecorded pairs are still needed to clear the next dimension, and to un-mark their birth
                        // as essential if its dimension was reduced earlier (against the direction of the matrix)
                        if record || !is_last || diagram.essential.contains(&entry.row_index) {
                            acc.apparent.push((entry.row_index, col, record));
                        }
                    }
                    Some(_) => acc.to_reduce.push(i),
                }
                acc
            })
            .reduce(Classified::default, Classified::merge);
        let Classified {
            cleared,
            zero,
            apparent,
            mut to_reduce,
        } = classified;
        for (col, birth) in cleared {
            reduction_columns.insert(col, ReductionColumn::Cleared(birth));
        }
        for col in zero {
            diagram.essential.insert(col);
        }
        for (pivot_row, col, record) in apparent {
            if !is_last {
                previous_pairs.insert(pivot_row.clone(), col.clone());
            }
            if record {
                diagram.pairings.push((pivot_row, col));
            } else {
                diagram.essential.remove(&pivot_row);
            }
        }

        // 2. Reduce the remaining columns in parallel (in basis order, for the pivot-stealing logic)
        to_reduce.par_sort_unstable();
        let cols: Vec<M::ColT> = to_reduce
            .into_iter()
            .map(|i| boundary.sub_basis(dim).element(i))
            .collect();
        let reducer =
            DimensionReducer::<M, S, C>::new(&boundary, cols, &clearing, columns, cache_columns);
        (0..reducer.cols.len())
            .into_par_iter()
            .with_max_len(MAX_TASK_LEN)
            .for_each(|j| reducer.reduce_column(j));

        // Rows paired in the previous dimension that weren't columns of this one aren't needed for clearing.
        // But if their pair was skipped, the row's column may be reduced in a later dimension and marked essential.
        // (When dimensions are visited in the direction of the matrix there are no such rows.)
        if skip_zero_apparent_pairs {
            for (row, col) in clearing.iter() {
                let was_cleared = matches!(
                    reduction_columns.get(&row),
                    Some(ReductionColumn::Cleared(_))
                );
                if !was_cleared
                    && boundary.filtration_value(row.clone())
                        == boundary.filtration_value(col.clone())
                {
                    skipped_births.insert(row);
                }
            }
        }

        // 3. Read off the results
        for (j, label) in reducer.cols.iter().enumerate() {
            let snapshot = reducer
                .state
                .load(j)
                .expect("Every column is published by the end of reduction");
            if let Some((pivot_row, _)) = &snapshot.pivot {
                if !is_last {
                    previous_pairs.insert(pivot_row.clone(), label.clone());
                }
                diagram.pairings.push((pivot_row.clone(), label.clone()));
            } else {
                diagram.essential.insert(label.clone());
            }
            // We don't store the diagonal of V, nor V at all when it is trivial
            if !snapshot.v.is_empty() {
                reduction_columns.insert(
                    label.clone(),
                    ReductionColumn::Reduced(
                        snapshot
                            .v
                            .iter()
                            .filter(|(_, k)| k != label)
                            .cloned()
                            .collect(),
                    ),
                );
            }
        }
    }

    let v = ClearedReductionMatrix::build_from_owned(boundary, reduction_columns);
    (v, diagram.finish(Some(&skipped_births)))
}
