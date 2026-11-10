//! A parallel, label-keyed variant of the clearing algorithm, adapted from
//! [Morozov and Nigmetov, *Towards Lockfree Persistent Homology*](https://doi.org/10.1145/3350755.3400244).
//!
//! Run it with `Clearing::new().parallel().reduce(matrix)`, see [`ParallelClearing`].
//!
//! Compared to the serial [`Clearing`] this reduces all (uncleared) columns of a
//! dimension concurrently, using the same out-of-order scheme as the paper:
//! a thread reduces a private copy of a column and, whenever it finds a pivot owned by a column to its *right*,
//! it publishes its own column, steals the pivot and carries on reducing the displaced column.
//! By the pairing uniqueness lemma the resulting diagram is the same as the serial algorithm's
//! (the reduction matrix `V` that comes out may differ, but is still a valid one).
//!
//! # Differences from the integer-indexed algorithm
//!
//! * **Rows are never indexed.**
//!   The pivot table is a concurrent hash map keyed by the row *label* ([`RowT`](MatrixOracle::RowT)),
//!   which only ever holds rows that are the pivot of some column (so at most one entry per column).
//!   Row order comes from the usual `(filtration value, label)` order on column entries.
//!   This means we never enumerate the row space, which is the main saving of cohomology.
//! * **Columns are indexed densely per dimension** by their position in the (uncleared part of the) column basis.
//!   That is exactly the order the serial algorithm sweeps in, and it's all the algorithm needs from the columns.
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

use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use dashmap::{mapref::entry::Entry, DashMap};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use super::clearing::{ClearedReductionMatrix, ReductionColumn};
use super::Diagram;
use super::{Clearing, Reduction, ReductionAlgorithm};
use crate::columns::{BHCol, ColumnEntry};
use crate::fields::{Invertible, NonZeroCoefficient};
use crate::matrices::{ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle, SquareMatrix};

/// Largest block of consecutive columns handed to a single rayon task.
/// Columns vary a lot in cost, so fine-grained tasks let rayon balance the load dynamically.
const MAX_TASK_LEN: usize = 16;

/// `FxHasher` followed by a finaliser: Fx puts its entropy in the high bits, but the concurrent maps also use the low ones.
#[derive(Default)]
pub struct MixedFx(FxHasher);

impl Hasher for MixedFx {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        self.0.write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i)
    }
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i)
    }
    fn finish(&self) -> u64 {
        let h = self.0.finish();
        let h = (h ^ (h >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^ (h >> 32)
    }
}

type FxBuild = BuildHasherDefault<MixedFx>;

// ====== Published columns =======================================

/// An immutable snapshot of a column: `V_j` (including its diagonal) and the pivot of `R_j = D V_j`.
struct PublishedColumn<RowT, CF> {
    /// Entries `(coefficient, dimension-local column index)`.
    /// Empty when `V_j = e_j` (the commonest case), so as not to allocate; use [`v_entries`].
    v: Vec<(CF, usize)>,
    /// Row label and coefficient of the pivot of `R_j`, or `None` if `R_j = 0`
    pivot: Option<(RowT, CF)>,
}

type Snapshot<RowT, CF> = Arc<PublishedColumn<RowT, CF>>;

/// The entries of `V_j` from its published snapshot (expanding the empty, trivial case to `e_j`).
fn v_entries<RowT, CF: NonZeroCoefficient>(
    snapshot: &PublishedColumn<RowT, CF>,
    j: usize,
) -> impl Iterator<Item = (CF, usize)> + '_ {
    let trivial = snapshot.v.is_empty().then(|| (CF::one(), j));
    trivial.into_iter().chain(snapshot.v.iter().copied())
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

struct DimensionReducer<'a, M: HasRowFiltration, S> {
    boundary: &'a M,
    /// The uncleared columns of this dimension, in basis order. Position in this vec is the local index.
    cols: Vec<M::ColT>,
    state: S,
    _marker: PhantomData<M>,
}

type RCol<M> = BHCol<
    <M as HasRowFiltration>::FiltrationT,
    <M as MatrixOracle>::RowT,
    <M as MatrixOracle>::CoefficientField,
>;
type VCol<M> = BHCol<(), usize, <M as MatrixOracle>::CoefficientField>;

impl<'a, M, S> DimensionReducer<'a, M, S>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
    S: Backend<M::RowT, M::CoefficientField>,
{
    fn new(boundary: &'a M, cols: Vec<M::ColT>) -> Self {
        Self {
            boundary,
            state: S::new(cols.len()),
            cols,
            _marker: PhantomData,
        }
    }

    /// The entries of `mult * D V` for the column `V` given as `(coefficient, local index)` pairs.
    fn d_times_v<'b>(
        &'b self,
        v: impl Iterator<Item = (M::CoefficientField, usize)> + 'b,
        mult: M::CoefficientField,
    ) -> impl Iterator<Item = ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>> + 'b {
        v.flat_map(move |(coeff, k)| {
            let scale = coeff * mult;
            self.boundary
                .column_with_filtration(self.cols[k].clone())
                .map(move |entry| entry * scale)
        })
    }

    /// State of a column that no-one has touched yet: `R_j = D_j`, `V_j = e_j`
    fn initial_state(&self, j: usize) -> (RCol<M>, VCol<M>) {
        let r = self.boundary.build_bhcol(self.cols[j].clone());
        let mut v = VCol::<M>::default();
        v.add_tuple((M::CoefficientField::one(), j, ()));
        (r, v)
    }

    /// Reconstruct the working state of a column from a published snapshot (recomputing `R = D V`)
    fn state_from(
        &self,
        snapshot: &PublishedColumn<M::RowT, M::CoefficientField>,
        j: usize,
    ) -> (RCol<M>, VCol<M>) {
        let mut r = self.boundary.empty_bhcol();
        r.add_entries(self.d_times_v(v_entries(snapshot, j), M::CoefficientField::one()));
        let mut v = VCol::<M>::default();
        v.add_tuples(v_entries(snapshot, j).map(|(coeff, k)| (coeff, k, ())));
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

    /// Find the column owning pivot row `l`, together with a consistent snapshot of it.
    /// Retries if the column has since changed pivot (see the paper's `Algorithm 3`).
    #[allow(clippy::type_complexity)]
    fn get_owner(&self, l: &M::RowT) -> Option<(usize, Snapshot<M::RowT, M::CoefficientField>)> {
        loop {
            let owner = self.state.pivot_owner(l)?;
            let snapshot = self
                .state
                .load(owner)
                .expect("A column is always published before it is made a pivot owner");
            if snapshot.pivot.as_ref().map(|(row, _)| row) == Some(l) {
                return Some((owner, snapshot));
            }
        }
    }

    /// Publish column `j` with `V_j = e_j` and the given pivot of `D_j`
    fn publish_trivial(&self, j: usize, pivot: Option<(M::RowT, M::CoefficientField)>) {
        self.state
            .store(j, Arc::new(PublishedColumn { v: vec![], pivot }));
    }

    fn reduce_column(&self, j: usize) {
        // Most columns need no reduction: their pivot is not yet claimed (or they are zero).
        // Try to finish those without building the whole column.
        match self.boundary.column_pivot(self.cols[j].clone()) {
            None => {
                self.publish_trivial(j, None);
                return;
            }
            Some(entry) if self.state.pivot_owner(&entry.row_index).is_none() => {
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
                Some((owner, snapshot)) if owner < w => {
                    let owner_coeff = snapshot.pivot.as_ref().unwrap().1;
                    let multiple = coeff.additive_inverse() * owner_coeff.mult_inverse();
                    r.add_entries(self.d_times_v(v_entries(&snapshot, owner), multiple));
                    v.add_entries(
                        v_entries(&snapshot, owner)
                            .map(|(c, k)| ColumnEntry::from((c * multiple, k, ()))),
                    );
                }
                Some((owner, snapshot)) if owner > w => {
                    self.publish(w, &mut v, Some((l.clone(), coeff)));
                    if self.state.compare_set_pivot(&l, Some(owner), w) {
                        // We now own the pivot and the displaced column is ours to reduce further
                        w = owner;
                        (r, v) = self.state_from(&snapshot, owner);
                    }
                }
                Some(_) => {
                    unreachable!("A column cannot own the pivot of the column being reduced")
                }
            }
        }
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
pub struct ParallelClearing<'p> {
    clearing: Clearing,
    pool: Option<&'p rayon::ThreadPool>,
    storage: Storage,
}

impl Clearing {
    /// Reduce the columns of each dimension in parallel, on rayon's global thread pool and with [`Storage::Locking`].
    pub fn parallel(self) -> ParallelClearing<'static> {
        ParallelClearing {
            clearing: self,
            pool: None,
            storage: Storage::default(),
        }
    }
}

impl<'p> ParallelClearing<'p> {
    /// Run on the given thread pool rather than rayon's global one.
    pub fn in_pool<'q>(self, pool: &'q rayon::ThreadPool) -> ParallelClearing<'q> {
        ParallelClearing {
            clearing: self.clearing,
            pool: Some(pool),
            storage: self.storage,
        }
    }

    /// Choose the concurrent data structures to use.
    pub fn storage(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }
}

impl<M> ReductionAlgorithm<M> for ParallelClearing<'_>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync + Send,
    M::CoefficientField: Invertible + Send + Sync + 'static,
    M::ColT: Hash + Send + Sync,
    M::RowT: Hash + Send + Sync + 'static,
{
    type V = ClearedReductionMatrix<'static, M>;

    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT> {
        let dimension_order = self.clearing.dimension_order(&boundary);
        let run = move || {
            let (v, diagram) = match self.storage {
                Storage::Locking => {
                    build_with_diagram_using::<M, LockingBackend<_, _>>(boundary, dimension_order)
                }
                #[cfg(feature = "lock-free")]
                Storage::LockFree => {
                    build_with_diagram_using::<M, LockFreeBackend<_, _>>(boundary, dimension_order)
                }
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
fn build_with_diagram_using<M, S>(
    boundary: M,
    dimension_order: Vec<usize>,
) -> (ClearedReductionMatrix<'static, M>, Diagram<M::ColT>)
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync,
    M::CoefficientField: Invertible + Send + Sync,
    M::ColT: Hash + Send + Sync,
    M::RowT: Hash + Send + Sync,
    S: Backend<M::RowT, M::CoefficientField>,
{
    let mut reduction_columns: FxHashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>> =
        FxHashMap::default();
    let mut essential = FxHashSet::default();
    let mut pairings = FxHashSet::default();

    for dim in dimension_order {
        // Collect the columns that haven't been cleared
        let cols: Vec<M::ColT> = {
            let basis = boundary.sub_basis(dim);
            (0..basis.size())
                .map(|i| basis.element(i))
                .filter(|elem| !reduction_columns.contains_key(elem))
                .collect()
        };

        let reducer = DimensionReducer::<M, S>::new(&boundary, cols);
        (0..reducer.cols.len())
            .into_par_iter()
            .with_max_len(MAX_TASK_LEN)
            .for_each(|j| reducer.reduce_column(j));

        // Read off the results sequentially
        for (j, label) in reducer.cols.iter().enumerate() {
            let snapshot = reducer
                .state
                .load(j)
                .expect("Every column is published by the end of reduction");
            if let Some((pivot_row, _)) = &snapshot.pivot {
                pairings.insert((pivot_row.clone(), label.clone()));
                // Clear the column labelled by the pivot, its V column is the R column of this one
                reduction_columns
                    .insert(pivot_row.clone(), ReductionColumn::Cleared(label.clone()));
            } else {
                essential.insert(label.clone());
            }
            // We don't store the diagonal of V, nor V at all when it is trivial
            if !snapshot.v.is_empty() {
                reduction_columns.insert(
                    label.clone(),
                    ReductionColumn::Reduced(
                        snapshot
                            .v
                            .iter()
                            .filter(|&&(_, k)| k != j)
                            .map(|&(coeff, k)| (coeff, reducer.cols[k].clone()))
                            .collect(),
                    ),
                );
            }
        }
    }

    let v = ClearedReductionMatrix::build_from_owned(boundary, reduction_columns);
    let mut diagram = Diagram {
        essential,
        pairings,
    };
    diagram.remove_births_from_essential();
    (v, diagram)
}
