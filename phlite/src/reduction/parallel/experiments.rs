//! Every pivot table and every kind of slot for [`ParallelClearing`]'s shared state, to be benchmarked against each
//! other with [`reduce_with`] (feature `storage-experiments`, not part of the stable API).
//!
//! This is a separate function rather than a choice of [`Storage`](super::Storage), so that the sixteen pairings are
//! only compiled for the matrices it is called with, not for every matrix every pipeline reduces in parallel.

use super::storage::{
    ArcSwapSlots, DashTable, MutexShardsTable, MutexSlots, PapayaTable, RetainingSlots, RwLockSlots, SccTable,
};
use super::{build_with_diagram_using, ParallelClearing};
use crate::columns::{ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow};
use crate::fields::Invertible;
use crate::matrices::{HasRowFiltration, HasSplitColBasis, MatrixOracle, SquareMatrix};
use crate::reduction::{clearing::ClearedReductionMatrix, Reduction};

/// The concurrent map from pivot rows to the columns owning them
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotTableKind {
    /// `dashmap`: shards behind read–write locks (as [`Storage::Locking`](super::Storage::Locking))
    DashMap,
    /// `papaya`: lock-free (as `Storage::LockFree`)
    Papaya,
    /// `scc`: lock-free reads, writes lock one bucket
    Scc,
    /// 64 `FxHashMap`s, each behind a `Mutex`
    MutexShards,
}

/// How each column's latest snapshot is held
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    /// An `Arc` behind a `RwLock` (as [`Storage::Locking`](super::Storage::Locking))
    RwLock,
    /// An `Arc` behind a `Mutex`
    Mutex,
    /// An `ArcSwapOption` (as `Storage::LockFree`)
    ArcSwap,
    /// An atomic pointer to a snapshot kept until the dimension is done (no reference counting)
    Retaining,
}

impl PivotTableKind {
    pub const ALL: [Self; 4] = [Self::DashMap, Self::Papaya, Self::Scc, Self::MutexShards];
}

impl SlotKind {
    pub const ALL: [Self; 4] = [Self::RwLock, Self::Mutex, Self::ArcSwap, Self::Retaining];
}

/// Reduce `boundary` as `parallel` would (its pool, columns and options), but with the pivot table `table` and the
/// slots `slots` for its shared state.
pub fn reduce_with<M, C>(
    parallel: ParallelClearing<'_, C>,
    boundary: M,
    table: PivotTableKind,
    slots: SlotKind,
) -> Reduction<ClearedReductionMatrix<'static, M>, M::ColT>
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync + Send,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::RowT: ColumnRow + 'static,
    M::FiltrationT: ColumnFiltration,
    C: ColumnRepresentation,
{
    let dimension_order = parallel.clearing.dimension_order(&boundary);
    let skip_zero_apparent_pairs = parallel.clearing.skip_zero_apparent_pairs;
    let columns = parallel.clearing.columns;
    let cache_columns = parallel.clearing.cache_columns;
    let run = move || {
        macro_rules! with {
            ($table:ident, $slots:ident) => {
                build_with_diagram_using::<M, $table<_>, $slots<_>, C>(
                    boundary,
                    dimension_order,
                    skip_zero_apparent_pairs,
                    &columns,
                    cache_columns,
                )
            };
        }
        use PivotTableKind as T;
        use SlotKind as S;
        let (v, diagram) = match (table, slots) {
            (T::DashMap, S::RwLock) => with!(DashTable, RwLockSlots),
            (T::DashMap, S::Mutex) => with!(DashTable, MutexSlots),
            (T::DashMap, S::ArcSwap) => with!(DashTable, ArcSwapSlots),
            (T::DashMap, S::Retaining) => with!(DashTable, RetainingSlots),
            (T::Papaya, S::RwLock) => with!(PapayaTable, RwLockSlots),
            (T::Papaya, S::Mutex) => with!(PapayaTable, MutexSlots),
            (T::Papaya, S::ArcSwap) => with!(PapayaTable, ArcSwapSlots),
            (T::Papaya, S::Retaining) => with!(PapayaTable, RetainingSlots),
            (T::Scc, S::RwLock) => with!(SccTable, RwLockSlots),
            (T::Scc, S::Mutex) => with!(SccTable, MutexSlots),
            (T::Scc, S::ArcSwap) => with!(SccTable, ArcSwapSlots),
            (T::Scc, S::Retaining) => with!(SccTable, RetainingSlots),
            (T::MutexShards, S::RwLock) => with!(MutexShardsTable, RwLockSlots),
            (T::MutexShards, S::Mutex) => with!(MutexShardsTable, MutexSlots),
            (T::MutexShards, S::ArcSwap) => with!(MutexShardsTable, ArcSwapSlots),
            (T::MutexShards, S::Retaining) => with!(MutexShardsTable, RetainingSlots),
        };
        Reduction { v, diagram }
    };
    match parallel.pool {
        Some(pool) => pool.install(run),
        None => run(),
    }
}
