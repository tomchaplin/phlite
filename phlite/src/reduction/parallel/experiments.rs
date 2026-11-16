//! Every pivot table and every kind of slot for [`ParallelClearing`](super::ParallelClearing)'s shared state, to be
//! benchmarked against each other with `Storage::Custom` (feature `storage-experiments`, not part of the stable API).

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
