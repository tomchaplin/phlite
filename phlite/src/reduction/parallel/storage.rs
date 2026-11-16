//! The concurrent state shared by the threads reducing a dimension: a **pivot table** (which column owns each pivot
//! row) and a **slot** per column holding its latest published snapshot.
//!
//! The two halves are independent, so a [`Backend`] is any [`PivotTable`] with any [`Slots`]. The public
//! [`Storage`](super::Storage) choices are particular pairs; with the `storage-experiments` feature every pair can be
//! picked (see [`super::experiments`]), which is how they were benchmarked against each other.

use std::hash::Hash;
use std::ops::Deref;
use std::sync::{Arc, RwLock};
#[cfg(feature = "storage-experiments")]
use std::sync::Mutex;

use dashmap::{mapref::entry::Entry, DashMap};

use crate::hash::FxBuild;

/// Which column owns each pivot row, as a dimension-local column index.
pub(super) trait PivotTable<RowT>: Sync {
    /// Room for `n` entries: a row is only in the table while it is some column's pivot, so there are never more
    fn with_capacity(n: usize) -> Self;
    /// The column that currently owns `row`
    fn owner(&self, row: &RowT) -> Option<usize>;
    /// Atomically set the owner of `row` to `new` if it is currently `current`. Returns whether this happened.
    fn compare_set(&self, row: &RowT, current: Option<usize>, new: usize) -> bool;
}

/// One slot per column, holding the latest snapshot published for it.
pub(super) trait Slots<T>: Sync {
    /// A handle on a loaded snapshot, which stays valid however the slot changes afterwards
    type Ref<'a>: Deref<Target = T>
    where
        Self: 'a,
        T: 'a;
    fn new(n: usize) -> Self;
    fn load(&self, j: usize) -> Option<Self::Ref<'_>>;
    fn store(&self, j: usize, snapshot: T);
}

/// A pivot table and the published snapshots: everything the threads of a dimension share.
pub(super) struct Backend<P, L> {
    pub(super) pivots: P,
    pub(super) published: L,
}

impl<P, L> Backend<P, L> {
    pub(super) fn new<RowT, T>(n_columns: usize) -> Self
    where
        P: PivotTable<RowT>,
        L: Slots<T>,
    {
        Self {
            pivots: P::with_capacity(n_columns),
            published: L::new(n_columns),
        }
    }
}

// ====== Pivot tables ============================================

/// [`DashMap`]: a hash map sharded behind read–write locks
pub(super) struct DashTable<RowT>(DashMap<RowT, usize, FxBuild>);

impl<RowT: Hash + Eq + Clone + Send + Sync> PivotTable<RowT> for DashTable<RowT> {
    fn with_capacity(n: usize) -> Self {
        Self(DashMap::with_capacity_and_hasher(n, FxBuild::default()))
    }
    fn owner(&self, row: &RowT) -> Option<usize> {
        self.0.get(row).map(|owner| *owner)
    }
    fn compare_set(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
        match self.0.entry(row.clone()) {
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
}

#[cfg(feature = "lock-free")]
pub(super) use lock_free::{ArcSwapSlots, PapayaTable};

#[cfg(feature = "lock-free")]
mod lock_free {
    use super::*;
    use arc_swap::ArcSwapOption;
    use papaya::{Compute, Operation};

    /// A [`papaya::HashMap`], updated with its atomic [`compute`](papaya::HashMap::compute) (a compare-and-swap loop).
    /// It is made with room for every pivot, so never resizes, the one operation `papaya` doesn't promise is lock-free.
    pub(in super::super) struct PapayaTable<RowT>(papaya::HashMap<RowT, usize, FxBuild>);

    impl<RowT: Hash + Eq + Clone + Send + Sync> PivotTable<RowT> for PapayaTable<RowT> {
        fn with_capacity(n: usize) -> Self {
            Self(papaya::HashMap::with_capacity_and_hasher(n, FxBuild::default()))
        }
        fn owner(&self, row: &RowT) -> Option<usize> {
            self.0.pin().get(row).copied()
        }
        fn compare_set(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
            let pivots = self.0.pin();
            let result = pivots.compute(row.clone(), |entry| {
                if entry.map(|(_, owner)| *owner) == current {
                    Operation::Insert(new)
                } else {
                    Operation::Abort(())
                }
            });
            !matches!(result, Compute::Aborted(_))
        }
    }

    /// An [`ArcSwapOption`] per column, whose loads and stores never block
    pub(in super::super) struct ArcSwapSlots<T>(Vec<ArcSwapOption<T>>);

    impl<T: Send + Sync> Slots<T> for ArcSwapSlots<T> {
        type Ref<'a>
            = Arc<T>
        where
            T: 'a;
        fn new(n: usize) -> Self {
            Self((0..n).map(|_| ArcSwapOption::empty()).collect())
        }
        fn load(&self, j: usize) -> Option<Arc<T>> {
            self.0[j].load_full()
        }
        fn store(&self, j: usize, snapshot: T) {
            self.0[j].store(Some(Arc::new(snapshot)));
        }
    }
}

/// A [`scc::HashMap`]: lock-free reads, and writes locking a single bucket
#[cfg(feature = "storage-experiments")]
pub(super) struct SccTable<RowT>(scc::HashMap<RowT, usize, FxBuild>);

#[cfg(feature = "storage-experiments")]
impl<RowT: Hash + Eq + Clone + Send + Sync> PivotTable<RowT> for SccTable<RowT> {
    fn with_capacity(n: usize) -> Self {
        Self(scc::HashMap::with_capacity_and_hasher(n, FxBuild::default()))
    }
    fn owner(&self, row: &RowT) -> Option<usize> {
        self.0.read_sync(row, |_, owner| *owner)
    }
    fn compare_set(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
        match self.0.entry_sync(row.clone()) {
            scc::hash_map::Entry::Occupied(mut entry) if current == Some(*entry.get()) => {
                *entry.get_mut() = new;
                true
            }
            scc::hash_map::Entry::Vacant(entry) if current.is_none() => {
                entry.insert_entry(new);
                true
            }
            _ => false,
        }
    }
}

/// A fixed number of plain hash maps, each behind a [`Mutex`]
#[cfg(feature = "storage-experiments")]
pub(super) struct MutexShardsTable<RowT> {
    shards: Box<[Mutex<rustc_hash::FxHashMap<RowT, usize>>]>,
    hasher: FxBuild,
}

#[cfg(feature = "storage-experiments")]
impl<RowT: Hash + Eq> MutexShardsTable<RowT> {
    const SHARDS: usize = 64;
    fn shard(&self, row: &RowT) -> std::sync::MutexGuard<'_, rustc_hash::FxHashMap<RowT, usize>> {
        use std::hash::BuildHasher;
        // The high bits: the low ones also pick the bucket within the shard
        let hash = self.hasher.hash_one(row);
        let shard = (hash >> 58) as usize % Self::SHARDS;
        self.shards[shard].lock().expect("no thread panicked holding a shard")
    }
}

#[cfg(feature = "storage-experiments")]
impl<RowT: Hash + Eq + Clone + Send + Sync> PivotTable<RowT> for MutexShardsTable<RowT> {
    fn with_capacity(n: usize) -> Self {
        let per_shard = n.div_ceil(Self::SHARDS);
        Self {
            shards: (0..Self::SHARDS)
                .map(|_| {
                    Mutex::new(rustc_hash::FxHashMap::with_capacity_and_hasher(
                        per_shard,
                        Default::default(),
                    ))
                })
                .collect(),
            hasher: FxBuild::default(),
        }
    }
    fn owner(&self, row: &RowT) -> Option<usize> {
        self.shard(row).get(row).copied()
    }
    fn compare_set(&self, row: &RowT, current: Option<usize>, new: usize) -> bool {
        let mut shard = self.shard(row);
        if shard.get(row).copied() == current {
            shard.insert(row.clone(), new);
            true
        } else {
            false
        }
    }
}

// ====== Slots ===================================================

/// An [`Arc`] behind a [`RwLock`] per column
pub(super) struct RwLockSlots<T>(Vec<RwLock<Option<Arc<T>>>>);

impl<T: Send + Sync> Slots<T> for RwLockSlots<T> {
    type Ref<'a>
        = Arc<T>
    where
        T: 'a;
    fn new(n: usize) -> Self {
        Self((0..n).map(|_| RwLock::new(None)).collect())
    }
    fn load(&self, j: usize) -> Option<Arc<T>> {
        // A poisoned lock means another thread panicked mid-store: that panic is already propagating
        self.0[j].read().expect("no thread panicked while publishing").clone()
    }
    fn store(&self, j: usize, snapshot: T) {
        *self.0[j].write().expect("no thread panicked while publishing") = Some(Arc::new(snapshot));
    }
}

/// An [`Arc`] behind a [`Mutex`] per column
#[cfg(feature = "storage-experiments")]
pub(super) struct MutexSlots<T>(Vec<Mutex<Option<Arc<T>>>>);

#[cfg(feature = "storage-experiments")]
impl<T: Send + Sync> Slots<T> for MutexSlots<T> {
    type Ref<'a>
        = Arc<T>
    where
        T: 'a;
    fn new(n: usize) -> Self {
        Self((0..n).map(|_| Mutex::new(None)).collect())
    }
    fn load(&self, j: usize) -> Option<Arc<T>> {
        self.0[j].lock().expect("no thread panicked while publishing").clone()
    }
    fn store(&self, j: usize, snapshot: T) {
        *self.0[j].lock().expect("no thread panicked while publishing") = Some(Arc::new(snapshot));
    }
}

/// A pointer per column, swapped atomically, to a snapshot that is **never freed before the slots are**.
///
/// The snapshots of a dimension are only needed while it is being reduced, so rather than count references (as an
/// [`Arc`] does, which makes every load write to memory shared with other threads) or track who might still be
/// reading (as epoch-based schemes do), superseded snapshots are kept on a lock-free list and freed together when the
/// dimension is done. Loads are then a single atomic read, and a loaded snapshot is a plain reference. The cost is
/// memory: every snapshot published in the dimension lives until its end (columns are typically published a few
/// times each).
#[cfg(feature = "storage-experiments")]
pub(super) struct RetainingSlots<T> {
    slots: Box<[std::sync::atomic::AtomicPtr<Node<T>>]>,
    /// Snapshots displaced from their slot, as a stack linked through [`Node::next`] (only ever pushed to)
    retired: std::sync::atomic::AtomicPtr<Node<T>>,
}

#[cfg(feature = "storage-experiments")]
pub(super) struct Node<T> {
    value: T,
    next: std::sync::atomic::AtomicPtr<Node<T>>,
}

/// SAFETY: the slots own their `T`s, which are only ever shared by reference
#[cfg(feature = "storage-experiments")]
unsafe impl<T: Send + Sync> Sync for RetainingSlots<T> {}
#[cfg(feature = "storage-experiments")]
unsafe impl<T: Send> Send for RetainingSlots<T> {}

#[cfg(feature = "storage-experiments")]
impl<T: Send + Sync> Slots<T> for RetainingSlots<T> {
    type Ref<'a>
        = &'a T
    where
        T: 'a;
    fn new(n: usize) -> Self {
        use std::sync::atomic::AtomicPtr;
        Self {
            slots: (0..n).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect(),
            retired: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
    fn load(&self, j: usize) -> Option<&T> {
        use std::sync::atomic::Ordering;
        let node = self.slots[j].load(Ordering::Acquire);
        // SAFETY: a non-null pointer came from `Box::into_raw` in `store` (published by its Release swap, which this
        // Acquire load synchronises with), and is only freed in `drop`, which can't run while `self` is borrowed
        unsafe { node.as_ref() }.map(|node| &node.value)
    }
    fn store(&self, j: usize, snapshot: T) {
        use std::sync::atomic::{AtomicPtr, Ordering};
        let node = Box::into_raw(Box::new(Node {
            value: snapshot,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }));
        let old = self.slots[j].swap(node, Ordering::AcqRel);
        if old.is_null() {
            return;
        }
        // Retire the displaced snapshot (readers may still hold it). Only this thread got `old` from the swap, so only
        // it writes `old.next`; a push-only stack has no ABA problem.
        // SAFETY: `old` is live (see `load`)
        let old_next = unsafe { &(*old).next };
        let mut head = self.retired.load(Ordering::Relaxed);
        loop {
            old_next.store(head, Ordering::Relaxed);
            match self
                .retired
                .compare_exchange_weak(head, old, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => head = current,
            }
        }
    }
}

#[cfg(feature = "storage-experiments")]
impl<T> Drop for RetainingSlots<T> {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        // SAFETY: we have `&mut self`, so no snapshot is borrowed; each node is in exactly one slot or on the stack
        for slot in self.slots.iter() {
            let node = slot.load(Ordering::Acquire);
            if !node.is_null() {
                drop(unsafe { Box::from_raw(node) });
            }
        }
        let mut node = self.retired.load(Ordering::Acquire);
        while !node.is_null() {
            let boxed = unsafe { Box::from_raw(node) };
            node = boxed.next.load(Ordering::Relaxed);
        }
    }
}

#[cfg(all(test, feature = "storage-experiments"))]
mod tests {
    use super::*;

    #[test]
    fn retaining_slots_keep_superseded_snapshots_alive() {
        let slots = RetainingSlots::<Vec<u32>>::new(3);
        assert!(slots.load(0).is_none());
        slots.store(1, vec![1]);
        let first = slots.load(1).unwrap();
        std::thread::scope(|scope| {
            for t in 0..4u32 {
                let slots = &slots;
                scope.spawn(move || {
                    for i in 0..1000 {
                        slots.store(1, vec![t, i]);
                        assert_eq!(slots.load(1).unwrap().len(), 2);
                    }
                });
            }
        });
        // The first snapshot is still readable after 4000 replacements
        assert_eq!(*first, vec![1]);
        assert_eq!(slots.load(1).unwrap().len(), 2);
        assert!(slots.load(2).is_none());
    }
}
