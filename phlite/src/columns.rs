//! Representations of matrix columns as linear combinations with ordered terms, whose leading term (the pivot) can be extracted.
//!
//! * [`BHCol`] stores every term in a heap.
//! * [`MergeCol`] lazily merges sorted runs of terms ([`SortedRun`]s), keeping one heap slot per run.
//!
//! The reduction algorithms are generic over how they represent the column `R_i = D V_i` that they are reducing,
//! via the [`ColumnRepresentation`] trait. The default, [`MergedColumns`], uses [`MergeCol`]s and [`HeapColumns`] uses [`BHCol`]s.
//!
//! Most of the content of this module is an implementation detail of the reduction algorithms and is likely to change in the future.
//! Unless you wish to access pivots of matrices yourself, or to plug a different column representation into the
//! reduction algorithms, you probably don't need this module.
use crate::{
    fields::NonZeroCoefficient,
    hash::FxBuild,
    matrices::{BasisElement, FiltrationValue, HasRowFiltration, SquareMatrix},
};
use dashmap::DashMap;
use std::{fmt::Debug, hash::Hash, iter::repeat, ops::Mul};

/// The heap behind [`BHCol`]: a 4-ary heap, which has half the depth of a binary heap and keeps each node's children
/// together in memory, so large columns (whose heaps don't fit in cache) need fewer memory accesses per push and pop.
type Heap<T> = dary_heap::QuaternaryHeap<T>;

#[derive(Clone, Copy)]
/// Represents an entry in a column of a matrix (or alternatively a term in a linear combination).
///
/// This is stored as the row index of the entry, together with the coefficient of that term.
/// The filtration value of the row is also stored so that [`ColumnEntry`]s can be efficiently sorted without having to recompute filtration values.
pub struct ColumnEntry<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// The filtration value corresponding to the row index of the column entry.
    pub filtration_value: FilT,
    /// The row index of this entry in the column/sum.
    pub row_index: RowT,
    /// The coefficient of this term in the sum/entry in the matrix.
    pub coeff: CF,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Debug for ColumnEntry<FilT, RowT, CF>
where
    FilT: Debug,
    RowT: Debug,
    CF: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "({:?} * {:?}) :: {:?}",
            self.coeff, self.row_index, self.filtration_value
        ))
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> From<(CF, RowT, FilT)>
    for ColumnEntry<FilT, RowT, CF>
{
    fn from((coeff, row_index, filtration_value): (CF, RowT, FilT)) -> Self {
        Self {
            filtration_value,
            row_index,
            coeff,
        }
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> From<ColumnEntry<FilT, RowT, CF>>
    for (CF, RowT, FilT)
{
    fn from(entry: ColumnEntry<FilT, RowT, CF>) -> Self {
        (entry.coeff, entry.row_index, entry.filtration_value)
    }
}

/// <div class="warning">
///
/// Equality only checks row index - to check correct coefficient and filtration value, convert to tuple.
///
/// </div>
impl<FilT: FiltrationValue, RowT: BasisElement, CF> PartialEq for ColumnEntry<FilT, RowT, CF> {
    // Equal row index implies equal filtration value
    fn eq(&self, other: &Self) -> bool {
        self.row_index.eq(&other.row_index)
    }
}
impl<FilT: FiltrationValue, RowT: BasisElement, CF> Eq for ColumnEntry<FilT, RowT, CF> {}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> PartialOrd for ColumnEntry<FilT, RowT, CF> {
    // Order by filtration value and then order on RowT
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Ord for ColumnEntry<FilT, RowT, CF> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.filtration_value, &self.row_index).cmp(&(&other.filtration_value, &other.row_index))
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient> Mul<CF>
    for ColumnEntry<FilT, RowT, CF>
{
    type Output = Self;

    fn mul(self, rhs: CF) -> Self::Output {
        ColumnEntry {
            coeff: self.coeff * rhs,
            filtration_value: self.filtration_value,
            row_index: self.row_index,
        }
    }
}

#[derive(Clone)]
/// Represents a column in a matrix where each [`ColumnEntry`] is stored in a heap (a 4-ary heap, see `dary_heap`).
/// The order of terms in this heap is determined by the filtration value and then the row index.
///
/// Since the same row index my appear more than once in the heap, it is more accurate to say that this represents a linear combination of elements in the row basis.
/// Terms are stored in a heap to allow efficient access to the leading term, otherwise known as the column pivot.
///
/// One should typically construct a [`BHCol`] by calling [`build_bhcol`](crate::matrices::HasRowFiltration::build_bhcol) or [`empty_bhcol`](crate::matrices::HasRowFiltration::empty_bhcol).
/// To add columns into the heap, one should probably call [`column_with_filtration`](crate::matrices::HasRowFiltration::column_with_filtration) and pass the resulting iterator to [`add_entries`](BHCol::add_entries).
/// Note, your matrix must implement [`HasRowFiltration`] in order to do this.
/// Consider using [`with_trivial_filtration`](crate::matrices::MatrixOracle::with_trivial_filtration) if you have no filtration that you care about (e.g. you are computing non-persistent homology).
pub struct BHCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    heap: Heap<ColumnEntry<FilT, RowT, CF>>,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Debug for BHCol<FilT, RowT, CF>
where
    ColumnEntry<FilT, RowT, CF>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(&self.heap).finish()
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Default for BHCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self {
            heap: Heap::default(),
        }
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> BHCol<FilT, RowT, CF> {
    /// Add a single entry to the heap.
    /// Note this __will not__ automatically coalesce terms with the same row index.
    pub fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        self.heap.push(entry);
    }

    /// Add each of the entries in the provided iterator to the binary heap.
    /// Note this __will not__ automatically coalesce terms with the same row index.
    /// If you can provide a helpful size hint to the iterator, that will be used.
    pub fn add_entries(&mut self, entries: impl IntoIterator<Item = ColumnEntry<FilT, RowT, CF>>) {
        let entries = entries.into_iter();
        let (lower_bound, _) = entries.size_hint();
        self.heap.reserve(lower_bound);
        // Internal iteration (rather than a `for` loop) lets chained/flattened iterators, which oracles often return,
        // run their own efficient fold instead of re-checking every adaptor on each `next`
        entries.for_each(|entry| self.add_entry(entry));
    }

    /// Convert tuple into [`ColumnEntry`]s and then pass to [`add_entry`](BHCol::add_entry).
    pub fn add_tuple(&mut self, tuple: (CF, RowT, FilT)) {
        self.add_entry(tuple.into());
    }

    /// Convert tuples into [`ColumnEntry`]s and then pass to [`add_entries`](BHCol::add_entries).
    pub fn add_tuples(&mut self, tuples: impl IntoIterator<Item = (CF, RowT, FilT)>) {
        self.add_entries(tuples.into_iter().map(Into::into));
    }

    /// This will return an iterator that iteratively pulls out the leading non-zero term of the linear combination.
    /// This __will__ coalesce terms with the same row index into a single term - if the coefficient becomes 0 then that term is omitted.
    /// Each time you pull an item out of this iterator, you will be removed all terms with that row index from `self`.
    pub fn drain_sorted(&mut self) -> impl Iterator<Item = ColumnEntry<FilT, RowT, CF>> + '_
    where
        CF: NonZeroCoefficient,
    {
        repeat(()).map_while(|()| self.pop_pivot())
    }

    /// Collect the output of [`drain_sorted`](BHCol::drain_sorted) into a [`Vec`].
    pub fn to_sorted_vec(mut self) -> Vec<ColumnEntry<FilT, RowT, CF>>
    where
        CF: NonZeroCoefficient,
    {
        self.drain_sorted().collect()
    }

    /// Remove and return the leading order non-zero term in the sum (i.e. the column pivot).
    /// This __will__ coalesce terms with the same row index into a single term - if the coefficient becomes 0 then that term is omitted and the next index is considered.
    pub fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>>
    where
        CF: NonZeroCoefficient,
    {
        // Pull out first entry
        let first_entry = self.heap.pop()?;
        let mut working_index: RowT = first_entry.row_index;
        let mut working_sum: Option<CF> = Some(first_entry.coeff);
        let mut working_filtration = first_entry.filtration_value;

        loop {
            // No more elements, break and report pivot
            let Some(next_entry) = self.heap.peek() else {
                break;
            };

            // Check if next index is different
            if next_entry.row_index != working_index {
                if working_sum.is_some() {
                    // Found the largest index with non-zero coefficent report
                    break;
                }
                // Otherwise we prepare to start adding the next largest index
                working_index = next_entry.row_index.clone();
                working_sum = None;
                working_filtration = next_entry.filtration_value.clone();
            }

            // Actually remove from heap
            let next_entry = self.heap.pop().expect("If None would have broke earlier");
            working_sum = next_entry.coeff + working_sum;
        }

        working_sum.map(|coeff| ColumnEntry {
            row_index: working_index,
            filtration_value: working_filtration,
            coeff,
        })
    }

    /// Essentially [`pop_pivot`](Self::pop_pivot) and then push a clone of the pivot back onto the heap.
    pub fn clone_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>>
    where
        ColumnEntry<FilT, RowT, CF>: Clone,
        CF: NonZeroCoefficient,
    {
        let pivot = self.pop_pivot();
        if let Some(pivot) = pivot {
            let ret = Some(pivot.clone());
            self.add_entry(pivot);
            ret
        } else {
            None
        }
    }

    /// Peek at the top of the binary heap (presuambly to figure out the column pivot).
    /// Note this may not correspond to the column pivot because there may be multiple elements in the sum with the same index.
    /// Moreover, these terms may sum to `0` and hence even the row index at the top of the heap may not correspond to the row index of the pivot.
    ///
    /// <div class="warning">
    ///
    /// This only yields the column pivot if previously called [`clone_pivot`](BHCol::clone_pivot) or [`add_entry`](BHCol::add_entry) to push on the new pivot (or you are lucky).
    ///
    /// </div>
    pub fn peek_pivot(&self) -> Option<&ColumnEntry<FilT, RowT, CF>> {
        self.heap.peek()
    }
}

/// A sorted, consolidated column (or linear combination) stored contiguously, largest entry first,
/// with each row appearing at most once.
/// Shared so that the same column can be added into many [`MergeCol`]s without copying it.
pub type SortedRun<FilT, RowT, CF> = std::sync::Arc<[ColumnEntry<FilT, RowT, CF>]>;

/// Sort `entries` into a [`SortedRun`], largest first, summing the coefficients of repeated rows (and dropping those that cancel).
pub fn sorted_run<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient>(
    entries: impl IntoIterator<Item = ColumnEntry<FilT, RowT, CF>>,
) -> SortedRun<FilT, RowT, CF> {
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_unstable_by(|a, b| b.cmp(a));
    let mut consolidated: Vec<ColumnEntry<FilT, RowT, CF>> = Vec::with_capacity(entries.len());
    let mut entries = entries.into_iter().peekable();
    while let Some(first) = entries.next() {
        let mut sum = Some(first.coeff);
        while let Some(next) = entries.next_if(|next| next.row_index == first.row_index) {
            sum = next.coeff + sum;
        }
        if let Some(coeff) = sum {
            consolidated.push(ColumnEntry { coeff, ..first });
        }
    }
    consolidated.into()
}

/// A position in a [`SortedRun`] being merged into a [`MergeCol`], ordered by its current `head` entry.
struct Cursor<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// The current entry of the run, already multiplied by `multiple`
    head: ColumnEntry<FilT, RowT, CF>,
    /// The rest of the run, or `None` for a single entry
    run: Option<SortedRun<FilT, RowT, CF>>,
    /// Index into `run` of the entry after `head`
    next: usize,
    multiple: CF,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> PartialEq for Cursor<FilT, RowT, CF> {
    fn eq(&self, other: &Self) -> bool {
        self.head.eq(&other.head)
    }
}
impl<FilT: FiltrationValue, RowT: BasisElement, CF> Eq for Cursor<FilT, RowT, CF> {}
impl<FilT: FiltrationValue, RowT: BasisElement, CF> PartialOrd for Cursor<FilT, RowT, CF> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<FilT: FiltrationValue, RowT: BasisElement, CF> Ord for Cursor<FilT, RowT, CF> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.head.cmp(&other.head)
    }
}

/// A linear combination of [`SortedRun`]s, whose terms are produced in descending order by lazily merging the runs.
///
/// This plays the same role as [`BHCol`], but the heap holds one cursor per summand rather than one slot per entry.
/// When columns are long, the heap of a [`BHCol`] can grow to millions of entries, while this heap stays small enough to
/// remain in cache. Moreover entries below the eventual pivot are never touched.
pub struct MergeCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    heap: Heap<Cursor<FilT, RowT, CF>>,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Default for MergeCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self {
            heap: Heap::default(),
        }
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient> MergeCol<FilT, RowT, CF> {
    /// Add `multiple` times the entries of `run` from index `skip` onwards.
    pub fn add_run(&mut self, run: SortedRun<FilT, RowT, CF>, multiple: CF, skip: usize) {
        let Some(head) = run.get(skip).cloned() else {
            return;
        };
        self.heap.push(Cursor {
            head: head * multiple,
            run: Some(run),
            next: skip + 1,
            multiple,
        });
    }

    /// Add a single entry.
    pub fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        self.heap.push(Cursor {
            head: entry,
            run: None,
            next: 0,
            multiple: CF::one(),
        });
    }

    /// Remove and return the leading non-zero term of the combination (i.e. the column pivot),
    /// summing all terms with the same row index (and moving on to the next row if they cancel).
    pub fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        loop {
            let top = self.heap.peek_mut()?;
            let pivot = top.head.clone();
            let mut sum = Some(pivot.coeff);
            Self::advance(top);
            while let Some(top) = self.heap.peek_mut() {
                if top.head.row_index != pivot.row_index {
                    break;
                }
                sum = top.head.coeff + sum;
                Self::advance(top);
            }
            if let Some(coeff) = sum {
                return Some(ColumnEntry { coeff, ..pivot });
            }
        }
    }

    /// Move the top cursor onto the next entry of its run, removing it if the run is exhausted.
    fn advance(mut top: dary_heap::PeekMut<'_, Cursor<FilT, RowT, CF>, 4>) {
        if let Some(entry) = top.run.as_ref().and_then(|run| run.get(top.next)).cloned() {
            let multiple = top.multiple;
            top.head = entry * multiple;
            top.next += 1;
        } else {
            dary_heap::PeekMut::pop(top);
        }
    }

    /// [`pop_pivot`](Self::pop_pivot) and then put (a copy of) the pivot back, so that the column is unchanged.
    pub fn clone_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        let pivot = self.pop_pivot()?;
        self.add_entry(pivot.clone());
        Some(pivot)
    }

    /// Pull out every remaining non-zero term, in descending order.
    pub fn drain_sorted(&mut self) -> impl Iterator<Item = ColumnEntry<FilT, RowT, CF>> + '_ {
        repeat(()).map_while(|()| self.pop_pivot())
    }
}

// ====== Column representations for the reduction algorithms ======

/// A working column being reduced: a linear combination of rows whose leading term (the pivot) can be extracted.
pub trait WorkingColumn<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// Add a single term. This __need not__ coalesce terms with the same row index.
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>);

    /// Remove and return the leading non-zero term (i.e. the column pivot), summing all terms with the same row index
    /// (and moving on to the next row if they cancel). Returns `None` if the column is zero.
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>>;

    /// [`pop_pivot`](Self::pop_pivot) and then put (a copy of) the pivot back, so that the column is unchanged.
    fn clone_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>>
    where
        ColumnEntry<FilT, RowT, CF>: Clone,
    {
        let pivot = self.pop_pivot()?;
        self.add_entry(pivot.clone());
        Some(pivot)
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient>
    WorkingColumn<FilT, RowT, CF> for BHCol<FilT, RowT, CF>
{
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        BHCol::add_entry(self, entry);
    }
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        BHCol::pop_pivot(self)
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient>
    WorkingColumn<FilT, RowT, CF> for MergeCol<FilT, RowT, CF>
{
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        MergeCol::add_entry(self, entry);
    }
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        MergeCol::pop_pivot(self)
    }
}

/// Filtration values usable by every [`ColumnRepresentation`] (blanket implemented).
pub trait ColumnFiltration: FiltrationValue + Send + Sync {}
impl<T: FiltrationValue + Send + Sync> ColumnFiltration for T {}
/// Row indices usable by every [`ColumnRepresentation`] (blanket implemented).
pub trait ColumnRow: BasisElement + Hash + Send + Sync {}
impl<T: BasisElement + Hash + Send + Sync> ColumnRow for T {}
/// Coefficients usable by every [`ColumnRepresentation`] (blanket implemented).
pub trait ColumnCoefficient: NonZeroCoefficient + Send + Sync {}
impl<T: NonZeroCoefficient + Send + Sync> ColumnCoefficient for T {}

/// How the reduction algorithms represent the column `R_i = D V_i` that they are reducing.
///
/// Every algorithm is generic over this (defaulting to [`MergedColumns`]), so that a different representation can be
/// plugged in, for example to compare against [`HeapColumns`]:
/// ```
/// # use phlite::{columns::HeapColumns, reduction::{Clearing, Standard}};
/// let clearing = Clearing::new().with_columns(HeapColumns);
/// let standard = Standard::new().with_columns(HeapColumns);
/// ```
///
/// A representation is a cheap value holding only its configuration.
/// Each column is a [`Column`](Self::Column), which only ever changes by having a term or a multiple of a column of `D` added to it,
/// see [`add_matrix_column`](Self::add_matrix_column). That method can use a [`Cache`](Self::Cache), which the algorithms create
/// once per dimension and share between all of the columns reduced in that dimension (across threads, for the parallel algorithm).
pub trait ColumnRepresentation: Debug + Clone + Default + Send + Sync {
    /// The working column.
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>: WorkingColumn<FilT, RowT, CF>
        + Default
        + Send;

    /// State shared by all columns reduced in one dimension, such as a cache of columns of `D`.
    type Cache<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>: Default
        + Send
        + Sync;

    /// Add `multiple` times column `k` of `matrix` to `column`.
    fn add_matrix_column<M>(
        &self,
        cache: &Self::Cache<M::FiltrationT, M::RowT, M::CoefficientField>,
        matrix: &M,
        column: &mut Self::Column<M::FiltrationT, M::RowT, M::CoefficientField>,
        k: M::ColT,
        multiple: M::CoefficientField,
    ) where
        M: HasRowFiltration + SquareMatrix,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient;
}

/// Represent columns as [`BHCol`]s: every term of every column added is pushed into a heap.
///
/// This is simple and needs no extra memory, but when the columns being added are long the heap can grow to millions of
/// terms, so that it no longer fits in cache.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeapColumns;

impl ColumnRepresentation for HeapColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        BHCol<FilT, RowT, CF>;
    type Cache<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> = ();

    fn add_matrix_column<M>(
        &self,
        _cache: &(),
        matrix: &M,
        column: &mut BHCol<M::FiltrationT, M::RowT, M::CoefficientField>,
        k: M::ColT,
        multiple: M::CoefficientField,
    ) where
        M: HasRowFiltration + SquareMatrix,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient,
    {
        column.add_entries(
            matrix
                .column_with_filtration(k)
                .map(|entry| entry * multiple),
        );
    }
}

/// Represent columns as [`MergeCol`]s (the default): each column of `D` added is a sorted run, merged lazily.
///
/// Each column of `D` is sorted the first time it is needed in a dimension and then kept (in the [`Cache`](ColumnRepresentation::Cache))
/// until the end of that dimension, so is never recomputed.
/// The heap holds one slot per column added rather than one per term, so it stays small enough to remain in cache,
/// and terms below the eventual pivot are never touched.
/// This is much faster when reduced columns are long, at the cost of keeping the sorted columns of `D` in memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergedColumns;

/// The sorted columns of `D` needed so far in a dimension, keyed by column.
pub struct SortedColumnCache<FilT: FiltrationValue, RowT: BasisElement + Hash, CF>(
    DashMap<RowT, SortedRun<FilT, RowT, CF>, FxBuild>,
);

impl<FilT: FiltrationValue, RowT: BasisElement + Hash, CF> Default
    for SortedColumnCache<FilT, RowT, CF>
{
    fn default() -> Self {
        Self(DashMap::default())
    }
}

impl ColumnRepresentation for MergedColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        MergeCol<FilT, RowT, CF>;
    type Cache<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        SortedColumnCache<FilT, RowT, CF>;

    fn add_matrix_column<M>(
        &self,
        cache: &SortedColumnCache<M::FiltrationT, M::RowT, M::CoefficientField>,
        matrix: &M,
        column: &mut MergeCol<M::FiltrationT, M::RowT, M::CoefficientField>,
        k: M::ColT,
        multiple: M::CoefficientField,
    ) where
        M: HasRowFiltration + SquareMatrix,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient,
    {
        let run = match cache.0.get(&k) {
            Some(run) => run.clone(),
            None => {
                // Sort outside of the map's lock (so two threads may sort the same column, which is harmless)
                let run = sorted_run(matrix.column_with_filtration(k.clone()));
                cache.0.entry(k).or_insert(run).clone()
            }
        };
        column.add_run(run, multiple, 0);
    }
}
