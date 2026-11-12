//! Representations of matrix columns as linear combinations with ordered terms, whose leading term (the pivot) can be extracted.
//!
//! The reduction algorithms reduce one column `R_i = D V_i` at a time: they repeatedly find its pivot (its largest
//! term, by filtration value and then row), look for an earlier column with the same pivot, and if there is one, add a
//! multiple of that column `R_j = D V_j` to cancel it. Since only `V` is stored, adding `R_j` means adding each column of
//! `D` that makes it up. So a working column needs two operations: *add a column of `D`*, and *pop the pivot*. How it
//! does them is its [`ColumnRepresentation`], chosen with `with_columns` on the algorithm
//! (e.g. [`Clearing::with_columns`](crate::reduction::Clearing::with_columns)). Every representation gives the same
//! result; only the time and memory differ.
//!
//! # The representations
//!
//! **[`MergedColumns`] (the default)** never builds the working column. It keeps a list of the columns of `D` added so
//! far, each sorted largest first, and a small heap with one cursor into each. Popping the pivot takes the largest
//! cursor, sums in any other cursors on the same row, and moves them along. Adding a column pushes one cursor.
//! Nothing below the pivot is ever read, and the heap has one slot per column added rather than one per term, so it
//! stays small and in cache.
//! * Good when reduced columns get long, or the columns added have many terms far below the pivot: it is the only
//!   representation that does well on every matrix we have tried, which is why it is the default.
//! * Each column of `D` must be sorted before it is added (see [caching](#caching-columns-of-d) and
//!   [`sorted_column`](crate::matrices::HasRowFiltration::sorted_column)), and each step of the pivot search is a heap
//!   operation.
//!
//! **[`HeapColumns`]** pushes every term of every column added onto one heap ([`BHCol`]), and pops terms off the top,
//! summing those with the same row. This is the classic approach.
//! * Good when the columns of `D` are short and few are added per column: there is no sorting at all, and a push or
//!   pop on a small heap is cheap. On Vietoris–Rips cohomology it matches [`MergedColumns`], with less memory.
//! * Bad when many terms are added: the heap holds every term, cancelled or not, including all those below the
//!   eventual pivot, and once it outgrows the cache every operation slows down.
//!
//! **[`SortedColumns`]** builds the working column as one vector, sorted ascending so the pivot is the last entry.
//! Columns added are collected unsorted, then at the next pivot query sorted together (they are usually few and
//! short) and merged into the vector, starting from their smallest entry: entries of the vector below that stay put,
//! and the rest are rewritten. Popping the pivot is then just taking the last entry.
//! * Good when the columns added stay close to the pivot, so each merge only rewrites the top of the vector. This is
//!   typical of homology (boundary matrices) of simplicial and cell complexes, where it is the fastest representation
//!   (about 1.5× faster than [`MergedColumns`] on Rips homology), and of small dense matrices.
//! * Bad, sometimes very bad, when the columns added reach far below the pivot: each addition then rewrites nearly the
//!   whole working column, so the cost grows with its length. On GrPPH and on the cohomology of cubical complexes
//!   it is 15–35× slower than [`MergedColumns`]. Only use it when you have measured it on your matrices.
//!
//! **[`PivotColumns`]** keeps the working column as a dense array with a slot for every row of the matrix, and a
//! 64-ary tree of bitsets saying which slots are in use, as PHAT's default `bit_tree_pivot_column` does. Adding an
//! entry flips a bit (equal entries cancel at once), and the pivot is the highest set bit, found with a few bit-scans.
//! It needs the rank of every row in filtration order, from
//! [`row_ranks`](crate::matrices::HasRowFiltration::row_ranks), and falls back to [`MergedColumns`] for matrices
//! without it.
//! * Good when many columns are added to each, as for explicit dense or cubical complexes, especially with
//!   [`store_r`](crate::reduction::Clearing::store_r): on those it is 1.4–5× faster than [`MergedColumns`], and it is
//!   the default for explicit complexes in `phlite_py`.
//! * Every entry added is visited (none are skipped lazily below the pivot), and the column costs a slot per row of the
//!   matrix, allocated once per reduction. Only the serial clearing algorithm reuses one column, so the parallel and
//!   standard algorithms fall back to [`MergedColumns`].
//!
//! # Caching columns of `D`
//!
//! Separately from the representation, the algorithms can keep each column of `D` the first time it is added, sorted,
//! in a [`ColumnCache`] shared by every column of that dimension (`cache_columns` on the algorithm, e.g.
//! [`Clearing::cache_columns`](crate::reduction::Clearing::cache_columns); on by default). Later additions of the same
//! column then neither recompute nor re-sort it.
//! * Worth it when columns are expensive to compute: on GrPPH, whose coboundary enumerates path cofaces of a digraph,
//!   caching makes [`MergedColumns`] about 3× faster.
//! * Not worth it when columns are cheap: the cache holds every column needed in a dimension until the dimension is
//!   done, which for Vietoris–Rips is most of the memory the reduction uses (23 MB rather than 5 MB for 400 points),
//!   and looking a column up costs about as much as recomputing it.
//! * Less useful when the matrix stores its columns sorted and says so, via
//!   [`sorted_column`](crate::matrices::HasRowFiltration::sorted_column) (as for explicit matrices in `phlite_py`):
//!   columns are then never sorted, only copied into the working column, and the cache saves just that copy. That
//!   helps when columns are long (about 10% on cell complexes) and costs a little when they are short (about 20% on
//!   cubical complexes, where looking a column up costs more than copying it).
//!
//! # Choosing
//!
//! Start with the default. If memory matters and your columns are cheap to compute, try turning caching off, and
//! [`HeapColumns`] if columns of `D` are short. If your matrix can give [`row_ranks`](crate::matrices::HasRowFiltration::row_ranks),
//! use [`PivotColumns`] with [`store_r`](crate::reduction::Clearing::store_r), and caching off if it gives
//! [`sorted_column`](crate::matrices::HasRowFiltration::sorted_column)s. The benchmarks in the repository
//! (`python-tests/BENCHMARKS.md`) compare every combination on several kinds of matrix.
//!
//! # The column types
//!
//! * [`BHCol`] stores every term in a heap.
//! * [`MergeCol`] lazily merges sorted runs of terms, keeping one heap slot per run.
//! * [`SortedCol`] keeps its terms in a sorted vector.
//! * [`PivotCol`] keeps its terms in a dense array over row ranks ([`BitCol`]), or merges them.
//!
//! Most of the content of this module is an implementation detail of the reduction algorithms and is likely to change in the future.
//! Unless you wish to access pivots of matrices yourself, or to plug a different column representation into the
//! reduction algorithms, you probably don't need this module.
use crate::{
    fields::NonZeroCoefficient,
    hash::FxBuild,
    matrices::{BasisElement, FiltrationValue, HasRowFiltration, RowRanks, SquareMatrix},
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
/// Where the entries after a [`Cursor`]'s head are.
enum Run<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// Nowhere: the cursor is a single entry
    Single,
    /// A run shared with other columns (from a [`ColumnCache`])
    Shared(SortedRun<FilT, RowT, CF>),
    /// The column's own arena, up to (but not including) index `end`
    Arena { end: usize },
}

struct Cursor<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// The current entry of the run, already multiplied by `multiple`
    head: ColumnEntry<FilT, RowT, CF>,
    /// Where the rest of the run is
    run: Run<FilT, RowT, CF>,
    /// Index (into the shared run, or into the column's arena) of the entry after `head`
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

/// A linear combination of sorted runs of terms, whose terms are produced in descending order by lazily merging the runs.
///
/// This plays the same role as [`BHCol`], but the heap holds one cursor per summand rather than one slot per entry.
/// When columns are long, the heap of a [`BHCol`] can grow to millions of entries, while this heap stays small enough to
/// remain in cache. Moreover entries below the eventual pivot are never touched.
///
/// A run is either a [`SortedRun`] shared with other columns (see [`add_run`](Self::add_run)), or copied into this
/// column's own arena (see [`add_sorted`](Self::add_sorted) and [`add_unsorted`](Self::add_unsorted)), which grows as
/// needed and is freed with the column, so adding a run allocates nothing on average.
pub struct MergeCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    heap: Heap<Cursor<FilT, RowT, CF>>,
    arena: Vec<ColumnEntry<FilT, RowT, CF>>,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Default for MergeCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self {
            heap: Heap::default(),
            arena: Vec::new(),
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
            run: Run::Shared(run),
            next: skip + 1,
            multiple,
        });
    }

    /// Add `multiple` times `entries`, which must be largest first with each row at most once.
    pub fn add_sorted(
        &mut self,
        entries: impl IntoIterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        let start = self.arena.len();
        self.arena.extend(entries);
        self.push_arena_run(start, multiple);
    }

    /// Add `multiple` times `entries`, in any order: they are sorted (and repeated rows summed) in the arena.
    pub fn add_unsorted(
        &mut self,
        entries: impl IntoIterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        let start = self.arena.len();
        self.arena.extend(entries);
        let run = &mut self.arena[start..];
        run.sort_unstable_by(|a, b| b.cmp(a));
        // Sum repeated rows in place (they are now adjacent), dropping those that cancel
        let mut write = start;
        let mut read = start;
        while read < self.arena.len() {
            let mut sum = Some(self.arena[read].coeff);
            let mut next = read + 1;
            while next < self.arena.len() && self.arena[next] == self.arena[read] {
                sum = self.arena[next].coeff + sum;
                next += 1;
            }
            if let Some(coeff) = sum {
                self.arena.swap(write, read);
                self.arena[write].coeff = coeff;
                write += 1;
            }
            read = next;
        }
        self.arena.truncate(write);
        self.push_arena_run(start, multiple);
    }

    /// Push a cursor for the run of the arena from `start` to its end.
    fn push_arena_run(&mut self, start: usize, multiple: CF) {
        let Some(head) = self.arena.get(start).cloned() else {
            return;
        };
        self.heap.push(Cursor {
            head: head * multiple,
            run: Run::Arena {
                end: self.arena.len(),
            },
            next: start + 1,
            multiple,
        });
    }

    /// Add a single entry.
    pub fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        self.heap.push(Cursor {
            head: entry,
            run: Run::Single,
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
            Self::advance(top, &self.arena);
            while let Some(top) = self.heap.peek_mut() {
                if top.head.row_index != pivot.row_index {
                    break;
                }
                sum = top.head.coeff + sum;
                Self::advance(top, &self.arena);
            }
            if let Some(coeff) = sum {
                return Some(ColumnEntry { coeff, ..pivot });
            }
        }
    }

    /// Move the top cursor onto the next entry of its run, removing it if the run is exhausted.
    fn advance(
        mut top: dary_heap::PeekMut<'_, Cursor<FilT, RowT, CF>, 4>,
        arena: &[ColumnEntry<FilT, RowT, CF>],
    ) {
        let next = match &top.run {
            Run::Single => None,
            Run::Shared(run) => run.get(top.next),
            Run::Arena { end } => (top.next < *end).then(|| &arena[top.next]),
        };
        if let Some(entry) = next.cloned() {
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

    /// Remove every term, keeping any allocated capacity, so that the column can be reused for the next one.
    fn clear(&mut self);

    /// Remove every remaining non-zero term, largest first (summing terms with the same row), onto the end of `out`.
    fn drain_into(&mut self, out: &mut Vec<ColumnEntry<FilT, RowT, CF>>) {
        while let Some(entry) = self.pop_pivot() {
            out.push(entry);
        }
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
    fn clear(&mut self) {
        self.heap.clear();
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
    fn clear(&mut self) {
        self.heap.clear();
        self.arena.clear();
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
/// see [`add_matrix_column`](Self::add_matrix_column). Columns of `D` come either straight from the matrix, in whatever
/// order it gives them ([`add_entries`](Self::add_entries)), or sorted from a [`ColumnCache`] ([`add_run`](Self::add_run)).
/// Whether to cache is a separate choice, made on the algorithm (e.g. [`Clearing::cache_columns`](crate::reduction::Clearing::cache_columns)).
pub trait ColumnRepresentation: Debug + Clone + Default + Send + Sync {
    /// The working column.
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>: WorkingColumn<FilT, RowT, CF>
        + Default
        + Send;

    /// A new, empty working column for reducing columns of `matrix`. By default [`Default::default`], which needs
    /// nothing from the matrix; [`PivotColumns`] uses the matrix's [`row_ranks`](HasRowFiltration::row_ranks).
    ///
    /// Algorithms that reuse one working column for many columns (clearing, on one thread) create it this way; others
    /// create a [`Default`] column per column.
    fn new_column<M>(
        &self,
        matrix: &M,
    ) -> Self::Column<M::FiltrationT, M::RowT, M::CoefficientField>
    where
        M: HasRowFiltration,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient,
    {
        let _ = matrix;
        Default::default()
    }

    /// Add `multiple` times a column of `D`, given as its entries in any order (with no repeated rows).
    fn add_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut Self::Column<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    );

    /// Add `multiple` times a column of `D`, given largest first with each row at most once (from
    /// [`sorted_column`](HasRowFiltration::sorted_column)). By default the same as [`add_entries`](Self::add_entries).
    fn add_sorted_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut Self::Column<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        self.add_entries(column, entries, multiple);
    }

    /// Add `multiple` times a column of `D`, given sorted (largest first) and consolidated.
    fn add_run<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut Self::Column<FilT, RowT, CF>,
        run: &SortedRun<FilT, RowT, CF>,
        multiple: CF,
    );

    /// Add `multiple` times column `k` of `matrix` to `column`: from `cache` if it is enabled (computing and sorting
    /// the column the first time it is needed), or else straight from the matrix. Either way, if the matrix gives the
    /// column already sorted ([`sorted_column`](HasRowFiltration::sorted_column)) it isn't sorted again.
    fn add_matrix_column<M>(
        &self,
        cache: &ColumnCache<M::FiltrationT, M::RowT, M::CoefficientField>,
        matrix: &M,
        column: &mut Self::Column<M::FiltrationT, M::RowT, M::CoefficientField>,
        k: M::ColT,
        multiple: M::CoefficientField,
    ) where
        M: HasRowFiltration + SquareMatrix,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient,
    {
        match &cache.0 {
            Some(map) => {
                let run = match map.get(&k) {
                    Some(run) => run.clone(),
                    None => {
                        // Sort outside of the map's lock (so two threads may sort the same column, which is harmless)
                        let run = match matrix.sorted_column(k.clone()) {
                            Some(sorted) => sorted.collect(),
                            None => sorted_run(matrix.column_with_filtration(k.clone())),
                        };
                        map.entry(k).or_insert(run).clone()
                    }
                };
                self.add_run(column, &run, multiple);
            }
            None => match matrix.sorted_column(k.clone()) {
                Some(sorted) => self.add_sorted_entries(column, sorted, multiple),
                None => self.add_entries(column, matrix.column_with_filtration(k), multiple),
            },
        }
    }
}

/// The sorted columns of `D` needed so far in a dimension, keyed by column, or nothing if caching is off.
///
/// The algorithms create one per dimension (or one for the whole reduction, for [`Standard`](crate::reduction::Standard)),
/// shared between all of the columns reduced in it, across threads for the parallel algorithm.
/// Caching saves recomputing and re-sorting columns of `D` that are added many times, which pays off when computing a
/// column is expensive, at the cost of memory for every column needed.
pub struct ColumnCache<FilT: FiltrationValue, RowT: BasisElement + Hash, CF>(
    Option<DashMap<RowT, SortedRun<FilT, RowT, CF>, FxBuild>>,
);

impl<FilT: FiltrationValue, RowT: BasisElement + Hash, CF> ColumnCache<FilT, RowT, CF> {
    /// An empty cache, or a disabled one (which caches nothing) if `enabled` is false.
    pub fn new(enabled: bool) -> Self {
        Self(enabled.then(DashMap::default))
    }
}

/// Represent columns as [`BHCol`]s: every term of every column added is pushed into one heap.
///
/// Needs no sorting, so is good when the columns of `D` are short and few are added per column (as for Vietoris–Rips
/// cohomology). But the heap holds every term added, including those below the eventual pivot, so when many are added
/// it can outgrow the cache. See the [module docs](self) for how the representations compare.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeapColumns;

impl ColumnRepresentation for HeapColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        BHCol<FilT, RowT, CF>;

    fn add_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut BHCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        column.add_entries(entries.map(|entry| entry * multiple));
    }

    fn add_run<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut BHCol<FilT, RowT, CF>,
        run: &SortedRun<FilT, RowT, CF>,
        multiple: CF,
    ) {
        column.add_entries(run.iter().map(|entry| entry.clone() * multiple));
    }
}

/// Represent columns as [`MergeCol`]s (the default): each column of `D` added is a sorted run, merged lazily.
///
/// The heap holds one slot per column added rather than one per term, so it stays small enough to remain in cache,
/// and terms below the eventual pivot are never touched. This is much faster when reduced columns are long, and is the
/// only representation that does well on every matrix we have tried.
/// Each column of `D` added must be sorted: with a [`ColumnCache`] that happens once per column and dimension, and
/// without one, every time the column is added, unless the matrix gives it sorted
/// ([`sorted_column`](crate::matrices::HasRowFiltration::sorted_column)). See the [module docs](self) for how the
/// representations compare.
#[derive(Debug, Clone, Copy, Default)]
pub struct MergedColumns;

impl ColumnRepresentation for MergedColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        MergeCol<FilT, RowT, CF>;

    fn add_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut MergeCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        column.add_unsorted(entries, multiple);
    }

    fn add_sorted_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut MergeCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        column.add_sorted(entries, multiple);
    }

    fn add_run<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut MergeCol<FilT, RowT, CF>,
        run: &SortedRun<FilT, RowT, CF>,
        multiple: CF,
    ) {
        column.add_run(run.clone(), multiple, 0);
    }
}

/// A working column stored as a sorted vector, with the terms added since the last pivot query kept unsorted until
/// they are needed.
///
/// `sorted` holds consolidated terms (each row at most once, none zero) in ascending order, so the pivot is the last
/// entry. Adding a term or a column of `D` only appends to `pending`; [`pop_pivot`](Self::pop_pivot) then sorts
/// `pending` (typically a handful of entries) and merges it into `sorted`. The merge starts at the smallest pending
/// entry, so terms of `sorted` below everything added are never moved.
pub struct SortedCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    sorted: Vec<ColumnEntry<FilT, RowT, CF>>,
    pending: Vec<ColumnEntry<FilT, RowT, CF>>,
    scratch: Vec<ColumnEntry<FilT, RowT, CF>>,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Default for SortedCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self {
            sorted: Vec::new(),
            pending: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient> SortedCol<FilT, RowT, CF> {
    /// Push `entry` onto `out` (sorted ascending), summing it into the last entry if they share a row.
    #[inline]
    fn push_consolidated(
        out: &mut Vec<ColumnEntry<FilT, RowT, CF>>,
        entry: ColumnEntry<FilT, RowT, CF>,
    ) {
        if let Some(last) = out.last_mut() {
            if last.row_index == entry.row_index {
                match entry.coeff + Some(last.coeff) {
                    Some(coeff) => last.coeff = coeff,
                    None => {
                        out.pop();
                    }
                }
                return;
            }
        }
        out.push(entry);
    }

    /// Sort the pending terms and merge them into `sorted`.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        self.pending.sort_unstable();
        // Terms of `sorted` strictly below every pending term stay where they are
        let start = self
            .sorted
            .partition_point(|entry| *entry < self.pending[0]);
        self.scratch.clear();
        {
            let mut old = self.sorted.drain(start..).peekable();
            let mut new = self.pending.drain(..).peekable();
            loop {
                let take_old = match (old.peek(), new.peek()) {
                    (Some(a), Some(b)) => a <= b,
                    (Some(_), None) => true,
                    (None, Some(_)) => false,
                    (None, None) => break,
                };
                let entry = if take_old { old.next() } else { new.next() }.expect("peeked");
                Self::push_consolidated(&mut self.scratch, entry);
            }
        }
        self.sorted.append(&mut self.scratch);
    }

    /// Add a single term. This __need not__ coalesce terms with the same row index.
    pub fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        self.pending.push(entry);
    }

    /// Add every term of `entries`.
    pub fn add_entries(&mut self, entries: impl IntoIterator<Item = ColumnEntry<FilT, RowT, CF>>) {
        self.pending.extend(entries);
    }

    /// Remove and return the leading non-zero term of the column (i.e. the column pivot).
    pub fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        self.flush();
        self.sorted.pop()
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient>
    WorkingColumn<FilT, RowT, CF> for SortedCol<FilT, RowT, CF>
{
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        SortedCol::add_entry(self, entry);
    }
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        SortedCol::pop_pivot(self)
    }
    fn clear(&mut self) {
        self.sorted.clear();
        self.pending.clear();
    }
}

/// Represent columns as [`SortedCol`]s: the column being reduced is a sorted vector, and the columns of `D` added
/// since the last pivot query are sorted together, on the fly, then merged in.
///
/// Each merge rewrites the vector from the smallest entry added upwards. So this is the fastest representation when
/// the columns added stay near the pivot (as for homology of simplicial and cell complexes), but very slow when they
/// reach far below it (15–35× slower than [`MergedColumns`] on GrPPH, or the cohomology of cubical complexes).
/// Only use it when you have measured it on your matrices; see the [module docs](self) for how the representations
/// compare.
#[derive(Debug, Clone, Copy, Default)]
pub struct SortedColumns;

impl ColumnRepresentation for SortedColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        SortedCol<FilT, RowT, CF>;

    fn add_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut SortedCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        column.add_entries(entries.map(|entry| entry * multiple));
    }

    fn add_run<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut SortedCol<FilT, RowT, CF>,
        run: &SortedRun<FilT, RowT, CF>,
        multiple: CF,
    ) {
        column.add_entries(run.iter().map(|entry| entry.clone() * multiple));
    }
}

/// A set of numbers `0 <= i < bound` as a 64-ary tree of bitsets, as in PHAT's `bit_tree_pivot_column`: the bottom
/// level has a bit per number, and each bit of a level above says whether a word of the level below is non-zero.
/// Inserting, removing and finding the largest element each touch one word per level (at most 4 for 16 million).
#[derive(Clone, Debug)]
struct BitTree {
    /// Every level in one vector, bottom level first; the last level is a single word
    words: Vec<u64>,
    /// `offsets[k]` is where level `k` starts in `words`, for `k < depth`
    offsets: [usize; BitTree::MAX_DEPTH],
    depth: usize,
}

impl BitTree {
    /// Enough for 64^8 = 2^48 elements
    const MAX_DEPTH: usize = 8;

    fn new(bound: usize) -> Self {
        let mut offsets = [0; Self::MAX_DEPTH];
        let mut depth = 0;
        let mut total = 0;
        let mut n = bound.max(1);
        loop {
            let words = n.div_ceil(64);
            offsets[depth] = total;
            total += words;
            depth += 1;
            if words == 1 {
                break;
            }
            n = words;
        }
        Self {
            words: vec![0; total],
            offsets,
            depth,
        }
    }

    #[inline]
    fn contains(&self, i: usize) -> bool {
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Insert `i`, which must not be in the set.
    #[inline]
    fn insert(&mut self, i: usize) {
        self.insert_from(0, i);
    }

    /// Insert `i` if it's not in the set, and remove it if it is. The bottom level is flipped without branching on
    /// whether `i` was there (which, when adding dense columns, is a coin toss the CPU would often mispredict); the
    /// levels above change only when a bottom word becomes empty or non-empty.
    #[inline(always)]
    fn toggle(&mut self, i: usize) {
        let word = &mut self.words[i >> 6];
        let before = *word;
        *word ^= 1 << (i & 63);
        let after = *word;
        if before == 0 || after == 0 {
            self.toggle_above(i >> 6, after != 0);
        }
    }

    /// Mark word `i` of the bottom level as non-empty (`inserted`) or empty, in the levels above.
    #[cold]
    fn toggle_above(&mut self, i: usize, inserted: bool) {
        if self.depth == 1 {
            return;
        }
        if inserted {
            self.insert_from(1, i);
        } else {
            self.remove_from(1, i);
        }
    }

    /// Insert `i` (a bit of level `level`), which must not be there, and mark the words it fills in the levels above.
    #[inline]
    fn insert_from(&mut self, level: usize, mut i: usize) {
        for &offset in &self.offsets[level..self.depth] {
            let word = &mut self.words[offset + (i >> 6)];
            let was_empty = *word == 0;
            *word |= 1 << (i & 63);
            if !was_empty {
                return;
            }
            i >>= 6;
        }
    }

    /// Remove `i` (a bit of level `level`), which must be there, and unmark the words it empties in the levels above.
    #[inline]
    fn remove_from(&mut self, level: usize, mut i: usize) {
        for &offset in &self.offsets[level..self.depth] {
            let word = &mut self.words[offset + (i >> 6)];
            *word &= !(1 << (i & 63));
            if *word != 0 {
                return;
            }
            i >>= 6;
        }
    }

    /// Remove `i`, which must be in the set.
    #[inline]
    fn remove(&mut self, i: usize) {
        self.remove_from(0, i);
    }

    #[inline]
    fn max(&self) -> Option<usize> {
        let word = *self.words.last()?;
        if word == 0 {
            return None;
        }
        let mut i = 63 - word.leading_zeros() as usize;
        for &offset in self.offsets[..self.depth - 1].iter().rev() {
            let word = self.words[offset + i];
            i = (i << 6) | (63 - word.leading_zeros() as usize);
        }
        Some(i)
    }
}

/// The working column of [`PivotColumns`], for a matrix with [`row_ranks`](HasRowFiltration::row_ranks): a dense
/// column indexed by rank, with a 64-ary tree of bitsets marking the ranks present.
pub struct BitCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    ranks: RowRanks,
    present: BitTree,
    /// `slots[rank]` is the entry with that rank, if `rank` is in `present` (and otherwise stale).
    /// Filled when the first entry arrives, so a column that is never used allocates nothing.
    slots: Vec<ColumnEntry<FilT, RowT, CF>>,
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient> BitCol<FilT, RowT, CF> {
    fn new(ranks: RowRanks) -> Self {
        Self {
            present: BitTree::new(ranks.bound()),
            ranks,
            slots: Vec::new(),
        }
    }

    #[inline(always)]
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        let rank = self.ranks.rank(&entry.row_index);
        if self.slots.is_empty() {
            self.slots = vec![entry.clone(); self.ranks.bound()];
        }
        if self.present.contains(rank) {
            let slot = &mut self.slots[rank];
            match entry.coeff + Some(slot.coeff) {
                Some(coeff) => slot.coeff = coeff,
                None => self.present.remove(rank),
            }
        } else {
            self.slots[rank] = entry;
            self.present.insert(rank);
        }
    }

    /// Add every entry (times `multiple`), as [`add_entry`](Self::add_entry) would, but with the column's fields
    /// held in locals and the direction of the ranks decided once, rather than per entry. Adding stored columns of R
    /// is nearly all the work in reducing dense matrices.
    #[inline]
    fn add_all<'a>(
        &mut self,
        entries: impl Iterator<Item = &'a ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) where
        ColumnEntry<FilT, RowT, CF>: 'a,
    {
        let mut entries = entries.peekable();
        let Some(&first) = entries.peek() else {
            return;
        };
        if self.slots.is_empty() {
            self.slots = vec![first.clone(); self.ranks.bound()];
        }
        let Self {
            ranks,
            present,
            slots,
        } = self;
        let (table, descending) = ranks.parts();
        if descending {
            let top = table.len() - 1;
            Self::add_ranked(present, slots, entries, multiple, |i| {
                top - table[i] as usize
            });
        } else {
            Self::add_ranked(present, slots, entries, multiple, |i| table[i] as usize);
        }
    }

    #[inline(always)]
    fn add_ranked<'a>(
        present: &mut BitTree,
        slots: &mut [ColumnEntry<FilT, RowT, CF>],
        entries: impl Iterator<Item = &'a ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
        rank: impl Fn(usize) -> usize,
    ) where
        ColumnEntry<FilT, RowT, CF>: 'a,
    {
        if std::mem::size_of::<CF>() == 0 {
            // A field with only one non-zero coefficient is Z/2, where adding an entry flips it: write the entry
            // whether or not it ends up present (a slot is only read while its rank is), and flip its bit.
            for entry in entries {
                let rank = rank(entry.row_index.dense_index());
                slots[rank].clone_from(entry);
                present.toggle(rank);
            }
            return;
        }
        for entry in entries {
            let rank = rank(entry.row_index.dense_index());
            let coeff = entry.coeff * multiple;
            if present.contains(rank) {
                let slot = &mut slots[rank];
                match coeff + Some(slot.coeff) {
                    Some(sum) => slot.coeff = sum,
                    None => present.remove(rank),
                }
            } else {
                let slot = &mut slots[rank];
                slot.clone_from(entry);
                slot.coeff = coeff;
                present.insert(rank);
            }
        }
    }

    #[inline]
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        let rank = self.present.max()?;
        self.present.remove(rank);
        Some(self.slots[rank].clone())
    }

    fn clear(&mut self) {
        while let Some(rank) = self.present.max() {
            self.present.remove(rank);
        }
    }
}

/// The working column of [`PivotColumns`]: a [`BitCol`] if the matrix gives row ranks, or else a [`MergeCol`].
pub enum PivotCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    /// A dense column over the ranks of rows
    Bits(BitCol<FilT, RowT, CF>),
    /// The matrix has no row ranks, so fall back to merging
    Merged(MergeCol<FilT, RowT, CF>),
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF> Default for PivotCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self::Merged(MergeCol::default())
    }
}

impl<FilT: FiltrationValue, RowT: BasisElement, CF: NonZeroCoefficient>
    WorkingColumn<FilT, RowT, CF> for PivotCol<FilT, RowT, CF>
{
    #[inline]
    fn add_entry(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        match self {
            Self::Bits(column) => column.add_entry(entry),
            Self::Merged(column) => column.add_entry(entry),
        }
    }
    #[inline]
    fn pop_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>> {
        match self {
            Self::Bits(column) => column.pop_pivot(),
            Self::Merged(column) => column.pop_pivot(),
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Bits(column) => column.clear(),
            Self::Merged(column) => WorkingColumn::clear(column),
        }
    }
}

/// Represent the column being reduced as a dense column over the rows, with a 64-ary tree of bitsets saying which rows
/// are present, as PHAT's default `bit_tree_pivot_column` does. Needs the matrix to give the rank of every row in
/// filtration order ([`row_ranks`](HasRowFiltration::row_ranks), as explicit matrices in `phlite_py` do); for other
/// matrices it falls back to [`MergedColumns`].
///
/// Adding a term is a lookup of its rank and a bit flip, cancelling terms vanish at once, and finding the pivot is a
/// few bit-scans, whatever the number of columns added. So this is the fastest representation when many columns are
/// added to each, as for dense matrices and the cubical complexes of images (with
/// [`store_r`](crate::reduction::Clearing::store_r)). It costs a slot per row of the matrix, allocated once per
/// reduction, and every entry added is visited (none are skipped lazily, unlike [`MergedColumns`]).
/// Only the serial clearing algorithm creates bit columns (see [`new_column`](ColumnRepresentation::new_column));
/// the parallel and standard algorithms fall back to merging. See the [module docs](self) for how the representations
/// compare.
#[derive(Debug, Clone, Copy, Default)]
pub struct PivotColumns;

impl ColumnRepresentation for PivotColumns {
    type Column<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient> =
        PivotCol<FilT, RowT, CF>;

    fn new_column<M>(&self, matrix: &M) -> PivotCol<M::FiltrationT, M::RowT, M::CoefficientField>
    where
        M: HasRowFiltration,
        M::FiltrationT: ColumnFiltration,
        M::RowT: ColumnRow,
        M::CoefficientField: ColumnCoefficient,
    {
        match matrix.row_ranks() {
            Some(ranks) if M::RowT::DENSE_INDEX => PivotCol::Bits(BitCol::new(ranks)),
            _ => PivotCol::default(),
        }
    }

    fn add_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut PivotCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        match column {
            PivotCol::Bits(column) => entries.for_each(|entry| column.add_entry(entry * multiple)),
            PivotCol::Merged(column) => column.add_unsorted(entries, multiple),
        }
    }

    fn add_sorted_entries<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut PivotCol<FilT, RowT, CF>,
        entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>,
        multiple: CF,
    ) {
        match column {
            PivotCol::Bits(column) => entries.for_each(|entry| column.add_entry(entry * multiple)),
            PivotCol::Merged(column) => column.add_sorted(entries, multiple),
        }
    }

    fn add_run<FilT: ColumnFiltration, RowT: ColumnRow, CF: ColumnCoefficient>(
        &self,
        column: &mut PivotCol<FilT, RowT, CF>,
        run: &SortedRun<FilT, RowT, CF>,
        multiple: CF,
    ) {
        match column {
            PivotCol::Bits(column) => column.add_all(run.iter(), multiple),
            PivotCol::Merged(column) => column.add_run(run.clone(), multiple, 0),
        }
    }
}

#[cfg(test)]
mod sorted_col_tests {
    use super::*;
    use crate::fields::{NonZeroCoefficient, Z3};

    fn entry(row: usize, coeff: Z3) -> ColumnEntry<usize, usize, Z3> {
        ColumnEntry {
            filtration_value: row,
            row_index: row,
            coeff,
        }
    }

    /// SortedCol and MergeCol (with unsorted batches, pre-sorted batches and shared runs) agree with BHCol, given the
    /// same terms in batches, popping pivots in between
    #[test]
    fn matches_bhcol() {
        let mut state = 12345u64;
        let mut next = move |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let summary = |e: Option<ColumnEntry<usize, usize, Z3>>| e.map(|e| (e.row_index, e.coeff));
        for _ in 0..500 {
            let mut sorted = SortedCol::default();
            let mut merged = MergeCol::default();
            let mut heap = BHCol::default();
            for _ in 0..next(6) {
                let one = Z3::one();
                let coeff = |c: u64| if c == 0 { one } else { one.additive_inverse() };
                // A batch, possibly with repeated rows
                let batch: Vec<_> = (0..next(10))
                    .map(|_| entry(next(20) as usize, coeff(next(2))))
                    .collect();
                let multiple = coeff(next(2));
                sorted.add_entries(batch.iter().map(|&e| e * multiple));
                heap.add_entries(batch.iter().map(|&e| e * multiple));
                // MergeCol gets the batch one of three ways
                match next(3) {
                    0 => merged.add_unsorted(batch, multiple),
                    1 => merged.add_sorted(sorted_run(batch).iter().copied(), multiple),
                    _ => merged.add_run(sorted_run(batch), multiple, 0),
                }
                if next(2) == 0 {
                    let expected = summary(heap.pop_pivot());
                    assert_eq!(summary(sorted.pop_pivot()), expected);
                    assert_eq!(summary(merged.pop_pivot()), expected);
                }
            }
            loop {
                let expected = summary(heap.pop_pivot());
                assert_eq!(summary(sorted.pop_pivot()), expected);
                assert_eq!(summary(merged.pop_pivot()), expected);
                if expected.is_none() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod pivot_col_tests {
    use super::*;
    use crate::fields::{NonZeroCoefficient, Z2, Z3};
    use std::cmp::Reverse;

    /// A densely numbered row, whose rank is a fixed permutation of its index
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Row(usize);
    impl BasisElement for Row {
        const DENSE_INDEX: bool = true;
        fn dense_index(&self) -> usize {
            self.0
        }
        fn from_dense_index(index: usize) -> Self {
            Row(index)
        }
    }

    const N: usize = 200;

    /// Filtration values with many ties, and ranks in the order (value, row)
    fn values_and_ranks() -> (Vec<usize>, RowRanks) {
        let values: Vec<usize> = (0..N).map(|i| (i * 37) % 23).collect();
        let mut order: Vec<usize> = (0..N).collect();
        order.sort_by_key(|&i| (values[i], i));
        let mut table = vec![0u32; N];
        for (rank, &i) in order.iter().enumerate() {
            table[i] = rank as u32;
        }
        (values, RowRanks::new(table.into()))
    }

    /// Random batches of entries, added to a heap one by one and to a bit column either one by one or as a batch,
    /// with pivots popped from both now and then, must give the same pivots.
    fn check<
        FilT: FiltrationValue + Debug,
        RowT: BasisElement + Debug,
        CF: NonZeroCoefficient + Debug,
    >(
        ranks: RowRanks,
        entry: impl Fn(usize, CF) -> ColumnEntry<FilT, RowT, CF>,
    ) {
        let mut state = 98765u64;
        let mut next = move |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let summary = |e: Option<ColumnEntry<FilT, RowT, CF>>| e.map(|e| (e.row_index, e.coeff));
        let mut bits = BitCol::new(ranks);
        for _ in 0..300 {
            let mut heap = BHCol::default();
            for _ in 0..next(8) {
                let one = CF::one();
                let coeff = |c: u64| if c == 0 { one } else { one.additive_inverse() };
                // A batch (rows may repeat), drawn from a few rows so that there is plenty of cancellation
                let spread = 1 + next(N as u64);
                let batch: Vec<_> = (0..next(30))
                    .map(|_| entry(next(spread) as usize, coeff(next(2))))
                    .collect();
                for e in &batch {
                    heap.add_entry(e.clone());
                }
                if next(2) == 0 {
                    bits.add_all(batch.iter(), one);
                } else {
                    batch.into_iter().for_each(|e| bits.add_entry(e));
                }
                if next(2) == 0 {
                    assert_eq!(summary(bits.pop_pivot()), summary(heap.pop_pivot()));
                }
            }
            // Either drain the column or clear it (the column is reused, as in the reduction)
            if next(2) == 0 {
                loop {
                    let expected = summary(heap.pop_pivot());
                    assert_eq!(summary(bits.pop_pivot()), expected);
                    if expected.is_none() {
                        break;
                    }
                }
            } else {
                bits.clear();
                assert!(bits.pop_pivot().is_none());
            }
        }
    }

    /// A bit column gives the same pivots as a heap, in the matrix's order and in the reversed order, over Z/3 and
    /// over Z/2 (where adding a batch flips bits without checking what's there)
    #[test]
    fn matches_bhcol() {
        fn both<CF: NonZeroCoefficient + Debug>() {
            let (values, ranks) = values_and_ranks();
            check::<_, _, CF>(ranks.clone(), |row, coeff| ColumnEntry {
                filtration_value: values[row],
                row_index: Row(row),
                coeff,
            });
            check::<_, _, CF>(ranks.reversed(), |row, coeff| ColumnEntry {
                filtration_value: Reverse(values[row]),
                row_index: Reverse(Row(row)),
                coeff,
            });
        }
        both::<Z3>();
        both::<Z2>();
    }

    /// Toggling bits, across several levels, against a set
    #[test]
    fn bit_tree_toggle() {
        for bound in [1, 64, 65, 4097, 300_000] {
            let mut tree = BitTree::new(bound);
            let mut present = std::collections::BTreeSet::new();
            let mut state = 12345u64;
            for _ in 0..5000 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Mostly a few clustered values, so words fill and empty, plus some anywhere
                let i = if state >> 63 == 0 {
                    ((state >> 33) % 130) as usize % bound
                } else {
                    ((state >> 20) % bound as u64) as usize
                };
                tree.toggle(i);
                if !present.remove(&i) {
                    present.insert(i);
                }
                assert_eq!(tree.contains(i), present.contains(&i));
                assert_eq!(tree.max(), present.last().copied());
            }
        }
    }

    /// The bit tree's largest element, across several levels
    #[test]
    fn bit_tree_max() {
        for bound in [1, 63, 64, 65, 4096, 4097, 300_000] {
            let mut tree = BitTree::new(bound);
            assert_eq!(tree.max(), None);
            let mut present = std::collections::BTreeSet::new();
            for i in (0..bound).step_by(1 + bound / 50).chain([bound - 1, 0]) {
                if present.insert(i) {
                    tree.insert(i);
                }
                assert_eq!(tree.max(), present.last().copied());
            }
            while let Some(&i) = present.iter().nth(present.len() / 2) {
                present.remove(&i);
                tree.remove(i);
                assert_eq!(tree.max(), present.last().copied());
                assert!(!tree.contains(i));
            }
        }
    }
}
