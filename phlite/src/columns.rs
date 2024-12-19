//! Binary heap representations of matrix columns, essentially corresponding to linear combinations with ordered terms.
//!
//! Most of the content of this module is an implementation detail of the reduction algorithms and is likely to change in the future.
//! Unless you wish to access pivots of matrices yourself, you probably don't need this module.
use crate::{
    fields::NonZeroCoefficient,
    matrices::{BasisElement, FiltrationValue},
};
use std::{collections::BinaryHeap, fmt::Debug, iter::repeat, ops::Mul};

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
/// Represents a column in a matrix where each [`ColumnEntry`] is stored in a [`BinaryHeap`].
/// The order of terms in this heap is determined by the filtration value and then the row index.
///
/// Since the same row index my appear more than once in the heap, it is more accurate to say that this represents a linear combination of elements in the row basis.
/// Terms are stored in a binary heap to allow efficient access to the leading term, otherwise known as the column pivot.
///
/// One should typically construct a [`BHCol`] by calling [`build_bhcol`](crate::matrices::HasRowFiltration::build_bhcol) or [`empty_bhcol`](crate::matrices::HasRowFiltration::empty_bhcol).
/// To add columns into the heap, one should probably call [`column_with_filtration`](crate::matrices::HasRowFiltration::column_with_filtration) and pass the resulting iterator to [`add_entries`](BHCol::add_entries).
/// Note, your matrix must implement [`HasRowFiltration`](crate::matrices::HasRowFiltration) in order to do this.
/// Consider using [`with_trivial_filtration`](crate::matrices::MatrixOracle::with_trivial_filtration) if you have no filtration that you care about (e.g. you are computing non-persistent homology).
pub struct BHCol<FilT: FiltrationValue, RowT: BasisElement, CF> {
    heap: BinaryHeap<ColumnEntry<FilT, RowT, CF>>,
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
            heap: BinaryHeap::default(),
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
        for entry in entries {
            self.add_entry(entry);
        }
        //self.add_tuples(entries.into_iter().map(Into::into));
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
