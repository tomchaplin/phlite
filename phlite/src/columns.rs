//! Binary heap representations of matrix columns, essentially corresponding to linear combinations with a leading term.
use crate::{
    fields::NonZeroCoefficient,
    matrices::{BasisElement, FiltrationT},
};
use std::{collections::BinaryHeap, fmt::Debug, iter::repeat, ops::Mul};

#[derive(Clone, Copy)]
pub struct ColumnEntry<FilT: FiltrationT, RowT: BasisElement, CF> {
    pub filtration_value: FilT,
    pub row_index: RowT,
    pub coeff: CF,
}

impl<FilT: FiltrationT, RowT: BasisElement, CF> Debug for ColumnEntry<FilT, RowT, CF>
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

impl<FilT: FiltrationT, RowT: BasisElement, CF> From<(CF, RowT, FilT)>
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

impl<FilT: FiltrationT, RowT: BasisElement, CF> From<ColumnEntry<FilT, RowT, CF>>
    for (CF, RowT, FilT)
{
    fn from(entry: ColumnEntry<FilT, RowT, CF>) -> Self {
        (entry.coeff, entry.row_index, entry.filtration_value)
    }
}

/// WARNING: Equality only checks row index - to check correct coefficient and filtration value, convert to tuple
impl<FilT: FiltrationT, RowT: BasisElement, CF> PartialEq for ColumnEntry<FilT, RowT, CF> {
    // Equal row index implies equal filtration value
    fn eq(&self, other: &Self) -> bool {
        self.row_index.eq(&other.row_index)
    }
}
impl<FilT: FiltrationT, RowT: BasisElement, CF> Eq for ColumnEntry<FilT, RowT, CF> {}

impl<FilT: FiltrationT, RowT: BasisElement, CF> PartialOrd for ColumnEntry<FilT, RowT, CF> {
    // Order by filtration value and then order on RowT
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<FilT: FiltrationT, RowT: BasisElement, CF> Ord for ColumnEntry<FilT, RowT, CF> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (&self.filtration_value, &self.row_index).cmp(&(&other.filtration_value, &other.row_index))
    }
}

impl<FilT: FiltrationT, RowT: BasisElement, CF: NonZeroCoefficient> Mul<CF>
    for ColumnEntry<FilT, RowT, CF>
{
    type Output = Self;

    fn mul(self: Self, rhs: CF) -> Self::Output {
        ColumnEntry {
            coeff: self.coeff * rhs,
            filtration_value: self.filtration_value,
            row_index: self.row_index,
        }
    }
}

#[derive(Clone)]
pub struct BHCol<FilT: FiltrationT, RowT: BasisElement, CF> {
    heap: BinaryHeap<ColumnEntry<FilT, RowT, CF>>,
}

impl<FilT: FiltrationT, RowT: BasisElement, CF> Debug for BHCol<FilT, RowT, CF>
where
    ColumnEntry<FilT, RowT, CF>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(&self.heap).finish()
    }
}

impl<FilT: FiltrationT, RowT: BasisElement, CF> Default for BHCol<FilT, RowT, CF> {
    fn default() -> Self {
        Self {
            heap: BinaryHeap::default(),
        }
    }
}

impl<FilT: FiltrationT, RowT: BasisElement, CF> BHCol<FilT, RowT, CF> {
    pub fn add_entries(&mut self, entries: impl Iterator<Item = ColumnEntry<FilT, RowT, CF>>) {
        self.add_tuples(entries.map(Into::into));
    }

    pub fn add_tuples(&mut self, tuples: impl Iterator<Item = (CF, RowT, FilT)>) {
        let (lower_bound, _) = tuples.size_hint();
        self.heap.reserve(lower_bound);
        for tuple in tuples {
            self.heap.push(tuple.into());
        }
    }

    pub fn add_tuple(&mut self, tuple: (CF, RowT, FilT)) {
        self.heap.push(tuple.into());
    }

    pub fn drain_sorted(&mut self) -> impl Iterator<Item = ColumnEntry<FilT, RowT, CF>> + '_
    where
        CF: NonZeroCoefficient,
    {
        repeat(()).map_while(|()| self.pop_pivot())
    }

    pub fn to_sorted_vec(mut self) -> Vec<ColumnEntry<FilT, RowT, CF>>
    where
        CF: NonZeroCoefficient,
    {
        self.drain_sorted().collect()
    }

    pub fn push(&mut self, entry: ColumnEntry<FilT, RowT, CF>) {
        self.heap.push(entry);
    }

    pub fn clone_pivot(&mut self) -> Option<ColumnEntry<FilT, RowT, CF>>
    where
        ColumnEntry<FilT, RowT, CF>: Clone,
        CF: NonZeroCoefficient,
    {
        let pivot = self.pop_pivot();
        if let Some(pivot) = pivot {
            let ret = Some(pivot.clone());
            self.push(pivot);
            ret
        } else {
            None
        }
    }

    /// WARNING: Only valid if previously called `clone_pivot` or pushed the new pivot.
    pub fn peek_pivot(&self) -> Option<&ColumnEntry<FilT, RowT, CF>> {
        self.heap.peek()
    }

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
                working_index = next_entry.row_index;
                working_sum = None;
                working_filtration = next_entry.filtration_value;
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
}
