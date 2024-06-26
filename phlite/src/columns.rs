use crate::matrices::HasRowFiltration;
use std::{collections::BinaryHeap, fmt::Debug, iter::repeat, ops::Mul};

#[derive(Clone, Copy)]
pub struct ColumnEntry<M: HasRowFiltration> {
    pub filtration_value: M::FiltrationT,
    pub row_index: M::RowT,
    pub coeff: M::CoefficientField,
}

impl<M: HasRowFiltration> Debug for ColumnEntry<M>
where
    M::FiltrationT: Debug,
    M::RowT: Debug,
    M::CoefficientField: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "({:?} * {:?}) :: {:?}",
            self.coeff, self.row_index, self.filtration_value
        ))
    }
}

impl<M: HasRowFiltration> From<(M::CoefficientField, M::RowT, M::FiltrationT)> for ColumnEntry<M> {
    fn from(
        (coeff, row_index, filtration_value): (M::CoefficientField, M::RowT, M::FiltrationT),
    ) -> Self {
        Self {
            filtration_value,
            row_index,
            coeff,
        }
    }
}

impl<M: HasRowFiltration> From<ColumnEntry<M>> for (M::CoefficientField, M::RowT, M::FiltrationT) {
    fn from(entry: ColumnEntry<M>) -> Self {
        (entry.coeff, entry.row_index, entry.filtration_value)
    }
}

/// WARNING: Equality only checks row index - to check correct coefficient and filtration value, convert to tuple
impl<M: HasRowFiltration> PartialEq for ColumnEntry<M> {
    // Equal row index implies equal filtration value
    fn eq(&self, other: &Self) -> bool {
        self.row_index.eq(&other.row_index)
    }
}
impl<M: HasRowFiltration> Eq for ColumnEntry<M> {}

impl<M: HasRowFiltration> PartialOrd for ColumnEntry<M> {
    // Order by filtration value and then order on RowT
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        ((&self.filtration_value, &self.row_index))
            .partial_cmp(&(&other.filtration_value, &other.row_index))
    }
}

impl<M: HasRowFiltration> Ord for ColumnEntry<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("Since underlying implement Ord, so does ColumnEntry")
    }
}

impl<M: HasRowFiltration> Mul<M::CoefficientField> for ColumnEntry<M> {
    type Output = Self;

    fn mul(self, rhs: M::CoefficientField) -> Self::Output {
        ColumnEntry {
            coeff: self.coeff * rhs,
            filtration_value: self.filtration_value,
            row_index: self.row_index,
        }
    }
}

pub struct BHCol<M: HasRowFiltration> {
    heap: BinaryHeap<ColumnEntry<M>>,
}

impl<M: HasRowFiltration> Debug for BHCol<M>
where
    ColumnEntry<M>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(&self.heap).finish()
    }
}

impl<M: HasRowFiltration> Default for BHCol<M> {
    fn default() -> Self {
        Self {
            heap: Default::default(),
        }
    }
}

impl<M: HasRowFiltration> BHCol<M> {
    pub fn add_entries<M2>(&mut self, entries: impl Iterator<Item = ColumnEntry<M2>>)
    where
        M2: HasRowFiltration<
            FiltrationT = M::FiltrationT,
            CoefficientField = M::CoefficientField,
            RowT = M::RowT,
        >,
    {
        self.add_tuples(entries.map(|e| e.into()))
    }

    pub fn add_tuples(
        &mut self,
        tuples: impl Iterator<Item = (M::CoefficientField, M::RowT, M::FiltrationT)>,
    ) {
        let (lower_bound, _) = tuples.size_hint();
        self.heap.reserve(lower_bound);
        for tuple in tuples {
            self.heap.push(tuple.into())
        }
    }

    pub fn add_tuple(&mut self, tuple: (M::CoefficientField, M::RowT, M::FiltrationT)) {
        self.heap.push(tuple.into())
    }

    pub fn drain_sorted<'a>(&'a mut self) -> impl Iterator<Item = ColumnEntry<M>> + 'a {
        repeat(()).map_while(|_| self.pop_pivot())
    }

    pub fn to_sorted_vec(mut self) -> Vec<ColumnEntry<M>> {
        self.drain_sorted().collect()
    }

    pub fn push(&mut self, entry: ColumnEntry<M>) {
        self.heap.push(entry)
    }

    pub fn clone_pivot(&mut self) -> Option<ColumnEntry<M>>
    where
        ColumnEntry<M>: Clone,
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
    pub fn peek_pivot(&self) -> Option<&ColumnEntry<M>> {
        self.heap.peek()
    }

    pub fn pop_pivot(&mut self) -> Option<ColumnEntry<M>> {
        // Pull out first entry
        let Some(first_entry) = self.heap.pop() else {
            return None;
        };
        let mut working_index: M::RowT = first_entry.row_index;
        let mut working_sum: Option<M::CoefficientField> = Some(first_entry.coeff);
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

        match working_sum {
            Some(coeff) => Some(ColumnEntry {
                row_index: working_index,
                filtration_value: working_filtration,
                coeff,
            }),
            None => None,
        }
    }
}
