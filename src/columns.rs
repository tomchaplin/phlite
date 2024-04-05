use crate::matricies::HasRowFiltration;
use std::{collections::BinaryHeap, fmt::Debug, iter::repeat};

pub struct ColumnEntry<M: HasRowFiltration> {
    pub(crate) filtration_value: M::FiltrationT,
    pub(crate) row_index: M::RowT,
    pub(crate) coeff: M::CoefficientField,
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
    pub fn add_entries(&mut self, entries: impl Iterator<Item = ColumnEntry<M>>) {
        let (lower_bound, _) = entries.size_hint();
        self.heap.reserve(lower_bound);
        for entry in entries {
            self.heap.push(entry)
        }
    }

    pub fn drain_sorted<'a>(&'a mut self) -> impl Iterator<Item = ColumnEntry<M>> + 'a {
        repeat(()).map_while(|_| self.pop_pivot())
    }

    pub fn to_sorted_vec(mut self) -> Vec<ColumnEntry<M>> {
        self.drain_sorted().collect()
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
