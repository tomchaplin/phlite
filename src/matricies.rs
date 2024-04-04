use std::{collections::BinaryHeap, fmt::Debug, marker::PhantomData};

use ordered_float::NotNan;

use crate::{fields::NonZeroCoefficient, PhliteError};

pub trait BasisElement: Ord {}
pub trait FiltrationT: Ord {}

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationT for NotNan<f32> {}
impl FiltrationT for NotNan<f64> {}
impl FiltrationT for () {}

pub trait MatrixOracle {
    type CoefficientField: NonZeroCoefficient;
    type ColT: BasisElement;
    type RowT: BasisElement + Copy;
    type FiltrationT: FiltrationT + Copy;

    /// Implement your oracle on the widest range of [`ColT`](Self::ColT) possible.
    /// To specify a given matrix, you will later provide an oracle, alongside a basis for the column space.
    /// If you are unable to produce a column, please return [`PhliteError::NotInDomain`].
    ///
    /// It is dis-advantageous to produce the rows in ascending order (see [`BinaryHeap::push`](std::collections::BinaryHeap::push)).
    /// Since checking and sorting by filtration values is typically slow, prefer to produce in descending order with respect to the ordering on [`RowT`](Self::RowT).
    /// Lower bounds on iterator size provided via [`Iterator::size_hint`] will be used to preallocate.
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<
        impl Iterator<Item = (Self::FiltrationT, Self::RowT, Self::CoefficientField)>,
        PhliteError,
    >;
}

pub struct ColumnEntry<M: MatrixOracle> {
    filtration_value: M::FiltrationT,
    row_index: M::RowT,
    coeff: M::CoefficientField,
}

impl<M: MatrixOracle> Debug for ColumnEntry<M>
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

impl<M: MatrixOracle> From<(M::FiltrationT, M::RowT, M::CoefficientField)> for ColumnEntry<M> {
    fn from(
        (filtration_value, row_index, coeff): (M::FiltrationT, M::RowT, M::CoefficientField),
    ) -> Self {
        Self {
            filtration_value,
            row_index,
            coeff,
        }
    }
}

impl<M: MatrixOracle> PartialEq for ColumnEntry<M> {
    // Equal row index implies equal filtration value
    fn eq(&self, other: &Self) -> bool {
        self.row_index.eq(&other.row_index)
    }
}

impl<M: MatrixOracle> Eq for ColumnEntry<M> {}

impl<M: MatrixOracle> PartialOrd for ColumnEntry<M> {
    // Order by filtration value and then order on RowT
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        ((&self.filtration_value, &self.row_index))
            .partial_cmp(&(&other.filtration_value, &other.row_index))
    }
}

impl<M: MatrixOracle> Ord for ColumnEntry<M> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("Since underlying implement Ord, so does ColumnEntry")
    }
}

pub struct BHCol<M: MatrixOracle> {
    heap: BinaryHeap<ColumnEntry<M>>,
}

impl<M: MatrixOracle> Debug for BHCol<M>
where
    ColumnEntry<M>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(&self.heap).finish()
    }
}

impl<M: MatrixOracle> Default for BHCol<M> {
    fn default() -> Self {
        Self {
            heap: Default::default(),
        }
    }
}

impl<M: MatrixOracle> BHCol<M> {
    pub fn add_entries(
        &mut self,
        entries: impl Iterator<Item = (M::FiltrationT, M::RowT, M::CoefficientField)>,
    ) {
        let (lower_bound, _) = entries.size_hint();
        self.heap.reserve(lower_bound);
        for entry in entries {
            self.heap.push(entry.into())
        }
    }

    pub fn to_sorted_vec(mut self) -> Vec<ColumnEntry<M>> {
        // Setup storage for output
        let mut out = vec![];
        loop {
            let next_pivot = self.pop_pivot();
            if let Some(pivot) = next_pivot {
                out.push(pivot)
            } else {
                break;
            }
        }
        return out;
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

pub struct VecVecMatrix<CF> {
    columns: Vec<Vec<usize>>,
    phantom: PhantomData<CF>,
}

impl<CF> From<Vec<Vec<usize>>> for VecVecMatrix<CF> {
    fn from(value: Vec<Vec<usize>>) -> Self {
        Self {
            columns: value,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> MatrixOracle for VecVecMatrix<CF> {
    type CoefficientField = CF;

    type ColT = usize;

    type RowT = usize;

    type FiltrationT = ();

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<
        impl Iterator<Item = (Self::FiltrationT, Self::RowT, Self::CoefficientField)>,
        PhliteError,
    > {
        Ok(self
            .columns
            .get(col)
            .ok_or(PhliteError::NotInDomain)?
            .iter()
            .map(|&row_idx| ((), row_idx, CF::one())))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        fields::{NonZeroCoefficient, Z2},
        matricies::{ColumnEntry, VecVecMatrix},
    };

    use super::{BHCol, MatrixOracle};

    #[test]
    fn test_add() {
        let matrix: VecVecMatrix<Z2> = vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ]
        .into();

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(5).unwrap());
        column.add_entries(matrix.column(4).unwrap());
        println!("{:?}", column);
        assert_eq!(
            column.pop_pivot(),
            Some(ColumnEntry {
                filtration_value: (),
                row_index: 1,
                coeff: Z2::one()
            })
        );
        assert_eq!(
            column.pop_pivot(),
            Some(ColumnEntry {
                filtration_value: (),
                row_index: 0,
                coeff: Z2::one()
            })
        );
        assert_eq!(column.pop_pivot(), None);

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(5).unwrap());
        column.add_entries(matrix.column(4).unwrap());
        column.add_entries(matrix.column(3).unwrap());
        assert_eq!(column.pop_pivot(), None);

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(6).unwrap());
        column.add_entries(matrix.column(6).unwrap());
        println!("{:?}", column);
        assert_eq!(column.to_sorted_vec().len(), 0);
    }
}
