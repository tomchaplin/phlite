use std::cmp::Reverse;

use itertools::equal;
use ordered_float::NotNan;

use crate::{
    columns::{BHCol, ColumnEntry},
    fields::NonZeroCoefficient,
    PhliteError,
};

use self::adaptors::{WithFuncFiltration, WithTrivialFiltration};

pub mod adaptors;
pub mod combinators;
pub mod implementors;
#[cfg(test)]
mod tests;

// ========= Traits for matrix indices and filtrations =========

pub trait BasisElement: Ord + Copy {}
pub trait FiltrationT: Ord + Copy {}

// Default implementors

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationT for NotNan<f32> {}
impl FiltrationT for NotNan<f64> {}
impl FiltrationT for usize {}
impl FiltrationT for isize {}
impl FiltrationT for () {}

impl<T> BasisElement for Reverse<T> where T: BasisElement {}
impl<T> FiltrationT for Reverse<T> where T: FiltrationT {}

// ======== Abstract matrix oracle trait =======================

// TODO: Try and get rid of Sized bounds, is there a better way to summarise ColumnEntry?
// TODO: Work out a better set of traits for withcolbasis, want ability to access basis element in index i and then we can use that to build a ColT ourselves

pub trait MatrixOracle {
    type CoefficientField: NonZeroCoefficient;
    type ColT: BasisElement;
    type RowT: BasisElement;

    /// Implement your oracle on the widest range of [`ColT`](Self::ColT) possible.
    /// To specify a given matrix, you will later provide an oracle, alongside a basis for the column space.
    /// If you are unable to produce a column, please return [`PhliteError::NotInDomain`].
    ///
    /// It is dis-advantageous to produce the rows in ascending order because inserting into binary heaps would require traversing the full height (see [`BinaryHeap::push`](std::collections::BinaryHeap::push)).
    /// Since checking and sorting by filtration values is typically slow, prefer to produce in descending order with respect to the ordering on [`RowT`](Self::RowT).
    /// Lower bounds on iterator size provided via [`Iterator::size_hint`] will be used to preallocate.
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError>;

    /// Checks that the matricies are equal on the specified col, ignoring ordering due to filtration values
    fn eq_on_col<M2>(&self, other: &M2, col: Self::ColT) -> bool
    where
        Self: Sized,
        M2: MatrixOracle<
                CoefficientField = Self::CoefficientField,
                ColT = Self::ColT,
                RowT = Self::RowT,
            > + Sized,
    {
        let self_trivial = self.with_trivial_filtration();
        let other_trivial = other.with_trivial_filtration();

        let mut self_col = self_trivial.build_bhcol(col).unwrap();
        let self_col_sorted = self_col
            .drain_sorted()
            .map(|e| Into::<(Self::CoefficientField, Self::RowT, ())>::into(e));
        let mut other_col = other_trivial.build_bhcol(col).unwrap();
        let other_col_sorted = other_col
            .drain_sorted()
            .map(|e| Into::<(Self::CoefficientField, Self::RowT, ())>::into(e));

        equal(self_col_sorted, other_col_sorted)
    }
}

pub trait HasRowFiltration: MatrixOracle + Sized {
    type FiltrationT: FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError>;

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = Result<ColumnEntry<Self>, PhliteError>>, PhliteError> {
        let column = self.column(col)?;
        Ok(column.map(|(coeff, row_index)| {
            let f_val = self.filtration_value(row_index)?;
            let entry = (coeff, row_index, f_val).into();
            Ok(entry)
        }))
    }

    fn empty_bhcol(&self) -> BHCol<Self> {
        BHCol::<Self>::default()
    }

    fn build_bhcol(&self, col: Self::ColT) -> Result<BHCol<Self>, PhliteError> {
        let mut output = self.empty_bhcol();
        output.add_entries(self.column_with_filtration(col)?.map(|e| e.unwrap()));
        Ok(output)
    }
}

pub trait FiniteOrderedColBasis: MatrixOracle<ColT = usize> {
    fn n_cols(&self) -> usize;
}

pub trait MatrixRef: MatrixOracle + Copy {
    fn with_trivial_filtration(self) -> WithTrivialFiltration<Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration { matrix_ref: self }
    }

    fn with_filtration<FT: FiltrationT, F: Fn(Self::RowT) -> Result<FT, PhliteError>>(
        self,
        filtration_function: F,
    ) -> WithFuncFiltration<Self, FT, F>
    where
        Self: Sized,
    {
        WithFuncFiltration {
            oracle: self,
            filtration: filtration_function,
        }
    }
}

impl<'a, M> MatrixOracle for &'a M
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        (*self).column(col)
    }
}

impl<'a, M> HasRowFiltration for &'a M
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (*self).filtration_value(row)
    }
}

impl<'a, M> FiniteOrderedColBasis for &'a M
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        (*self).n_cols()
    }
}

impl<'a, M> MatrixRef for &'a M where M: MatrixOracle {}
