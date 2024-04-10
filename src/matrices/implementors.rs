// ======== Default matrix oracles =============================

use std::{borrow::Cow, marker::PhantomData};

use crate::{
    fields::{NonZeroCoefficient, Z2},
    PhliteError,
};

use super::{BasisElement, ColBasis, HasColBasis, MatrixOracle};

use std::fmt::Debug;

// ====== VecVecMatrix =========================

pub struct VecVecMatrix<'a, CF: NonZeroCoefficient, RowT: BasisElement> {
    columns: Cow<'a, Vec<Vec<(CF, RowT)>>>,
    phantom: PhantomData<CF>,
    basis: StandardBasis,
}

impl<'a, RowT: BasisElement + Debug> Debug for VecVecMatrix<'a, Z2, RowT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.columns.fmt(f)
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<Cow<'a, Vec<Vec<(CF, RowT)>>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: Cow<'a, Vec<Vec<(CF, RowT)>>>) -> Self {
        Self {
            basis: StandardBasis {
                n_cols: value.len(),
            },
            columns: value,
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<&'a Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: &'a Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            basis: StandardBasis {
                n_cols: value.len(),
            },
            columns: Cow::Borrowed(value),
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'static, CF, RowT>
{
    fn from(value: Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            basis: StandardBasis {
                n_cols: value.len(),
            },
            columns: Cow::Owned(value),
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> MatrixOracle for VecVecMatrix<'a, CF, RowT> {
    type CoefficientField = CF;

    type ColT = usize;

    type RowT = RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        Ok(self
            .columns
            .get(col)
            .ok_or(PhliteError::NotInDomain)?
            .iter()
            .copied())
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> HasColBasis for VecVecMatrix<'a, CF, RowT> {
    type BasisT = StandardBasis;

    fn basis(&self) -> &Self::BasisT {
        &self.basis
    }
}

#[allow(non_snake_case)]
pub fn simple_Z2_matrix(cols: Vec<Vec<usize>>) -> VecVecMatrix<'static, Z2, usize> {
    let cols_with_coeffs = cols
        .into_iter()
        .map(|col| {
            col.into_iter()
                .map(|col_idx| (Z2::one(), col_idx))
                .collect()
        })
        .collect::<Vec<Vec<(Z2, usize)>>>();

    <VecVecMatrix<'_, Z2, usize>>::from(Cow::Owned(cols_with_coeffs))
}

#[derive(Clone, Copy)]
pub struct StandardBasis {
    n_cols: usize,
}

impl StandardBasis {
    pub fn new(n_cols: usize) -> Self {
        Self { n_cols }
    }
}

impl ColBasis for StandardBasis {
    type ElemT = usize;

    fn element(&self, index: usize) -> Self::ElemT {
        index
    }

    fn size(&self) -> usize {
        self.n_cols
    }
}
