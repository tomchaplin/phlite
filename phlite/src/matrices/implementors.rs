// ======== Default matrix oracles =============================

use std::borrow::Cow;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::{
    fields::{NonZeroCoefficient, Z2},
    PhliteError,
};

use super::{BasisElement, ColBasis, HasColBasis, MatrixOracle};

use std::fmt::Debug;

// ====== VecVecMatrix =========================

pub struct VecVecMatrix<'a, CF: NonZeroCoefficient, RowT: BasisElement> {
    columns: Cow<'a, Vec<Vec<(CF, RowT)>>>,
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
        }
    }
}

impl<CF: NonZeroCoefficient, RowT: BasisElement> From<Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'static, CF, RowT>
{
    fn from(value: Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            basis: StandardBasis {
                n_cols: value.len(),
            },
            columns: Cow::Owned(value),
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
            .cloned())
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

// ====== MapVecMatrix =========================

// TODO: Should give this a basis.
// Can we do this by saving a reference to the basis or do I need to use an ordered map?

pub struct MapVecMatrix<'a, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    columns: Cow<'a, FxHashMap<ColT, Vec<(CF, RowT)>>>,
}

impl<'a, CF, ColT, RowT> MatrixOracle for MapVecMatrix<'a, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    type CoefficientField = CF;
    type ColT = ColT;
    type RowT = RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        Ok(self
            .columns
            .get(&col)
            .ok_or(PhliteError::NotInDomain)?
            .iter()
            .cloned())
    }
}

impl<'a, CF, ColT, RowT> From<Cow<'a, FxHashMap<ColT, Vec<(CF, RowT)>>>>
    for MapVecMatrix<'a, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    fn from(value: Cow<'a, FxHashMap<ColT, Vec<(CF, RowT)>>>) -> Self {
        Self { columns: value }
    }
}

impl<'a, CF, ColT, RowT> From<&'a FxHashMap<ColT, Vec<(CF, RowT)>>>
    for MapVecMatrix<'a, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    fn from(value: &'a FxHashMap<ColT, Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: Cow::Borrowed(value),
        }
    }
}

impl<CF, ColT, RowT> From<FxHashMap<ColT, Vec<(CF, RowT)>>>
    for MapVecMatrix<'static, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    fn from(value: FxHashMap<ColT, Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: Cow::Owned(value),
        }
    }
}
