//! Concrete types that implement [`MatrixOracle`] and [`ColBasis`].

// ======== Default matrix oracles =============================

use std::hash::Hash;
use std::{borrow::Cow, ops::Deref};

use rustc_hash::FxHashMap;

use crate::fields::{NonZeroCoefficient, Z2};

use super::{BasisElement, ColBasis, HasColBasis, MatrixOracle};

use std::fmt::Debug;

// TODO: A VecSmallvecMatrix might be more cache efficient?

// ====== VecVecMatrix =========================

/// A sprase matrix stored as a [`Vec<Column>`] where each `Column` is a vector of the non-zero entries.
/// The columns are indexed by [`usize`] and the rows are indexed by an arbitrary type.
///
/// This is similar to the typical sparse matrix representation in other frameworks.
/// Although note this is not in CSR/CSC format!
pub struct VecVecMatrix<'a, CF: NonZeroCoefficient, RowT: BasisElement> {
    /// The columns of the matrix.
    /// This is stored behind a [`Cow`] to allow construction whether you own `columns` or not
    pub columns: Cow<'a, Vec<Vec<(CF, RowT)>>>,
}

impl<RowT: BasisElement + Debug> Debug for VecVecMatrix<'_, Z2, RowT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.columns.fmt(f)
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<Cow<'a, Vec<Vec<(CF, RowT)>>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: Cow<'a, Vec<Vec<(CF, RowT)>>>) -> Self {
        Self { columns: value }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<&'a Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: &'a Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: Cow::Borrowed(value),
        }
    }
}

impl<CF: NonZeroCoefficient, RowT: BasisElement> From<Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'static, CF, RowT>
{
    fn from(value: Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
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
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.columns.get(col).unwrap().iter().cloned()
    }
}

impl<CF: NonZeroCoefficient, RowT: BasisElement> HasColBasis for VecVecMatrix<'_, CF, RowT> {
    type BasisT = StandardBasis;
    type BasisRef<'b>
        = StandardBasis
    where
        Self: 'b;

    fn basis(&self) -> Self::BasisRef<'_> {
        StandardBasis {
            n_cols: self.columns.len(),
        }
    }
}

impl<CF: NonZeroCoefficient, RowT: BasisElement> VecVecMatrix<'static, CF, RowT> {
    /// Constructor for `VecVecMatrix<'static, CF, RowT>`, typically `I` is a `Vec<Vec<(CF, RowT)>>`.
    pub fn new<I, J>(columns: I) -> Self
    where
        J: IntoIterator<Item = (CF, RowT)>,
        I: IntoIterator<Item = J>,
    {
        let cols_with_coeffs = columns
            .into_iter()
            .map(|col| col.into_iter().collect())
            .collect::<Vec<Vec<(CF, RowT)>>>();

        <VecVecMatrix<'_, CF, RowT>>::from(Cow::Owned(cols_with_coeffs))
    }

    /// Calls [`Self::new`] with all coefficients set to `CF::one()`, typically `I` is a `Vec<Vec<RowT>>` and `CF` is [`Z2`].
    pub fn new_all_one<I, J>(columns: I) -> Self
    where
        J: IntoIterator<Item = RowT>,
        I: IntoIterator<Item = J>,
    {
        Self::new(
            columns
                .into_iter()
                .map(|col| col.into_iter().map(|row| (CF::one(), row))),
        )
    }
}

/// A sparse [`Z2`]-matrix implementation, with columns and rows indexed by `usize`.
pub type SimpleZ2Matrix = VecVecMatrix<'static, Z2, usize>;

#[derive(Clone, Copy)]
/// A basis for a matrix indexed by [`usize`] in which the element in index `i` is `i`.
/// Useful in conjunction with [`VecVecMatrix`].
pub struct StandardBasis {
    n_cols: usize,
}

impl StandardBasis {
    /// Constructor for a [`StandardBasis`] with columns indexed by the numbers `0..n_cols`.
    pub fn new(n_cols: usize) -> Self {
        Self { n_cols }
    }
}

impl Deref for StandardBasis {
    type Target = StandardBasis;

    fn deref(&self) -> &Self::Target {
        self
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

/// A sprase matrix stored as a [`FxHashMap<Column>`] where each `Column` is a vector of the non-zero entries.
/// The columns are indexed by any [`Hash`] type and the rows are indexed by an arbitrary type.
///
/// This is similar to the typical sparse matrix representation in other frameworks.
pub struct MapVecMatrix<'a, CF, ColT, RowT>
where
    CF: NonZeroCoefficient,
    ColT: BasisElement + Hash,
    RowT: BasisElement,
{
    /// The columns of the matrix.
    /// This is stored behind a [`Cow`] to allow construction whether you own `columns` or not
    pub columns: Cow<'a, FxHashMap<ColT, Vec<(CF, RowT)>>>,
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
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.columns.get(&col).unwrap().iter().cloned()
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
