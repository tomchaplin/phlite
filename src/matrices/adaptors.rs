// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use crate::{columns::BHCol, PhliteError};

use super::{ColBasis, FiltrationT, HasColBasis, HasRowFiltration, MatrixOracle, MatrixRef};

#[derive(Clone, Copy)]
pub struct WithTrivialFiltration<M: MatrixRef> {
    pub(crate) oracle: M,
}

impl<M: MatrixRef> MatrixOracle for WithTrivialFiltration<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.oracle.column(col)
    }
}

impl<M: MatrixRef> HasRowFiltration for WithTrivialFiltration<M> {
    type FiltrationT = ();

    fn filtration_value(&self, _row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        Ok(())
    }
}

impl<M: MatrixRef> HasColBasis for WithTrivialFiltration<M>
where
    M: HasColBasis,
{
    type BasisT = M::BasisT;

    fn basis(&self) -> &Self::BasisT {
        self.oracle.basis()
    }
}

// ====== WithFuncFiltration ===================

#[derive(Clone, Copy)]
pub struct WithFuncFiltration<
    M: MatrixRef,
    FT: FiltrationT,
    F: Fn(M::RowT) -> Result<FT, PhliteError>,
> {
    pub(crate) oracle: M,
    pub(crate) filtration: F,
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>>
    WithFuncFiltration<M, FT, F>
{
    pub fn discard_filtration(self) -> M {
        self.oracle
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> MatrixOracle
    for WithFuncFiltration<M, FT, F>
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.oracle.column(col)
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> HasRowFiltration
    for WithFuncFiltration<M, FT, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (self.filtration)(row)
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> HasColBasis
    for WithFuncFiltration<M, FT, F>
where
    M: HasColBasis,
{
    type BasisT = M::BasisT;

    fn basis(&self) -> &Self::BasisT {
        self.oracle.basis()
    }
}

// ====== Consolidator =========================

pub fn consolidate<M: MatrixRef>(oracle: M) -> Consolidator<M> {
    Consolidator { oracle }
}

#[derive(Clone, Copy)]
pub struct Consolidator<M: MatrixRef> {
    oracle: M,
}

pub struct ConsolidatorColumn<M: MatrixRef> {
    bh_col: BHCol<WithTrivialFiltration<M>>,
}

impl<M: MatrixRef> Iterator for ConsolidatorColumn<M> {
    type Item = (M::CoefficientField, M::RowT);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.bh_col.pop_pivot()?;
        let (coef, index, _) = next.into();
        Some((coef, index))
    }
}

impl<M: MatrixRef> MatrixOracle for Consolidator<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        // Take all the entries in the column and store them in a binary heap
        let bh_col = self.oracle.with_trivial_filtration().build_bhcol(col)?;
        // This iterator will consolidate all entries with the same row index into a new iterator
        Ok(ConsolidatorColumn { bh_col })
    }
}

impl<M: MatrixRef> HasRowFiltration for Consolidator<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixRef> HasColBasis for Consolidator<M>
where
    M: HasColBasis,
{
    type BasisT = M::BasisT;

    fn basis(&self) -> &Self::BasisT {
        self.oracle.basis()
    }
}

// ====== MatrixWithBasis ======================

#[derive(Debug, Clone, Copy)]
pub struct MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    pub matrix: M,
    pub basis: B,
}

impl<M, B> MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    pub fn new(matrix: M, basis: B) -> Self {
        Self { matrix, basis }
    }
}

impl<M, B> MatrixOracle for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.matrix.column(col)
    }
}

impl<M, B> HasRowFiltration for MatrixWithBasis<M, B>
where
    M: MatrixOracle + HasRowFiltration,
    B: ColBasis<ElemT = M::ColT>,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.matrix.filtration_value(row)
    }
}

impl<M, B> HasColBasis for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    type BasisT = B;

    fn basis(&self) -> &Self::BasisT {
        &self.basis
    }
}

// ====== UsingColBasisIndex ==================

#[derive(Clone, Copy)]
pub struct UsingColBasisIndex<M: MatrixRef + HasColBasis> {
    pub(crate) oracle: M,
}

impl<M> MatrixOracle for UsingColBasisIndex<M>
where
    M: MatrixRef + HasColBasis,
{
    type CoefficientField = M::CoefficientField;

    type ColT = usize;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.oracle.column(self.oracle.basis().element(col))
    }
}

impl<M> HasRowFiltration for UsingColBasisIndex<M>
where
    M: MatrixRef + HasColBasis + HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

// TODO: Should we give it the standard clumn basis?
