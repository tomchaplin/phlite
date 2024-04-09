// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use crate::{columns::BHCol, PhliteError};

use super::{FiltrationT, FiniteOrderedColBasis, HasRowFiltration, MatrixOracle, MatrixRef};

#[derive(Clone, Copy)]
pub struct WithTrivialFiltration<M: MatrixRef> {
    pub(crate) matrix_ref: M,
}

impl<M: MatrixRef> MatrixOracle for WithTrivialFiltration<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.matrix_ref.column(col)
    }
}

impl<M: MatrixRef> MatrixRef for WithTrivialFiltration<M> {}

impl<M: MatrixRef> HasRowFiltration for WithTrivialFiltration<M> {
    type FiltrationT = ();

    fn filtration_value(&self, _row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        Ok(())
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for WithTrivialFiltration<M>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.matrix_ref.n_cols()
    }
}

// ====== WithFuncFiltration ===================

#[derive(Clone)]
pub struct WithFuncFiltration<
    M: MatrixRef,
    FT: FiltrationT,
    F: Fn(M::RowT) -> Result<FT, PhliteError>,
> {
    pub(crate) oracle: M,
    pub(crate) filtration: F,
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> Copy
    for WithFuncFiltration<M, FT, F>
where
    F: Copy,
{
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

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> MatrixRef
    for WithFuncFiltration<M, FT, F>
where
    F: Copy,
{
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> HasRowFiltration
    for WithFuncFiltration<M, FT, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (self.filtration)(row)
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> FiniteOrderedColBasis
    for WithFuncFiltration<M, FT, F>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.oracle.n_cols()
    }
}

// ====== WithOrderedBasis =====================

// TODO: Change so it only stores a ref?
pub struct WithOrderedColBasis<M: MatrixRef> {
    oracle: M,
    pub col_basis: Vec<M::ColT>,
}

impl<M: MatrixRef> WithOrderedColBasis<M> {
    pub fn new(oracle: M, col_basis: Vec<M::ColT>) -> Self {
        Self { oracle, col_basis }
    }
}

impl<M: MatrixRef> MatrixOracle for WithOrderedColBasis<M> {
    type CoefficientField = M::CoefficientField;

    type ColT = usize;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (M::CoefficientField, M::RowT)>, PhliteError> {
        let col_idx = self.col_basis.get(col).ok_or(PhliteError::NotInDomain)?;
        self.oracle.column(*col_idx)
    }
}

impl<M: MatrixRef> HasRowFiltration for WithOrderedColBasis<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for WithOrderedColBasis<M> {
    fn n_cols(&self) -> usize {
        self.col_basis.len()
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

impl<M: MatrixRef> MatrixRef for Consolidator<M> {}

impl<M: MatrixRef> HasRowFiltration for Consolidator<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for Consolidator<M>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.oracle.n_cols()
    }
}
