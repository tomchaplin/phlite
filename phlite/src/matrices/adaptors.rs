// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use std::cmp::Reverse;

use crate::{
    columns::{BHCol, ColumnEntry},
    PhliteError,
};

use super::{
    BasisElement, ColBasis, FiltrationT, HasColBasis, HasRowFiltration, MatrixOracle, MatrixRef,
    SplitByDimension,
};

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

// TODO: Should we give it the standard column basis?

// ====== WithSubBasis =========================

#[derive(Clone, Copy)]
pub struct WithSubBasis<M: MatrixRef + HasColBasis<BasisT: SplitByDimension>> {
    pub(crate) oracle: M,
    pub(crate) dimension: usize,
}

impl<M> MatrixOracle for WithSubBasis<M>
where
    M: MatrixRef + HasColBasis<BasisT: SplitByDimension>,
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

impl<M> HasRowFiltration for WithSubBasis<M>
where
    M: MatrixRef + HasColBasis<BasisT: SplitByDimension> + HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M> HasColBasis for WithSubBasis<M>
where
    M: MatrixRef + HasColBasis<BasisT: SplitByDimension>,
{
    type BasisT = <M::BasisT as SplitByDimension>::SubBasisT;

    fn basis(&self) -> &Self::BasisT {
        self.oracle.basis().in_dimension(self.dimension)
    }
}

// ====== ReverseMatrix ========================

pub struct ReverseMatrix<M> {
    pub(crate) oracle: M,
}

impl<M> ReverseMatrix<M> {
    pub fn unreverse(self) -> M {
        self.oracle
    }
}

impl<M> MatrixOracle for ReverseMatrix<M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;

    type ColT = Reverse<M::ColT>;

    type RowT = Reverse<M::RowT>;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        let normal_col = self.oracle.column(col.0)?;
        let reverse_col = normal_col.map(|(coeff, row)| (coeff, Reverse(row)));
        Ok(reverse_col)
    }
}

impl<M> HasRowFiltration for ReverseMatrix<M>
where
    M: MatrixOracle + HasRowFiltration,
{
    type FiltrationT = Reverse<M::FiltrationT>;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        Ok(Reverse(self.oracle.filtration_value(row.0)?))
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = crate::columns::ColumnEntry<Self>>, PhliteError> {
        let normal_col = self.oracle.column_with_filtration(col.0)?;
        let reverse_col = normal_col.map(|entry| ColumnEntry {
            filtration_value: Reverse(entry.filtration_value),
            row_index: Reverse(entry.row_index),
            coeff: entry.coeff,
        });
        Ok(reverse_col)
    }
}

impl<M> HasColBasis for ReverseMatrix<M>
where
    M: MatrixOracle + HasColBasis,
{
    type BasisT = ReverseBasis<M::BasisT>;

    fn basis(&self) -> &Self::BasisT {
        unsafe { std::mem::transmute(self.oracle.basis()) }
    }
}

#[repr(transparent)]
/// Note that layout is `repr(transparent)` so can transmute references.
pub struct ReverseBasis<B>(pub B);

impl<B> ColBasis for ReverseBasis<B>
where
    B: ColBasis,
{
    type ElemT = Reverse<B::ElemT>;

    fn element(&self, index: usize) -> Self::ElemT {
        Reverse(self.0.element(self.0.size() - 1 - index))
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<B> SplitByDimension for ReverseBasis<B>
where
    B: SplitByDimension,
{
    type SubBasisT = ReverseBasis<B::SubBasisT>;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        unsafe { std::mem::transmute(self.0.in_dimension(dimension)) }
    }
}

// ====== UnreverseMatrix ======================
// Used to un-reverse a matrix/basis that is indexed + filtered by Reverse<T>
// Returns a matrix/basis that is index + filtered by T
// If you naively call reverse again, you will not be able to multiply matrices because
// Reverse<Reverse<T>> != T
// Could we fix this by just requiring Into<ColT> on traits?

pub struct UnreverseMatrix<M> {
    pub(crate) oracle: M,
}

impl<M, ColT, RowT> MatrixOracle for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
{
    type CoefficientField = M::CoefficientField;

    type ColT = ColT;

    type RowT = RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        let reversed = self.oracle.column(Reverse(col))?;
        Ok(reversed.map(|(coeff, rev_row)| (coeff, rev_row.0)))
    }
}

impl<M, ColT, RowT, FilT> HasRowFiltration for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasRowFiltration<FiltrationT = Reverse<FilT>>,
    FilT: FiltrationT,
{
    type FiltrationT = FilT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(Reverse(row)).map(|ft| ft.0)
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = ColumnEntry<Self>>, PhliteError> {
        let rev_col = self.oracle.column_with_filtration(Reverse(col))?;
        Ok(rev_col.map(|entry| ColumnEntry {
            filtration_value: entry.filtration_value.0,
            row_index: entry.row_index.0,
            coeff: entry.coeff,
        }))
    }
}

impl<M, ColT, RowT, BasisT> HasColBasis for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasColBasis<BasisT = BasisT>,
    BasisT: ColBasis<ElemT = Reverse<ColT>>,
{
    type BasisT = UnreverseBasis<BasisT>;

    fn basis(&self) -> &Self::BasisT {
        let rev_basis = self.oracle.basis();
        unsafe { std::mem::transmute(rev_basis) }
    }
}

#[repr(transparent)]
pub struct UnreverseBasis<B>(pub B);

impl<B, T> ColBasis for UnreverseBasis<B>
where
    B: ColBasis<ElemT = Reverse<T>>,
    T: BasisElement,
{
    type ElemT = T;

    fn element(&self, index: usize) -> Self::ElemT {
        let reversed = self.0.element(self.0.size() - 1 - index);
        reversed.0
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<B, T, SbT> SplitByDimension for UnreverseBasis<B>
where
    B: ColBasis<ElemT = Reverse<T>>,
    B: SplitByDimension<SubBasisT = SbT>,
    SbT: ColBasis<ElemT = Reverse<T>>,
    T: BasisElement,
{
    type SubBasisT = UnreverseBasis<SbT>;
    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        let reversed = self.0.in_dimension(dimension);
        unsafe { std::mem::transmute(reversed) }
    }
}
