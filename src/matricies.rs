use crate::{fields::CoefficientField, PhliteError};
pub trait BasisElement: Ord {}

pub trait MatrixOracle {
    type CoefficientField: CoefficientField;
    type ColT: BasisElement;
    type RowT: BasisElement;

    /// Implement your oracle on the widest range of [`ColT`](Self::ColT) possible.
    /// To specify a given matrix, you will later provide an oracle, alongside a basis for the column space.
    /// If you are unable to produce a column, please return [`PhliteError::NotInDomain`].
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::RowT, Self::CoefficientField)>, PhliteError>;
}
