//! The boundary matrix of a cubical complex, as a lazy oracle.

use std::marker::PhantomData;

use ordered_float::NotNan;
use phlite::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{HasRowFiltration, MatrixOracle},
};

use super::{
    cohomology::{coefficient, not_nan},
    Cube, CubicalComplex,
};

/// The boundary oracle of a [`CubicalComplex`]: column `c` has an entry in the row of every face of `c`. Its transpose
/// is [`CubicalCoboundary`](super::cohomology::CubicalCoboundary).
///
/// The faces of a cell in the complex are always in it, since they enter no later.
pub struct CubicalBoundary<'a, CF: NonZeroCoefficient + Invertible, const D: usize> {
    complex: &'a CubicalComplex<D>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> Clone for CubicalBoundary<'_, CF, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> Copy for CubicalBoundary<'_, CF, D> {}

impl<'a, CF: NonZeroCoefficient + Invertible, const D: usize> CubicalBoundary<'a, CF, D> {
    /// The boundary oracle on its own, without a column basis attached (see [`build_bases`](super::build_bases)).
    /// The complex must have at least one pixel.
    pub fn new(complex: &'a CubicalComplex<D>) -> Self {
        Self {
            complex,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> MatrixOracle
    for CubicalBoundary<'_, CF, D>
{
    type CoefficientField = CF;
    type ColT = Cube;
    type RowT = Cube;

    fn column(&self, col: Cube) -> impl Iterator<Item = (CF, Cube)> {
        let grid = self.complex.grid_ref();
        grid.faces(&grid.decode(col))
            .into_iter()
            .map(|face| (coefficient(face.negative), face.cube))
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> HasRowFiltration
    for CubicalBoundary<'_, CF, D>
{
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Cube) -> NotNan<f64> {
        not_nan(self.complex.internal_value_of(row))
    }

    fn column_with_filtration(
        &self,
        col: Cube,
    ) -> impl Iterator<Item = ColumnEntry<NotNan<f64>, Cube, CF>> {
        let grid = self.complex.grid_ref();
        grid.faces(&grid.decode(col))
            .into_iter()
            .map(|face| ColumnEntry {
                filtration_value: not_nan(self.complex.internal_value(&face.decoded)),
                row_index: face.cube,
                coeff: coefficient(face.negative),
            })
    }

    // The pivot is the youngest face, and the first face with the cell's own value is it, see
    // `CubicalComplex::youngest_face`
    fn column_pivot(&self, col: Cube) -> Option<ColumnEntry<NotNan<f64>, Cube, CF>> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(col);
        let value = self.complex.internal_value(&cell);
        self.complex
            .youngest_face(&cell, value)
            .map(|(value, face)| ColumnEntry {
                filtration_value: not_nan(value),
                row_index: face.cube,
                coeff: coefficient(face.negative),
            })
    }

    // Row `r` has entries in the columns of its cofaces, so the first is its oldest coface (in the complex). This is
    // what detects apparent pairs.
    fn first_column_in_row(&self, row: Cube) -> Option<Cube> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(row);
        let value = self.complex.internal_value(&cell);
        self.complex
            .oldest_coface(&cell, value)
            .map(|(_, coface)| coface.cube)
    }

    fn last_column_in_row(&self, row: Cube) -> Option<Cube> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(row);
        grid.cofaces(&cell)
            .into_iter()
            .filter_map(|coface| {
                let value = self.complex.internal_value(&coface.decoded);
                (value != f64::INFINITY).then(|| (not_nan(value), coface.cube))
            })
            .max()
            .map(|(_, cube)| cube)
    }
}
