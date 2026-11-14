//! The coboundary matrix of a cubical complex, as a lazy oracle.

use std::marker::PhantomData;

use ordered_float::NotNan;
use phlite::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{HasRowFiltration, MatrixOracle},
};

use super::{grid::Neighbour, Cube, CubicalComplex};

/// The coefficient `-1` if `negative`, else `1`.
#[inline]
pub(crate) fn coefficient<CF: NonZeroCoefficient>(negative: bool) -> CF {
    if negative {
        CF::one().additive_inverse()
    } else {
        CF::one()
    }
}

/// The value of a cell, as a filtration value.
#[inline]
pub(crate) fn not_nan(value: f64) -> NotNan<f64> {
    NotNan::new(value).expect("cell values are never NaN")
}

/// The coboundary oracle of a [`CubicalComplex`]: column `c` has an entry in the row of every coface of `c` that is in
/// the complex. Its transpose is [`CubicalBoundary`](super::homology::CubicalBoundary).
pub struct CubicalCoboundary<'a, CF: NonZeroCoefficient + Invertible, const D: usize> {
    complex: &'a CubicalComplex<D>,
    phantom: PhantomData<CF>,
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> Clone for CubicalCoboundary<'_, CF, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> Copy for CubicalCoboundary<'_, CF, D> {}

impl<'a, CF: NonZeroCoefficient + Invertible, const D: usize> CubicalCoboundary<'a, CF, D> {
    /// The coboundary oracle on its own, without a column basis attached (see
    /// [`build_bases`](super::build_bases)). The complex must have at least one pixel.
    pub fn new(complex: &'a CubicalComplex<D>) -> Self {
        Self {
            complex,
            phantom: PhantomData,
        }
    }

    /// The cofaces of `col` that are in the complex, with their values.
    #[inline]
    fn cofaces(&self, col: Cube) -> impl Iterator<Item = (f64, Neighbour<D>)> + '_ {
        let grid = self.complex.grid_ref();
        grid.cofaces(&grid.decode(col))
            .into_iter()
            .filter_map(|coface| {
                let value = self.complex.internal_value(&coface.decoded);
                (value != f64::INFINITY).then_some((value, coface))
            })
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> MatrixOracle
    for CubicalCoboundary<'_, CF, D>
{
    type CoefficientField = CF;
    type ColT = Cube;
    type RowT = Cube;

    fn column(&self, col: Cube) -> impl Iterator<Item = (CF, Cube)> {
        self.cofaces(col)
            .map(|(_, coface)| (coefficient(coface.negative), coface.cube))
    }
}

impl<CF: NonZeroCoefficient + Invertible, const D: usize> HasRowFiltration
    for CubicalCoboundary<'_, CF, D>
{
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Cube) -> NotNan<f64> {
        not_nan(self.complex.internal_value_of(row))
    }

    fn column_with_filtration(
        &self,
        col: Cube,
    ) -> impl Iterator<Item = ColumnEntry<NotNan<f64>, Cube, CF>> {
        self.cofaces(col).map(|(value, coface)| ColumnEntry {
            filtration_value: not_nan(value),
            row_index: coface.cube,
            coeff: coefficient(coface.negative),
        })
    }

    // The pivot of the anti-transpose (which is what cohomology reduces) is the oldest coface, and the first coface
    // with the cell's own value is it (an emergent pair), see `CubicalComplex::oldest_coface`
    fn column_min(&self, col: Cube) -> Option<ColumnEntry<NotNan<f64>, Cube, CF>> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(col);
        let value = self.complex.internal_value(&cell);
        self.complex
            .oldest_coface(&cell, value)
            .map(|(value, coface)| ColumnEntry {
                filtration_value: not_nan(value),
                row_index: coface.cube,
                coeff: coefficient(coface.negative),
            })
    }

    // Row `r` has entries in the columns of its faces, so the last is its youngest face. This is what detects apparent
    // pairs in the anti-transpose (which reverses the order).
    fn last_column_in_row(&self, row: Cube) -> Option<Cube> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(row);
        let value = self.complex.internal_value(&cell);
        self.complex
            .youngest_face(&cell, value)
            .map(|(_, face)| face.cube)
    }

    fn first_column_in_row(&self, row: Cube) -> Option<Cube> {
        let grid = self.complex.grid_ref();
        let cell = grid.decode(row);
        grid.faces(&cell)
            .into_iter()
            .map(|face| {
                (
                    not_nan(self.complex.internal_value(&face.decoded)),
                    face.cube,
                )
            })
            .min()
            .map(|(_, cube)| cube)
    }
}
