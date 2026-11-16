//! `FilteredComplex.extended_persistence`: extended persistence (see [`crate::extended`]), returned as a
//! `phlite.core.ExtendedBarcode`.

use phlite::{columns::ColumnRepresentation, extended::ExtendedBar};
use phlite_bindings::AlgorithmVisitor;

use crate::complex::{Cell, Complex};
use crate::extended::extended_persistence;
use crate::field::IntField;

/// Compute extended persistence with whichever algorithm is configured.
pub(crate) struct ExtendedRun<'a, CF> {
    pub(crate) complex: &'a Complex<CF>,
    pub(crate) method: phlite::barcode::Method,
}

impl<CF: IntField> AlgorithmVisitor for ExtendedRun<'_, CF> {
    type Output = Vec<ExtendedBar<Cell, f64>>;

    fn visit<R: ColumnRepresentation>(self, algorithm: phlite::barcode::Algorithm<'_, R>) -> Self::Output {
        extended_persistence(self.complex, self.method, algorithm)
    }
}
