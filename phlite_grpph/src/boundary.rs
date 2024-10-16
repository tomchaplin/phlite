use std::{
    iter::{self, once},
    marker::PhantomData,
};

use ordered_float::NotNan;
use phlite::{
    fields::NonZeroCoefficient,
    matrices::{adaptors::MatrixWithBasis, HasRowFiltration, MatrixOracle},
};

use crate::coboundary::{
    two_path_time, DigraphEdgeSet, DigraphFiltration, PathHom2Cell, PathHomCell, PathHomSingleBasis,
};

pub struct GrPPHBoundary<'a, CF, F: DigraphFiltration> {
    filtration: F,
    edge_set: &'a DigraphEdgeSet,
    phantom: PhantomData<CF>,
}

impl<'a, CF: NonZeroCoefficient, F: DigraphFiltration> GrPPHBoundary<'a, CF, F> {
    pub fn build(
        filtration: F,
        edge_set: &'a DigraphEdgeSet,
        basis: PathHomSingleBasis,
    ) -> MatrixWithBasis<Self, PathHomSingleBasis> {
        MatrixWithBasis {
            matrix: Self {
                filtration,
                edge_set,
                phantom: PhantomData,
            },
            basis,
        }
    }
}

impl<'a, CF, F: DigraphFiltration> MatrixOracle for GrPPHBoundary<'a, CF, F>
where
    CF: NonZeroCoefficient,
{
    type CoefficientField = CF;

    type ColT = PathHomCell;

    type RowT = PathHomCell;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let boundary: Box<dyn Iterator<Item = (Self::CoefficientField, Self::RowT)>> = match col {
            PathHomCell::Node(_n) => Box::new(iter::empty()),
            PathHomCell::Edge(s, t) => {
                let target = once((CF::one(), PathHomCell::Node(t)));
                let source = once((CF::one().additive_inverse(), PathHomCell::Node(s)));
                Box::new(target.chain(source))
            }
            PathHomCell::TwoCell(two_cell) => match two_cell {
                PathHom2Cell::DoubleEdge(a, b) => {
                    let ab = once((CF::one(), PathHomCell::Edge(a, b)));
                    let ba = once((CF::one(), PathHomCell::Edge(b, a)));
                    Box::new(ab.chain(ba))
                }
                PathHom2Cell::DirectedTriangle(a, b, c) => {
                    let bc = once((CF::one(), PathHomCell::Edge(b, c)));
                    let ac = once((CF::one().additive_inverse(), PathHomCell::Edge(a, c)));
                    let ab = once((CF::one(), PathHomCell::Edge(a, b)));
                    Box::new(bc.chain(ac).chain(ab))
                }
                PathHom2Cell::LongSquare(a, b, c, d) => {
                    let cd = once((CF::one(), PathHomCell::Edge(c, d)));
                    let ac = once((CF::one(), PathHomCell::Edge(a, c)));
                    let bd = once((CF::one().additive_inverse(), PathHomCell::Edge(b, d)));
                    let ab = once((CF::one().additive_inverse(), PathHomCell::Edge(a, b)));
                    Box::new(cd.chain(ac).chain(bd).chain(ab))
                }
            },
        };
        boundary
    }
}

impl<CF, F> HasRowFiltration for GrPPHBoundary<'_, CF, F>
where
    CF: NonZeroCoefficient,
    F: DigraphFiltration,
{
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        match row {
            PathHomCell::Node(_s) => unsafe { NotNan::new_unchecked(0.0) },
            PathHomCell::Edge(s, t) => {
                // This is the grounding - edges in graph are born at 0
                if self.edge_set.contains(&(s, t)) {
                    unsafe { NotNan::new_unchecked(0.0) }
                } else if let Some(time) = self.filtration.edge_time(&s, &t) {
                    time
                } else {
                    panic!("Asked for filtration value of unknown edge.")
                }
            }
            PathHomCell::TwoCell(cell) => match cell {
                PathHom2Cell::DoubleEdge(a, b) => {
                    two_path_time(&self.filtration, &a, &b, &a).unwrap()
                }
                PathHom2Cell::DirectedTriangle(a, b, c) => {
                    let abc_time = two_path_time(&self.filtration, &a, &b, &c).unwrap();
                    let ac_time = self.filtration.edge_time(&a, &c).unwrap();
                    abc_time.max(ac_time)
                }
                PathHom2Cell::LongSquare(a, b, c, d) => {
                    let abd_time = two_path_time(&self.filtration, &a, &b, &d).unwrap();
                    let acd_time = two_path_time(&self.filtration, &a, &c, &d).unwrap();
                    abd_time.max(acd_time)
                }
            },
        }
    }
}
