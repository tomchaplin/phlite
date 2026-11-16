//! Extended persistence of a filtered complex, by `phlite::extended`, going back down by the least value of each cell
//! and its faces.
//!
//! The core's [`Coned`] complex is built on the complex's own boundary and coboundary views, with the reverse value
//! `g(σ) = min_{τ ⊆ σ} f(τ)`, so that the superlevel subcomplexes `{g ≥ a}` are the largest subcomplexes of the
//! superlevel sets. For the lower-star filtration of values on the vertices (each cell at its highest vertex, as
//! `Chain::extend` makes it from the vertices' values) this is the upper-star filtration, the classical setting.

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
    extended::{extended_bars, Coned, ExtendedBar},
    matrices::MatrixOracle,
};

use crate::complex::{Cell, Complex};
use crate::field::IntField;

/// The extended persistence of `complex`'s filtration, computed by `method` and `algorithm` over its field: bars in
/// no particular order, with values and input cells, without those of length zero.
pub fn extended_persistence<CF: IntField, R: ColumnRepresentation>(
    complex: &Complex<CF>,
    method: Method,
    algorithm: Algorithm<'_, R>,
) -> Vec<ExtendedBar<Cell, f64>> {
    let n = complex.n_cells();
    let boundary = complex.boundary();
    // By rank: each cell's value going up (f), its value going down (g: the least f of it and its faces, which come
    // first), and its dimension
    let mut up: Vec<NotNan<f64>> = Vec::with_capacity(n);
    let mut down: Vec<NotNan<f64>> = Vec::with_capacity(n);
    let top = complex.top_dimension().unwrap_or(0);
    let mut by_dimension: Vec<Vec<Cell>> = vec![vec![]; top + 1];
    for rank in 0..n as u32 {
        let (_, value, dimension) = complex.at_rank(Cell(rank));
        let value = NotNan::new(value).expect("values are not NaN");
        let least = boundary.column(Cell(rank)).map(|(_, face)| down[face.idx()]).fold(value, NotNan::min);
        up.push(value);
        down.push(least);
        by_dimension[dimension].push(Cell(rank));
    }
    // The views are filtered by rank; going up, the real values give the same order (ranks follow value, then index),
    // and are what extended persistence compares with the values going down
    let values = up.clone();
    let filtered = complex.boundary().with_filtration(move |rank: Cell| values[rank.idx()]);
    let coned = Coned::new(filtered, complex.coboundary(), &by_dimension, |rank: Cell| down[rank.idx()]);
    extended_bars(&coned, method, algorithm)
        .into_iter()
        .map(|bar| ExtendedBar {
            kind: bar.kind,
            dimension: bar.dimension,
            birth: bar.birth.into_inner(),
            death: bar.death.into_inner(),
            birth_cell: complex.cell(bar.birth_cell),
            death_cell: complex.cell(bar.death_cell),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::{Chain, RawChain};
    use phlite::extended::Kind;
    use phlite::fields::{Z2, Z3};
    use phlite::reduction::Clearing;

    /// A graph (vertices then edges, as input cells) filtered by the lower-star filtration of `values`.
    fn graph<CF: IntField>(values: &[f64], edges: &[(i64, i64)]) -> Complex<CF> {
        let n = values.len();
        let mut raw = RawChain { dimensions: vec![0; n], indptr: vec![0; n + 1], indices: vec![], coefficients: Some(vec![]) };
        for &(u, v) in edges {
            raw.dimensions.push(1);
            raw.indices.extend([u, v]);
            raw.coefficients.as_mut().unwrap().extend([-1, 1]);
            raw.indptr.push(raw.indices.len() as i64);
        }
        let mut all = values.iter().map(|&v| Some(v)).collect::<Vec<_>>();
        all.extend(edges.iter().map(|_| None));
        let chain = Chain::<CF>::from_raw(raw).unwrap();
        let filled = chain.extend(all).unwrap();
        chain.filter(Some(filled)).unwrap()
    }

    fn run<CF: IntField>(complex: &Complex<CF>, method: Method) -> Vec<(Kind, usize, f64, f64)> {
        let mut bars: Vec<_> = extended_persistence(complex, method, Algorithm::Clearing(Clearing::new()))
            .into_iter()
            .map(|b| (b.kind, b.dimension, b.birth, b.death))
            .collect();
        bars.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bars
    }

    #[test]
    fn a_cycle() {
        // GUDHI: Ord (1, 2) in H0, Rel (2, 1) in H1, Ext+ (0, 3) in H0, Ext- (3, 0) in H1
        let expected = vec![
            (Kind::Ordinary, 0, 1.0, 2.0),
            (Kind::Relative, 1, 2.0, 1.0),
            (Kind::ExtendedPlus, 0, 0.0, 3.0),
            (Kind::ExtendedMinus, 1, 3.0, 0.0),
        ];
        let (values, edges) = ([0.0, 2.0, 1.0, 3.0], [(0, 1), (1, 2), (2, 3), (3, 0)]);
        for method in [Method::Cohomology, Method::Homology] {
            assert_eq!(run(&graph::<Z2>(&values, &edges), method), expected);
            assert_eq!(run(&graph::<Z3>(&values, &edges), method), expected);
        }
    }

    #[test]
    fn two_components() {
        let bars = run(&graph::<Z2>(&[0.0, 1.0, 5.0, 7.0], &[(0, 1), (2, 3)]), Method::Cohomology);
        assert_eq!(bars, vec![(Kind::ExtendedPlus, 0, 0.0, 1.0), (Kind::ExtendedPlus, 0, 5.0, 7.0)]);
    }
}
