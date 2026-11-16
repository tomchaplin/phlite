//! Extended persistence (Cohen-Steiner, Edelsbrunner and Harer) of any filtration that ends at a finite complex, given
//! a "reverse" filtration to come back down by.
//!
//! Ordinary persistence of a filtration of a complex `K` leaves the homology of `K` itself essential: a component of
//! a mesh is born at its lowest point and never dies, and so is a loop of a graph. Extended persistence goes up
//! through the filtration (each cell `σ` at its value `f(σ)`), then back down through the pairs `(K, K^{≥ a})`,
//! where `K^{≥ a} = {σ : g(σ) ≥ a}` is given by a second value on each cell, `g`, no more than its faces'
//! (`g(τ) ≥ g(σ)` for a face `τ` of `σ`, so each `K^{≥ a}` is a subcomplex). Then every class dies, and each bar is
//! one of four [`Kind`]s:
//! * **ordinary**: born and dying on the way up (as in ordinary persistence);
//! * **relative**: born and dying on the way down;
//! * **extended**: born on the way up and dying on the way down: **Ext⁺** if it dies above its birth (a component,
//!   from its lowest point to its highest), **Ext⁻** below (a loop, from its highest point to its lowest).
//!
//! The classical setting is a function on the vertices: `f(σ)` the largest of `σ`'s vertices' values (lower-star),
//! `g(σ)` the smallest (upper-star). But any pair will do: `g(σ) = min_{τ ⊆ σ} f(τ)` for any filtration `f`, say, or
//! a flag complex's shortest edge.
//!
//! # How
//!
//! By coning, as GUDHI does: a cone vertex `ω`, then `K` in its filtration order, then the cone `ω * σ` on every cell
//! in decreasing order of `g`, with boundary `∂(ω * σ) = σ - ω * ∂σ` (and `∂(ω * v) = v - ω` for a vertex). Here the
//! coned complex is never listed: [`Coned`] wraps the complex's boundary matrix and its transpose (any phlite
//! oracles), and gives the coned complex's boundary and coboundary as oracles ([`ConeCell`]s filtered by
//! [`ConeValue`]s), reduced by any [`Algorithm`]. [`extended_bars`] reads the four kinds off where each bar's cells
//! are. The cone vertex's component is the only essential class, and is left out.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::{
    barcode::{cohomology, homology, Algorithm, Cell, Field, Method},
    columns::ColumnRepresentation,
    fields::NonZeroCoefficient,
    matrices::{bases::SplitBasis, BasisElement, FiltrationValue, HasRowFiltration, MatrixOracle, RowRanks},
};

/// A cell of the coned complex: the cone vertex, a cell of the complex, or the cone on one. The derived order (apex,
/// then the complex's cells, then the cones, each in the complex's own order) breaks ties in filtration value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConeCell<C> {
    /// The cone vertex `ω`, first.
    Apex,
    /// A cell `σ` of the complex, entering on the way up.
    Up(C),
    /// The cone `ω * σ` on a cell, entering on the way down.
    Down(C),
}

/// Densely numbered when the complex's cells are: the apex 0, then `2i + 1` going up and `2i + 2` for the cone, where
/// `i` is the cell's own number.
impl<C: BasisElement> BasisElement for ConeCell<C> {
    const DENSE_INDEX: bool = C::DENSE_INDEX;

    #[inline]
    fn dense_index(&self) -> usize {
        match self {
            ConeCell::Apex => 0,
            ConeCell::Up(c) => 2 * c.dense_index() + 1,
            ConeCell::Down(c) => 2 * c.dense_index() + 2,
        }
    }

    #[inline]
    fn from_dense_index(index: usize) -> Self {
        match index {
            0 => ConeCell::Apex,
            i if i % 2 == 1 => ConeCell::Up(C::from_dense_index(i / 2)),
            i => ConeCell::Down(C::from_dense_index(i / 2 - 1)),
        }
    }
}

/// When a cell of the coned complex enters: the apex first, then the complex by its own values, then the cones by
/// decreasing reverse value (so the derived order is the filtration order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConeValue<F> {
    Apex,
    Up(F),
    Down(Reverse<F>),
}

impl<F: FiltrationValue> FiltrationValue for ConeValue<F> {}

/// The four kinds of extended persistence bar (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Ordinary,
    Relative,
    ExtendedPlus,
    ExtendedMinus,
}

impl Kind {
    /// `'ordinary'`, `'relative'`, `'extended+'` or `'extended-'`.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Ordinary => "ordinary",
            Kind::Relative => "relative",
            Kind::ExtendedPlus => "extended+",
            Kind::ExtendedMinus => "extended-",
        }
    }
}

/// A bar of extended persistence: its kind; its dimension (for a relative bar, its dimension in the coned complex,
/// one more than its cells', as GUDHI reports it); its birth and death values (`f` going up, `g` going down); and
/// the complex's cells whose entry gives them (on the way down, the cell `σ` whose cone `ω * σ` enters).
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedBar<C, F> {
    pub kind: Kind,
    pub dimension: usize,
    pub birth: F,
    pub death: F,
    pub birth_cell: C,
    pub death_cell: C,
}

/// The coned complex of a filtered complex (see the [module docs](self)): its boundary `boundary` (filtered by `f`,
/// with every cell, split by dimension, in `basis`), its transpose `coboundary`, and the reverse values `down` (`g`).
pub struct Coned<M, T, G, C> {
    boundary: M,
    coboundary: T,
    down: G,
    /// The cells of dimension 0, whose cones' boundaries have the apex
    vertices: HashSet<C>,
    /// Each cell's dimension in the complex
    dimensions: HashMap<C, usize>,
    /// The coned complex's cells, by dimension, in filtration order
    basis: SplitBasis<ConeCell<C>>,
    /// If the cells are densely numbered, each cone cell's rank in the order on (value, cell), by dense index: so
    /// that [`PivotColumns`](crate::columns::PivotColumns) can reduce the coned complex
    ranks: Option<RowRanks>,
}

impl<C, F, M, T, G> Coned<M, T, G, C>
where
    C: Cell,
    F: FiltrationValue,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = M::CoefficientField>,
    G: Fn(C) -> F,
{
    /// Cone the complex whose cells are `basis` (by dimension, each in increasing filtration order: the order its
    /// pipeline reduces it in), with boundary `boundary` and coboundary `coboundary` (its transpose), coming back
    /// down by `down`.
    ///
    /// # Panics
    /// In debug builds, if `down` increases from a face to a coface (`down(τ) < down(σ)` for a face `τ` of `σ`).
    pub fn new(boundary: M, coboundary: T, basis: &[Vec<C>], down: G) -> Self {
        let vertices: HashSet<C> = basis.first().map_or_else(HashSet::new, |v| v.iter().cloned().collect());
        let dimensions: HashMap<C, usize> =
            basis.iter().enumerate().flat_map(|(d, cells)| cells.iter().map(move |c| (c.clone(), d))).collect();
        #[cfg(debug_assertions)]
        for cell in dimensions.keys() {
            for (_, face) in boundary.column(cell.clone()) {
                debug_assert!(down(face.clone()) >= down(cell.clone()), "the reverse values must not increase from a face to a coface");
            }
        }
        // Dimension d of the coned complex: (the apex), the complex's d-cells going up, then the cones on its
        // (d-1)-cells, by decreasing reverse value (ties as the complex orders them, faces first)
        let top = basis.len();
        let mut dims: Vec<Vec<ConeCell<C>>> = vec![vec![]; top + 1];
        dims[0].push(ConeCell::Apex);
        // (Every dimension's cells going up first, then the cones, which all come after them)
        for (d, cells) in basis.iter().enumerate() {
            dims[d].extend(cells.iter().cloned().map(ConeCell::Up));
        }
        for (d, cells) in basis.iter().enumerate() {
            let mut cones: Vec<C> = cells.clone();
            cones.sort_by(|a, b| down(b.clone()).cmp(&down(a.clone())).then(a.cmp(b)));
            dims[d + 1].extend(cones.into_iter().map(ConeCell::Down));
        }
        let mut coned = Self {
            boundary,
            coboundary,
            down,
            vertices,
            dimensions,
            basis: SplitBasis::from_dimensions(dims.clone()),
            ranks: None,
        };
        if C::DENSE_INDEX {
            coned.ranks = Some(coned.rank_table(dims.into_iter().flatten()));
        }
        coned
    }

    /// The rank of every cone cell in the order on (value, cell), as a table by dense index. Numbers that are no cell's
    /// get the ranks after every cell's, so that the ranks are distinct.
    fn rank_table(&self, cells: impl Iterator<Item = ConeCell<C>>) -> RowRanks {
        let mut cells: Vec<(ConeValue<F>, ConeCell<C>)> = cells.map(|cell| (self.value(&cell), cell)).collect();
        cells.sort_unstable();
        let size = cells.iter().map(|(_, cell)| cell.dense_index() + 1).max().unwrap_or(0);
        let mut table = vec![u32::MAX; size];
        for (rank, (_, cell)) in cells.iter().enumerate() {
            table[cell.dense_index()] = u32::try_from(rank).expect("fewer than 2^32 cells");
        }
        let mut next = cells.len() as u32;
        for rank in table.iter_mut().filter(|rank| **rank == u32::MAX) {
            *rank = next;
            next += 1;
        }
        RowRanks::new(table.into())
    }

    /// The coned complex's boundary matrix, ready for [`homology`].
    pub fn boundary(&self) -> ConedBoundary<'_, M, T, G, C> {
        ConedBoundary(self)
    }

    /// The coned complex's coboundary matrix, ready for [`cohomology`].
    pub fn coboundary(&self) -> ConedCoboundary<'_, M, T, G, C> {
        ConedCoboundary(self)
    }

    /// The coned complex's basis, by dimension.
    pub fn basis(&self) -> &SplitBasis<ConeCell<C>> {
        &self.basis
    }

    fn value(&self, cell: &ConeCell<C>) -> ConeValue<F> {
        match cell {
            ConeCell::Apex => ConeValue::Apex,
            ConeCell::Up(c) => ConeValue::Up(self.boundary.filtration_value(c.clone())),
            ConeCell::Down(c) => ConeValue::Down(Reverse((self.down)(c.clone()))),
        }
    }
}

/// The boundary matrix of a [`Coned`] complex.
pub struct ConedBoundary<'a, M, T, G, C>(&'a Coned<M, T, G, C>);
/// The coboundary matrix of a [`Coned`] complex.
pub struct ConedCoboundary<'a, M, T, G, C>(&'a Coned<M, T, G, C>);

impl<M, T, G, C> Clone for ConedBoundary<'_, M, T, G, C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M, T, G, C> Copy for ConedBoundary<'_, M, T, G, C> {}
impl<M, T, G, C> Clone for ConedCoboundary<'_, M, T, G, C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M, T, G, C> Copy for ConedCoboundary<'_, M, T, G, C> {}

impl<C, F, M, T, G> MatrixOracle for ConedBoundary<'_, M, T, G, C>
where
    C: Cell,
    F: FiltrationValue,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = M::CoefficientField>,
    G: Fn(C) -> F,
{
    type CoefficientField = M::CoefficientField;
    type ColT = ConeCell<C>;
    type RowT = ConeCell<C>;

    fn column(&self, col: ConeCell<C>) -> impl Iterator<Item = (M::CoefficientField, ConeCell<C>)> {
        let coned = self.0;
        let one = M::CoefficientField::one();
        let (faces, cone): (Vec<_>, Vec<_>) = match col {
            ConeCell::Apex => (vec![], vec![]),
            ConeCell::Up(c) => (coned.boundary.column(c).map(|(a, f)| (a, ConeCell::Up(f))).collect(), vec![]),
            // ∂(ω * σ) = σ - ω * ∂σ, and ∂(ω * v) = v - ω
            ConeCell::Down(c) => {
                let mut entries = vec![(one, ConeCell::Up(c.clone()))];
                if coned.vertices.contains(&c) {
                    entries.push((one.additive_inverse(), ConeCell::Apex));
                } else {
                    entries.extend(coned.boundary.column(c).map(|(a, f)| (a.additive_inverse(), ConeCell::Down(f))));
                }
                (entries, vec![])
            }
        };
        faces.into_iter().chain(cone)
    }
}

impl<C, F, M, T, G> MatrixOracle for ConedCoboundary<'_, M, T, G, C>
where
    C: Cell,
    F: FiltrationValue,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = M::CoefficientField>,
    G: Fn(C) -> F,
{
    type CoefficientField = M::CoefficientField;
    type ColT = ConeCell<C>;
    type RowT = ConeCell<C>;

    /// The transpose of [`ConedBoundary`]'s: `δω = -Σ_v ω * v`, `δσ = δ_K σ + ω * σ`, and `δ(ω * σ) = -ω * δ_K σ`.
    fn column(&self, col: ConeCell<C>) -> impl Iterator<Item = (M::CoefficientField, ConeCell<C>)> {
        let coned = self.0;
        let one = M::CoefficientField::one();
        let entries: Vec<_> = match col {
            ConeCell::Apex => coned.vertices.iter().map(|v| (one.additive_inverse(), ConeCell::Down(v.clone()))).collect(),
            ConeCell::Up(c) => coned
                .coboundary
                .column(c.clone())
                .map(|(a, f)| (a, ConeCell::Up(f)))
                .chain(std::iter::once((one, ConeCell::Down(c))))
                .collect(),
            ConeCell::Down(c) => {
                coned.coboundary.column(c).map(|(a, f)| (a.additive_inverse(), ConeCell::Down(f))).collect()
            }
        };
        entries.into_iter()
    }
}

impl<C, F, M, T, G> HasRowFiltration for ConedBoundary<'_, M, T, G, C>
where
    C: Cell,
    F: FiltrationValue,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = M::CoefficientField>,
    G: Fn(C) -> F,
{
    type FiltrationT = ConeValue<F>;
    fn filtration_value(&self, row: ConeCell<C>) -> ConeValue<F> {
        self.0.value(&row)
    }
    fn row_ranks(&self) -> Option<RowRanks> {
        self.0.ranks.clone()
    }
}

impl<C, F, M, T, G> HasRowFiltration for ConedCoboundary<'_, M, T, G, C>
where
    C: Cell,
    F: FiltrationValue,
    M: MatrixOracle<ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = M::CoefficientField>,
    G: Fn(C) -> F,
{
    type FiltrationT = ConeValue<F>;
    fn filtration_value(&self, row: ConeCell<C>) -> ConeValue<F> {
        self.0.value(&row)
    }
    fn row_ranks(&self) -> Option<RowRanks> {
        self.0.ranks.clone()
    }
}

/// The extended persistence of the complex `coned` was made from, by `method` and `algorithm`, as
/// [`ExtendedBar`]s in no particular order. `is_zero_length(birth, death)` says when a bar's two values are equal
/// (its length zero), to leave it out.
pub fn extended_bars<C, F, CF, M, T, G, R>(
    coned: &Coned<M, T, G, C>,
    method: Method,
    algorithm: Algorithm<'_, R>,
) -> Vec<ExtendedBar<C, F>>
where
    C: Cell,
    F: FiltrationValue + Send + Sync,
    CF: Field,
    M: MatrixOracle<ColT = C, RowT = C, CoefficientField = CF> + HasRowFiltration<FiltrationT = F> + crate::barcode::MaybeSync,
    T: MatrixOracle<ColT = C, RowT = C, CoefficientField = CF> + crate::barcode::MaybeSync,
    G: Fn(C) -> F + crate::barcode::MaybeSync,
    R: ColumnRepresentation,
{
    use crate::matrices::MatrixOracle as _;
    // (A copy of the basis: a matrix with a basis split by dimension owns it)
    let bars = match method {
        Method::Homology => homology(&coned.boundary().with_basis(coned.basis().clone()), algorithm, false),
        Method::Cohomology => cohomology(&coned.coboundary().with_basis(coned.basis().clone()), algorithm),
    };
    let up = |c: &C| coned.boundary.filtration_value(c.clone());
    let down = |c: &C| (coned.down)(c.clone());
    let dim = |c: &C| coned.dimensions[c];
    let mut out = vec![];
    for bar in bars {
        // Only the apex's component is essential
        let Some((death, _)) = bar.death else { continue };
        let extended = match (bar.birth, death) {
            (ConeCell::Up(b), ConeCell::Up(d)) => {
                ExtendedBar { kind: Kind::Ordinary, dimension: dim(&b), birth: up(&b), death: up(&d), birth_cell: b, death_cell: d }
            }
            (ConeCell::Down(b), ConeCell::Down(d)) => ExtendedBar {
                kind: Kind::Relative,
                dimension: dim(&b) + 1,
                birth: down(&b),
                death: down(&d),
                birth_cell: b,
                death_cell: d,
            },
            (ConeCell::Up(b), ConeCell::Down(d)) => {
                let (birth, death) = (up(&b), down(&d));
                let kind = if birth < death { Kind::ExtendedPlus } else { Kind::ExtendedMinus };
                ExtendedBar { kind, dimension: dim(&b), birth, death, birth_cell: b, death_cell: d }
            }
            _ => unreachable!("the apex is never paired, and the cones come after the complex"),
        };
        if extended.birth != extended.death {
            out.push(extended);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{Z2, Z3};
    use crate::matrices::implementors::VecVecMatrix;
    use crate::reduction::Clearing;

    /// The 4-cycle 0-1-2-3 with vertex values 0, 2, 1, 3: vertices are cells 0..4, edges 4..8 (with signs, so it
    /// works over any field), filtered lower-star and coming back upper-star.
    fn cycle<CF: NonZeroCoefficient>() -> (Vec<Vec<(CF, usize)>>, Vec<Vec<(CF, usize)>>, Vec<usize>) {
        let (one, minus) = (CF::one(), CF::one().additive_inverse());
        let values = [0usize, 2, 1, 3];
        let edges = [(0, 1), (1, 2), (2, 3), (0, 3)];
        let mut boundary: Vec<Vec<(CF, usize)>> = vec![vec![]; 4];
        let mut coboundary: Vec<Vec<(CF, usize)>> = vec![vec![]; 8];
        for (i, &(u, v)) in edges.iter().enumerate() {
            boundary.push(vec![(minus, u), (one, v)]);
            coboundary[u].push((minus, 4 + i));
            coboundary[v].push((one, 4 + i));
        }
        let mut f: Vec<usize> = values.to_vec();
        f.extend(edges.iter().map(|&(u, v)| values[u].max(values[v])));
        (boundary, coboundary, f)
    }

    fn run<CF: Field>(method: Method) -> Vec<(Kind, usize, usize, usize)> {
        let (boundary, coboundary, f) = cycle::<CF>();
        let values = [0usize, 2, 1, 3];
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (0, 3)];
        let g = |c: usize| if c < 4 { values[c] } else { let (u, v) = edges[c - 4]; values[u].min(values[v]) };
        let fv = f.clone();
        // Each dimension's cells in increasing (f, cell) order
        let mut vertices: Vec<usize> = (0..4).collect();
        vertices.sort_by_key(|&c| (fv[c], c));
        let mut edge_cells: Vec<usize> = (4..8).collect();
        edge_cells.sort_by_key(|&c| (fv[c], c));
        let matrix = VecVecMatrix::from(boundary).with_filtration(move |c: usize| fv[c]);
        let coned = Coned::new(matrix, VecVecMatrix::from(coboundary), &[vertices, edge_cells], g);
        {
            use crate::matrices::contract::{check_boundary, check_coboundary, check_split_basis, check_transposes, Direction};
            let b = coned.boundary().with_basis(coned.basis().clone());
            let c = coned.coboundary().with_basis(coned.basis().clone());
            assert_eq!(check_boundary(&b), Ok(()), "boundary");
            assert_eq!(check_split_basis(&b, Direction::Boundary), Ok(()), "boundary basis");
            assert_eq!(check_coboundary(&c), Ok(()), "coboundary");
            assert_eq!(check_split_basis(&c, Direction::Coboundary), Ok(()), "coboundary basis");
            assert_eq!(check_transposes(&b, coned.coboundary()), Ok(()), "transposes");
        }
        let mut bars: Vec<_> = extended_bars(&coned, method, Algorithm::Clearing(Clearing::new()))
            .into_iter()
            .map(|b| (b.kind, b.dimension, b.birth, b.death))
            .collect();
        bars.sort();
        bars
    }

    #[test]
    fn a_cycle_as_gudhi() {
        // GUDHI: Ord (1, 2) in H0, Rel (2, 1) in H1, Ext+ (0, 3) in H0, Ext- (3, 0) in H1
        let expected = vec![
            (Kind::Ordinary, 0, 1, 2),
            (Kind::Relative, 1, 2, 1),
            (Kind::ExtendedPlus, 0, 0, 3),
            (Kind::ExtendedMinus, 1, 3, 0),
        ];
        for method in [Method::Homology, Method::Cohomology] {
            assert_eq!(run::<Z2>(method), expected, "{method:?}");
            assert_eq!(run::<Z3>(method), expected);
        }
    }
}
