//! High-level helpers for computing a persistence barcode, optionally with a representative for every bar.
//!
//! The [`reduction`](crate::reduction) algorithms give you a reduction matrix `V` and a [`Diagram`] of paired *columns*.
//! In applications you usually want *bars*: a birth and (optionally) a death, both with filtration values,
//! and perhaps a cycle representing the homology class.
//! This module turns a matrix (plus a choice of [`Algorithm`]) into a list of [`Bar`]s, handling the
//! cohomology-vs-homology bookkeeping for you.
//!
//! There are two routes:
//! * [`homology`] reduces a *boundary* matrix. Representatives come for free: the column of `R = D V`
//!   at the death cell for finite bars, and the column of `V` at the birth cell for essential bars.
//! * [`cohomology`] reduces the anti-transpose of a *coboundary* matrix, which is typically much faster.
//!   For representatives use [`cohomology_with_involution`], which additionally takes a boundary oracle
//!   and runs a small homology reduction restricted to the cells that matter
//!   (the death cells of finite bars and the birth cells of essential bars), as in Čufar's "involuted" persistence.
//!
//! In both cases the bars are reported in *homology* conventions: `birth` is the lower dimensional cell.

use std::{cmp::Reverse, fmt::Debug, hash::Hash};

use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{
        combinators::product, BasisElement, FiltrationValue, HasRowFiltration, HasSplitColBasis,
        MatrixOracle,
    },
    reduction::{Clearing, Diagram, Reduction, ReductionAlgorithm, Standard},
};

#[cfg(feature = "parallel")]
use crate::reduction::ParallelClearing;

/// Which reduction algorithm to run, see the [`reduction`](crate::reduction) module docs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Algorithm<'p> {
    /// [`Standard`], the standard algorithm with no optimisations.
    Standard(Standard),
    /// [`Clearing`], the standard algorithm with the clearing optimisation.
    Clearing(Clearing),
    /// [`ParallelClearing`], clearing with each dimension reduced in parallel (feature `parallel`).
    #[cfg(feature = "parallel")]
    ParallelClearing(ParallelClearing<'p>),
    #[doc(hidden)]
    #[cfg(not(feature = "parallel"))]
    _Lifetime(std::marker::PhantomData<&'p ()>, std::convert::Infallible),
}

impl Default for Algorithm<'_> {
    /// [`Clearing`] with the dimension order detected from the matrix.
    fn default() -> Self {
        Algorithm::Clearing(Clearing::new())
    }
}

/// Whether to compute a barcode by reducing the coboundary or the boundary matrix.
///
/// The functions in this module take matrices rather than a [`Method`], since the two methods need different matrices.
/// This is for applications that build the right matrix for you (such as `phlite_rips` and `phlite_grpph`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    /// Reduce the anti-transposed coboundary matrix (see [`cohomology`]), computing representatives by involution.
    /// Typically much faster.
    #[default]
    Cohomology,
    /// Reduce the boundary matrix (see [`homology`]).
    Homology,
}

/// A bar in a persistence barcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bar<C, F, CF> {
    /// The cell whose appearance creates the homology class.
    pub birth: C,
    /// The filtration value of [`birth`](Self::birth).
    pub birth_value: F,
    /// The cell whose appearance kills the homology class, together with its filtration value, or `None` for an essential bar.
    pub death: Option<(C, F)>,
    /// A cycle representing the homology class, as a linear combination of cells of the same dimension as [`birth`](Self::birth),
    /// in decreasing filtration order. `None` if representatives were not requested.
    ///
    /// For a finite bar this is a cycle whose pivot (leading term) is `birth` and which becomes a boundary at `death`.
    pub representative: Option<Vec<(CF, C)>>,
}

impl<C, F: PartialEq, CF> Bar<C, F, CF> {
    /// Whether this bar is finite with `birth_value == death_value`.
    /// Most applications discard such bars.
    pub fn is_zero_length(&self) -> bool {
        matches!(&self.death, Some((_, death_value)) if *death_value == self.birth_value)
    }
}

// ====== Bounds needed to run every algorithm ==========================

/// `Send + Sync` when the `parallel` feature is enabled (so that [`ParallelClearing`] can be used), otherwise nothing.
/// Blanket-implemented.
#[cfg(feature = "parallel")]
pub trait MaybeSync: Send + Sync {}
#[cfg(feature = "parallel")]
impl<T: Send + Sync> MaybeSync for T {}

/// `Send + Sync` when the `parallel` feature is enabled (so that `ParallelClearing` can be used), otherwise nothing.
/// Blanket-implemented.
#[cfg(not(feature = "parallel"))]
pub trait MaybeSync {}
#[cfg(not(feature = "parallel"))]
impl<T> MaybeSync for T {}

/// Requirements on cells (row and column indices) for every [`Algorithm`]. Blanket-implemented.
pub trait Cell: BasisElement + Hash + Debug + MaybeSync + 'static {}
impl<T: BasisElement + Hash + Debug + MaybeSync + 'static> Cell for T {}

/// Requirements on the coefficient field for every [`Algorithm`]. Blanket-implemented.
pub trait Field: Invertible + MaybeSync + 'static {}
impl<T: Invertible + MaybeSync + 'static> Field for T {}

/// A filtered matrix, with rows and columns indexed by `C` and a column basis split by dimension, that every [`Algorithm`] can reduce.
/// Blanket-implemented, you should never need to implement this yourself.
/// Typically this is a reference to a [`MatrixWithBasis`](crate::matrices::adaptors::MatrixWithBasis).
pub trait Reducible<C, F, CF>:
    MatrixOracle<CoefficientField = CF, ColT = C, RowT = C>
    + HasRowFiltration<FiltrationT = F>
    + HasSplitColBasis
    + MaybeSync
    + Copy
{
}

impl<C, F, CF, M> Reducible<C, F, CF> for M
where
    C: BasisElement,
    F: FiltrationValue,
    CF: NonZeroCoefficient,
    M: MatrixOracle<CoefficientField = CF, ColT = C, RowT = C>
        + HasRowFiltration<FiltrationT = F>
        + HasSplitColBasis
        + MaybeSync
        + Copy,
{
}

/// Generic over the type of `V`, which a closure cannot be.
trait ReductionVisitor<C, CF, Out> {
    fn visit<V>(self, reduction: Reduction<V, C>) -> Out
    where
        V: MatrixOracle<CoefficientField = CF, ColT = C, RowT = C>;
}

/// Run `algorithm` on `matrix` and pass the result to `then`.
/// The type of `V` depends on the algorithm, hence the visitor.
fn run<C, F, CF, M, Out>(
    matrix: M,
    algorithm: Algorithm<'_>,
    then: impl ReductionVisitor<C, CF, Out>,
) -> Out
where
    C: Cell,
    F: FiltrationValue + MaybeSync,
    CF: Field,
    M: Reducible<C, F, CF>,
{
    match algorithm {
        Algorithm::Standard(alg) => then.visit(alg.reduce(matrix)),
        Algorithm::Clearing(alg) => then.visit(alg.reduce(matrix)),
        #[cfg(feature = "parallel")]
        Algorithm::ParallelClearing(alg) => then.visit(alg.reduce(matrix)),
        #[cfg(not(feature = "parallel"))]
        Algorithm::_Lifetime(_, never) => match never {},
    }
}

/// Collect a column of `matrix` into a consolidated linear combination, in decreasing order with respect to the matrix's row filtration.
fn collect_column<M>(matrix: M, col: M::ColT) -> Vec<(M::CoefficientField, M::RowT)>
where
    M: HasRowFiltration,
{
    matrix
        .build_bhcol(col)
        .drain_sorted()
        .map(|entry| (entry.coeff, entry.row_index))
        .collect()
}

// ====== Homology =======================================================

/// Compute the barcode of a filtered chain complex by reducing its `boundary` matrix.
///
/// Every cell in the column basis of `boundary` is reduced, so to get correct bars in dimensions `0..=max_dim`
/// you need the basis to contain cells up to dimension `max_dim + 1`.
/// Essential bars born in the top dimension of the basis are then spurious (nothing was there to kill them) and you should discard them.
///
/// If `representatives` is true then every bar comes with a representative cycle:
/// the column of `R = D V` at the death cell for finite bars, or the column of `V` at the birth cell for essential bars.
pub fn homology<C, F, CF, M>(
    boundary: M,
    algorithm: Algorithm<'_>,
    representatives: bool,
) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + MaybeSync,
    CF: Field,
    M: Reducible<C, F, CF>,
{
    struct Visitor<M>(M, bool);
    impl<C, F, CF, M> ReductionVisitor<C, CF, Vec<Bar<C, F, CF>>> for Visitor<M>
    where
        C: Cell,
        F: FiltrationValue,
        CF: Field,
        M: Reducible<C, F, CF>,
    {
        fn visit<V>(self, reduction: Reduction<V, C>) -> Vec<Bar<C, F, CF>>
        where
            V: MatrixOracle<CoefficientField = CF, ColT = C, RowT = C>,
        {
            let Visitor(boundary, representatives) = self;
            let v = &reduction.v;
            let r = product(boundary, v);
            let Diagram {
                essential,
                pairings,
            } = reduction.diagram;

            let finite = pairings.into_iter().map(|(birth, death)| Bar {
                birth_value: boundary.filtration_value(birth.clone()),
                representative: representatives.then(|| collect_column(&r, death.clone())),
                death: Some((death.clone(), boundary.filtration_value(death))),
                birth,
            });
            let essential = essential.into_iter().map(|birth| Bar {
                birth_value: boundary.filtration_value(birth.clone()),
                representative: representatives
                    .then(|| collect_column(v.with_trivial_filtration(), birth.clone())),
                death: None,
                birth,
            });
            finite.chain(essential).collect()
        }
    }
    run(boundary, algorithm, Visitor(boundary, representatives))
}

// ====== Cohomology =====================================================

struct DiagramVisitor;

impl<C, CF> ReductionVisitor<C, CF, Diagram<C>> for DiagramVisitor {
    fn visit<V>(self, reduction: Reduction<V, C>) -> Diagram<C>
    where
        V: MatrixOracle<CoefficientField = CF, ColT = C, RowT = C>,
    {
        reduction.diagram
    }
}

/// Compute the barcode of a filtered chain complex from its `coboundary` matrix.
///
/// We reduce the anti-transpose ([`reverse`](MatrixOracle::reverse)) of `coboundary`, so pass the coboundary itself.
/// Unlike [`homology`], the column basis only needs cells up to dimension `max_dim` to get correct bars in dimensions `0..=max_dim`.
///
/// No representatives are computed, see [`cohomology_with_involution`].
pub fn cohomology<C, F, CF, M>(coboundary: M, algorithm: Algorithm<'_>) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + MaybeSync,
    CF: Field,
    M: Reducible<C, F, CF>,
{
    let diagram: Diagram<Reverse<C>> = run(coboundary.reverse(), algorithm, DiagramVisitor);

    // The anti-transpose has columns in dimension k and rows in dimension k + 1,
    // so a pairing (pivot row, column) is (homology death, homology birth).
    let finite = diagram
        .pairings
        .into_iter()
        .map(|(Reverse(death), Reverse(birth))| Bar {
            birth_value: coboundary.filtration_value(birth.clone()),
            birth,
            death: Some((death.clone(), coboundary.filtration_value(death))),
            representative: None,
        });
    let essential = diagram.essential.into_iter().map(|Reverse(birth)| Bar {
        birth_value: coboundary.filtration_value(birth.clone()),
        birth,
        death: None,
        representative: None,
    });
    finite.chain(essential).collect()
}

/// As [`cohomology`], but also computes a representative cycle for every bar via the involution trick.
///
/// You must provide `boundary`, an oracle for the boundary matrix of the same filtered chain complex
/// (no column basis needed), with the same row filtration as `coboundary`.
/// We then reduce `boundary` restricted to the death cells of the finite bars and the birth cells of the essential bars,
/// which is typically a tiny fraction of the full boundary matrix.
///
/// # Panics
/// In debug builds, if the involuted reduction disagrees with the cohomology pairing.
/// This indicates that `boundary` and `coboundary` disagree on the filtration (or the boundary itself).
pub fn cohomology_with_involution<C, F, CF, M, B>(
    coboundary: M,
    boundary: B,
    algorithm: Algorithm<'_>,
) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + MaybeSync,
    CF: Field,
    M: Reducible<C, F, CF>,
    B: MatrixOracle<CoefficientField = CF, ColT = C, RowT = C> + HasRowFiltration<FiltrationT = F>,
{
    let mut bars = cohomology(coboundary, algorithm);

    // The columns of the involuted boundary matrix: the death cells of finite bars (whose R columns are representatives)
    // and the birth cells of essential bars (whose V columns are representatives).
    // Reducing a column only ever adds earlier columns with a pivot, i.e. other death cells,
    // so these are all the columns we need.
    let mut involution_basis: Vec<(F, C)> = bars
        .iter()
        .map(|bar| match &bar.death {
            Some((death, death_value)) => (death_value.clone(), death.clone()),
            None => (bar.birth_value.clone(), bar.birth.clone()),
        })
        .collect();
    involution_basis.sort_unstable();
    let involution_basis: Vec<C> = involution_basis
        .into_iter()
        .map(|(_f, cell)| cell)
        .collect();

    let restricted = (&boundary).with_basis(involution_basis);
    let Reduction { v, diagram: _ } = Standard::new().reduce(&restricted);
    let r = product(&restricted, &v);

    for bar in bars.iter_mut() {
        let representative = match &bar.death {
            Some((death, _)) => {
                let rep = collect_column(&r, death.clone());
                debug_assert!(
                    rep.first().map(|(_, cell)| cell) == Some(&bar.birth),
                    "Involuted representative should have pivot at the birth cell; do `boundary` and `coboundary` agree?"
                );
                rep
            }
            None => collect_column((&v).with_trivial_filtration(), bar.birth.clone()),
        };
        bar.representative = Some(representative);
    }
    bars
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fields::Z2,
        matrices::{implementors::SimpleZ2Matrix, SplitBasis},
    };

    /// A filled triangle {0, 1, 2} with an extra edge {1, 2} entering last, plus a lone vertex 3.
    /// Cells, in filtration order (filtration value = index):
    /// 0,1,2,3: vertices; 4: {0,1}; 5: {0,2}; 6: {1,2}; 7: triangle {0,1,2}
    fn boundary() -> SimpleZ2Matrix {
        SimpleZ2Matrix::new_all_one(vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![4, 5, 6],
        ])
    }

    fn coboundary() -> SimpleZ2Matrix {
        let d = boundary();
        let mut cols = vec![vec![]; 8];
        for (j, col) in (0..8).map(|j| (j, d.column(j).map(|(_c, i)| i).collect::<Vec<_>>())) {
            for i in col {
                cols[i].push(j);
            }
        }
        SimpleZ2Matrix::new_all_one(cols)
    }

    fn basis(max_dim: usize) -> SplitBasis<usize> {
        let dims: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7]];
        SplitBasis::from_dimensions(dims.into_iter().take(max_dim + 1))
    }

    fn summary(bars: &[Bar<usize, usize, Z2>]) -> Vec<(usize, Option<usize>)> {
        let mut out: Vec<_> = bars
            .iter()
            .map(|bar| (bar.birth, bar.death.map(|(d, _)| d)))
            .collect();
        out.sort();
        out
    }

    fn algorithms() -> Vec<Algorithm<'static>> {
        vec![
            Algorithm::Standard(Standard::new()),
            Algorithm::Clearing(Clearing::new()),
            #[cfg(feature = "parallel")]
            Algorithm::ParallelClearing(Clearing::new().parallel()),
        ]
    }

    fn rep_cells(bar: &Bar<usize, usize, Z2>) -> Vec<usize> {
        let mut cells: Vec<_> = bar
            .representative
            .as_ref()
            .unwrap()
            .iter()
            .map(|(_c, cell)| *cell)
            .collect();
        cells.sort();
        cells
    }

    #[test]
    fn homology_and_cohomology_agree() {
        // Filtration value = index
        let d = boundary().with_filtration(|i| i);
        let d = (&d).with_basis(basis(2));
        let delta = coboundary().with_filtration(|i| i);
        let delta = (&delta).with_basis(basis(1));
        let expected = vec![
            (1, Some(4)),
            (2, Some(5)),
            (3, None),
            (0, None),
            (6, Some(7)),
        ];
        let mut expected = expected;
        expected.sort();

        for algorithm in algorithms() {
            let hom = homology(&d, algorithm.clone(), true);
            assert_eq!(summary(&hom), expected);
            let cohom = cohomology(&delta, algorithm.clone());
            assert_eq!(summary(&cohom), expected);
            let invol =
                cohomology_with_involution(&delta, boundary().with_filtration(|i| i), algorithm);
            assert_eq!(summary(&invol), expected);

            for bars in [&hom, &invol] {
                for bar in bars.iter() {
                    let expected_rep: Vec<usize> = match (bar.birth, bar.death) {
                        (0, None) => vec![0],
                        (3, None) => vec![3],
                        (1, Some(_)) => vec![0, 1],
                        (2, Some(_)) => vec![0, 2],
                        (6, Some(_)) => vec![4, 5, 6],
                        other => panic!("Unexpected bar {other:?}"),
                    };
                    assert_eq!(rep_cells(bar), expected_rep, "{bar:?}");
                }
            }
        }
    }
}
