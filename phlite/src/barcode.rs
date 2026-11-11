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
//!   For representatives use [`cohomology_with_involution`], which needs the coboundary to know its transpose
//!   (the boundary matrix, attached with [`with_transpose`](crate::matrices::MatrixOracle::with_transpose)).
//!   It then runs a small homology reduction restricted to the cells that matter
//!   (the death cells of finite bars and the birth cells of essential bars), as in Čufar's "involuted" persistence.
//!
//! In both cases the bars are reported in *homology* conventions: `birth` is the lower dimensional cell.
//! Bars of length zero (`birth_value == death_value`) are omitted, since they carry no information
//! and there are typically far more of them than of any other bar.
//!
//! # Performance
//!
//! Attach the transpose of your matrix with [`with_transpose`](crate::matrices::MatrixOracle::with_transpose)
//! (the boundary for a coboundary matrix, and vice versa), even if you don't want representatives:
//! the clearing algorithms then recognise *apparent pairs*, which need no reduction or storage.
//! The clearing algorithms also [skip zero-persistence apparent pairs](crate::reduction::Clearing::skip_zero_apparent_pairs)
//! by default, since those pairs would only become zero-length bars.
//! This means your column basis needn't contain the cells of zero-persistence apparent pairs at all,
//! which in many filtrations (such as Vietoris-Rips) are nearly all of them (see `phlite_rips::build_reduction_bases`).
//! The [`Standard`] algorithm doesn't look for apparent pairs, so needs the full basis.

use std::{cmp::Reverse, fmt::Debug, hash::Hash};

use crate::{
    columns::{ColumnRepresentation, MergedColumns},
    fields::{Invertible, NonZeroCoefficient},
    matrices::{
        combinators::product, BasisElement, FiltrationValue, HasRowFiltration, HasSplitColBasis,
        HasTranspose, MatrixOracle, SplitBasis,
    },
    reduction::{Clearing, Diagram, Reduction, ReductionAlgorithm, Standard},
};

#[cfg(feature = "parallel")]
use crate::reduction::ParallelClearing;

/// Which reduction algorithm to run, see the [`reduction`](crate::reduction) module docs.
///
/// Every algorithm represents the columns it reduces with `R`, see [`ColumnRepresentation`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Algorithm<'p, R = MergedColumns> {
    /// [`Standard`], the standard algorithm with no optimisations.
    Standard(Standard<R>),
    /// [`Clearing`], the standard algorithm with the clearing optimisation.
    Clearing(Clearing<R>),
    /// [`ParallelClearing`], clearing with each dimension reduced in parallel (feature `parallel`).
    #[cfg(feature = "parallel")]
    ParallelClearing(ParallelClearing<'p, R>),
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
    /// The functions in this module never return such bars.
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
pub trait Cell: BasisElement + Hash + Debug + Send + Sync + 'static {}
impl<T: BasisElement + Hash + Debug + Send + Sync + 'static> Cell for T {}

/// Requirements on the coefficient field for every [`Algorithm`]. Blanket-implemented.
pub trait Field: Invertible + Send + Sync + 'static {}
impl<T: Invertible + Send + Sync + 'static> Field for T {}

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
fn run<C, F, CF, M, R, Out>(
    matrix: M,
    algorithm: Algorithm<'_, R>,
    then: impl ReductionVisitor<C, CF, Out>,
) -> Out
where
    C: Cell,
    F: FiltrationValue + Send + Sync,
    CF: Field,
    M: Reducible<C, F, CF>,
    R: ColumnRepresentation,
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
pub fn homology<C, F, CF, M, R>(
    boundary: M,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + Send + Sync,
    CF: Field,
    M: Reducible<C, F, CF>,
    R: ColumnRepresentation,
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

            let finite = pairings.into_iter().filter_map(|(birth, death)| {
                let birth_value = boundary.filtration_value(birth.clone());
                let death_value = boundary.filtration_value(death.clone());
                (birth_value != death_value).then(|| Bar {
                    birth_value,
                    representative: representatives.then(|| collect_column(&r, death.clone())),
                    death: Some((death, death_value)),
                    birth,
                })
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

/// Read the bars of non-zero length (without representatives) off the diagram of the anti-transposed coboundary,
/// converting to homology conventions.
fn cohomology_bars<C, F, CF, M>(coboundary: M, diagram: &Diagram<Reverse<C>>) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue,
    CF: Field,
    M: Reducible<C, F, CF>,
{
    // The anti-transpose has columns in dimension k and rows in dimension k + 1,
    // so a pairing (pivot row, column) is (homology death, homology birth).
    let finite = diagram
        .pairings
        .iter()
        .filter_map(|(Reverse(death), Reverse(birth))| {
            let birth_value = coboundary.filtration_value(birth.clone());
            let death_value = coboundary.filtration_value(death.clone());
            (birth_value != death_value).then(|| Bar {
                birth_value,
                birth: birth.clone(),
                death: Some((death.clone(), death_value)),
                representative: None,
            })
        });
    let essential = diagram.essential.iter().map(|Reverse(birth)| Bar {
        birth_value: coboundary.filtration_value(birth.clone()),
        birth: birth.clone(),
        death: None,
        representative: None,
    });
    finite.chain(essential).collect()
}

/// Compute the barcode of a filtered chain complex from its `coboundary` matrix.
///
/// We reduce the anti-transpose ([`reverse`](MatrixOracle::reverse)) of `coboundary`, so pass the coboundary itself.
/// Unlike [`homology`], the column basis only needs cells up to dimension `max_dim` to get correct bars in dimensions `0..=max_dim`.
///
/// No representatives are computed, see [`cohomology_with_involution`].
pub fn cohomology<C, F, CF, M, R>(coboundary: M, algorithm: Algorithm<'_, R>) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + Send + Sync,
    CF: Field,
    M: Reducible<C, F, CF>,
    R: ColumnRepresentation,
{
    let diagram: Diagram<Reverse<C>> = run(coboundary.reverse(), algorithm, DiagramVisitor);
    cohomology_bars(coboundary, &diagram)
}

/// As [`cohomology`], but also computes a representative cycle for every bar via the involution trick.
///
/// The `coboundary` must know its transpose, the boundary matrix, typically attached with
/// [`with_transpose`](MatrixOracle::with_transpose) (which also lets the reduction use apparent pairs).
/// We reduce the boundary restricted to the death cells of the finite bars and the birth cells of the essential bars,
/// which is typically a tiny fraction of the full boundary matrix.
/// The filtration on the boundary's rows is that of the coboundary's columns, so they can't disagree.
///
/// # Panics
/// In debug builds, if the involuted reduction disagrees with the cohomology pairing.
/// This indicates that the attached transpose is not really the transpose.
pub fn cohomology_with_involution<C, F, CF, M, R>(
    coboundary: M,
    algorithm: Algorithm<'_, R>,
) -> Vec<Bar<C, F, CF>>
where
    C: Cell,
    F: FiltrationValue + Send + Sync,
    CF: Field,
    M: Reducible<C, F, CF> + HasTranspose,
    R: ColumnRepresentation,
{
    let diagram: Diagram<Reverse<C>> = run(coboundary.reverse(), algorithm, DiagramVisitor);
    let mut bars = cohomology_bars(coboundary, &diagram);

    // The columns of the involuted boundary matrix: the death cells of finite bars (whose R columns are representatives)
    // and the birth cells of essential bars (whose V columns are representatives).
    // Reducing a column only ever adds earlier columns with a pivot, i.e. other death cells
    // (including those of zero-length bars, so we take them from the diagram), so these are all the columns we need.
    let mut involution_basis: Vec<(F, C)> = diagram
        .pairings
        .into_iter()
        .map(|(Reverse(death), _)| death)
        .chain(diagram.essential.into_iter().map(|Reverse(birth)| birth))
        .map(|cell| (coboundary.filtration_value(cell.clone()), cell))
        .collect();
    involution_basis.sort_unstable();
    let involution_basis: Vec<C> = involution_basis
        .into_iter()
        .map(|(_f, cell)| cell)
        .collect();

    // The boundary, with the coboundary as its transpose so that the reduction can find apparent pairs:
    // zero-persistence apparent pairs may be missing from the diagram, and so their death cells from the basis.
    // A single "dimension" (the cells needn't share a dimension, since cells of different dimensions never interact).
    let boundary = coboundary
        .transpose()
        .with_filtration(|cell: C| coboundary.filtration_value(cell));
    let boundary = (&boundary).with_transpose(coboundary);
    let restricted = (&boundary).with_basis(SplitBasis::from_dimensions([involution_basis]));
    let Reduction { v, diagram: _ } = Clearing::new().reduce(&restricted);
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
    use crate::columns::{HeapColumns, SortedColumns};
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

    /// Every algorithm, representing columns with `R`
    /// Every algorithm representing columns with `R`, with and without caching columns
    fn algorithms<R: ColumnRepresentation>() -> Vec<Algorithm<'static, R>> {
        [true, false]
            .into_iter()
            .flat_map(|cache| {
                let clearing = Clearing::new()
                    .with_columns(R::default())
                    .cache_columns(cache);
                vec![
                    Algorithm::Standard(
                        Standard::new()
                            .with_columns(R::default())
                            .cache_columns(cache),
                    ),
                    #[cfg(feature = "parallel")]
                    Algorithm::ParallelClearing(clearing.clone().parallel()),
                    Algorithm::Clearing(clearing.clone().store_r(true)),
                    Algorithm::Clearing(clearing),
                ]
            })
            .collect()
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
        homology_and_cohomology_agree_with::<MergedColumns>();
        homology_and_cohomology_agree_with::<HeapColumns>();
        homology_and_cohomology_agree_with::<SortedColumns>();
    }

    fn homology_and_cohomology_agree_with<R: ColumnRepresentation>() {
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

        for algorithm in algorithms::<R>() {
            let hom = homology(&d, algorithm.clone(), true);
            assert_eq!(summary(&hom), expected);
            let cohom = cohomology(&delta, algorithm.clone());
            assert_eq!(summary(&cohom), expected);
            let b = boundary();
            let invol = cohomology_with_involution((&delta).with_transpose(&b), algorithm);
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

    #[test]
    fn first_and_last_columns_in_row() {
        use crate::matrices::HasRowFiltration;
        // Rows of the coboundary are the cofaces, so the columns in row 7 (the triangle) are its edges 4, 5, 6
        let co = coboundary();
        let delta = (&co).with_filtration(|i| i);
        let b = boundary();
        let with_t = (&delta).with_transpose(&b);
        assert_eq!(with_t.first_column_in_row(7), Some(4));
        assert_eq!(with_t.last_column_in_row(7), Some(6));
        // And reversing swaps them
        let rev = (&with_t).reverse();
        assert_eq!(rev.first_column_in_row(Reverse(7)), Some(Reverse(6)));
        assert_eq!(rev.last_column_in_row(Reverse(7)), Some(Reverse(4)));
        // Without a transpose nothing is known
        assert_eq!(delta.first_column_in_row(7), None);
    }

    /// Attaching transposes turns on apparent pairs, which must not change the bars or representatives
    #[test]
    fn transposes_give_same_bars() {
        transposes_give_same_bars_with::<MergedColumns>();
        transposes_give_same_bars_with::<HeapColumns>();
    }

    fn transposes_give_same_bars_with<R: ColumnRepresentation>() {
        let b = boundary();
        let co = coboundary();
        let d = (&b).with_filtration(|i| i);
        let d = (&d).with_basis(basis(2));
        let d = (&d).with_transpose(&co);
        let delta = (&co).with_filtration(|i| i);
        let delta = (&delta).with_basis(basis(1));
        let delta = (&delta).with_transpose(&b);
        let expected = {
            let mut e = vec![
                (1, Some(4)),
                (2, Some(5)),
                (3, None),
                (0, None),
                (6, Some(7)),
            ];
            e.sort();
            e
        };
        for algorithm in algorithms::<R>() {
            // `d` and `delta` are already references (WithTranspose of references is Copy)
            let hom = homology(d, algorithm.clone(), true);
            assert_eq!(summary(&hom), expected, "{algorithm:?}");
            assert_eq!(summary(&cohomology(delta, algorithm.clone())), expected);
            let invol = cohomology_with_involution(delta, algorithm.clone());
            assert_eq!(summary(&invol), expected);
            for bar in hom.iter().chain(invol.iter()) {
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

    /// Skipping zero-persistence apparent pairs must give the same diagram as not skipping (minus those pairs),
    /// whatever order the dimensions are reduced in, serially and in parallel.
    #[test]
    fn skipping_is_correct_in_any_dimension_order() {
        use crate::reduction::{Diagram, ReductionAlgorithm};
        // Edges 4 = {0,1} and 5 = {0,2} enter with the vertices, so are zero-persistence apparent pairs
        let value = |i: usize| [0usize, 0, 0, 0, 0, 0, 1, 2][i];
        let b = boundary();
        let co = coboundary();
        let d = (&b).with_filtration(value);
        let d = (&d).with_basis(basis(2));
        let d = (&d).with_transpose(&co);
        let delta = (&co).with_filtration(value);
        let delta = (&delta).with_basis(basis(2));
        let delta = (&delta).with_transpose(&b);
        let anti = (&delta).reverse();

        fn check<T: Eq + std::fmt::Debug>(full: &Diagram<T>, skipped: &Diagram<T>, context: &str) {
            assert_eq!(skipped.essential, full.essential, "{context}");
            assert!(
                skipped
                    .pairings
                    .iter()
                    .all(|pair| full.pairings.contains(pair)),
                "{context}"
            );
        }

        // Dimensions beyond the matrix (3 and up) are ignored
        for order in [
            vec![0, 1, 2],
            vec![2, 1, 0],
            vec![1, 0, 2],
            vec![2, 0, 1],
            vec![0, 1, 2, 3, 7],
            vec![5, 2, 1, 0],
        ] {
            let clearing = Clearing::new().with_dimension_order(order.clone());
            let context = format!("order {order:?}");
            let full = clearing
                .clone()
                .skip_zero_apparent_pairs(false)
                .reduce(d)
                .diagram
                .sorted();
            assert_eq!(full.essential, Standard::new().reduce(d).diagram.essential);
            check(
                &full,
                &clearing.clone().reduce(d).diagram.sorted(),
                &context,
            );
            #[cfg(feature = "parallel")]
            check(
                &full,
                &clearing.clone().parallel().reduce(d).diagram.sorted(),
                &context,
            );
            // And the anti-transposed coboundary, as used for cohomology
            let full = clearing
                .clone()
                .skip_zero_apparent_pairs(false)
                .reduce(anti)
                .diagram
                .sorted();
            check(
                &full,
                &clearing.clone().reduce(anti).diagram.sorted(),
                &context,
            );
            #[cfg(feature = "parallel")]
            check(
                &full,
                &clearing.clone().parallel().reduce(anti).diagram.sorted(),
                &context,
            );
        }
    }
}
