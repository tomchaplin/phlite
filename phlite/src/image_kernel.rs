//! Persistent homology of images, kernels and cokernels: for a filtered complex `K` and a subcomplex `L` (with the
//! restricted filtration), the barcodes of the image, kernel and cokernel of `H(L_t) -> H(K_t)` (Cohen-Steiner,
//! Edelsbrunner, Harer and Morozov, *Persistent homology for kernels, images, and cokernels*, 2009), as phimaker
//! computes them, for any boundary oracle.
//!
//! Like phimaker, it reads the three barcodes off a few reductions (and the relative barcode of `(K, L)` off one more):
//!
//! * `D_f`, the boundary of `K`, and `D_g`, that of `L`: which cells are negative (kill a class), and cycles `z_j`
//!   (with youngest cell `j`) for every positive cell, from the reductions' `V`;
//! * `D_im`: `D_f` with the rows of `L` first. Only the columns negative in `D_f` need reducing (a column reduces to
//!   zero whatever the order of the rows), so the others are skipped, as clearing would;
//! * `D_ker`: the cycles `z_j` of `K` (any basis of cycles compatible with the filtration has the same pivots), rows
//!   of `L` first;
//! * `D_cok`: `D_f` with the columns of cells positive in `L` replaced by `L`'s cycles.
//!
//! Then, with `low_im(j)` the pivot of column `j` of the reduced `D_im`:
//!
//! * **image**: a class is born at each cell of `L` positive in `L`, and dies at a cell `j` negative in `K` whose
//!   `low_im(j)` is in `L`, killing the class born there;
//! * **kernel**: a class is born at each cell `j` not in `L`, negative in `K`, with `low_im(j)` in `L`, and dies at
//!   each cell `j` of `L` negative in `L` but positive in `K`, killing the class born at the pivot of `z_j` in the
//!   reduced `D_ker`;
//! * **cokernel**: a class is born at each cell positive in `K` and not positive in `L`, and dies at each cell `j`
//!   negative in `K` with `low_im(j)` not in `L`, killing the class born at the pivot of column `j` of the reduced
//!   `D_cok`;
//! * **relative**: the barcode of the quotient `K / L`.
//!
//! [`inclusion_six_pack`] also gives a representative for every bar of each of the six barcodes (those of `L` and `K`
//! too), read off the same reductions (see its docs); [`crate::six_pack`] extends all of this to any filtered chain
//! map, by its mapping cylinder.
//!
//! Over any field. The order on cells, (filtration value, cell), must be a filtration order (faces first).

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::columns::{ColumnCoefficient, ColumnFiltration, ColumnRepresentation, ColumnRow};
use crate::fields::{Invertible, NonZeroCoefficient};
use crate::matrices::{BasisElement, FiltrationValue, HasRowFiltration, MatrixOracle, SplitBasis};
use crate::reduction::{Clearing, Diagram, ReductionAlgorithm, Standard, StandardReductionMatrix};

/// A filtration value with the rows of the subcomplex first: `(false, value)` for a cell of `L`, `(true, value)` for
/// one of `K` outside it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubcomplexFirst<F>(pub bool, pub F);

impl<F: FiltrationValue> FiltrationValue for SubcomplexFirst<F> {}

/// The barcodes of the image, kernel and cokernel of `H(L) -> H(K)`, and of `(K, L)` (bars as pairs of cells: `(birth,
/// Some(death))`, or `(birth, None)` if essential), with those of `K` and `L` themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageKernelCokernel<C> {
    pub image: Diagram<C>,
    pub kernel: Diagram<C>,
    pub cokernel: Diagram<C>,
    pub relative: Diagram<C>,
    pub complex: Diagram<C>,
    pub subcomplex: Diagram<C>,
}

/// A boundary oracle restricted to some of its cells (a subcomplex, or the cells of a quotient): its columns and row
/// values, and the shortcuts that only depend on a column (its pivot), but not those that look along a row (the oldest
/// or youngest coface of a cell, or row ranks), which would see cofaces outside the restricted basis.
#[derive(Clone)]
pub struct Restricted<M>(pub M);

impl<M: MatrixOracle> MatrixOracle for Restricted<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(&self, col: M::ColT) -> impl Iterator<Item = (M::CoefficientField, M::RowT)> {
        self.0.column(col)
    }
}

impl<M: HasRowFiltration> HasRowFiltration for Restricted<M> {
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: M::RowT) -> M::FiltrationT {
        self.0.filtration_value(row)
    }

    fn same_filtration_value(&self, a: M::RowT, b: M::RowT) -> bool {
        self.0.same_filtration_value(a, b)
    }

    fn column_pivot(
        &self,
        col: M::ColT,
    ) -> Option<crate::columns::ColumnEntry<M::FiltrationT, M::RowT, M::CoefficientField>> {
        self.0.column_pivot(col)
    }
}

/// A matrix with its columns given by a lookup (a column of another matrix, or a stored chain), for `D_ker` and
/// `D_cok`.
struct Columns<'a, C, CF, F> {
    columns: &'a HashMap<C, Vec<(CF, C)>>,
    filtration: &'a F,
}

impl<C: BasisElement + Hash, CF: crate::fields::NonZeroCoefficient, F> MatrixOracle
    for Columns<'_, C, CF, F>
{
    type CoefficientField = CF;
    type ColT = C;
    type RowT = C;

    fn column(&self, col: C) -> impl Iterator<Item = (CF, C)> {
        self.columns.get(&col).into_iter().flatten().cloned()
    }
}

impl<
        C: BasisElement + Hash,
        CF: crate::fields::NonZeroCoefficient,
        FT: FiltrationValue,
        F: Fn(C) -> FT,
    > HasRowFiltration for Columns<'_, C, CF, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: C) -> FT {
        (self.filtration)(row)
    }
}

/// A chain with repeated cells summed (and those that cancel dropped): the columns of `V` for cleared cells are
/// products `D V_d`, sums of boundaries full of repeats.
pub fn consolidated<C: BasisElement + Hash, CF: crate::fields::NonZeroCoefficient>(
    column: impl Iterator<Item = (CF, C)>,
) -> Vec<(CF, C)> {
    let mut sums: HashMap<C, Option<CF>> = HashMap::new();
    for (coefficient, row) in column {
        let entry = sums.entry(row).or_insert(None);
        *entry = match *entry {
            None => Some(coefficient),
            Some(c) => c + coefficient,
        };
    }
    sums.into_iter()
        .filter_map(|(row, c)| c.map(|c| (c, row)))
        .collect()
}

/// Run both, in parallel on the current rayon thread pool when phlite is built with it.
fn join<A: Send, B: Send>(a: impl FnOnce() -> A + Send, b: impl FnOnce() -> B + Send) -> (A, B) {
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}

/// The pivots `column -> low` of a reduced matrix, and its `V`.
type Pivots<M> = (
    HashMap<<M as MatrixOracle>::ColT, <M as MatrixOracle>::RowT>,
    StandardReductionMatrix<<M as MatrixOracle>::CoefficientField, <M as MatrixOracle>::ColT>,
);

/// The pivots `column -> low` of the reduced matrix, by the standard algorithm over the columns in `basis` (in order),
/// and its `V`.
fn pivots<M, R>(matrix: M, basis: Vec<M::ColT>, columns: R) -> Pivots<M>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::RowT: ColumnRow + Hash + 'static,
    R: ColumnRepresentation,
{
    let reduction = Standard::new()
        .with_columns(columns)
        .reduce(matrix.with_basis(basis));
    (
        reduction
            .diagram
            .pairings
            .into_iter()
            .map(|(low, col)| (col, low))
            .collect(),
        reduction.v,
    )
}

/// `sum_i x_i D_i` for the column `sum_i x_i e_i`, with `D_i` given by `column`, consolidated.
fn combine<C, CF, I>(chain: impl Iterator<Item = (CF, C)>, column: impl Fn(C) -> I) -> Vec<(CF, C)>
where
    C: BasisElement + Hash,
    CF: crate::fields::NonZeroCoefficient,
    I: Iterator<Item = (CF, C)>,
{
    consolidated(chain.flat_map(|(x, i)| column(i).map(move |(y, r)| (x * y, r))))
}

/// A bar of one of the barcodes of a six-pack, on cells: its dimension, birth and death cells with their values, and a
/// representative (if asked for) as `(coefficient, cell)` pairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellBar<C, F, CF> {
    pub dimension: usize,
    pub birth: (C, F),
    /// `None` for an essential bar
    pub death: Option<(C, F)>,
    pub representative: Option<Vec<(CF, C)>>,
}

/// The six barcodes of the inclusion `L -> K` of a subcomplex, on cells (see [`inclusion_six_pack`]). Bars of length
/// zero (birth and death at the same value) are included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellSixPack<C, F, CF> {
    /// `L`
    pub domain: Vec<CellBar<C, F, CF>>,
    /// `K`
    pub codomain: Vec<CellBar<C, F, CF>>,
    pub image: Vec<CellBar<C, F, CF>>,
    pub kernel: Vec<CellBar<C, F, CF>>,
    pub cokernel: Vec<CellBar<C, F, CF>>,
    /// `(K, L)`
    pub relative: Vec<CellBar<C, F, CF>>,
}

fn to_diagram<C, F, CF>(bars: Vec<CellBar<C, F, CF>>) -> Diagram<C> {
    let mut diagram = Diagram {
        essential: vec![],
        pairings: vec![],
    };
    for bar in bars {
        match bar.death {
            Some((death, _)) => diagram.pairings.push((bar.birth.0, death)),
            None => diagram.essential.push(bar.birth.0),
        }
    }
    diagram
}

/// The barcodes of the image, kernel and cokernel of `H(L) -> H(K)` and of `(K, L)` (see the [module docs](self)):
/// `boundary` is the boundary of `K` (rows and columns its cells), `cells` its cells by dimension (each in filtration
/// order), and `in_subcomplex` says which are in `L` (a subcomplex: every face of a cell of `L` is in `L`).
pub fn image_kernel_cokernel<M, R>(
    boundary: M,
    cells: Vec<Vec<M::ColT>>,
    in_subcomplex: impl Fn(&M::ColT) -> bool + Sync,
    columns: R,
) -> ImageKernelCokernel<M::ColT>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration + Clone + Send + Sync,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::ColT: ColumnRow + Hash + Debug + 'static,
    R: ColumnRepresentation,
{
    let pack = inclusion_six_pack(boundary, cells, in_subcomplex, columns, false);
    ImageKernelCokernel {
        image: to_diagram(pack.image),
        kernel: to_diagram(pack.kernel),
        cokernel: to_diagram(pack.cokernel),
        relative: to_diagram(pack.relative),
        complex: to_diagram(pack.codomain),
        subcomplex: to_diagram(pack.domain),
    }
}

/// The six barcodes of the inclusion of a subcomplex `L` in `K`: of `L` and `K`, of the image, kernel and cokernel of
/// `H(L) -> H(K)`, and of `(K, L)`, with (if `representatives`) a representative for every bar. Arguments as
/// [`image_kernel_cokernel`]. A kernel bar's dimension is that of its classes, one less than its birth cell's.
///
/// The representatives, from the reductions (`R = D V` of each):
///
/// * `L`, `K`, `(K, L)`: `R_d` for a bar dying at `d`, `V_b` for an essential bar born at `b`;
/// * image: a cycle of `L`: `R_im,j` (all of whose cells are in `L`, as its pivot is) for a bar dying at `j`, the
///   cycle `V_b` of `L` for an essential one;
/// * kernel: a cycle of `L` that is a boundary in `K` from the bar's birth: `R_im,b` for an essential bar born at
///   `b`; for one dying at `j`, with `z = R_ker,j` (a cycle of `K` whose cells outside `L` are no later than the birth
///   `b`, the pivot), `-∂(z outside L) = ∂(z in L)`, which lies in `L` and is a boundary in `K` from `b`;
/// * cokernel: a cycle of `K`: `R_cok,j` for a bar dying at `j`, `V_b` of `K` for an essential one.
pub fn inclusion_six_pack<M, R>(
    boundary: M,
    cells: Vec<Vec<M::ColT>>,
    in_subcomplex: impl Fn(&M::ColT) -> bool + Sync,
    columns: R,
    representatives: bool,
) -> CellSixPack<M::ColT, M::FiltrationT, M::CoefficientField>
where
    M: MatrixOracle<RowT = <M as MatrixOracle>::ColT> + HasRowFiltration + Clone + Send + Sync,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::FiltrationT: ColumnFiltration,
    M::ColT: ColumnRow + Hash + Debug + 'static,
    R: ColumnRepresentation,
{
    type Cell<M> = <M as MatrixOracle>::ColT;
    type Chain<M> = Vec<(
        <M as MatrixOracle>::CoefficientField,
        <M as MatrixOracle>::ColT,
    )>;
    let key = |c: &Cell<M>| (boundary.filtration_value(c.clone()), c.clone());
    let mut all: Vec<Cell<M>> = cells.iter().flatten().cloned().collect();
    all.sort_by_cached_key(key);
    let dimension_of: HashMap<Cell<M>, usize> = cells
        .iter()
        .enumerate()
        .flat_map(|(d, dim)| dim.iter().map(move |c| (c.clone(), d)))
        .collect();
    let l_cells: Vec<Vec<Cell<M>>> = cells
        .iter()
        .map(|dim| dim.iter().filter(|c| in_subcomplex(c)).cloned().collect())
        .collect();
    let quotient_cells: Vec<Vec<Cell<M>>> = cells
        .iter()
        .map(|dim| dim.iter().filter(|c| !in_subcomplex(c)).cloned().collect())
        .collect();

    // D_f and D_g: positive and negative cells, and cycles for the positive ones
    let full = || {
        Clearing::new()
            .with_columns(columns.clone())
            .full_basis(true)
            .skip_zero_apparent_pairs(false)
    };
    let (k_basis, l_basis) = (
        SplitBasis::from_dimensions(cells.clone()),
        SplitBasis::from_dimensions(l_cells),
    );
    let (k, l) = join(
        || full().reduce(boundary.clone().with_basis(k_basis)),
        || full().reduce(Restricted(boundary.clone()).with_basis(l_basis)),
    );
    let negative_k: HashSet<Cell<M>> = k.diagram.pairings.iter().map(|(_, d)| d.clone()).collect();
    let negative_l: HashSet<Cell<M>> = l.diagram.pairings.iter().map(|(_, d)| d.clone()).collect();
    let positive_l = |c: &Cell<M>| in_subcomplex(c) && !negative_l.contains(c);
    let subcomplex_first =
        |row: Cell<M>| SubcomplexFirst(!in_subcomplex(&row), boundary.filtration_value(row));
    let row_value = |row: Cell<M>| boundary.filtration_value(row);

    // D_im, on the columns negative in K
    let negatives: Vec<Cell<M>> = all
        .iter()
        .filter(|c| negative_k.contains(c))
        .cloned()
        .collect();
    let image_matrix = || {
        pivots(
            boundary.clone().with_filtration(&subcomplex_first),
            negatives.clone(),
            columns.clone(),
        )
    };

    // D_ker: the cycles of K (columns of V for the positive cells), rows of L first. Only the pivots of kernel deaths
    // (cells of L negative in L, positive in K) are read, and a column only meets those of its own dimension, so only
    // the positive cells of dimensions with a kernel death, up to the last one, are reduced
    let mut last_death: HashMap<usize, usize> = HashMap::new();
    for (i, c) in all.iter().enumerate() {
        if in_subcomplex(c) && negative_l.contains(c) && !negative_k.contains(c) {
            last_death.insert(dimension_of[c], i);
        }
    }
    let positives: Vec<Cell<M>> = all
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            !negative_k.contains(*c)
                && last_death
                    .get(&dimension_of[*c])
                    .is_some_and(|&last| *i <= last)
        })
        .map(|(_, c)| c.clone())
        .collect();
    let kernel_matrix = || {
        let columns_map: HashMap<Cell<M>, Chain<M>> = positives
            .iter()
            .map(|c| (c.clone(), consolidated(k.v.column(c.clone()))))
            .collect();
        let reduced = pivots(
            Columns {
                columns: &columns_map,
                filtration: &subcomplex_first,
            },
            positives.clone(),
            columns.clone(),
        );
        (reduced, columns_map)
    };

    // D_cok: D_f with the cells positive in L replaced by L's cycles. A column positive in K and not positive in L is
    // its boundary, in the span of the earlier boundaries, which the replaced ones (positive in L) are in the span of
    // L's earlier columns: so it reduces to zero, and is left out
    let cokernel_matrix = || {
        let needed: Vec<Cell<M>> = all
            .iter()
            .filter(|c| negative_k.contains(c) || positive_l(c))
            .cloned()
            .collect();
        let columns_map: HashMap<Cell<M>, Chain<M>> = needed
            .iter()
            .map(|c| {
                let column: Vec<_> = if positive_l(c) {
                    consolidated(l.v.column(c.clone()))
                } else {
                    boundary.column(c.clone()).collect()
                };
                (c.clone(), column)
            })
            .collect();
        let reduced = pivots(
            Columns {
                columns: &columns_map,
                filtration: &row_value,
            },
            needed,
            columns.clone(),
        );
        (reduced, columns_map)
    };

    // (K, L): the quotient's boundary, dropping faces in L. Without representatives, by cohomology (its transpose,
    // with clearing); with them, by homology, for V
    let relative_matrix = || -> Vec<CellBar<Cell<M>, M::FiltrationT, M::CoefficientField>> {
        let quotient_basis = SplitBasis::from_dimensions(quotient_cells.clone());
        let bar = |birth: Cell<M>, death: Option<Cell<M>>, representative| CellBar {
            dimension: dimension_of[&birth],
            birth: (birth.clone(), row_value(birth)),
            death: death.map(|d| (d.clone(), row_value(d))),
            representative,
        };
        if representatives {
            let quotient_map: HashMap<Cell<M>, Chain<M>> = all
                .iter()
                .filter(|c| !in_subcomplex(c))
                .map(|c| {
                    (
                        c.clone(),
                        boundary
                            .column(c.clone())
                            .filter(|(_, f)| !in_subcomplex(f))
                            .collect(),
                    )
                })
                .collect();
            let quotient = Columns {
                columns: &quotient_map,
                filtration: &row_value,
            };
            let reduction = full().reduce((&quotient).with_basis(quotient_basis));
            let column = |c: Cell<M>| quotient_map.get(&c).into_iter().flatten().cloned();
            let mut bars: Vec<_> = reduction
                .diagram
                .pairings
                .iter()
                .map(|(b, d)| {
                    bar(
                        b.clone(),
                        Some(d.clone()),
                        Some(combine(reduction.v.column(d.clone()), column)),
                    )
                })
                .collect();
            bars.extend(reduction.diagram.essential.iter().map(|b| {
                bar(
                    b.clone(),
                    None,
                    Some(consolidated(reduction.v.column(b.clone()))),
                )
            }));
            return bars;
        }
        let mut coboundary: HashMap<Cell<M>, Chain<M>> = HashMap::new();
        for c in all.iter().filter(|c| !in_subcomplex(c)) {
            coboundary.entry(c.clone()).or_default();
            for (x, f) in boundary.column(c.clone()) {
                if !in_subcomplex(&f) {
                    coboundary.entry(f).or_default().push((x, c.clone()));
                }
            }
        }
        let quotient = Columns {
            columns: &coboundary,
            filtration: &row_value,
        }
        .with_basis(quotient_basis);
        let algorithm =
            crate::barcode::Algorithm::Clearing(Clearing::new().with_columns(columns.clone()));
        crate::barcode::cohomology(&quotient, algorithm)
            .into_iter()
            .map(|b| bar(b.birth, b.death.map(|(d, _)| d), None))
            .collect()
    };
    let (
        ((low_im, v_im), ((low_ker, v_ker), ker_columns)),
        (((low_cok, v_cok), cok_columns), relative),
    ) = join(
        || join(image_matrix, kernel_matrix),
        || join(cokernel_matrix, relative_matrix),
    );

    // The barcodes
    let value = |c: &Cell<M>| boundary.filtration_value(c.clone());
    let bar = |dimension: usize,
               birth: &Cell<M>,
               death: Option<&Cell<M>>,
               representative: Option<Chain<M>>| CellBar {
        dimension,
        birth: (birth.clone(), value(birth)),
        death: death.map(|d| (d.clone(), value(d))),
        representative,
    };
    let boundary_of = |c: Cell<M>| boundary.column(c);
    let want = |f: &dyn Fn() -> Chain<M>| representatives.then(f);
    let mut image = vec![];
    let mut kernel = vec![];
    let mut cokernel = vec![];
    let (mut image_dies, mut kernel_dies, mut cokernel_dies) =
        (HashSet::new(), HashSet::new(), HashSet::new());
    for j in &all {
        if negative_k.contains(j) {
            let low = low_im
                .get(j)
                .expect("a negative column has a pivot, whatever the row order");
            if in_subcomplex(low) {
                let rep = want(&|| combine(v_im.column(j.clone()), boundary_of));
                image.push(bar(dimension_of[low], low, Some(j), rep));
                image_dies.insert(low.clone());
            } else {
                let birth = low_cok
                    .get(j)
                    .expect("a cokernel death has a pivot")
                    .clone();
                let rep = want(&|| {
                    combine(v_cok.column(j.clone()), |c| {
                        cok_columns.get(&c).into_iter().flatten().cloned()
                    })
                });
                cokernel.push(bar(dimension_of[&birth], &birth, Some(j), rep));
                cokernel_dies.insert(birth);
            }
        }
        if in_subcomplex(j) && negative_l.contains(j) && !negative_k.contains(j) {
            let birth = low_ker.get(j).expect("a kernel death has a pivot").clone();
            let rep = want(&|| {
                let z = combine(v_ker.column(j.clone()), |c| {
                    ker_columns.get(&c).into_iter().flatten().cloned()
                });
                let outside = z
                    .into_iter()
                    .filter(|(_, c)| !in_subcomplex(c))
                    .map(|(x, c)| (x.additive_inverse(), c));
                combine(outside, boundary_of)
            });
            kernel.push(bar(dimension_of[&birth] - 1, &birth, Some(j), rep));
            kernel_dies.insert(birth);
        }
    }
    for j in &all {
        let negative = negative_k.contains(j);
        if positive_l(j) && !image_dies.contains(j) {
            image.push(bar(
                dimension_of[j],
                j,
                None,
                want(&|| consolidated(l.v.column(j.clone()))),
            ));
        }
        if !in_subcomplex(j)
            && negative
            && low_im.get(j).is_some_and(&in_subcomplex)
            && !kernel_dies.contains(j)
        {
            let rep = want(&|| combine(v_im.column(j.clone()), boundary_of));
            kernel.push(bar(dimension_of[j] - 1, j, None, rep));
        }
        if !negative && !positive_l(j) && !cokernel_dies.contains(j) {
            cokernel.push(bar(
                dimension_of[j],
                j,
                None,
                want(&|| consolidated(k.v.column(j.clone()))),
            ));
        }
    }

    // L and K themselves
    let own = |diagram: &Diagram<Cell<M>>, v: &dyn Fn(Cell<M>) -> Chain<M>| {
        let mut bars: Vec<_> = diagram
            .pairings
            .iter()
            .map(|(b, d)| {
                bar(
                    dimension_of[b],
                    b,
                    Some(d),
                    want(&|| combine(v(d.clone()).into_iter(), boundary_of)),
                )
            })
            .collect();
        bars.extend(
            diagram
                .essential
                .iter()
                .map(|b| bar(dimension_of[b], b, None, want(&|| v(b.clone())))),
        );
        bars
    };
    let domain = own(&l.diagram, &|c| consolidated(l.v.column(c)));
    let codomain = own(&k.diagram, &|c| consolidated(k.v.column(c)));
    CellSixPack {
        domain,
        codomain,
        image,
        kernel,
        cokernel,
        relative,
    }
}
