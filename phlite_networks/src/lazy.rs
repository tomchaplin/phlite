//! What the directed flag, ordered-tuple and Dowker complexes have in common: a filtered complex of [`Simplex`]es
//! described lazily, by listing the cofacets and facets of any simplex, from which this module builds the coboundary
//! and boundary oracles, the column bases and the barcode, much as `phlite_clouds::rips` does for Vietoris–Rips.
//!
//! A complex implements [`LazyComplex`]: the filtration value of a simplex, its cofacets **in increasing order** (with
//! a lower bound on their values), and its facets. From these:
//!
//! * [`Coboundary`] and [`Boundary`] are the two oracles, each the other's transpose. They override the
//!   [optional hooks](phlite::matrices#optional-performance-hooks) as Ripser does: since cofacets come in increasing
//!   order and none comes before its lower bound, the first cofacet *at* the lower bound is the oldest, so the search
//!   for a column's pivot in cohomology (an [emergent pair](phlite::reduction)) usually stops after a few cofacets;
//!   and the youngest facet and oldest cofacet of a simplex give apparent pairs without the transpose.
//! * [`build_bases`] lists the simplices of each dimension in filtration order, leaving out those in zero-persistence
//!   apparent pairs (usually most of them), generating each simplex once from its [extensions](LazyComplex::extensions)
//!   and never storing the top dimension in full.
//! * [`barcode`] computes the barcode by cohomology or homology, with any of `phlite`'s algorithms.

use std::marker::PhantomData;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar, Method},
    columns::{ColumnEntry, ColumnRepresentation},
    fields::{NonZeroCoefficient, Z2},
    matrices::{HasRowFiltration, MatrixOracle, SplitBasis},
};
use rayon::prelude::*;

use crate::simplex::{Codec, Simplex};

/// A cofacet of a simplex: the simplex with one more vertex, inserted at `position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cofacet {
    /// Where the new vertex is (its coefficient in the coboundary is `(-1)^position`)
    pub position: usize,
    pub simplex: Simplex,
    /// Its filtration value
    pub value: NotNan<f64>,
}

/// A filtered complex of [`Simplex`]es, described by the cofacets and facets of each simplex (see the
/// [module docs](self)). Every simplex passed to these methods is in the complex.
pub trait LazyComplex: Sync {
    /// How simplices are packed.
    fn codec(&self) -> Codec;

    /// The number of vertices.
    fn n_vertices(&self) -> u32;

    /// When `simplex` enters the filtration.
    fn value(&self, simplex: Simplex) -> NotNan<f64>;

    /// The cofacets of `simplex` that are in the complex, **in increasing order of [`Simplex`]**, together with a lower
    /// bound on their values (usually the value of `simplex` itself).
    ///
    /// Both must be right: the search for the oldest cofacet stops at the first one whose value is the lower bound.
    fn cofacets(&self, simplex: Simplex) -> (NotNan<f64>, impl Iterator<Item = Cofacet> + '_);

    /// The cofacets of `simplex` with the new vertex last (at position `simplex.n_vertices()`), with their values,
    /// in any order. Every simplex of dimension at least one must be such a cofacet of exactly one simplex (for
    /// instance, of its facet without its last vertex), so that [`build_bases`] lists each simplex once.
    fn extensions(&self, simplex: Simplex) -> impl Iterator<Item = (Simplex, NotNan<f64>)> + '_ {
        let last = simplex.n_vertices();
        self.cofacets(simplex)
            .1
            .filter(move |cofacet| cofacet.position == last)
            .map(|cofacet| (cofacet.simplex, cofacet.value))
    }

    /// The facets of `simplex` in the complex, each with the position of the vertex removed (its coefficient in the
    /// boundary is `(-1)^position`), in any order. A vertex has none.
    fn facets(&self, simplex: Simplex) -> impl Iterator<Item = (usize, Simplex)> + '_;

    /// The oldest cofacet of `simplex` (the smallest by value, then [`Simplex`]), stopping at the first cofacet whose
    /// value is the lower bound.
    fn oldest_cofacet(&self, simplex: Simplex) -> Option<Cofacet> {
        let (bound, cofacets) = self.cofacets(simplex);
        let mut oldest: Option<Cofacet> = None;
        for cofacet in cofacets {
            // On ties keep the earlier, smaller, simplex
            if oldest.is_none_or(|current| cofacet.value < current.value) {
                oldest = Some(cofacet);
                if cofacet.value == bound {
                    break;
                }
            }
        }
        oldest
    }

    /// The youngest facet of `simplex` (the largest by value, then [`Simplex`]), with the position of the vertex
    /// removed and its value; `None` for a vertex.
    fn youngest_facet(&self, simplex: Simplex) -> Option<(usize, Simplex, NotNan<f64>)> {
        self.facets(simplex)
            .map(|(position, facet)| (position, facet, self.value(facet)))
            .max_by_key(|&(_, facet, value)| (value, facet))
    }
}

#[inline]
fn sign<CF: NonZeroCoefficient>(position: usize) -> CF {
    if position.is_multiple_of(2) {
        CF::one()
    } else {
        CF::one().additive_inverse()
    }
}

/// The coboundary oracle of a [`LazyComplex`].
pub struct Coboundary<'a, K, CF> {
    complex: &'a K,
    phantom: PhantomData<CF>,
}

impl<'a, K, CF> Coboundary<'a, K, CF> {
    pub fn new(complex: &'a K) -> Self {
        Self {
            complex,
            phantom: PhantomData,
        }
    }
}

impl<K: LazyComplex, CF: NonZeroCoefficient> MatrixOracle for Coboundary<'_, K, CF> {
    type CoefficientField = CF;
    type ColT = Simplex;
    type RowT = Simplex;

    fn column(&self, col: Simplex) -> impl Iterator<Item = (CF, Simplex)> {
        self.complex
            .cofacets(col)
            .1
            .map(|cofacet| (sign(cofacet.position), cofacet.simplex))
    }
}

impl<K: LazyComplex, CF: NonZeroCoefficient> HasRowFiltration for Coboundary<'_, K, CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Simplex) -> NotNan<f64> {
        self.complex.value(row)
    }

    fn column_with_filtration(
        &self,
        col: Simplex,
    ) -> impl Iterator<Item = ColumnEntry<NotNan<f64>, Simplex, CF>> {
        self.complex
            .cofacets(col)
            .1
            .map(|cofacet| ColumnEntry {
                filtration_value: cofacet.value,
                row_index: cofacet.simplex,
                coeff: sign(cofacet.position),
            })
    }

    // The pivot of the anti-transpose (which cohomology reduces) is the smallest entry: the oldest cofacet
    fn column_min(&self, col: Simplex) -> Option<ColumnEntry<NotNan<f64>, Simplex, CF>> {
        self.complex
            .oldest_cofacet(col)
            .map(|cofacet| ColumnEntry {
                filtration_value: cofacet.value,
                row_index: cofacet.simplex,
                coeff: sign(cofacet.position),
            })
    }

    // Row `tau` has entries in the columns of its facets, so the last is its youngest facet
    fn last_column_in_row(&self, row: Simplex) -> Option<Simplex> {
        self.complex
            .youngest_facet(row)
            .map(|(_, facet, _)| facet)
    }
}

/// The boundary oracle of a [`LazyComplex`].
pub struct Boundary<'a, K, CF> {
    complex: &'a K,
    phantom: PhantomData<CF>,
}

impl<'a, K, CF> Boundary<'a, K, CF> {
    pub fn new(complex: &'a K) -> Self {
        Self {
            complex,
            phantom: PhantomData,
        }
    }
}

impl<K: LazyComplex, CF: NonZeroCoefficient> MatrixOracle for Boundary<'_, K, CF> {
    type CoefficientField = CF;
    type ColT = Simplex;
    type RowT = Simplex;

    fn column(&self, col: Simplex) -> impl Iterator<Item = (CF, Simplex)> {
        self.complex
            .facets(col)
            .map(|(position, facet)| (sign(position), facet))
    }
}

impl<K: LazyComplex, CF: NonZeroCoefficient> HasRowFiltration for Boundary<'_, K, CF> {
    type FiltrationT = NotNan<f64>;

    fn filtration_value(&self, row: Simplex) -> NotNan<f64> {
        self.complex.value(row)
    }

    // The pivot of the boundary is the youngest facet
    fn column_pivot(&self, col: Simplex) -> Option<ColumnEntry<NotNan<f64>, Simplex, CF>> {
        self.complex
            .youngest_facet(col)
            .map(|(position, facet, value)| ColumnEntry {
                filtration_value: value,
                row_index: facet,
                coeff: sign(position),
            })
    }

    // Row `sigma` has entries in the columns of its cofacets, so the first is its oldest cofacet
    fn first_column_in_row(&self, row: Simplex) -> Option<Simplex> {
        self.complex
            .oldest_cofacet(row)
            .map(|cofacet| cofacet.simplex)
    }
}

/// Whether `simplex` (with value `value`) is in a zero-persistence apparent pair, on either side.
///
/// Such simplices need no reduction and only give zero-length bars, so [`build_bases`] leaves them out.
/// If `check_cofacets` is false then only check whether `simplex` is the *death* of such a pair (so that the
/// cofacets of the top dimension, which may not fit in a [`Simplex`], are never looked at).
pub fn in_zero_apparent_pair<K: LazyComplex>(
    complex: &K,
    simplex: Simplex,
    value: NotNan<f64>,
    check_cofacets: bool,
) -> bool {
    // The birth: its oldest cofacet has the same value, and it is that cofacet's youngest facet
    if check_cofacets {
        if let Some(cofacet) = complex.oldest_cofacet(simplex) {
            if cofacet.value == value
                && complex
                    .youngest_facet(cofacet.simplex)
                    .map(|(_, facet, _)| facet)
                    == Some(simplex)
            {
                return true;
            }
        }
    }
    // The death: its youngest facet has the same value, and it is that facet's oldest cofacet
    if let Some((_, facet, facet_value)) = complex.youngest_facet(simplex) {
        if facet_value == value
            && complex
                .oldest_cofacet(facet)
                .map(|cofacet| cofacet.simplex)
                == Some(simplex)
        {
            return true;
        }
    }
    false
}

/// The column bases: the simplices of each dimension up to `max_dim`, in filtration order (by value, then
/// [`Simplex`]).
///
/// If `skip_zero_apparent_pairs`, leave out every simplex in a zero-persistence apparent pair (see
/// [`in_zero_apparent_pair`]): only for the clearing algorithms, on a matrix with a transpose (they recognise apparent
/// pairs again when needed); the standard algorithm needs every simplex. Set `check_top_cofacets` to false if the
/// cofacets of the top dimension might not fit in a [`Simplex`]. If `parallel`, build on the current rayon thread
/// pool.
pub fn build_bases<K: LazyComplex>(
    complex: &K,
    max_dim: usize,
    skip_zero_apparent_pairs: bool,
    check_top_cofacets: bool,
    parallel: bool,
) -> SplitBasis<Simplex> {
    let codec = complex.codec();
    let keep = |dim: usize| {
        let check_cofacets = dim < max_dim || check_top_cofacets;
        move |&(value, simplex): &(NotNan<f64>, Simplex)| {
            !skip_zero_apparent_pairs
                || !in_zero_apparent_pair(complex, simplex, value, check_cofacets)
        }
    };
    let sorted = |mut simplices: Vec<(NotNan<f64>, Simplex)>| {
        if parallel {
            simplices.par_sort_unstable();
        } else {
            simplices.sort_unstable();
        }
        simplices
            .into_iter()
            .map(|(_, simplex)| simplex)
            .collect::<Vec<_>>()
    };
    let generate = |&(_, simplex): &(NotNan<f64>, Simplex)| {
        complex
            .extensions(simplex)
            .map(|(cofacet, value)| (value, cofacet))
    };

    // Every simplex of the current dimension (needed to generate the next), then those we keep
    let mut all: Vec<(NotNan<f64>, Simplex)> = (0..complex.n_vertices())
        .map(|v| {
            let vertex = codec.encode(&[v]);
            (complex.value(vertex), vertex)
        })
        .collect();
    let mut dims = vec![];
    for dim in 0..=max_dim {
        let kept: Vec<_> = if parallel {
            all.par_iter().copied().filter(keep(dim)).collect()
        } else {
            all.iter().copied().filter(keep(dim)).collect()
        };
        dims.push(sorted(kept));
        if dim == max_dim {
            break;
        }
        if dim + 1 == max_dim {
            // Generate and filter the top dimension in one go, so that it is never stored in full
            let kept: Vec<_> = if parallel {
                all.par_iter()
                    .flat_map_iter(generate)
                    .filter(keep(max_dim))
                    .collect()
            } else {
                all.iter().flat_map(generate).filter(keep(max_dim)).collect()
            };
            dims.push(sorted(kept));
            break;
        }
        all = if parallel {
            all.par_iter().flat_map_iter(generate).collect()
        } else {
            all.iter().flat_map(generate).collect()
        };
    }
    SplitBasis::from_dimensions(dims)
}

/// A bar of the barcode of a [`LazyComplex`], see [`phlite::barcode::Bar`].
pub type SimplexBar = Bar<Simplex, NotNan<f64>, Z2>;

/// Whether simplices of dimension `max_dim + 1` (which both methods need) fit in a [`Simplex`] for `complex`.
pub fn fits<K: LazyComplex>(complex: &K, max_dim: usize) -> bool {
    max_dim + 2 <= complex.codec().max_vertices()
}

/// The barcode of `complex` in dimensions `0..=max_dim`, over Z/2, by cohomology or homology and with any of
/// `phlite`'s algorithms, optionally with a representative cycle for every bar.
///
/// Bars are returned in no particular order, and zero-length bars are omitted.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` don't fit in a [`Simplex`] (see [`fits`]).
pub fn barcode<K: LazyComplex, R: ColumnRepresentation>(
    complex: &K,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<SimplexBar> {
    #[allow(unreachable_patterns)]
    let parallel = matches!(algorithm, Algorithm::ParallelClearing(_));
    barcode_with(complex, max_dim, method, algorithm, representatives, parallel, true)
}

/// As [`barcode`], choosing whether to build the bases in parallel on the current rayon thread pool
/// (`parallel_setup`, which [`barcode`] does only for the parallel algorithm), whatever the algorithm, and whether
/// (by cohomology, without representatives) to find dimension 0 by union-find (`union_find`, which [`barcode`] does):
/// every edge's boundary is the difference of its two vertices, so reducing dimension 0 is a union-find on the edges
/// in filtration order ([`phlite::union_find`]), and the edges that kill a component are left out of dimension 1, as
/// clearing would.
///
/// # Panics
/// As [`barcode`]: if simplices of dimension `max_dim + 1` don't fit in a [`Simplex`] (see [`fits`]).
pub fn barcode_with<K: LazyComplex, R: ColumnRepresentation>(
    complex: &K,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    parallel_setup: bool,
    union_find: bool,
) -> Vec<SimplexBar> {
    assert!(
        fits(complex, max_dim),
        "simplices of dimension {} don't fit in a Simplex on {} vertices",
        max_dim + 1,
        complex.n_vertices()
    );
    let parallel = parallel_setup;
    // The standard algorithm can't find the owners of omitted apparent pairs, so give it every simplex
    let skip = !matches!(algorithm, Algorithm::Standard(_));
    let coboundary = Coboundary::<K, Z2>::new(complex);
    let boundary = Boundary::<K, Z2>::new(complex);
    match method {
        Method::Cohomology if union_find && !representatives => {
            let (mut bars, deaths) = components(complex, parallel);
            if max_dim >= 1 {
                let basis = without(build_bases(complex, max_dim, skip, true, parallel), &deaths);
                let coboundary = coboundary.with_basis(basis);
                let coboundary = coboundary.by_ref().with_transpose(&boundary);
                bars.extend(barcode::cohomology(coboundary, algorithm));
            }
            bars
        }
        Method::Cohomology => {
            let basis = build_bases(complex, max_dim, skip, true, parallel);
            let coboundary = coboundary.with_basis(basis);
            let coboundary = coboundary.by_ref().with_transpose(&boundary);
            if representatives {
                barcode::cohomology_with_involution(coboundary, algorithm)
            } else {
                barcode::cohomology(coboundary, algorithm)
            }
        }
        Method::Homology => {
            // The cofacets of the top dimension (max_dim + 1) might not fit, so don't look at them
            let basis = build_bases(complex, max_dim + 1, skip, false, parallel);
            let boundary = boundary.with_basis(basis);
            let boundary = boundary.by_ref().with_transpose(&coboundary);
            let mut bars = barcode::homology(boundary, algorithm, representatives);
            // Nothing kills the top-dimensional simplices, so they look essential, but we didn't compute that dimension
            bars.retain(|bar| bar.birth.dimension() <= max_dim);
            bars
        }
    }
}

/// The bars of dimension 0 by union-find on the edges, in filtration order (by value, then [`Simplex`], as the
/// reductions order them), with the elder rule (the root of a component is its oldest vertex, by value then number):
/// so the pairs a reduction would find. Zero-length bars are left out. Also returns the edges that kill a component
/// (zero-length bars' included), sorted.
fn components<K: LazyComplex>(complex: &K, parallel: bool) -> (Vec<SimplexBar>, Vec<Simplex>) {
    use phlite::union_find::IncidenceReduction;
    let codec = complex.codec();
    let n = complex.n_vertices();
    let vertex = |v: u32| codec.encode(&[v]);
    // Every edge once, as an extension of its first vertex
    let edges_of = |v: u32| complex.extensions(vertex(v)).map(|(edge, value)| (value, edge));
    let mut edges: Vec<(NotNan<f64>, Simplex)> = if parallel {
        (0..n).into_par_iter().flat_map_iter(edges_of).collect()
    } else {
        (0..n).flat_map(edges_of).collect()
    };
    if parallel {
        edges.par_sort_unstable();
    } else {
        edges.sort_unstable();
    }
    let values: Vec<NotNan<f64>> = (0..n).map(|v| complex.value(vertex(v))).collect();
    let mut union_find = IncidenceReduction::<Z2, NotNan<f64>>::with_keys(values.iter().copied());
    let mut bars = vec![];
    let mut deaths = vec![];
    for (value, edge) in edges {
        let (a, b) = (codec.vertex(edge, 0), codec.vertex(edge, 1));
        let Some(younger) = union_find.join(a, Some(b)) else {
            continue;
        };
        deaths.push(edge);
        if values[younger as usize] != value {
            bars.push(Bar {
                birth: vertex(younger),
                birth_value: values[younger as usize],
                death: Some((edge, value)),
                representative: None,
            });
        }
    }
    for v in 0..n {
        if union_find.is_free_root(v) {
            bars.push(Bar { birth: vertex(v), birth_value: values[v as usize], death: None, representative: None });
        }
    }
    deaths.sort_unstable();
    (bars, deaths)
}

/// `basis` without dimension 0 (found by union-find) and without the edges in `excluded` (sorted).
fn without(basis: SplitBasis<Simplex>, excluded: &[Simplex]) -> SplitBasis<Simplex> {
    use phlite::matrices::SplitByDimension;
    let dims: Vec<Vec<Simplex>> = (0..SplitByDimension::n_dimensions(&&basis))
        .map(|dim| {
            let slice = (&basis).in_dimension(dim).0;
            match dim {
                0 => vec![],
                1 => slice.iter().copied().filter(|e| excluded.binary_search(e).is_err()).collect(),
                _ => slice.to_vec(),
            }
        })
        .collect();
    SplitBasis::from_dimensions(dims)
}
