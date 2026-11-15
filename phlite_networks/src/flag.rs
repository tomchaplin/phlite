//! Directed flag complexes and ordered-tuple complexes of weighted digraphs, and the grounded directed flag complex.
//!
//! An *ordered simplex* of a digraph is a sequence of vertices `(x_0, ..., x_k)` in which `x_i -> x_j` is an edge
//! whenever `i < j` and `x_i != x_j`. Two complexes are built from them:
//!
//! * the **directed flag complex** ([`Kind::DirectedFlag`]), as in flagser [[1]](#1): the vertices are distinct, so
//!   each simplex is a directed clique with a total order (a source, then a vertex with one edge in, and so on);
//! * the **ordered-tuple complex** ([`Kind::OrderedTuples`]), as in Turner [[2]](#2): vertices may repeat, but not
//!   consecutively (`(a, b, a)` is a simplex when `a -> b` and `b -> a` are edges, `(a, a, b)` is degenerate). The
//!   boundary of a simplex drops its degenerate faces (this is the normalised chain complex), so `(a, b, a)` has
//!   boundary `(b, a) + (a, b)`.
//!
//! The faces of a simplex are its subsequences (one vertex removed), with the usual signs. With edge `a -> b`
//! entering at `w(a, b)` (see [`crate::digraph`]) and nodes at 0, a simplex enters when its last pair of vertices is
//! joined: at `max_{i < j, x_i != x_j} w(x_i, x_j)` (flagser's `max` filtration).
//!
//! The **grounded** directed flag complex follows grounded persistent path homology [[3]](#3): the filtration is the
//! directed flag complex of the shortest-path digraph, except that the edges of the original digraph (and every node)
//! are present from time 0. Only dimensions 0 and 1 are meaningful: the grounded homology in dimension 1 is the cycles
//! of the digraph itself, until they become boundaries.
//!
//! Everything here is lazy, as for Vietoris–Rips (see [`crate::lazy`]): cofacets are listed on the fly, in increasing
//! order, by intersecting adjacency lists, and the zero-persistence apparent pairs are left out of the reduction.
//!
//! # References
//!
//! <a id="1">[1]</a> Lütgehetmann, D., Govc, D., Smith, J.P. and Levi, R., 2020. Computing persistent homology of
//! directed flag complexes. Algorithms, 13(1), 19.
//!
//! <a id="2">[2]</a> Turner, K., 2019. Rips filtrations for quasimetric spaces and asymmetric functions with
//! stability results. Algebraic & Geometric Topology, 19(3), 1135–1170.
//!
//! <a id="3">[3]</a> Chaplin, T., Harrington, H.A. and Tillmann, U., 2022. Grounded persistent path homology: a
//! stable, topological descriptor for weighted digraphs. arXiv:2210.11274.

use ordered_float::NotNan;

use crate::{
    digraph::Digraph,
    lazy::{Cofacet, LazyComplex},
    simplex::{Codec, Simplex, Vertices},
};

/// Which ordered simplices are allowed, see the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Distinct vertices: the directed flag complex.
    DirectedFlag,
    /// Vertices may repeat, but not consecutively: the ordered-tuple complex.
    OrderedTuples,
}

/// A directed flag or ordered-tuple complex of a weighted digraph, possibly grounded (see the [module docs](self)).
#[derive(Debug, Clone)]
pub struct FlagComplex {
    kind: Kind,
    codec: Codec,
    /// The digraph whose ordered simplices make the complex; edge `a -> b` enters at its weight
    graph: Digraph,
    /// If grounded: the edges (1-simplices), with the base digraph's edges at time 0 and the others at their weight
    grounded_edges: Option<Digraph>,
}

const ZERO: NotNan<f64> = unsafe { NotNan::new_unchecked(0.0) };

impl FlagComplex {
    /// The directed flag (or ordered-tuple) complex of `graph`.
    pub fn new(kind: Kind, graph: Digraph) -> Self {
        Self {
            kind,
            codec: Codec::new(graph.n_vertices()),
            graph,
            grounded_edges: None,
        }
    }

    /// The complex of `graph` grounded at `base_edges`: those edges (whatever their weight in `graph`, or even if
    /// they aren't edges of `graph`) are present from time 0, as 1-simplices. Only the barcode in dimensions 0 and 1
    /// is meaningful.
    pub fn grounded(kind: Kind, graph: Digraph, base_edges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let edges = graph
            .edges()
            .map(|(a, b, w)| (a, b, w.into_inner()))
            .chain(base_edges.into_iter().map(|(a, b)| (a, b, 0.0)));
        let grounded_edges = Digraph::new(graph.n_vertices(), edges);
        Self {
            grounded_edges: Some(grounded_edges),
            ..Self::new(kind, graph)
        }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn graph(&self) -> &Digraph {
        &self.graph
    }

    /// The edges of the complex (1-simplices), with their values.
    fn edges(&self) -> &Digraph {
        self.grounded_edges.as_ref().unwrap_or(&self.graph)
    }

    /// The vertices of `simplex`.
    pub fn vertices(&self, simplex: Simplex) -> Vertices {
        self.codec.vertices(simplex)
    }

    /// The simplex with these vertices (which needn't be in the complex).
    pub fn simplex(&self, vertices: &[u32]) -> Simplex {
        self.codec.encode(vertices)
    }

    /// When the ordered simplex with these vertices enters the ungrounded filtration, or `None` if it never does.
    /// Doesn't check that consecutive vertices differ.
    pub fn ungrounded_value(&self, vertices: &[u32]) -> Option<NotNan<f64>> {
        let mut value = ZERO;
        for (i, &a) in vertices.iter().enumerate() {
            for &b in &vertices[i + 1..] {
                if a == b {
                    if self.kind == Kind::DirectedFlag {
                        return None;
                    }
                    continue;
                }
                value = value.max(self.graph.weight(a, b)?);
            }
        }
        Some(value)
    }

    /// The cofacets of `simplex`, in increasing order; or only those with the new vertex last if `only_last`.
    fn cofacet_iter(&self, simplex: Simplex, only_last: bool) -> (NotNan<f64>, CofacetIter<'_>) {
        let vertices = self.codec.vertices(simplex);
        // A vertex's cofacets are the edges at it, which may be grounded; otherwise they are (ungrounded) simplices
        // of dimension at least two, so if the simplex isn't in the ungrounded filtration (a grounded edge that never
        // enters it) it has none
        let (graph, base) = if vertices.len() == 1 {
            (self.edges(), Some(ZERO))
        } else {
            (&self.graph, self.ungrounded_value(&vertices))
        };
        let mut iter = CofacetIter {
            complex: self,
            graph,
            simplex,
            vertices,
            base: base.unwrap_or(ZERO),
            phase: if only_last { Phase::Last } else { Phase::Down },
            only_last,
            position: if only_last { vertices.len() } else { 0 },
            driver: Driver::None,
            candidates: &[],
            next: 0,
        };
        if base.is_none() {
            iter.phase = Phase::Done;
        } else {
            iter.start_segment();
        }
        (iter.base, iter)
    }
}

/// The three phases of listing cofacets in increasing order, see [`CofacetIter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Positions `0, ..., k`, new vertices below the one they displace
    Down,
    /// Position `k + 1`, the new vertex last
    Last,
    /// Positions `k, ..., 0`, new vertices above the one they displace
    Up,
    Done,
}

/// Which edge the list of candidate vertices comes from (so needn't be checked again).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Driver {
    None,
    /// The candidates `w` are the out-neighbours of the vertex at this index: `x_i -> w`
    Out(usize),
    /// The candidates `w` are the in-neighbours of the vertex at this index: `w -> x_j`
    In(usize),
}

/// The cofacets of an ordered simplex `(x_0, ..., x_k)`, in increasing (lexicographic) order.
///
/// The cofacet with `w` inserted at position `p` is `(x_0, ..., x_{p-1}, w, x_p, ..., x_k)`. Two cofacets with
/// `p < p'` agree before position `p`, where one has `w` and the other `x_p` (and `w != x_p`, since consecutive
/// vertices differ), so the first is smaller exactly when `w < x_p`. The order is therefore: for `p = 0, ..., k`,
/// the `w < x_p` (increasing); then `p = k + 1`, every `w`; then for `p = k, ..., 0`, the `w > x_p`.
///
/// At position `p`, `w` must be an out-neighbour of `x_{p-1}` and an in-neighbour of `x_p` (neither can equal `w`);
/// the candidates are the shorter of these two sorted adjacency lists, and the other conditions are looked up.
struct CofacetIter<'a> {
    complex: &'a FlagComplex,
    /// The digraph the new edges come from: the (possibly grounded) edges for a vertex, otherwise the filtration's
    graph: &'a Digraph,
    simplex: Simplex,
    vertices: Vertices,
    /// The value of the simplex in the ungrounded filtration, a lower bound on its cofacets'
    base: NotNan<f64>,
    phase: Phase,
    only_last: bool,
    position: usize,
    driver: Driver,
    /// The candidates for the new vertex in this segment, with the weight of the driving edge, and the next to try
    candidates: &'a [(u32, NotNan<f64>)],
    next: usize,
}

impl CofacetIter<'_> {
    /// Set up the candidates for the current phase and position.
    fn start_segment(&mut self) {
        let k = self.vertices.len() - 1;
        let p = self.position;
        let out = (p > 0).then(|| (self.graph.out_edges(self.vertices[p - 1]), Driver::Out(p - 1)));
        let into = (p <= k).then(|| (self.graph.in_edges(self.vertices[p]), Driver::In(p)));
        let (candidates, driver) = match (out, into) {
            (Some(out), Some(into)) => {
                if out.0.len() <= into.0.len() {
                    out
                } else {
                    into
                }
            }
            (Some(only), None) | (None, Some(only)) => only,
            (None, None) => unreachable!("a simplex has a vertex"),
        };
        self.driver = driver;
        self.candidates = candidates;
        self.next = 0;
        match self.phase {
            Phase::Down => {
                let displaced = self.vertices[p];
                let end = candidates.partition_point(|&(w, _)| w < displaced);
                self.candidates = &candidates[..end];
            }
            Phase::Up => {
                let displaced = self.vertices[p];
                self.next = candidates.partition_point(|&(w, _)| w <= displaced);
            }
            Phase::Last | Phase::Done => {}
        }
    }

    /// Move to the next segment, returning false when there are none left.
    fn next_segment(&mut self) -> bool {
        let k = self.vertices.len() - 1;
        match self.phase {
            Phase::Down if self.position < k => self.position += 1,
            Phase::Down => {
                self.phase = Phase::Last;
                self.position = k + 1;
            }
            Phase::Last if !self.only_last => {
                self.phase = Phase::Up;
                self.position = k;
            }
            Phase::Up if self.position > 0 => self.position -= 1,
            _ => {
                self.phase = Phase::Done;
                return false;
            }
        }
        self.start_segment();
        true
    }

    /// The value of the cofacet with `w` inserted at the current position (`driving` is the weight of the edge it
    /// was found by), if it is in the complex.
    #[inline]
    fn check(&self, w: u32, driving: NotNan<f64>) -> Option<NotNan<f64>> {
        let p = self.position;
        let repeats = self.complex.kind == Kind::OrderedTuples;
        let mut value = self.base.max(driving);
        for (i, &x) in self.vertices.iter().enumerate() {
            if (i < p && self.driver == Driver::Out(i)) || (i >= p && self.driver == Driver::In(i)) {
                continue;
            }
            if x == w {
                // Only an ordered tuple may repeat a vertex, and never next to itself
                if !repeats || i + 1 == p || i == p {
                    return None;
                }
                continue;
            }
            let weight = if i < p {
                self.graph.weight(x, w)?
            } else {
                self.graph.weight(w, x)?
            };
            value = value.max(weight);
        }
        Some(value)
    }
}

impl Iterator for CofacetIter<'_> {
    type Item = Cofacet;

    #[inline]
    fn next(&mut self) -> Option<Cofacet> {
        loop {
            if self.phase == Phase::Done {
                return None;
            }
            while self.next < self.candidates.len() {
                let (w, driving) = self.candidates[self.next];
                self.next += 1;
                if let Some(value) = self.check(w, driving) {
                    return Some(Cofacet {
                        position: self.position,
                        simplex: self.complex.codec.insert(self.simplex, self.position, w),
                        value,
                    });
                }
            }
            if !self.next_segment() {
                return None;
            }
        }
    }
}

impl LazyComplex for FlagComplex {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn n_vertices(&self) -> u32 {
        self.graph.n_vertices()
    }

    fn value(&self, simplex: Simplex) -> NotNan<f64> {
        let vertices = self.codec.vertices(simplex);
        match *vertices {
            [_] => Some(ZERO),
            [a, b] => self.edges().weight(a, b),
            _ => self.ungrounded_value(&vertices),
        }
        .unwrap_or_else(|| panic!("{vertices:?} is not in the complex"))
    }

    fn cofacets(&self, simplex: Simplex) -> (NotNan<f64>, impl Iterator<Item = Cofacet> + '_) {
        self.cofacet_iter(simplex, false)
    }

    fn extensions(&self, simplex: Simplex) -> impl Iterator<Item = (Simplex, NotNan<f64>)> + '_ {
        self.cofacet_iter(simplex, true)
            .1
            .map(|cofacet| (cofacet.simplex, cofacet.value))
    }

    fn facets(&self, simplex: Simplex) -> impl Iterator<Item = (usize, Simplex)> + '_ {
        let vertices = self.codec.vertices(simplex);
        let k = vertices.len() - 1;
        let n_facets = if k == 0 { 0 } else { k + 1 };
        (0..n_facets)
            // Removing a vertex between two copies of the same vertex gives a degenerate face
            .filter(move |&p| p == 0 || p == k || vertices[p - 1] != vertices[p + 1])
            .map(move |p| (p, self.codec.remove(simplex, p)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{barcode, build_bases, Boundary, Coboundary};
    use phlite::{
        barcode::{Algorithm, Method},
        columns::{HeapColumns, MergedColumns},
        fields::Z3,
        matrices::{ColBasis, HasRowFiltration, MatrixOracle},
        reduction::{Clearing, Standard},
    };

    /// Every ordered simplex of `complex` (up to `max_vertices` vertices), with its value, by brute force.
    fn brute_force(complex: &FlagComplex, max_vertices: usize) -> Vec<(Vec<u32>, f64)> {
        let n = complex.n_vertices();
        let mut out = vec![];
        let mut current: Vec<Vec<u32>> = (0..n).map(|v| vec![v]).collect();
        for len in 1..=max_vertices {
            for vertices in &current {
                let value = match vertices.len() {
                    1 => Some(ZERO),
                    2 => complex.edges().weight(vertices[0], vertices[1]),
                    _ => complex.ungrounded_value(vertices),
                };
                if let Some(value) = value {
                    out.push((vertices.clone(), value.into_inner()));
                }
            }
            if len == max_vertices {
                break;
            }
            current = current
                .iter()
                .flat_map(|vertices| {
                    (0..n).filter_map(move |w| {
                        (w != *vertices.last().unwrap()).then(|| {
                            let mut longer = vertices.clone();
                            longer.push(w);
                            longer
                        })
                    })
                })
                .filter(|vertices| {
                    // The faces must be in the complex too (always true for flag complexes, but not for the grounded
                    // edges, which aren't edges of the triangles)
                    vertices.len() < 3 || complex.ungrounded_value(vertices).is_some()
                })
                .collect();
        }
        out.sort_by(|a, b| (a.0.len(), &a.0).cmp(&(b.0.len(), &b.0)));
        out
    }

    fn graphs() -> Vec<(u32, Vec<(u32, u32, f64)>)> {
        let mut graphs = vec![
            // A directed 4-cycle with a chord
            (4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 2.0)]),
            // A bidirected edge, and a transitive triangle
            (3, vec![(0, 1, 1.0), (1, 0, 2.0), (1, 2, 1.0), (0, 2, 3.0)]),
            // A complete digraph with tied weights
            (
                5,
                (0..5)
                    .flat_map(|a| (0..5).map(move |b| (a, b)))
                    .filter(|(a, b)| a != b)
                    .map(|(a, b)| (a, b, ((a * 3 + b * 5) % 4) as f64))
                    .collect(),
            ),
        ];
        // Random digraphs with lots of ties
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for n in [6, 8, 9] {
            let edges = (0..n)
                .flat_map(|a| (0..n).map(move |b| (a, b)))
                .filter(|(a, b)| a != b)
                .filter_map(|(a, b)| (next() % 2 == 0).then(|| (a, b, (next() % 4) as f64)))
                .collect();
            graphs.push((n, edges));
        }
        graphs
    }

    fn complexes() -> Vec<FlagComplex> {
        graphs()
            .into_iter()
            .flat_map(|(n, edges)| {
                let graph = Digraph::new(n, edges.iter().copied());
                let base = edges.iter().step_by(2).map(|&(a, b, _)| (a, b));
                [
                    FlagComplex::new(Kind::DirectedFlag, graph.clone()),
                    FlagComplex::new(Kind::OrderedTuples, graph.clone()),
                    FlagComplex::grounded(Kind::DirectedFlag, graph, base),
                ]
            })
            .collect()
    }

    /// The cofacets, in increasing order with the right values and signs, and the facets, agree with brute force;
    /// and so do the bases built from extensions.
    #[test]
    fn cofacets_facets_and_bases_match_brute_force() {
        for complex in complexes() {
            let max_vertices = 4;
            let all = brute_force(&complex, max_vertices);
            let simplices: Vec<Vec<u32>> = all.iter().map(|(s, _)| s.clone()).collect();
            for (vertices, value) in &all {
                let simplex = complex.simplex(vertices);
                assert_eq!(complex.value(simplex).into_inner(), *value, "{vertices:?}");
                if vertices.len() < max_vertices {
                    let (bound, cofacets) = complex.cofacets(simplex);
                    let cofacets: Vec<_> = cofacets.collect();
                    let expected: Vec<_> = all
                        .iter()
                        .filter(|(cofacet, _)| cofacet.len() == vertices.len() + 1)
                        .filter_map(|(cofacet, value)| {
                            (0..cofacet.len())
                                .find(|&p| {
                                    let mut face = cofacet.clone();
                                    face.remove(p);
                                    face == *vertices
                                })
                                .map(|p| (p, complex.simplex(cofacet), *value))
                        })
                        .collect();
                    let got: Vec<_> = cofacets
                        .iter()
                        .map(|c| (c.position, c.simplex, c.value.into_inner()))
                        .collect();
                    assert_eq!(got, expected, "cofacets of {vertices:?}");
                    assert!(cofacets.iter().all(|c| c.value >= bound));
                }
                let facets: Vec<_> = complex.facets(simplex).collect();
                for &(p, facet) in &facets {
                    let mut face = vertices.clone();
                    face.remove(p);
                    assert_eq!(complex.vertices(facet).to_vec(), face);
                    assert!(simplices.contains(&face), "{face:?} is a face of {vertices:?}");
                }
            }
            let bases = build_bases(&complex, max_vertices - 1, false, true, false);
            let mut expected: Vec<_> = all
                .iter()
                .map(|(vertices, value)| (vertices.len(), NotNan::new(*value).unwrap(), complex.simplex(vertices)))
                .collect();
            expected.sort();
            let got: Vec<_> = (0..bases.size()).map(|i| bases.element(i)).collect();
            assert_eq!(
                got,
                expected.into_iter().map(|(_, _, s)| s).collect::<Vec<_>>()
            );
        }
    }

    /// The hooks agree with their defaults, and the boundary is the transpose of the coboundary (over Z/3, so that
    /// signs matter); `d d = 0`.
    #[test]
    fn hooks_and_transpose() {
        for complex in complexes() {
            let coboundary = Coboundary::<_, Z3>::new(&complex);
            let boundary = Boundary::<_, Z3>::new(&complex);
            let bases = build_bases(&complex, 2, false, true, false);
            for i in 0..bases.size() {
                let simplex = bases.element(i);
                let generic_min = coboundary
                    .column_with_filtration(simplex)
                    .min()
                    .map(|e| (e.filtration_value, e.row_index, e.coeff));
                let min = coboundary
                    .column_min(simplex)
                    .map(|e| (e.filtration_value, e.row_index, e.coeff));
                assert_eq!(min, generic_min);
                let generic_pivot = boundary
                    .column(simplex)
                    .map(|(coeff, row)| (boundary.filtration_value(row), row, coeff))
                    .max_by_key(|&(value, row, _)| (value, row));
                let pivot = boundary
                    .column_pivot(simplex)
                    .map(|e| (e.filtration_value, e.row_index, e.coeff));
                assert_eq!(pivot, generic_pivot);
                // Transpose: every cofacet has this simplex as a facet, with the same coefficient
                for (coeff, cofacet) in coboundary.column(simplex) {
                    let entries: Vec<_> = boundary
                        .column(cofacet)
                        .filter(|(_, facet)| *facet == simplex)
                        .collect();
                    assert_eq!(entries, vec![(coeff, simplex)]);
                }
                for (coeff, facet) in boundary.column(simplex) {
                    let entries: Vec<_> = coboundary
                        .column(facet)
                        .filter(|(_, cofacet)| *cofacet == simplex)
                        .collect();
                    // A grounded edge outside the filtration has no cofacets, and so isn't a facet of any triangle
                    assert_eq!(entries, vec![(coeff, simplex)]);
                }
                // d d = 0
                let mut dd: std::collections::HashMap<Simplex, Option<Z3>> = Default::default();
                for (c1, facet) in boundary.column(simplex) {
                    for (c2, face) in boundary.column(facet) {
                        let entry = dd.entry(face).or_insert(None);
                        *entry = (c1 * c2) + *entry;
                    }
                }
                assert!(dd.values().all(Option::is_none), "d d != 0 on {simplex:?}");
            }
        }
    }

    fn summary(bars: &[crate::lazy::SimplexBar]) -> Vec<(usize, f64, Option<f64>)> {
        let mut out: Vec<_> = bars
            .iter()
            .map(|bar| {
                (
                    bar.birth.dimension(),
                    bar.birth_value.into_inner(),
                    bar.death.map(|(_, d)| d.into_inner()),
                )
            })
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    /// Every method and algorithm (and skipping apparent pairs or not) gives the same barcode
    #[test]
    fn every_method_and_algorithm_agree() {
        for complex in complexes() {
            let max_dim = if complex.grounded_edges.is_some() { 1 } else { 2 };
            let expected = summary(&barcode(
                &complex,
                max_dim,
                Method::Homology,
                Algorithm::Standard(Standard::new()),
                false,
            ));
            for method in [Method::Cohomology, Method::Homology] {
                for representatives in [false, true] {
                    for algorithm in [
                        Algorithm::Clearing(Clearing::new()),
                        Algorithm::ParallelClearing(Clearing::new().parallel()),
                        Algorithm::Standard(Standard::new()),
                    ] {
                        let bars = barcode(&complex, max_dim, method, algorithm.clone(), representatives);
                        assert_eq!(summary(&bars), expected, "{method:?} {algorithm:?}");
                    }
                    let bars = barcode(
                        &complex,
                        max_dim,
                        method,
                        Algorithm::Clearing(Clearing::new().with_columns(HeapColumns).cache_columns(true)),
                        representatives,
                    );
                    assert_eq!(summary(&bars), expected);
                    let bars = barcode(
                        &complex,
                        max_dim,
                        method,
                        Algorithm::Clearing(Clearing::new().with_columns(MergedColumns).store_r(true)),
                        representatives,
                    );
                    assert_eq!(summary(&bars), expected);
                }
            }
        }
    }

    fn bars_of(kind: Kind, n: u32, edges: &[(u32, u32, f64)], max_dim: usize) -> Vec<(usize, f64, Option<f64>)> {
        let complex = FlagComplex::new(kind, Digraph::new(n, edges.iter().copied()));
        summary(&barcode(
            &complex,
            max_dim,
            Method::Cohomology,
            Algorithm::default(),
            true,
        ))
    }

    /// `bars`, sorted as [`summary`] sorts (an essential bar, `None`, before the finite ones)
    fn sorted(mut bars: Vec<(usize, f64, Option<f64>)>) -> Vec<(usize, f64, Option<f64>)> {
        bars.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bars
    }

    #[test]
    fn hand_checked_examples() {
        let inf = None;
        // A directed 3-cycle: no 2-simplices in either complex (a cycle isn't transitive)
        let cycle = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)];
        assert_eq!(
            bars_of(Kind::DirectedFlag, 3, &cycle, 2),
            sorted(vec![(0, 0.0, Some(1.0)), (0, 0.0, Some(2.0)), (0, 0.0, inf), (1, 3.0, inf)])
        );
        // A transitive triangle fills in when its last edge arrives
        let transitive = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
        assert_eq!(
            bars_of(Kind::DirectedFlag, 3, &transitive, 2),
            sorted(vec![(0, 0.0, Some(1.0)), (0, 0.0, Some(2.0)), (0, 0.0, inf)])
        );
        // A bidirected edge: in the directed flag complex, a loop of two edges that nothing fills; in the
        // ordered-tuple complex, filled by (0, 1, 0) and (1, 0, 1) as soon as both edges are there
        let bidirected = [(0, 1, 1.0), (1, 0, 2.0)];
        assert_eq!(
            bars_of(Kind::DirectedFlag, 2, &bidirected, 2),
            sorted(vec![(0, 0.0, Some(1.0)), (0, 0.0, inf), (1, 2.0, inf)])
        );
        assert_eq!(
            bars_of(Kind::OrderedTuples, 2, &bidirected, 2),
            sorted(vec![(0, 0.0, Some(1.0)), (0, 0.0, inf)])
        );
        // A directed square a -> b -> d, a -> c -> d: two transitive triangles once a -> d arrives
        let square = [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 1.0), (0, 3, 5.0)];
        assert_eq!(
            bars_of(Kind::DirectedFlag, 4, &square, 2),
            sorted(vec![(0, 0.0, Some(1.0)), (0, 0.0, Some(1.0)), (0, 0.0, Some(1.0)), (0, 0.0, inf), (1, 1.0, Some(5.0))])
        );
        // The complete digraph on 3 vertices: in the directed flag complex 3 vertices, 6 edges and 6 2-simplices (every
        // ordering), Euler characteristic 3 - 6 + 6 = 3; with one component and no H_1 (checked below), that leaves
        // b_2 = 3 - 1 = 2 essential 2-dimensional classes
        let complete: Vec<_> = (0..3)
            .flat_map(|a| (0..3).map(move |b| (a, b)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| (a, b, 1.0))
            .collect();
        let bars = bars_of(Kind::DirectedFlag, 3, &complete, 2);
        assert_eq!(bars.iter().filter(|b| b.0 == 2 && b.2.is_none()).count(), 2);
        assert_eq!(bars.iter().filter(|b| b.0 == 1).count(), 0);
    }

    /// Grounded: the cycles of the digraph are born at 0 and die when filled in the shortest-path filtration
    #[test]
    fn grounded_examples() {
        // The directed 4-cycle (each edge 1): at 2 the triangles (0, 1, 2) and (2, 3, 0) are in, but their boundaries
        // leave 0 -> 2 and 2 -> 0, different edges, so the cycle dies only at 3, when triangles such as (0, 2, 3)
        // arrive (as grpphati says too)
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
        let complex = FlagComplex::grounded(
            Kind::DirectedFlag,
            Digraph::shortest_paths(4, &edges),
            edges.iter().map(|&(a, b, _)| (a, b)),
        );
        let bars = summary(&barcode(&complex, 1, Method::Cohomology, Algorithm::default(), true));
        assert_eq!(bars, vec![(0, 0.0, None), (1, 0.0, Some(3.0))]);
    }
}
