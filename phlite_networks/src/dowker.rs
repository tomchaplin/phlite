//! The Dowker sink and source complexes of a weighted digraph (Chowdhury and Mémoli [[1]](#1)).
//!
//! Think of the weight of the edge `a -> b` as a dissimilarity `ω(a, b)`, with `ω(a, a) = 0` and `ω(a, b)` infinite
//! when there is no edge. A set of nodes is a simplex of the **sink** complex at time `t` when they have a common
//! *sink*: a node `x` (perhaps one of them) with `ω(v, x) <= t` for each of them. So a simplex enters at
//!
//! `min_x max_{v in σ} ω(v, x)`,
//!
//! and in the **source** complex likewise with `ω(x, v)`: a common *source*. By Dowker duality the two have the same
//! barcode [[1]](#1) (a property the tests check), though not the same simplices.
//!
//! Simplices are ordinary (unordered) sets of nodes, stored as their nodes in increasing order. The complex is
//! described lazily, as for the directed flag complex (see [`crate::lazy`]): the cofacets of a simplex, adding a node
//! larger or smaller than all of its own, come in increasing order by adding nodes in increasing order, and each one's
//! value is found from the simplex's own `max_{v in σ} ω(v, x)` for every `x`, so listing them takes `O(n)` per
//! cofacet. Every set of nodes with a common sink is a simplex, so Dowker complexes are dense: they suit small digraphs
//! (up to a few hundred nodes) and low dimensions.
//!
//! # References
//!
//! <a id="1">[1]</a> Chowdhury, S. and Mémoli, F., 2018. A functorial Dowker theorem and persistent homology of
//! asymmetric networks. Journal of Applied and Computational Topology, 2(1), 115–175.

use ordered_float::NotNan;

use crate::{
    digraph::Digraph,
    lazy::{Cofacet, LazyComplex},
    simplex::{Codec, Simplex, Vertices},
};

/// Which side the witness is on, see the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// A common sink: `ω(v, x) <= t` for every vertex `v`.
    Sink,
    /// A common source: `ω(x, v) <= t` for every vertex `v`.
    Source,
}

/// The Dowker sink or source complex of a weighted digraph.
#[derive(Debug, Clone)]
pub struct DowkerComplex {
    codec: Codec,
    n: usize,
    /// `toward[v * n + x]`: how far `v` is from having `x` as its witness, `ω(v, x)` for a sink and `ω(x, v)` for a
    /// source (0 for `v == x`, infinite for no edge)
    toward: Vec<f64>,
}

impl DowkerComplex {
    /// The Dowker complex of `graph` on the given side.
    pub fn new(graph: &Digraph, side: Side) -> Self {
        let n = graph.n_vertices() as usize;
        let mut toward = vec![f64::INFINITY; n * n];
        for v in 0..n {
            toward[v * n + v] = 0.0;
        }
        for (a, b, w) in graph.edges() {
            let (v, x) = match side {
                Side::Sink => (a, b),
                Side::Source => (b, a),
            };
            toward[v as usize * n + x as usize] = w.into_inner();
        }
        Self {
            codec: Codec::new(graph.n_vertices()),
            n,
            toward,
        }
    }

    /// The vertices of `simplex`, in increasing order.
    pub fn vertices(&self, simplex: Simplex) -> Vertices {
        self.codec.vertices(simplex)
    }

    /// The simplex with these vertices (in increasing order).
    pub fn simplex(&self, vertices: &[u32]) -> Simplex {
        self.codec.encode(vertices)
    }

    /// For each potential witness `x`, the largest `ω` from a vertex of `vertices` to `x`.
    fn reach(&self, vertices: &[u32]) -> Vec<f64> {
        let mut reach = vec![0.0f64; self.n];
        for &v in vertices {
            let row = &self.toward[v as usize * self.n..(v as usize + 1) * self.n];
            for (r, &w) in reach.iter_mut().zip(row) {
                *r = r.max(w);
            }
        }
        reach
    }

    /// When the set of nodes `vertices` enters (infinite if it never does).
    pub fn value_of(&self, vertices: &[u32]) -> f64 {
        self.reach(vertices).into_iter().fold(f64::INFINITY, f64::min)
    }
}

/// The cofacets of a simplex of a [`DowkerComplex`], in increasing order (by the node added).
struct DowkerCofacets<'a> {
    complex: &'a DowkerComplex,
    simplex: Simplex,
    vertices: Vertices,
    reach: Vec<f64>,
    /// The next node to try adding, and the position it would go in
    next: u32,
    position: usize,
}

impl Iterator for DowkerCofacets<'_> {
    type Item = Cofacet;

    fn next(&mut self) -> Option<Cofacet> {
        let n = self.complex.n as u32;
        while self.next < n {
            let w = self.next;
            self.next += 1;
            // Skip the simplex's own vertices, keeping track of where w would go
            while self.position < self.vertices.len() && self.vertices[self.position] < w {
                self.position += 1;
            }
            if self.vertices.get(self.position) == Some(&w) {
                continue;
            }
            let row = &self.complex.toward[w as usize * self.complex.n..(w as usize + 1) * self.complex.n];
            let value = self
                .reach
                .iter()
                .zip(row)
                .map(|(&r, &d)| r.max(d))
                .fold(f64::INFINITY, f64::min);
            if value.is_finite() {
                return Some(Cofacet {
                    position: self.position,
                    simplex: self.complex.codec.insert(self.simplex, self.position, w),
                    value: NotNan::new(value).unwrap(),
                });
            }
        }
        None
    }
}

impl LazyComplex for DowkerComplex {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn n_vertices(&self) -> u32 {
        self.n as u32
    }

    fn value(&self, simplex: Simplex) -> NotNan<f64> {
        let vertices = self.codec.vertices(simplex);
        let value = self.value_of(&vertices);
        NotNan::new(value)
            .ok()
            .filter(|v| v.is_finite())
            .unwrap_or_else(|| panic!("{vertices:?} is not in the complex"))
    }

    fn cofacets(&self, simplex: Simplex) -> (NotNan<f64>, impl Iterator<Item = Cofacet> + '_) {
        let vertices = self.codec.vertices(simplex);
        let reach = self.reach(&vertices);
        let value = reach.iter().copied().fold(f64::INFINITY, f64::min);
        (
            NotNan::new(value).unwrap(),
            DowkerCofacets {
                complex: self,
                simplex,
                vertices,
                reach,
                next: 0,
                position: 0,
            },
        )
    }

    fn extensions(&self, simplex: Simplex) -> impl Iterator<Item = (Simplex, NotNan<f64>)> + '_ {
        // Adding a node above every vertex: start after the last
        let vertices = self.codec.vertices(simplex);
        let next = vertices.last().map_or(0, |&v| v + 1);
        let reach = self.reach(&vertices);
        DowkerCofacets {
            complex: self,
            simplex,
            vertices,
            reach,
            next,
            position: vertices.len(),
        }
        .map(|cofacet| (cofacet.simplex, cofacet.value))
    }

    fn facets(&self, simplex: Simplex) -> impl Iterator<Item = (usize, Simplex)> + '_ {
        let k = simplex.n_vertices();
        let n_facets = if k == 1 { 0 } else { k };
        (0..n_facets).map(move |p| (p, self.codec.remove(simplex, p)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{barcode, build_bases, Boundary, Coboundary};
    use phlite::{
        barcode::{Algorithm, Method},
        fields::Z3,
        matrices::{ColBasis, HasRowFiltration, MatrixOracle},
        reduction::{Clearing, Standard},
    };

    fn graphs() -> Vec<(u32, Vec<(u32, u32, f64)>)> {
        let mut graphs = vec![
            (4, vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 2.0)]),
            (3, vec![(0, 1, 1.0), (1, 0, 2.0), (1, 2, 1.0), (0, 2, 3.0)]),
        ];
        let mut state = 11u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for n in [5, 7, 8] {
            let edges = (0..n)
                .flat_map(|a| (0..n).map(move |b| (a, b)))
                .filter(|(a, b)| a != b)
                .filter_map(|(a, b)| (next() % 3 != 0).then(|| (a, b, (next() % 4) as f64)))
                .collect();
            graphs.push((n, edges));
        }
        graphs
    }

    /// Every subset of nodes (up to `max_vertices`) with its value, by brute force
    fn brute_force(complex: &DowkerComplex, max_vertices: usize) -> Vec<(Vec<u32>, f64)> {
        let n = complex.n as u32;
        let mut out = vec![];
        let mut current: Vec<Vec<u32>> = (0..n).map(|v| vec![v]).collect();
        for len in 1..=max_vertices {
            for vertices in &current {
                let value = complex.value_of(vertices);
                if value.is_finite() {
                    out.push((vertices.clone(), value));
                }
            }
            if len == max_vertices {
                break;
            }
            current = current
                .iter()
                .flat_map(|vs| ((vs.last().unwrap() + 1)..n).map(move |w| [vs.clone(), vec![w]].concat()))
                .collect();
        }
        out.sort_by(|a, b| (a.0.len(), &a.0).cmp(&(b.0.len(), &b.0)));
        out
    }

    #[test]
    fn cofacets_and_bases_match_brute_force() {
        for (n, edges) in graphs() {
            let graph = Digraph::new(n, edges.iter().copied());
            for side in [Side::Sink, Side::Source] {
                let complex = DowkerComplex::new(&graph, side);
                let all = brute_force(&complex, 4);
                for (vertices, value) in &all {
                    let simplex = complex.simplex(vertices);
                    assert_eq!(complex.value(simplex).into_inner(), *value);
                    if vertices.len() < 4 {
                        let (bound, cofacets) = complex.cofacets(simplex);
                        let got: Vec<_> = cofacets.map(|c| (c.position, c.simplex, c.value.into_inner())).collect();
                        let expected: Vec<_> = all
                            .iter()
                            .filter(|(c, _)| c.len() == vertices.len() + 1 && vertices.iter().all(|v| c.contains(v)))
                            .map(|(c, value)| {
                                let position = (0..c.len()).find(|&p| !vertices.contains(&c[p])).unwrap();
                                (position, complex.simplex(c), *value)
                            })
                            .collect();
                        assert_eq!(got, expected, "{side:?} cofacets of {vertices:?}");
                        assert!(got.iter().all(|c| c.2 >= bound.into_inner()));
                    }
                }
                let bases = build_bases(&complex, 3, false, true, false);
                let mut expected: Vec<_> = all
                    .iter()
                    .map(|(vs, value)| (vs.len(), NotNan::new(*value).unwrap(), complex.simplex(vs)))
                    .collect();
                expected.sort();
                let got: Vec<_> = (0..bases.size()).map(|i| bases.element(i)).collect();
                assert_eq!(got, expected.into_iter().map(|(_, _, s)| s).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn transpose_and_hooks() {
        for (n, edges) in graphs() {
            let complex = DowkerComplex::new(&Digraph::new(n, edges.iter().copied()), Side::Sink);
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
                for (coeff, cofacet) in coboundary.column(simplex) {
                    let entries: Vec<_> = boundary.column(cofacet).filter(|(_, f)| *f == simplex).collect();
                    assert_eq!(entries, vec![(coeff, simplex)]);
                }
            }
        }
    }

    /// Dowker duality: the sink and source complexes have the same barcode, by every method and algorithm
    #[test]
    fn dowker_duality() {
        let summary = |bars: Vec<crate::lazy::SimplexBar>| {
            let mut out: Vec<_> = bars
                .iter()
                .filter(|bar| bar.death.is_none_or(|(_, d)| d != bar.birth_value))
                .map(|bar| (bar.birth.dimension(), bar.birth_value, bar.death.map(|(_, d)| d)))
                .collect();
            out.sort();
            out
        };
        for (n, edges) in graphs() {
            let graph = Digraph::new(n, edges.iter().copied());
            let sink = DowkerComplex::new(&graph, Side::Sink);
            let source = DowkerComplex::new(&graph, Side::Source);
            let expected = summary(barcode(&sink, 2, Method::Homology, Algorithm::Standard(Standard::new()), false));
            for complex in [&sink, &source] {
                for method in [Method::Cohomology, Method::Homology] {
                    for algorithm in [
                        Algorithm::Clearing(Clearing::new()),
                        Algorithm::Standard(Standard::new()),
                        Algorithm::ParallelClearing(Clearing::new().parallel()),
                    ] {
                        let bars = barcode(complex, 2, method, algorithm, true);
                        assert_eq!(summary(bars), expected, "{n} {method:?}");
                    }
                }
            }
        }
    }
}
