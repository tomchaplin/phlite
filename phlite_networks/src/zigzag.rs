//! Zigzag persistence of networks that change: snapshots of a graph or digraph on shared node labels (contacts
//! by day, a network at successive thresholds of a time-varying weight), each turned into a complex (the clique
//! complex, or the directed flag complex, whose simplices are directed cliques) and joined to the next by a union or
//! an intersection; computed by FastZigzag ([`phlite_complexes::zigzag::ComplexZigzag`]).
//!
//! Path homology doesn't fit this scheme: the ∂-invariant paths `Ω(G)` have no basis of cells, and `Ω` doesn't
//! respect unions (`Ω(G ∪ H)` can be larger than `Ω(G) + Ω(H)`, so there is no Mayer–Vietoris diamond for
//! FastZigzag's reduction to an up-down filtration to rest on). The directed flag complex is the nearest cellular
//! relative.
//!
//! ```
//! use phlite::barcode::Algorithm;
//! use phlite::reduction::Clearing;
//! use phlite_complexes::zigzag::{Join, LevelBar};
//! use phlite_networks::zigzag::{network_zigzag, NetworkComplex, Snapshot};
//!
//! // A triangle's edges, two at a time: each snapshot a path, each union the whole triangle's boundary (a loop)
//! let a = Snapshot::new([], [(0, 1), (1, 2)]);
//! let b = Snapshot::new([], [(1, 2), (2, 0)]);
//! let bars = network_zigzag(&[a, b], NetworkComplex::Clique, 1, Join::Union, Algorithm::Clearing(Clearing::new()));
//! assert!(bars.contains(&LevelBar { dimension: 1, birth: 1, death: 1 }));
//! ```

use std::collections::{BTreeMap, BTreeSet};

use phlite::barcode::Algorithm;
use phlite::columns::ColumnRepresentation;
use phlite_complexes::zigzag::{ComplexZigzag, Join, LevelBar, SimplicialComplex};

/// A snapshot of a network: its nodes and (directed, for a directed flag complex) edges, by label. Every endpoint of
/// an edge is a node; self-loops are ignored.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Snapshot {
    pub nodes: BTreeSet<u32>,
    pub edges: BTreeSet<(u32, u32)>,
}

impl Snapshot {
    /// The nodes given and the endpoints of the edges, with the edges (less self-loops).
    pub fn new(nodes: impl IntoIterator<Item = u32>, edges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut snapshot = Self { nodes: nodes.into_iter().collect(), edges: BTreeSet::new() };
        for (a, b) in edges {
            snapshot.nodes.insert(a);
            snapshot.nodes.insert(b);
            if a != b {
                snapshot.edges.insert((a, b));
            }
        }
        snapshot
    }

    /// The edges of a weighted network with weight at most `threshold` (and every node of `nodes`).
    pub fn at(nodes: impl IntoIterator<Item = u32>, edges: &[(u32, u32, f64)], threshold: f64) -> Self {
        Self::new(nodes, edges.iter().filter(|e| e.2 <= threshold).map(|&(a, b, _)| (a, b)))
    }
}

/// Which complex of a network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NetworkComplex {
    /// The clique (flag) complex of the underlying undirected graph: a simplex for each clique.
    #[default]
    Clique,
    /// The directed flag complex: an ordered simplex `(x_0, ..., x_k)` for each directed clique (`x_i -> x_j` for
    /// every `i < j`), as in flagser.
    DirectedFlag,
}

/// The complex of `snapshot` up to dimension `top`, on its labels (simplices sorted for the clique complex, ordered
/// for the directed flag complex).
pub fn network_complex(snapshot: &Snapshot, complex: NetworkComplex, top: usize) -> SimplicialComplex<u32> {
    let directed = complex == NetworkComplex::DirectedFlag;
    // Out-neighbours (all neighbours, for cliques)
    let mut out: BTreeMap<u32, BTreeSet<u32>> = snapshot.nodes.iter().map(|&v| (v, BTreeSet::new())).collect();
    for &(a, b) in &snapshot.edges {
        out.entry(a).or_default().insert(b);
        if !directed {
            out.entry(b).or_default().insert(a);
        }
    }
    let adjacent = |a: u32, b: u32| out.get(&a).is_some_and(|n| n.contains(&b));
    let mut simplices: Vec<Vec<u32>> = vec![];
    let mut layer: Vec<Vec<u32>> = snapshot.nodes.iter().map(|&v| vec![v]).collect();
    for _ in 0..=top {
        let mut next = vec![];
        for s in &layer {
            if s.len() <= top {
                let last = *s.last().expect("a node");
                for &w in &out[&last] {
                    // Cliques: increasing; directed cliques: every earlier vertex points to w
                    let fresh = if directed { !s.contains(&w) } else { w > last };
                    if fresh && s.iter().all(|&v| adjacent(v, w)) {
                        let mut t = s.clone();
                        t.push(w);
                        next.push(t);
                    }
                }
            }
        }
        simplices.append(&mut layer);
        layer = next;
    }
    SimplicialComplex::new(simplices)
}

/// The zigzag barcode, in levels (snapshot `i` at `2 i`, see [`LevelBar`]), of `snapshots`, each as `complex` (up to
/// dimension `max_dim + 1`), consecutive ones joined by `join`.
pub fn network_zigzag<R: ColumnRepresentation>(
    snapshots: &[Snapshot],
    complex: NetworkComplex,
    max_dim: usize,
    join: Join,
    algorithm: Algorithm<'_, R>,
) -> Vec<LevelBar> {
    let mut zigzag = ComplexZigzag::new(join);
    for snapshot in snapshots {
        zigzag.push(&network_complex(snapshot, complex, max_dim + 1));
    }
    let mut bars = zigzag.barcode(algorithm).expect("a network's complex is closed under faces");
    bars.retain(|b| b.dimension <= max_dim);
    bars
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    fn run(snapshots: &[Snapshot], complex: NetworkComplex, join: Join) -> Vec<LevelBar> {
        network_zigzag(snapshots, complex, 2, join, Algorithm::Clearing(Clearing::new()))
    }

    #[test]
    fn clique_complexes() {
        let k4 = Snapshot::new([], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        assert_eq!(network_complex(&k4, NetworkComplex::Clique, 3).len(), 15);
        let square = Snapshot::new([], [(0, 1), (1, 2), (2, 3), (3, 0)]);
        // A square, then filled in (K4), then a square again: a loop that dies in the union and is born again
        let bars = run(&[square.clone(), k4, square], NetworkComplex::Clique, Join::Union);
        let loops: Vec<_> = bars.iter().filter(|b| b.dimension == 1).copied().collect();
        assert_eq!(loops, vec![LevelBar { dimension: 1, birth: 0, death: 0 }, LevelBar { dimension: 1, birth: 4, death: 4 }]);
    }

    #[test]
    fn directed_flag_complexes() {
        // A directed 3-cycle has no 2-simplex (no directed clique): a loop; a transitive triangle has one: none
        let cycle = Snapshot::new([], [(0, 1), (1, 2), (2, 0)]);
        let transitive = Snapshot::new([], [(0, 1), (1, 2), (0, 2)]);
        assert_eq!(network_complex(&cycle, NetworkComplex::DirectedFlag, 2).len(), 6);
        assert_eq!(network_complex(&transitive, NetworkComplex::DirectedFlag, 2).len(), 7);
        let bars = run(&[cycle.clone(), transitive.clone()], NetworkComplex::DirectedFlag, Join::Intersection);
        // The intersection is the path 0 -> 1 -> 2: the cycle's loop lives in level 0 only
        assert!(bars.contains(&LevelBar { dimension: 1, birth: 0, death: 0 }), "{bars:?}");
        // Undirected, both are triangles: the cycle's clique complex is filled
        let bars = run(&[cycle, transitive], NetworkComplex::Clique, Join::Union);
        assert!(bars.iter().all(|b| b.dimension == 0));
        // Reciprocal edges: (0, 1) and (1, 0) both 1-simplices, a loop of two edges
        let both = Snapshot::new([], [(0, 1), (1, 0)]);
        let bars = run(&[both], NetworkComplex::DirectedFlag, Join::Union);
        assert!(bars.contains(&LevelBar { dimension: 1, birth: 0, death: 0 }));
    }

    /// Random digraphs: the directed flag complex has exactly the ordered tuples of distinct nodes with an edge from
    /// each to every later one, and the clique complex the sets of pairwise adjacent nodes (brute force)
    #[test]
    fn complexes_by_brute_force() {
        let mut seed = 77u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..100 {
            let n = 2 + (rand() % 5) as u32;
            let edges: Vec<(u32, u32)> =
                (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).filter(|&(a, b)| a != b).filter(|_| rand() % 3 == 0).collect();
            let snapshot = Snapshot::new(0..n, edges.iter().copied());
            let has = |a: u32, b: u32| edges.contains(&(a, b));
            // Every tuple of distinct nodes of length up to 4
            let mut tuples: Vec<Vec<u32>> = (0..n).map(|v| vec![v]).collect();
            let mut all = tuples.clone();
            for _ in 1..4 {
                tuples = tuples
                    .iter()
                    .flat_map(|t| (0..n).filter(|v| !t.contains(v)).map(move |v| [t.clone(), vec![v]].concat()))
                    .collect();
                all.extend(tuples.iter().cloned());
            }
            let directed: BTreeSet<Vec<u32>> =
                all.iter().filter(|t| (0..t.len()).all(|i| (i + 1..t.len()).all(|j| has(t[i], t[j])))).cloned().collect();
            assert_eq!(network_complex(&snapshot, NetworkComplex::DirectedFlag, 3).simplices, directed, "{edges:?}");
            let cliques: BTreeSet<Vec<u32>> = all
                .iter()
                .filter(|t| t.windows(2).all(|w| w[0] < w[1]))
                .filter(|t| (0..t.len()).all(|i| (i + 1..t.len()).all(|j| has(t[i], t[j]) || has(t[j], t[i]))))
                .cloned()
                .collect();
            assert_eq!(network_complex(&snapshot, NetworkComplex::Clique, 3).simplices, cliques, "{edges:?}");
        }
    }

    #[test]
    fn thresholds() {
        let edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 0, 3.0)];
        let snapshots: Vec<Snapshot> = [1.0, 2.0, 3.0, 2.0].iter().map(|&t| Snapshot::at(0..3, &edges, t)).collect();
        let bars = run(&snapshots, NetworkComplex::Clique, Join::Union);
        let h0_at = |l: usize| bars.iter().filter(|b| b.dimension == 0 && b.birth <= l && l <= b.death).count();
        assert_eq!((h0_at(0), h0_at(2), h0_at(4), h0_at(6)), (2, 1, 1, 1));
    }
}
