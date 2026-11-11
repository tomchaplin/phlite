//! Check that the parallel label-keyed reduction agrees with the serial clearing algorithm on random digraphs.

use ordered_float::NotNan;
use petgraph::{algo::dijkstra, graph::NodeIndex, Graph};
use phlite::{
    fields::Z2,
    matrices::{combinators::product, ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle},
    reduction::{Clearing, Diagram, ReductionAlgorithm, Storage},
};
use phlite_grpph::boundary::GrPPHBoundary;
use phlite_grpph::coboundary::{GrPPHCoboundary, VecDigraphFiltration};
use rustc_hash::{FxHashMap, FxHashSet};

/// Tiny deterministic generator so we don't need a dev-dependency
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Random digraph with integer edge weights, completed to a shortest-path filtration
fn random_digraph(n: u16, p: f64, seed: u64) -> (VecDigraphFiltration, FxHashSet<(u16, u16)>) {
    let mut rng = Lcg(seed);
    let mut g = Graph::<(), f64>::new();
    for _ in 0..n {
        g.add_node(());
    }
    let mut edge_set = FxHashSet::default();
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                let w = 1.0 + (rng.next_f64() * 4.0).floor();
                g.add_edge(NodeIndex::from(i as u32), NodeIndex::from(j as u32), w);
                edge_set.insert((i, j));
            }
        }
    }
    let filtration = (0..n)
        .map(|i| {
            let sp = dijkstra(&g, NodeIndex::from(i as u32), None, |e| *e.weight());
            (0..n)
                .map(|j| {
                    if i == j {
                        return None;
                    }
                    sp.get(&NodeIndex::from(j as u32))
                        .copied()
                        .map(|t| NotNan::new(t).unwrap())
                })
                .collect()
        })
        .collect();
    (filtration, edge_set)
}

fn assert_same_diagram<T: Eq + std::hash::Hash + std::fmt::Debug>(a: &Diagram<T>, b: &Diagram<T>) {
    assert_eq!(a.essential, b.essential, "essential classes differ");
    assert_eq!(a.pairings, b.pairings, "pairings differ");
}

/// The storage backends to check: the locking one always, the lock-free one only with the `lock-free` feature
fn storages() -> Vec<Storage> {
    #[allow(unused_mut)]
    let mut storages = vec![Storage::Locking];
    #[cfg(feature = "lock-free")]
    storages.push(Storage::LockFree);
    storages
}

fn check(n: u16, p: f64, seed: u64, storage: Storage) {
    let (filtration, edge_set) = random_digraph(n, p, seed);
    let d = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n);
    check_matrix((&d).reverse(), storage);
    // With the transpose attached, apparent pairs are skipped by the reduction (and V refers to columns outside it)
    let boundary = GrPPHBoundary::<Z2, _>::new(&filtration, &edge_set);
    let with_t = (&d).with_transpose(&boundary);
    check_matrix((&with_t).reverse(), storage);
}

fn check_matrix<M>(d_rev: M, storage: Storage)
where
    M: MatrixOracle<CoefficientField = Z2, RowT = <M as MatrixOracle>::ColT>
        + HasRowFiltration
        + HasSplitColBasis
        + Copy
        + Sync
        + Send,
    M::ColT: std::hash::Hash + std::fmt::Debug + Send + Sync + Copy + 'static,
    M::FiltrationT: Send + Sync,
{
    // Keep every pair (including zero-persistence apparent ones), since we compare against the full pairing of R = D V
    let serial = Clearing::new()
        .skip_zero_apparent_pairs(false)
        .reduce(d_rev)
        .diagram;
    let reduction = Clearing::new()
        .skip_zero_apparent_pairs(false)
        .parallel()
        .storage(storage)
        .reduce(d_rev);
    let (v_par, parallel) = (reduction.v, reduction.diagram);
    assert_same_diagram(&serial, &parallel);

    // Independently check that V really reduces D: recompute R = D V and read the pairing off its pivots
    let v_trivial = (&v_par).with_trivial_filtration();
    let r = product(d_rev, &v_trivial);
    let mut pivot_of_row: FxHashMap<_, _> = FxHashMap::default();
    let mut essential = FxHashSet::default();
    for dim in 0..=1 {
        let basis = d_rev.sub_basis(dim);
        for i in 0..basis.size() {
            let col = basis.element(i);
            match r.build_bhcol(col).pop_pivot() {
                Some(entry) => {
                    let previous = pivot_of_row.insert(entry.row_index, col);
                    assert!(
                        previous.is_none(),
                        "two columns of R share a pivot, so R is not reduced"
                    );
                }
                None => {
                    essential.insert(col);
                }
            }
        }
    }
    // Columns paired as a pivot row are cleared and so have zero R column, they are not essential
    for row in pivot_of_row.keys() {
        essential.remove(row);
    }
    let pairings: FxHashSet<_> = pivot_of_row.into_iter().collect();
    assert_eq!(
        pairings, parallel.pairings,
        "pairings read from R = D V differ"
    );
    assert_eq!(
        essential, parallel.essential,
        "essential classes read from R = D V differ"
    );
}

#[test]
fn parallel_agrees_with_serial_on_random_digraphs() {
    let mut checked = 0;
    for &n in &[4u16, 7, 10, 14, 18] {
        for &p in &[0.15, 0.3, 0.6, 1.0] {
            for seed in 0..6u64 {
                for storage in storages() {
                    check(n, p, seed * 1000 + n as u64, storage);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 120 * storages().len());
}

/// Hammer the same input repeatedly so that different thread interleavings are exercised
#[test]
fn parallel_is_stable_across_runs() {
    let (filtration, edge_set) = random_digraph(24, 0.35, 7);
    let d = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, 24);
    let d_rev = (&d).reverse();
    let serial = Clearing::new().reduce(&d_rev).diagram;
    for _ in 0..50 {
        for storage in storages() {
            let parallel = Clearing::new()
                .parallel()
                .storage(storage)
                .reduce(&d_rev)
                .diagram;
            assert_same_diagram(&serial, &parallel);
        }
    }
}
