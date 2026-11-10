//! The GrPPH boundary oracle must be exactly the transpose of the coboundary oracle (over Z2),
//! since we use it to find apparent pairs.

use phlite::{
    fields::Z2,
    matrices::{ColBasis, HasSplitColBasis, MatrixOracle},
};
use phlite_grpph::{
    boundary::GrPPHBoundary, build_edge_set, build_filtration, coboundary::GrPPHCoboundary,
};
use rustc_hash::{FxHashMap, FxHashSet};

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

/// Entries of a column over Z2: the rows appearing an odd number of times
fn z2_support<T: std::hash::Hash + Eq>(entries: impl Iterator<Item = T>) -> FxHashSet<T> {
    let mut counts: FxHashMap<T, usize> = FxHashMap::default();
    for e in entries {
        *counts.entry(e).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, c)| c % 2 == 1)
        .map(|(e, _)| e)
        .collect()
}

#[test]
fn boundary_is_transpose_of_coboundary() {
    let mut checked_cells = 0;
    for &n in &[3u32, 5, 7, 9] {
        for &p in &[0.2, 0.4, 0.7, 1.0] {
            for seed in 0..5u64 {
                let mut rng = Lcg(seed * 7919 + n as u64);
                let mut edges = vec![];
                for i in 0..n {
                    for j in 0..n {
                        if i != j && rng.next_f64() < p {
                            let w = if rng.next_f64() < 0.1 {
                                f64::INFINITY
                            } else {
                                1.0 + (rng.next_f64() * 3.0).floor()
                            };
                            edges.push((i, j, w));
                        }
                    }
                }
                let filtration = build_filtration(n, &edges);
                let edge_set = build_edge_set(&edges);
                let coboundary = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n as u16);
                let boundary = GrPPHBoundary::<Z2, _>::new(&filtration, &edge_set);

                // Every entry (row tau, column sigma) of the coboundary, over Z2
                let mut transpose: FxHashMap<_, FxHashSet<_>> = FxHashMap::default();
                let mut columns = FxHashSet::default();
                for dim in 0..=1 {
                    let basis = coboundary.sub_basis(dim);
                    for i in 0..basis.size() {
                        let sigma = basis.element(i);
                        columns.insert(sigma);
                        for tau in z2_support(coboundary.column(sigma).map(|(_c, row)| row)) {
                            transpose.entry(tau).or_default().insert(sigma);
                        }
                    }
                }
                // Row tau of the coboundary must be exactly the boundary of tau (restricted to cells in the complex)
                for (tau, from_coboundary) in &transpose {
                    let from_boundary = z2_support(boundary.column(*tau).map(|(_c, row)| row));
                    // The boundary never mentions cells outside the complex, so the first/last column in a row is a real column
                    assert!(
                        from_boundary.iter().all(|cell| columns.contains(cell)),
                        "cell {tau:?}"
                    );
                    assert_eq!(
                        &from_boundary, from_coboundary,
                        "n={n} p={p} seed={seed} cell {tau:?}"
                    );
                    checked_cells += 1;
                }
            }
        }
    }
    assert!(checked_cells > 1000, "only checked {checked_cells} cells");
}
