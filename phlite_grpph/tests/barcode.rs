//! Check that every method and algorithm give the same GrPPH barcode on random digraphs,
//! and that every representative is a cycle with the right birth.

use phlite::{
    barcode::Algorithm,
    reduction::{Clearing, Standard},
};
use phlite_grpph::{
    build_edge_set, build_filtration, coboundary::PathHomCell, grpph_barcode, GrPPHBar, Method,
};

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

/// Random digraph with integer edge weights, occasionally infinite
fn random_edges(n: u32, p: f64, seed: u64) -> Vec<(u32, u32, f64)> {
    let mut rng = Lcg(seed);
    let mut edges = vec![];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                let w = if rng.next_f64() < 0.05 {
                    f64::INFINITY
                } else {
                    1.0 + (rng.next_f64() * 4.0).floor()
                };
                edges.push((i, j, w));
            }
        }
    }
    edges
}

/// Non-zero-length bars as (dimension, birth cell, birth, death), sorted
fn summary(bars: &[GrPPHBar]) -> Vec<(usize, PathHomCell, f64, Option<f64>)> {
    let mut out: Vec<_> = bars
        .iter()
        .filter(|bar| !bar.is_zero_length())
        .map(|bar| {
            (
                bar.birth.dimension(),
                bar.birth,
                bar.birth_value.into_inner(),
                bar.death.map(|(_, d)| d.into_inner()),
            )
        })
        .collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

fn check_representatives(n: u32, bars: &[GrPPHBar]) {
    for bar in bars {
        let rep = bar
            .representative
            .as_ref()
            .expect("asked for representatives");
        assert!(!rep.is_empty(), "{bar:?} has empty representative");
        // Every sum of nodes is a 0-cycle, but a sum of edges is a 1-cycle (over Z2) iff every node meets an even number of them
        if bar.birth.dimension() == 1 {
            let mut degree = vec![0usize; n as usize];
            for (_, cell) in rep {
                let PathHomCell::Edge(a, b) = cell else {
                    panic!("H_1 representative of {bar:?} contains {cell:?}")
                };
                degree[*a as usize] += 1;
                degree[*b as usize] += 1;
            }
            assert!(
                degree.iter().all(|d| d.is_multiple_of(2)),
                "{bar:?} is not a cycle"
            );
        }
        // The birth cell is the leading term of every finite representative
        if bar.death.is_some() {
            assert_eq!(rep[0].1, bar.birth, "{bar:?} has the wrong pivot");
        }
    }
}

#[test]
fn methods_and_algorithms_agree_on_random_digraphs() {
    for &n in &[3u32, 6, 9, 12] {
        for &p in &[0.2, 0.4, 0.8] {
            for seed in 0..4u64 {
                let edges = random_edges(n, p, seed * 1000 + n as u64);
                let filtration = build_filtration(n, &edges);
                let edge_set = build_edge_set(&edges);
                let reference = summary(&grpph_barcode(
                    &filtration,
                    &edge_set,
                    n as u16,
                    Method::Cohomology,
                    Algorithm::default(),
                    false,
                ));
                for method in [Method::Cohomology, Method::Homology] {
                    for algorithm in [
                        Algorithm::Standard(Standard::new()),
                        Algorithm::Clearing(Clearing::new()),
                        Algorithm::ParallelClearing(Clearing::new().parallel()),
                    ] {
                        let bars = grpph_barcode(
                            &filtration,
                            &edge_set,
                            n as u16,
                            method,
                            algorithm.clone(),
                            true,
                        );
                        assert_eq!(
                            summary(&bars),
                            reference,
                            "n={n} p={p} seed={seed} {method:?} {algorithm:?}"
                        );
                        check_representatives(n, &bars);
                    }
                }
            }
        }
    }
}
