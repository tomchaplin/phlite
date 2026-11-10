//! Benchmark the serial and parallel clearing reductions on a random digraph.
//!
//! Usage: `bench_grpph_parallel <n> <edge_probability> <seed> <serial|par:THREADS|lf:THREADS>`

use ordered_float::NotNan;
use petgraph::{algo::dijkstra, graph::NodeIndex, Graph};
use phlite::{
    fields::Z2,
    matrices::MatrixOracle,
    reduction::{Clearing, ReductionAlgorithm, Storage},
};
use phlite_grpph::coboundary::GrPPHCoboundary;
use rustc_hash::FxHashSet;
use std::time::Instant;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u16 = args[1].parse().unwrap();
    let p: f64 = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();
    let mode = args[4].as_str();

    let mut rng = Lcg(seed);
    let mut g = Graph::<(), f64>::new();
    for _ in 0..n {
        g.add_node(());
    }
    let mut edge_set = FxHashSet::default();
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                let w = 1.0 + (rng.next_f64() * 20.0).floor();
                g.add_edge(NodeIndex::from(i as u32), NodeIndex::from(j as u32), w);
                edge_set.insert((i, j));
            }
        }
    }
    let filtration: Vec<Vec<Option<NotNan<f64>>>> = (0..n)
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

    let d = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n);
    let d_rev = (&d).reverse();

    let start = Instant::now();
    let diagram = if mode == "serial" {
        Clearing::new().reduce(&d_rev).diagram
    } else {
        let (storage, threads) = if let Some(threads) = mode.strip_prefix("par:") {
            (Storage::Locking, threads)
        } else if let Some(threads) = mode.strip_prefix("lf:") {
            #[cfg(not(feature = "lock-free"))]
            panic!("mode `lf:N` needs `--features lock-free`, got {threads}");
            #[cfg(feature = "lock-free")]
            (Storage::LockFree, threads)
        } else {
            panic!("mode is serial, par:N or lf:N")
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.parse().unwrap())
            .build()
            .unwrap();
        Clearing::new()
            .parallel()
            .in_pool(&pool)
            .storage(storage)
            .reduce(&d_rev)
            .diagram
    };
    let elapsed = start.elapsed();
    println!(
        "{mode}: reduce {:.3}s  essential={} pairings={}",
        elapsed.as_secs_f64(),
        diagram.essential.len(),
        diagram.pairings.len()
    );
}
