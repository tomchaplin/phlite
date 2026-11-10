//! Benchmark serial vs parallel clearing cohomology on a random point cloud in the unit cube.
//!
//! Usage: `bench_rips_parallel <n_points> <max_dim> <seed> <serial|par:THREADS|lf:THREADS>`

use ordered_float::NotNan;
use phlite::{
    fields::Z2,
    matrices::MatrixOracle,
    reduction::{Clearing, ReductionAlgorithm, Storage},
};
use phlite_rips::cohomology::build_rips_coboundary_matrix;
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
    let n: usize = args[1].parse().unwrap();
    let max_dim: usize = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();
    let mode = args[4].as_str();

    let mut rng = Lcg(seed);
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()])
        .collect();
    let dist: Vec<Vec<NotNan<f64>>> = pts
        .iter()
        .map(|a| {
            pts.iter()
                .map(|b| {
                    let d2: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum();
                    NotNan::new(d2.sqrt()).unwrap()
                })
                .collect()
        })
        .collect();

    let setup = Instant::now();
    let coboundary = build_rips_coboundary_matrix::<Z2>(dist, max_dim).reverse();
    let setup = setup.elapsed();

    let start = Instant::now();
    let diagram = if mode == "serial" {
        Clearing::new().reduce(&coboundary).diagram
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
            .reduce(&coboundary)
            .diagram
    };
    println!(
        "{mode}: setup {:.3}s reduce {:.3}s essential={} pairings={}",
        setup.as_secs_f64(),
        start.elapsed().as_secs_f64(),
        diagram.essential.len(),
        diagram.pairings.len()
    );
}
