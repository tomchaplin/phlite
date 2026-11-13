//! Criterion benchmarks of the clearing reduction on the (anti-transposed) GrPPH coboundary matrix of random digraphs.
//!
//! Run with `cargo bench -p phlite_grpph` (or `just bench-rust` at the root). Filter by name, e.g.
//! `cargo bench -p phlite_grpph -- 'n=100/parallel'`, and compare with a saved baseline using criterion's
//! `--save-baseline NAME` and `--baseline NAME`.
//!
//! Each benchmark reduces the same matrix (built once, untimed) with: the serial algorithm with merged and heap
//! columns, and the parallel algorithm with locking (and, with the `lock-free` feature, lock-free) storage on
//! 1, 2 and 4 threads.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phlite::{
    columns::{HeapColumns, SortedColumns},
    fields::Z2,
    matrices::MatrixOracle,
    reduction::{Clearing, ReductionAlgorithm, Storage},
};
use phlite_grpph::{build_edge_set, build_filtration, coboundary::GrPPHCoboundary};
use std::hint::black_box;

/// (number of nodes, edge probability)
const CASES: [(u32, f64); 2] = [(50, 0.1), (100, 0.1)];
const THREADS: [usize; 3] = [1, 2, 4];

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

/// A random digraph on `n` nodes, each edge present with probability `p`, with integer weights 1 to 20.
fn random_edges(n: u32, p: f64, seed: u64) -> Vec<(u32, u32, f64)> {
    let mut rng = Lcg(seed);
    let mut edges = vec![];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                edges.push((i, j, 1.0 + (rng.next_f64() * 20.0).floor()));
            }
        }
    }
    edges
}

fn storages() -> Vec<(&'static str, Storage)> {
    #[allow(unused_mut)]
    let mut storages = vec![("locking", Storage::Locking)];
    #[cfg(feature = "lock-free")]
    storages.push(("lock-free", Storage::LockFree));
    storages
}

fn reduction(c: &mut Criterion) {
    for (n, p) in CASES {
        let edges = random_edges(n, p, 5);
        let filtration = build_filtration(n, &edges);
        let edge_set = build_edge_set(&edges);
        let coboundary = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n as u16);
        let matrix = (&coboundary).reverse();
        let mut group = c.benchmark_group(format!("grpph n={n} p={p}"));
        group.sample_size(10);

        // Storing the reduced columns of R (merged columns, cached)
        group.bench_function("serial merged store-r", |b| {
            b.iter(|| black_box(Clearing::new().store_r(true).reduce(&matrix)))
        });
        // Every column representation, with and without caching columns of the matrix
        for cache in [true, false] {
            let suffix = if cache { "cached" } else { "uncached" };
            group.bench_function(format!("serial merged {suffix}"), |b| {
                b.iter(|| black_box(Clearing::new().cache_columns(cache).reduce(&matrix)))
            });
            group.bench_function(format!("serial heap {suffix}"), |b| {
                b.iter(|| {
                    black_box(
                        Clearing::new()
                            .with_columns(HeapColumns)
                            .cache_columns(cache)
                            .reduce(&matrix),
                    )
                })
            });
            group.bench_function(format!("serial sorted {suffix}"), |b| {
                b.iter(|| {
                    black_box(
                        Clearing::new()
                            .with_columns(SortedColumns)
                            .cache_columns(cache)
                            .reduce(&matrix),
                    )
                })
            });
        }
        for (name, storage) in storages() {
            for threads in THREADS {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                group.bench_with_input(
                    BenchmarkId::new(format!("parallel {name}"), threads),
                    &threads,
                    |b, _| {
                        b.iter(|| {
                            black_box(
                                Clearing::new()
                                    .parallel()
                                    .in_pool(&pool)
                                    .storage(storage)
                                    .reduce(&matrix),
                            )
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, reduction);
criterion_main!(benches);
