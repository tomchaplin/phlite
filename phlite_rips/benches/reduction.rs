//! Criterion benchmarks of the clearing reduction on the Rips coboundary matrix of random points in the unit cube.
//!
//! Run with `cargo bench -p phlite_rips --features lock-free` (or `just bench-rust` at the root). Filter by name,
//! e.g. `cargo bench -p phlite_rips -- 'n=160/parallel'`, and compare with a saved baseline using criterion's
//! `--save-baseline NAME` and `--baseline NAME`.
//!
//! Each benchmark reduces the same matrix (built once, untimed) with: the serial algorithm with merged and heap
//! columns, and the parallel algorithm with locking (and, with the `lock-free` feature, lock-free) storage on
//! 1, 2 and 4 threads.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ordered_float::NotNan;
use phlite::{
    columns::{HeapColumns, SortedColumns},
    fields::Z2,
    matrices::MatrixOracle,
    reduction::{Clearing, ReductionAlgorithm, Storage},
};
use phlite_rips::cohomology::build_rips_coboundary_matrix;
use std::hint::black_box;

/// (number of points, maximum dimension)
const CASES: [(usize, usize); 2] = [(100, 2), (160, 2)];
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

/// The distance matrix of `n` points uniformly in the unit cube.
fn distances(n: usize, seed: u64) -> Vec<Vec<NotNan<f64>>> {
    let mut rng = Lcg(seed);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()])
        .collect();
    points
        .iter()
        .map(|a| {
            points
                .iter()
                .map(|b| {
                    let d2: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum();
                    NotNan::new(d2.sqrt()).unwrap()
                })
                .collect()
        })
        .collect()
}

fn storages() -> Vec<(&'static str, Storage)> {
    #[allow(unused_mut)]
    let mut storages = vec![("locking", Storage::Locking)];
    #[cfg(feature = "lock-free")]
    storages.push(("lock-free", Storage::LockFree));
    storages
}

fn reduction(c: &mut Criterion) {
    for (n, max_dim) in CASES {
        let coboundary = build_rips_coboundary_matrix::<Z2>(distances(n, 5), max_dim).reverse();
        let mut group = c.benchmark_group(format!("rips n={n} dim≤{max_dim}"));
        group.sample_size(10);

        // Storing the reduced columns of R (merged columns, cached)
        group.bench_function("serial merged store-r", |b| {
            b.iter(|| black_box(Clearing::new().store_r(true).reduce(&coboundary)))
        });
        // Every column representation, with and without caching columns of the matrix
        for cache in [true, false] {
            let suffix = if cache { "cached" } else { "uncached" };
            group.bench_function(format!("serial merged {suffix}"), |b| {
                b.iter(|| black_box(Clearing::new().cache_columns(cache).reduce(&coboundary)))
            });
            group.bench_function(format!("serial heap {suffix}"), |b| {
                b.iter(|| {
                    black_box(
                        Clearing::new()
                            .with_columns(HeapColumns)
                            .cache_columns(cache)
                            .reduce(&coboundary),
                    )
                })
            });
            group.bench_function(format!("serial sorted {suffix}"), |b| {
                b.iter(|| {
                    black_box(
                        Clearing::new()
                            .with_columns(SortedColumns)
                            .cache_columns(cache)
                            .reduce(&coboundary),
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
                                    .reduce(&coboundary),
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
