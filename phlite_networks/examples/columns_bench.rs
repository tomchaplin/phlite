//! Time the serial clearing reduction with heap columns, flagser-style dynamic heap columns (at several thresholds)
//! and merged columns, with and without caching columns of `D`: on directed flag complexes (as flagser's
//! benchmarks), Rips cohomology and GrPPH.
//!
//! `cargo run --release --example columns_bench -p phlite_networks -- [repeats] [case filter] [cached only]`
//!
//! Prints CSV: `case,columns,cache,seconds` (the minimum over the repeats), checking every barcode against the first.

use std::time::Instant;

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::{ColumnRepresentation, DynamicHeapColumns, HeapColumns, MergedColumns},
    fields::Z2,
    matrices::MatrixOracle,
    reduction::{Clearing, ReductionAlgorithm},
};
use phlite_clouds::rips::cohomology::build_rips_coboundary_matrix;
use phlite_networks::{
    digraph::Digraph,
    flag::{FlagComplex, Kind},
    grpph::{build_edge_set, build_filtration, coboundary::GrPPHCoboundary},
    lazy::barcode,
};

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

fn digraph(n: u32, p: f64, seed: u64, integer: bool) -> Vec<(u32, u32, f64)> {
    let mut rng = Lcg(seed);
    let mut edges = vec![];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                let w = rng.next_f64();
                edges.push((i, j, if integer { 1.0 + (w * 20.0).floor() } else { w }));
            }
        }
    }
    edges
}

fn cube(n: usize, seed: u64) -> Vec<Vec<NotNan<f64>>> {
    let mut rng = Lcg(seed);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()])
        .collect();
    points
        .iter()
        .map(|a| {
            points
                .iter()
                .map(|b| NotNan::new((0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()).unwrap())
                .collect()
        })
        .collect()
}

/// Every representation to compare, by name
fn each<F: FnMut(&str, &dyn Fn(bool) -> u64)>(mut f: F, run: &dyn Fn(Rep, bool) -> u64) {
    // `BENCH_COLUMNS=heap|merged|...` runs only those representations, by name
    let only = std::env::var("BENCH_COLUMNS").ok();
    let mut f = |name: &str, go: &dyn Fn(bool) -> u64| {
        if only.as_ref().is_none_or(|only| only.split('|').any(|o| o == name)) {
            f(name, go)
        }
    };
    f("heap", &|cache| run(Rep::Heap, cache));
    for threshold in [256, 4096] {
        for repeats in [0.0, 1.5, 2.0, 4.0] {
            f(&format!("dynamic {threshold} x{repeats}"), &move |cache| run(Rep::Dynamic(threshold, repeats), cache));
        }
    }
    f("merged", &|cache| run(Rep::Merged, cache));
}

#[derive(Clone, Copy)]
enum Rep {
    Heap,
    Dynamic(usize, f64),
    Merged,
}

/// Run `body` with the representation `rep` (monomorphised for each)
macro_rules! with_rep {
    ($rep:expr, |$r:ident| $body:expr) => {
        match $rep {
            Rep::Heap => {
                let $r = HeapColumns;
                $body
            }
            Rep::Dynamic(threshold, repeats) => {
                let $r = DynamicHeapColumns { threshold, repeats };
                $body
            }
            Rep::Merged => {
                let $r = MergedColumns;
                $body
            }
        }
    };
}

fn time(case: &str, repeats: usize, run: &dyn Fn(Rep, bool) -> u64) {
    let filter = std::env::args().nth(2).unwrap_or_default();
    if !case.contains(&filter) {
        return;
    }
    let caches: &[bool] = if std::env::args().nth(3).is_some() { &[true] } else { &[true, false] };
    let mut expected = None;
    each(
        |name, go| {
            for &cache in caches {
                let mut best = f64::INFINITY;
                for _ in 0..repeats {
                    let start = Instant::now();
                    let fingerprint = go(cache);
                    best = best.min(start.elapsed().as_secs_f64());
                    assert_eq!(*expected.get_or_insert(fingerprint), fingerprint, "{case} {name}");
                }
                println!("{case},{name},{cache},{best:.5}");
            }
        },
        run,
    );
}

fn clearing<R: ColumnRepresentation>(r: R, cache: bool) -> Clearing<R> {
    Clearing::new().with_columns(r).cache_columns(cache)
}

fn main() {
    let repeats: usize = std::env::args().nth(1).map_or(3, |a| a.parse().unwrap());
    println!("case,columns,cache,seconds");
    for (n, p, max_dim) in [(100, 0.2, 2), (200, 0.1, 2), (60, 0.3, 3), (500, 0.05, 2)] {
        let complex = FlagComplex::new(Kind::DirectedFlag, Digraph::new(n, digraph(n, p, 7, false)).unwrap());
        time(&format!("directed flag n={n} p={p} dim<={max_dim}"), repeats, &|rep, cache| {
            with_rep!(rep, |r| barcode(
                &complex,
                max_dim,
                Method::Cohomology,
                Algorithm::Clearing(clearing(r, cache)),
                false
            )
            .len() as u64)
        });
    }
    for (n, max_dim) in [(160, 2), (300, 2), (1000, 1)] {
        let rips = build_rips_coboundary_matrix::<Z2>(cube(n, 5), max_dim).reverse();
        time(&format!("rips cube n={n} dim<={max_dim}"), repeats, &|rep, cache| {
            with_rep!(rep, |r| clearing(r, cache).reduce(&rips).diagram.pairings.len() as u64)
        });
    }
    let n = 100;
    let edges = digraph(n, 0.1, 5, true);
    let filtration = build_filtration(n, &edges).unwrap();
    let edge_set = build_edge_set(&edges);
    let coboundary = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n as u16);
    let matrix = coboundary.by_ref().reverse();
    time("grpph n=100 p=0.1", repeats, &|rep, cache| {
        with_rep!(rep, |r| clearing(r, cache).reduce(&matrix).diagram.pairings.len() as u64)
    });
}
