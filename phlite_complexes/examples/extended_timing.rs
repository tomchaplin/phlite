//! Time `extended_persistence` on the lower-star filtrations of random graphs and their flag complexes (the cases of
//! python-tests/benchmarks/bench_extended_persistence.py), by the complexes' default algorithm (pivot columns,
//! storing R) and with merged columns.
//!
//! `cargo run --release --example extended_timing -p phlite_complexes -- [repeats]`

use std::collections::HashMap;
use std::time::Instant;

use phlite::{
    barcode::{Algorithm, Method},
    columns::{MergedColumns, PivotColumns},
    fields::Z2,
    reduction::Clearing,
};
use phlite_complexes::complex::{Chain, Complex, RawChain};
use phlite_complexes::extended::extended_persistence;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// G(n, p) (and its triangles if `flag`), filtered by the lower-star filtration of random vertex values
fn complex(n: usize, p: f64, flag: bool) -> Complex<Z2> {
    let mut rng = Lcg(7);
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let mut simplices: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
    let mut neighbours = vec![vec![]; n];
    for u in 0..n {
        for v in u + 1..n {
            if rng.next_f64() < p {
                simplices.push(vec![u, v]);
                neighbours[u].push(v);
                neighbours[v].push(u);
            }
        }
    }
    if flag {
        let edges: Vec<(usize, usize)> = simplices.iter().filter(|s| s.len() == 2).map(|s| (s[0], s[1])).collect();
        for (u, v) in edges {
            for &w in &neighbours[u] {
                if w > v && neighbours[v].contains(&w) {
                    simplices.push(vec![u, v, w]);
                }
            }
        }
    }
    let index: HashMap<Vec<usize>, i64> = simplices.iter().enumerate().map(|(i, s)| (s.clone(), i as i64)).collect();
    let mut raw = RawChain { dimensions: vec![], indptr: vec![0], indices: vec![], coefficients: None };
    for s in &simplices {
        raw.dimensions.push(s.len() as i64 - 1);
        if s.len() > 1 {
            for i in 0..s.len() {
                let face: Vec<usize> = s.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v).collect();
                raw.indices.push(index[&face]);
            }
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    let filtration: Vec<f64> = simplices.iter().map(|s| s.iter().map(|&v| values[v]).fold(f64::MIN, f64::max)).collect();
    Chain::<Z2>::from_raw(raw).unwrap().filter(Some(filtration)).unwrap()
}

fn main() {
    let repeats: usize = std::env::args().nth(1).map_or(3, |a| a.parse().unwrap());
    for (n, p, flag) in [(2000, 0.002, false), (20000, 0.0002, false), (300, 0.05, true), (1000, 0.02, true)] {
        let complex = complex(n, p, flag);
        let name = format!("{} n={n} p={p} ({} cells)", if flag { "flag" } else { "graph" }, complex.n_cells());
        let mut expected = None;
        let mut time = |label: &str, run: &dyn Fn() -> usize| {
            let mut best = f64::INFINITY;
            for _ in 0..repeats {
                let start = Instant::now();
                let bars = run();
                best = best.min(start.elapsed().as_secs_f64());
                assert_eq!(*expected.get_or_insert(bars), bars, "{name} {label}");
            }
            println!("{name:42} {label:28} {best:8.4}s");
        };
        for method in [Method::Cohomology, Method::Homology] {
            time(&format!("{method:?} pivot, store R"), &|| {
                let clearing = Clearing::new().with_columns(PivotColumns).store_r(true);
                extended_persistence(&complex, method, Algorithm::Clearing(clearing)).len()
            });
            time(&format!("{method:?} merged"), &|| {
                let clearing = Clearing::new().with_columns(MergedColumns);
                extended_persistence(&complex, method, Algorithm::Clearing(clearing)).len()
            });
        }
    }
}
