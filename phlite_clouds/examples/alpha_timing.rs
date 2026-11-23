//! Time the reduction of alpha complexes of random points, with and without union-find (in dimension 0 and the top
//! dimension; see `phlite_complexes::complex::Complex::barcode_by_rank_with`).
//!
//! ```text
//! cargo run --release -p phlite_clouds --example alpha_timing -- [n] [dimension] [repetitions]
//! ```

use std::time::Instant;

use phlite::barcode::{Algorithm, Method};
use phlite::columns::PivotColumns;
use phlite::reduction::Clearing;
use phlite_clouds::alpha::alpha_complex;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(100_000, |a| a.parse().expect("n"));
    let dim: usize = args.get(2).map_or(3, |a| a.parse().expect("dimension"));
    let repetitions: usize = args.get(3).map_or(5, |a| a.parse().expect("repetitions"));
    // Uniform in the unit cube, from a fixed LCG
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut uniform = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let points: Vec<Vec<f64>> = (0..n).map(|_| (0..dim).map(|_| uniform()).collect()).collect();
    let start = Instant::now();
    let triangulation = phlite_clouds::alpha::delaunay(&points).expect("a triangulation");
    println!(
        "delaunay {:8.1} ms ({} top simplices)",
        start.elapsed().as_secs_f64() * 1e3,
        triangulation.simplices.len()
    );
    let start = Instant::now();
    let complex = alpha_complex(&points).expect("an alpha complex");
    let filtered = complex.filtered(true);
    println!(
        "complex {:8.1} ms ({} simplices)",
        start.elapsed().as_secs_f64() * 1e3,
        complex.n_simplices()
    );
    for union_find in [false, true] {
        let mut best = f64::INFINITY;
        let mut bars = 0;
        for _ in 0..repetitions {
            let algorithm = Algorithm::Clearing(Clearing::new().with_columns(PivotColumns).cache_columns(false).store_r(true));
            let start = Instant::now();
            bars = filtered.barcode_by_rank_with(None, Method::Cohomology, algorithm, false, union_find).len();
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!("union-find {union_find:5}: {:8.1} ms (best of {repetitions}; {bars} bars)", best * 1e3);
    }
}
