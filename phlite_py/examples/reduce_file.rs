//! Reduce a complex dumped to a file, for profiling the reduction without Python (see `python-tests/benchmarks`).
//!
//! The file is little-endian: `n` and `nnz` (u64), then the dimension of every cell (`n` i64), the CSC column pointers
//! (`n + 1` i64), the faces (`nnz` i64) and the filtration values (`n` f64).
//!
//! ```text
//! cargo run --release -p phlite_py --example reduce_file -- complex.bin [repetitions] [homology]
//! ```
//!
//! It prints the time of each stage: building the chain, filtering it, and computing the barcode (the configuration
//! `phlite_py` uses by default: serial clearing with pivot columns, storing R, no cache).

use std::time::Instant;

use phlite::barcode::{Algorithm, Method};
use phlite::columns::PivotColumns;
use phlite::fields::Z2;
use phlite::reduction::Clearing;
use phlite_py::complex::{Chain, RawChain};

fn read(path: &str) -> (RawChain, Vec<f64>) {
    let bytes = std::fs::read(path).expect("can't read the file");
    let mut words = bytes
        .chunks_exact(8)
        .map(|w| <[u8; 8]>::try_from(w).unwrap());
    let mut u64s = |k: usize| -> Vec<[u8; 8]> { words.by_ref().take(k).collect() };
    let header = u64s(2);
    let n = u64::from_le_bytes(header[0]) as usize;
    let nnz = u64::from_le_bytes(header[1]) as usize;
    let ints = |w: Vec<[u8; 8]>| w.into_iter().map(i64::from_le_bytes).collect::<Vec<_>>();
    let dimensions = ints(u64s(n));
    let indptr = ints(u64s(n + 1));
    let indices = ints(u64s(nnz));
    let values = u64s(n).into_iter().map(f64::from_le_bytes).collect();
    let raw = RawChain {
        dimensions,
        indptr,
        indices,
        coefficients: None,
    };
    (raw, values)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: reduce_file FILE [repetitions] [homology]");
    let repetitions: usize = args.get(2).map_or(5, |r| r.parse().expect("repetitions"));
    let method = match args.get(3).map(String::as_str) {
        Some("homology") => Method::Homology,
        _ => Method::Cohomology,
    };
    let (raw, values) = read(path);

    let start = Instant::now();
    let chain = Chain::<Z2>::from_raw(raw).expect("a valid chain complex");
    println!("chain   {:8.1} ms", start.elapsed().as_secs_f64() * 1e3);
    let start = Instant::now();
    let complex = chain.filter(Some(values)).expect("a valid filtration");
    println!("filter  {:8.1} ms", start.elapsed().as_secs_f64() * 1e3);

    let mut best = f64::INFINITY;
    let mut bars = 0;
    for _ in 0..repetitions {
        let algorithm = Algorithm::Clearing(
            Clearing::new()
                .with_columns(PivotColumns)
                .cache_columns(false)
                .store_r(true),
        );
        let start = Instant::now();
        bars = complex.barcode(None, method, algorithm, false).len();
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!(
        "barcode {:8.1} ms (best of {repetitions}; {bars} bars)",
        best * 1e3
    );
    let algorithm = Algorithm::Clearing(
        Clearing::new()
            .with_columns(PivotColumns)
            .cache_columns(false)
            .store_r(true),
    );
    let by_rank = complex.barcode_by_rank(None, method, algorithm, false).len();
    println!("({by_rank} bars before dropping zero-length ones)");
}
