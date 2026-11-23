//! Time Rips persistence of a distance matrix with and without edge collapse (see `phlite_clouds::rips::collapse`).
//!
//! The file is little-endian: `n` (u64), then the `n × n` distances (f64), row by row.
//!
//! ```text
//! cargo run --release -p phlite_clouds --example rips_collapse -- matrix.dist max_dim [repetitions]
//! ```

use std::time::Instant;

use ordered_float::NotNan;
use phlite::barcode::{Algorithm, Method};
use phlite::columns::HeapColumns;
use phlite::reduction::Clearing;
use phlite::values::INFINITY;
use phlite_clouds::rips::{collapse::collapse_edges, enclosing_radius, rips_barcode_with};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1]).expect("can't read the file");
    let max_dim: usize = args[2].parse().expect("max_dim");
    let repetitions: usize = args.get(3).map_or(3, |r| r.parse().expect("repetitions"));
    let words: Vec<f64> = bytes[8..].chunks_exact(8).map(|w| f64::from_le_bytes(w.try_into().unwrap())).collect();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let distances: Vec<Vec<NotNan<f64>>> =
        (0..n).map(|i| (0..n).map(|j| NotNan::new(words[i * n + j]).unwrap()).collect()).collect();
    let start = Instant::now();
    let (_, stats) = collapse_edges(&distances, enclosing_radius(&distances));
    println!("collapse alone {:8.1} ms: {stats:?}", start.elapsed().as_secs_f64() * 1e3);
    for collapse in [false, true] {
        let mut best = f64::INFINITY;
        let mut bars = 0;
        for _ in 0..repetitions {
            // phlite.clouds.rips' defaults: serial clearing, heap columns, no cache
            let algorithm = Algorithm::Clearing(Clearing::new().with_columns(HeapColumns).cache_columns(false));
            let start = Instant::now();
            bars = rips_barcode_with::<_, usize, _>(
                &distances, max_dim, INFINITY, Method::Cohomology, algorithm, false, true, false, collapse,
            )
            .len();
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!("collapse {collapse:5}: {:9.1} ms (best of {repetitions}; {bars} bars)", best * 1e3);
    }
}
