//! Time building the cohomology basis (enumerating paths) against the full barcode, on random networks.
//! Usage: cargo run --release --example enumeration -- n max_dim [threads] [density percent]

use std::time::Instant;

use phlite::{
    barcode::{Algorithm, Method},
    fields::Z2,
};
use phlite_networks::pph::{cohomology_basis, pph_barcode, Network, PPHBar};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (n, max_dim) = (args[0] as u16, args[1]);
    let threads = args.get(2).copied().unwrap_or(0);
    let density = args.get(3).copied().unwrap_or(100) as u64;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    // xorshift weights in 1..=20
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut edges = vec![];
    for a in 0..n {
        for b in 0..n {
            if a != b {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let weight = (1 + state % 20) as f64;
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state % 100 < density {
                    edges.push((a, b, weight));
                }
            }
        }
    }
    let network = Network::from_edges(n, &edges);
    let start = Instant::now();
    let basis = cohomology_basis(&network, max_dim);
    let enumeration = start.elapsed();
    let size = phlite::matrices::ColBasis::size(&basis);
    drop(basis);
    let start = Instant::now();
    let bars: Vec<PPHBar<Z2>> = pph_barcode(
        &network,
        max_dim,
        Method::Cohomology,
        Algorithm::default(),
        false,
    );
    let total = start.elapsed();
    println!(
        "n={n} max_dim={max_dim} threads={} basis={size} enumerate={:.3}s barcode={:.3}s bars={}",
        rayon::current_num_threads(),
        enumeration.as_secs_f64(),
        total.as_secs_f64(),
        bars.len()
    );
}
