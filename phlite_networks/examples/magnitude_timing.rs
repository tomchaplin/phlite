//! Time magnitude homology (`phlite_networks::magnitude`) of a graph read from a file: the number of vertices on the
//! first line, then one edge `a b` per line (both directions are added unless `directed`).
//!
//! ```text
//! cargo run --release -p phlite_networks --example magnitude_timing -- graph.edges l_max [pair S T | all] [directed] [moduli 2,251]
//! ```
//!
//! Prints the time and the non-zero ranks over each modulus (summed over the pairs).

use std::time::Instant;

use phlite_networks::magnitude::{magnitude_ranks, total_ranks, MagnitudeGraph, MagnitudeQuery};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).expect("can't read the graph");
    let l_max: usize = args[2].parse().expect("l_max");
    let mut lines = text.lines();
    let n: usize = lines.next().expect("n").trim().parse().expect("n");
    let directed = args.iter().any(|a| a == "directed");
    let mut edges: Vec<(u32, u32)> = vec![];
    for line in lines {
        let v: Vec<u32> = line.split_whitespace().map(|x| x.parse().expect("vertex")).collect();
        if v.len() == 2 {
            edges.push((v[0], v[1]));
            if !directed {
                edges.push((v[1], v[0]));
            }
        }
    }
    let pairs = args.iter().position(|a| a == "pair").map(|i| {
        vec![(args[i + 1].parse::<u32>().expect("s"), args[i + 2].parse::<u32>().expect("t"))]
    });
    let moduli: Vec<u64> = args
        .iter()
        .position(|a| a == "moduli")
        .map_or(vec![2], |i| args[i + 1].split(',').map(|m| m.parse().expect("modulus")).collect());
    let start = Instant::now();
    let graph = MagnitudeGraph::new(n, &edges);
    let query = MagnitudeQuery { l_max, k_max: None, pairs };
    let results = magnitude_ranks(&graph, &query, &moduli, true);
    let elapsed = start.elapsed().as_secs_f64();
    println!("{} vertices, {} edges, l <= {l_max}: {elapsed:.3} s", n, edges.len());
    for (m, modulus) in moduli.iter().enumerate() {
        let total = total_ranks(&results, m, l_max, None);
        let nonzero: Vec<String> = total
            .iter()
            .enumerate()
            .flat_map(|(l, row)| row.iter().enumerate().filter(|(_, &r)| r > 0).map(move |(k, r)| format!("({k},{l}):{r}")))
            .collect();
        println!("Z/{modulus}: {}", nonzero.join(" "));
    }
}
