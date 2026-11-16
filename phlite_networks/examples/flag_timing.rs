//! Time the flag complex of a random graph G(n, p) with random weights: implicitly (Rips' machinery, as
//! `undirected::flag_barcode`) and explicitly (every clique listed, reduced by `phlite_complexes`).
//!
//! `cargo run --release --example flag_timing -p phlite_networks -- 300 0.1 2`

use std::collections::HashMap;
use std::time::Instant;

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::{HeapColumns, PivotColumns},
    fields::Z2,
    reduction::Clearing,
    values::INFINITY,
};
use phlite_complexes::complex::{Chain, RawChain};
use phlite_networks::undirected::{cliques_within, flag_barcode, Graph};
use phlite_clouds::rips::Distances;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map_or(300, |a| a.parse().unwrap());
    let p: f64 = args.get(2).map_or(0.1, |a| a.parse().unwrap());
    let max_dim: usize = args.get(3).map_or(2, |a| a.parse().unwrap());
    // A small deterministic generator
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut edges = vec![];
    for u in 0..n {
        for v in u + 1..n {
            if random() < p {
                edges.push((u, v, random()));
            }
        }
    }
    let graph = Graph::new(n, &edges, None).unwrap();
    println!("n = {n}, {} edges", graph.n_edges());

    let start = Instant::now();
    let bars = flag_barcode::<_, usize>(
        &graph,
        max_dim,
        INFINITY,
        Method::Cohomology,
        Algorithm::Clearing(Clearing::new().with_columns(HeapColumns)),
        false,
        true,
        false,
    );
    println!("implicit (Rips' machinery): {:>8.3}s, {} bars", start.elapsed().as_secs_f64(), bars.len());

    let start = Instant::now();
    let simplices: Vec<Vec<u32>> = cliques_within(&graph, max_dim + 1, INFINITY, usize::MAX)
        .expect("no budget")
        .into_iter()
        .map(|(_, simplex)| simplex)
        .collect();
    let value = |s: &[u32]| -> f64 {
        let mut v = 0.0f64;
        for i in 0..s.len() {
            for j in i + 1..s.len() {
                v = v.max(graph.distance(s[i] as usize, s[j] as usize).into_inner());
            }
        }
        v
    };
    let mut order: Vec<(NotNan<f64>, usize, &Vec<u32>)> =
        simplices.iter().map(|s| (NotNan::new(value(s)).unwrap(), s.len(), s)).collect();
    order.sort();
    let index: HashMap<&[u32], i64> = order.iter().enumerate().map(|(i, (_, _, s))| (s.as_slice(), i as i64)).collect();
    let mut raw = RawChain { dimensions: vec![], indptr: vec![0], indices: vec![], coefficients: None };
    for (_, len, s) in &order {
        raw.dimensions.push(*len as i64 - 1);
        if *len > 1 {
            for i in 0..*len {
                let face: Vec<u32> = s.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v).collect();
                raw.indices.push(index[face.as_slice()]);
            }
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    let values: Vec<f64> = order.iter().map(|(v, _, _)| v.into_inner()).collect();
    let built = start.elapsed().as_secs_f64();
    let complex = Chain::<Z2>::from_raw(raw).unwrap().filter(Some(values)).unwrap();
    let bars = complex.barcode(
        Some(max_dim),
        Method::Cohomology,
        Algorithm::Clearing(Clearing::new().with_columns(PivotColumns).store_r(true)),
        false,
    );
    println!(
        "explicit (phlite_complexes): {:>8.3}s ({built:.3}s building {} cells), {} bars",
        start.elapsed().as_secs_f64(),
        order.len(),
        bars.len()
    );
}
