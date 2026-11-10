//! Classify the columns of the (anti-transposed) GrPPH coboundary matrix by how much work the clearing algorithm needs for them,
//! to see where the time goes.
//!
//! Usage: `grpph_stats <n_vertices> <edge_probability> <seed>` (random digraph with integer weights 1..=5)

use phlite::{
    fields::Z2,
    matrices::{ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle},
    reduction::{Clearing, ReductionAlgorithm},
};
use phlite_grpph::{
    boundary::GrPPHBoundary, build_edge_set, build_filtration, coboundary::GrPPHCoboundary,
};
use rustc_hash::{FxHashMap, FxHashSet};

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args[1].parse().unwrap();
    let p: f64 = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();

    let mut rng = Lcg(seed);
    let mut edges = vec![];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                edges.push((i, j, 1.0 + (rng.next_f64() * 5.0).floor()));
            }
        }
    }
    let filtration = build_filtration(n, &edges);
    let edge_set = build_edge_set(&edges);
    let coboundary = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n as u16);
    let boundary = GrPPHBoundary::<Z2, _>::new(&filtration, &edge_set);
    let with_t = (&coboundary).with_transpose(&boundary);
    let anti = (&with_t).reverse();

    let start = std::time::Instant::now();
    let diagram = Clearing::new().reduce(&anti).diagram;
    let reduce_time = start.elapsed();
    let final_pivot: FxHashMap<_, _> = diagram
        .pairings
        .iter()
        .map(|(row, col)| (*col, *row))
        .collect();
    let cleared: FxHashSet<_> = diagram.pairings.iter().map(|(row, _)| *row).collect();

    println!(
        "n={n} p={p} edges={} reduce={:.3}s",
        edges.len(),
        reduce_time.as_secs_f64()
    );
    println!(
        "{:>3} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>12}",
        "dim", "columns", "cleared", "apparent", "zero-add", "reduced", "zero col", "cofacets"
    );
    for dim in 0..=1 {
        let basis = anti.sub_basis(dim);
        let (mut n_cleared, mut apparent, mut zero_add, mut reduced, mut zero_col, mut cofacets) =
            (0, 0, 0, 0, 0, 0usize);
        let start = std::time::Instant::now();
        for i in 0..basis.size() {
            let col = basis.element(i);
            if cleared.contains(&col) {
                n_cleared += 1;
                continue;
            }
            cofacets += anti.column(col).count();
            let Some(initial) = anti.column_pivot(col).map(|e| e.row_index) else {
                zero_col += 1;
                continue;
            };
            if final_pivot.get(&col) == Some(&initial) {
                if anti.first_column_in_row(initial) == Some(col) {
                    apparent += 1;
                } else {
                    zero_add += 1;
                }
            } else {
                reduced += 1;
            }
        }
        println!(
            "{dim:>3} {:>9} {n_cleared:>9} {apparent:>9} {zero_add:>9} {reduced:>9} {zero_col:>9} {cofacets:>12}   (classify {:.3}s)",
            basis.size(),
            start.elapsed().as_secs_f64()
        );
    }
}
