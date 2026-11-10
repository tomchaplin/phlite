//! Classify the columns of the (anti-transposed) Rips coboundary matrix by how much work the clearing algorithm needs for them,
//! to see how much the apparent pairs and emergent pairs (zero-addition columns) optimisations could save.
//!
//! Usage: `rips_stats <n_points> <max_dim> <seed>` (points uniform in the unit cube)

use std::cmp::Reverse;

use ordered_float::NotNan;
use phlite::{
    fields::Z2,
    matrices::{ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle},
    reduction::{Clearing, ReductionAlgorithm},
};
use phlite_rips::{cohomology::build_rips_coboundary_matrix, homology::RipsBoundary};
use rustc_hash::FxHashMap;

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
    let n: usize = args[1].parse().unwrap();
    let max_dim: usize = args[2].parse().unwrap();
    let seed: u64 = args[3].parse().unwrap();

    let mut rng = Lcg(seed);
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()])
        .collect();
    let dist: Vec<Vec<NotNan<f64>>> = pts
        .iter()
        .map(|a| {
            pts.iter()
                .map(|b| {
                    let d2: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum();
                    NotNan::new(d2.sqrt()).unwrap()
                })
                .collect()
        })
        .collect();

    let boundary = RipsBoundary::<Z2>::new(dist.clone());
    let coboundary = build_rips_coboundary_matrix::<Z2>(dist, max_dim);
    let anti = (&coboundary).reverse();
    let diagram = Clearing::new().reduce(&anti).diagram;
    // Final pivot of each paired column: (death row, birth column) in the anti-transpose
    let final_pivot: FxHashMap<_, _> = diagram
        .pairings
        .iter()
        .map(|(row, col)| (*col, *row))
        .collect();
    let cleared: rustc_hash::FxHashSet<_> = diagram.pairings.iter().map(|(row, _)| *row).collect();

    println!("n={n} max_dim={max_dim}");
    println!(
        "{:>3} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>12}",
        "dim", "columns", "cleared", "apparent", "zero-add", "reduced", "zero col", "cofacets"
    );
    for dim in 0..=max_dim {
        let basis = anti.sub_basis(dim);
        let (mut n_cleared, mut apparent, mut zero_add, mut reduced, mut zero_col, mut cofacets) =
            (0, 0, 0, 0, 0, 0usize);
        for i in 0..basis.size() {
            let col = basis.element(i);
            if cleared.contains(&col) {
                n_cleared += 1;
                continue;
            }
            let mut heap = anti.build_bhcol(col);
            cofacets += anti.column(col).count();
            let Some(initial) = heap.pop_pivot().map(|e| e.row_index) else {
                zero_col += 1;
                continue;
            };
            if final_pivot.get(&col) == Some(&initial) {
                // No additions needed; is it also an apparent pair?
                // The youngest facet of tau is the max of its boundary in (filtration, index) order
                let Reverse(tau) = initial;
                let youngest = boundary
                    .column(tau)
                    .map(|(_c, facet)| (boundary.filtration_value(facet), facet))
                    .max()
                    .map(|(_f, facet)| facet);
                if youngest == Some(col.0) {
                    apparent += 1;
                } else {
                    zero_add += 1;
                }
            } else {
                reduced += 1;
            }
        }
        println!(
            "{dim:>3} {:>9} {n_cleared:>9} {apparent:>9} {zero_add:>9} {reduced:>9} {zero_col:>9} {cofacets:>12}",
            basis.size()
        );
    }
}
