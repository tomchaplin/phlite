//! Time the parallel reduction with every pivot table and every kind of slot for its shared state (see
//! `phlite::reduction::parallel::experiments`), on Rips and GrPPH coboundary matrices, at 1 to 8 threads.
//!
//! `cargo run --release --example storage_bench -p phlite_networks --features storage-experiments -- [repeats]`
//!
//! Prints CSV: `case,table,slots,threads,seconds` (the minimum over the repeats), checking every diagram against the
//! serial algorithm's.

use std::time::Instant;

use ordered_float::NotNan;
use phlite::{
    columns::{ColumnCoefficient, ColumnFiltration, ColumnRow},
    fields::{Invertible, Z2},
    matrices::{HasRowFiltration, HasSplitColBasis, MatrixOracle, SquareMatrix},
    reduction::{
        parallel::experiments::{reduce_with, PivotTableKind, SlotKind},
        Clearing, ReductionAlgorithm,
    },
};
use phlite_clouds::rips::{
    build_rips_bases,
    cohomology::{build_rips_coboundary_matrix, RipsCoboundary},
    homology::RipsBoundary,
};
use phlite_networks::grpph::{build_edge_set, build_filtration, coboundary::GrPPHCoboundary};

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

fn digraph(n: u32, p: f64, seed: u64) -> Vec<(u32, u32, f64)> {
    let mut rng = Lcg(seed);
    let mut edges = vec![];
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next_f64() < p {
                edges.push((i, j, 1.0 + (rng.next_f64() * 20.0).floor()));
            }
        }
    }
    edges
}

fn run<M>(case: &str, matrix: M, repeats: usize)
where
    M: MatrixOracle + SquareMatrix + HasRowFiltration + HasSplitColBasis + Sync + Send + Copy,
    M::ColT: Ord,
    M::CoefficientField: Invertible + ColumnCoefficient + 'static,
    M::RowT: ColumnRow + 'static,
    M::FiltrationT: ColumnFiltration,
    Clearing: ReductionAlgorithm<M>,
{
    let start = Instant::now();
    let expected = Clearing::new().reduce(matrix).diagram.sorted();
    eprintln!("{case}: serial {:.3}s, {} pairs", start.elapsed().as_secs_f64(), expected.pairings.len());
    for threads in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        for table in PivotTableKind::ALL {
            for slots in SlotKind::ALL {
                let mut best = f64::INFINITY;
                for _ in 0..repeats {
                    let start = Instant::now();
                    let reduction = reduce_with(Clearing::new().parallel().in_pool(&pool), matrix, table, slots);
                    best = best.min(start.elapsed().as_secs_f64());
                    assert!(reduction.diagram.sorted() == expected, "{case} {table:?} {slots:?} {threads}");
                }
                println!("{case},{table:?},{slots:?},{threads},{best:.5}");
            }
        }
    }
}

fn main() {
    let repeats: usize = std::env::args().nth(1).map_or(5, |a| a.parse().unwrap());
    println!("case,table,slots,threads,seconds");
    let rips = build_rips_coboundary_matrix::<Z2>(cube(160, 5), 2).reverse();
    run("rips cube n=160 dim<=2", &rips, repeats);
    drop(rips);
    // Homology: far more columns need reducing, so far more pivots are contended
    let distances = cube(60, 5);
    let coboundary = RipsCoboundary::<Z2, usize, _>::new(&distances);
    let boundary = RipsBoundary::<Z2, usize, _>::new(&distances)
        .with_basis(build_rips_bases::<usize, _>(&distances, 3))
        .with_transpose(&coboundary);
    run("rips homology cube n=60 dim<=2", &boundary, repeats);
    for (n, p) in [(100, 0.1)] {
        let edges = digraph(n, p, 5);
        let filtration = build_filtration(n, &edges).unwrap();
        let edge_set = build_edge_set(&edges);
        let coboundary = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n as u16);
        let matrix = coboundary.by_ref().reverse();
        run(&format!("grpph n={n} p={p}"), &matrix, repeats);
    }
}
