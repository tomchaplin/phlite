use ordered_float::NotNan;
use petgraph::{algo::dijkstra, graph::NodeIndex, Graph};
use phlite::{
    fields::Z2, filtrations::grpph::GrPPHCoboundary, matrices::HasRowFiltration,
    reduction::ClearedReductionMatrix,
};
use rustc_hash::FxHashSet;

fn main() {
    let n = 500;

    let mut g = Graph::<(), NotNan<f64>>::new();

    g.extend_with_edges(
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| (i, j, NotNan::new(1.0).unwrap())),
    );

    println!("Built graph");

    let mut filtration = vec![];
    for i in 0..n {
        let sp_lengths = dijkstra(&g, NodeIndex::from(i), None, |e| e.weight().into_inner());
        filtration.push(
            (0..n)
                .map(|j| {
                    sp_lengths
                        .get(&NodeIndex::from(j))
                        .copied()
                        .map(|t| NotNan::new(t).unwrap())
                })
                .collect(),
        );
    }

    println!("Computed filtration");

    let edge_set: FxHashSet<(u16, u16)> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i as u16, j as u16)))
        .filter(|(i, j)| i != j)
        .collect();

    let d = GrPPHCoboundary::<Z2, _>::build(filtration, edge_set, n as u16);

    println!("Built coboundary matrix");

    let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(&d, 0..=1);

    let mut count = 0;

    // Report
    println!("Essential:");
    for idx in diagram.essential.iter() {
        let f_val = d.filtration_value(*idx).unwrap().0;
        println!(" birth={idx:?}, f=({f_val}, ∞)");
    }
    println!("\nPairings:");
    for tup in diagram.pairings.iter() {
        let idx_tup = (tup.1, tup.0);
        let birth_f = d.filtration_value(tup.1).unwrap().0;
        let death_f = d.filtration_value(tup.0).unwrap().0;
        if death_f == birth_f {
            continue;
        }
        count += 1;
        println!(" pair={idx_tup:?}, f=({birth_f}, {death_f})");
    }
    println!("{count}");

    // Initial graph is complete and single component so Euler charcteristic tells us the circuit rank
    let n = n as usize;
    assert_eq!(count, n * (n - 1) - n + 1);
    // Only essential cycle is the initial component
    assert_eq!(diagram.essential.len(), 1);
}
