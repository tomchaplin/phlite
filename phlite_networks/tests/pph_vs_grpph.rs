//! `pph` of the shortest-path network agrees with the GrPPH oracle without grounding (PPH of the shortest-path
//! filtration, in dimensions 0 and 1): two independent implementations of the same thing.

use phlite::{
    barcode::{Algorithm, Method},
    columns::MergedColumns,
    fields::Z2,
    reduction::Clearing,
};
use phlite_networks::{
    grpph::{build_filtration, coboundary::DigraphEdgeSet, grpph_barcode},
    pph::{pph_barcode, Network, PPHBar},
};

/// A small deterministic random number generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// All-pairs shortest path lengths (Floyd–Warshall), infinite if there is no path.
fn shortest_paths(n: usize, edges: &[(u32, u32, f64)]) -> Vec<Vec<f64>> {
    let mut d = vec![vec![f64::INFINITY; n]; n];
    for (v, row) in d.iter_mut().enumerate() {
        row[v] = 0.0;
    }
    for &(a, b, w) in edges {
        let slot = &mut d[a as usize][b as usize];
        *slot = slot.min(w);
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through = d[i][k] + d[k][j];
                if through < d[i][j] {
                    d[i][j] = through;
                }
            }
        }
    }
    d
}

fn sorted(mut bars: Vec<(usize, f64, f64)>) -> Vec<(usize, f64, f64)> {
    bars.sort_by(|a, b| a.partial_cmp(b).unwrap());
    bars
}

#[test]
fn pph_of_shortest_paths_matches_grpph_without_grounding() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for trial in 0..300 {
        let n = 2 + rng.below(7) as usize;
        let density = [20, 40, 70][rng.below(3) as usize];
        let mut edges = vec![];
        for a in 0..n as u32 {
            for b in 0..n as u32 {
                if a != b && rng.below(100) < density {
                    let w = if rng.below(10) == 0 { f64::INFINITY } else { rng.below(4) as f64 };
                    edges.push((a, b, w));
                }
            }
        }
        let d = shortest_paths(n, &edges);
        let network = Network::from_dense(&d);

        let filtration = build_filtration(n as u32, &edges);
        let expected = grpph_barcode(
            &filtration,
            &DigraphEdgeSet::default(),
            n as u16,
            Method::Cohomology,
            Algorithm::<MergedColumns>::Clearing(Clearing::new()),
            false,
        );
        let expected = sorted(
            expected
                .iter()
                .map(|bar| {
                    let death = bar.death.map_or(f64::INFINITY, |(_, d)| d.into_inner());
                    (bar.birth.dimension(), bar.birth_value.into_inner(), death)
                })
                .collect(),
        );
        for method in [Method::Cohomology, Method::Homology] {
            let bars: Vec<PPHBar<Z2>> = pph_barcode(
                &network,
                1,
                method,
                Algorithm::<MergedColumns>::Clearing(Clearing::new()),
                false,
            );
            let got = sorted(
                bars.iter()
                    .map(|bar| {
                        let death = bar.death.map_or(f64::INFINITY, |(_, d)| d.into_inner());
                        (bar.birth.dimension(), bar.birth_value.into_inner(), death)
                    })
                    .collect(),
            );
            assert_eq!(got, expected, "trial {trial}, {method:?}, edges {edges:?}");
        }
    }
}
