use std::collections::HashMap;

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::MergedColumns,
    fields::{Invertible, NonZeroCoefficient, Z2, Z3},
    matrices::{ColBasis, MatrixOracle},
    reduction::{Clearing, Standard},
};
use phlite_networks::pph::{
    cohomology_basis, homology_basis, pph_barcode, Network, PPHBar, PPHBoundary, PPHCoboundary,
    RegularPath,
};

/// A small deterministic PRNG (xorshift64*), to avoid a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random network on `n` vertices: each edge present with probability `density_percent`%, with weight in `1..=levels`.
fn random_network(rng: &mut Rng, n: u16, density_percent: u64, levels: u64) -> Network {
    let mut edges = vec![];
    for a in 0..n {
        for b in 0..n {
            if a != b && rng.below(100) < density_percent {
                edges.push((a, b, (1 + rng.below(levels)) as f64));
            }
        }
    }
    Network::from_edges(n, &edges)
}

fn algorithms() -> Vec<(&'static str, Algorithm<'static, MergedColumns>)> {
    vec![
        ("standard", Algorithm::Standard(Standard::new())),
        ("clearing", Algorithm::Clearing(Clearing::new())),
        (
            "clearing (keep zero apparent)",
            Algorithm::Clearing(Clearing::new().skip_zero_apparent_pairs(false)),
        ),
        (
            "parallel",
            Algorithm::ParallelClearing(Clearing::new().parallel()),
        ),
    ]
}

/// Bars as sorted (dim, birth, death) triples, for comparing barcodes.
fn summary<CF>(bars: &[PPHBar<CF>]) -> Vec<(usize, NotNan<f64>, Option<NotNan<f64>>)> {
    let mut out: Vec<_> = bars
        .iter()
        .map(|bar| {
            (
                bar.birth.dimension(),
                bar.birth_value,
                bar.death.map(|(_, d)| d),
            )
        })
        .collect();
    out.sort();
    out
}

fn cycle_network(n: u16) -> Network {
    // a -> b at time (b - a) mod n, as in pph-matlab's cycle_nets.mat
    let edges: Vec<_> = (0..n)
        .flat_map(|a| {
            (0..n)
                .filter(move |&b| b != a)
                .map(move |b| (a, b, ((b + n - a) % n) as f64))
        })
        .collect();
    Network::from_edges(n, &edges)
}

#[test]
fn cycle_networks_have_one_bar_in_dimension_one() {
    // Chowdhury and Mémoli: Dgm_1 of the cycle network on n nodes is {(1, ceil(n / 2))}
    for n in 3..=9u16 {
        let network = cycle_network(n);
        for method in [Method::Homology, Method::Cohomology] {
            for (name, algorithm) in algorithms() {
                let bars: Vec<PPHBar<Z2>> = pph_barcode(&network, 1, method, algorithm, false);
                let dim_one: Vec<_> = summary(&bars).into_iter().filter(|b| b.0 == 1).collect();
                let expected = vec![(
                    1,
                    NotNan::new(1.0).unwrap(),
                    Some(NotNan::new(n.div_ceil(2) as f64).unwrap()),
                )];
                assert_eq!(dim_one, expected, "n = {n}, {method:?}, {name}");
            }
        }
    }
}

#[test]
fn coboundary_is_transpose_of_boundary() {
    let mut rng = Rng(7);
    for _ in 0..20 {
        let network = random_network(&mut rng, 5, 50, 3);
        let boundary = PPHBoundary::<Z3>::new(&network);
        let coboundary = PPHCoboundary::<Z3>::new(&network);
        let basis = homology_basis(&network, 2, false);
        let mut from_boundary = HashMap::new();
        for i in 0..basis.size() {
            let tau = basis.element(i);
            for (c, sigma) in boundary.column(tau) {
                from_boundary.insert((sigma, tau), c);
            }
        }
        let mut from_coboundary = HashMap::new();
        let rows: Vec<RegularPath> = from_boundary.keys().map(|(sigma, _)| *sigma).collect();
        for sigma in rows {
            for (c, tau) in coboundary.column(sigma) {
                if tau.dimension() <= 3 {
                    from_coboundary.insert((sigma, tau), c);
                }
            }
        }
        assert_eq!(from_boundary, from_coboundary);
    }
}

fn agreement<CF>(seed: u64)
where
    CF: NonZeroCoefficient + Invertible + Send + Sync + 'static + std::fmt::Debug + PartialEq,
{
    let mut rng = Rng(seed);
    for trial in 0..40 {
        let n = 3 + rng.below(4) as u16;
        let max_dim = if n <= 4 { 3 } else { 2 };
        // sparse networks have never-allowed paths
        let density = [30, 60, 100][trial % 3];
        let network = random_network(&mut rng, n, density, 4);
        let reference = summary(&pph_barcode::<CF, _>(
            &network,
            max_dim,
            Method::Homology,
            Algorithm::Standard(Standard::new()),
            false,
        ));
        for method in [Method::Homology, Method::Cohomology] {
            for (name, algorithm) in algorithms() {
                for representatives in [false, true] {
                    let bars: Vec<PPHBar<CF>> = pph_barcode(
                        &network,
                        max_dim,
                        method,
                        algorithm.clone(),
                        representatives,
                    );
                    assert_eq!(
                        summary(&bars),
                        reference,
                        "trial {trial}: {method:?}, {name}, representatives {representatives}"
                    );
                    if representatives {
                        check_representatives(&network, &bars);
                    }
                }
            }
        }
        // There is exactly one essential bar in dimension 0 per weakly connected component, all born at 0
        // (a sanity check that the reference isn't trivially wrong)
        assert!(reference.iter().any(|b| b.0 == 0 && b.2.is_none()));
    }
}

/// Every representative is a cycle made of paths allowed by the bar's birth, with the birth path as a term.
fn check_representatives<CF>(network: &Network, bars: &[PPHBar<CF>])
where
    CF: NonZeroCoefficient + Invertible + std::fmt::Debug + PartialEq,
{
    let boundary = PPHBoundary::<CF>::new(network);
    for bar in bars {
        let rep = bar
            .representative
            .as_ref()
            .expect("asked for representatives");
        assert!(rep.iter().any(|(_, path)| *path == bar.birth));
        let mut total: HashMap<RegularPath, Option<CF>> = HashMap::new();
        for (c, path) in rep {
            assert_eq!(path.dimension(), bar.birth.dimension());
            assert!(
                network.allow_time(*path) <= bar.birth_value,
                "representative uses a later path"
            );
            for (d, face) in boundary.column(*path) {
                let entry = total.entry(face).or_insert(None);
                *entry = (*c * d) + *entry;
            }
        }
        assert!(
            total.values().all(|c| c.is_none()),
            "representative is not a cycle"
        );
    }
}

#[test]
fn methods_and_algorithms_agree_z2() {
    agreement::<Z2>(1);
}

#[test]
fn methods_and_algorithms_agree_z3() {
    agreement::<Z3>(2);
}

#[test]
fn cohomology_basis_includes_never_allowed_faces() {
    // a -> b -> c without a -> c: the face [a, c] of [a, b, c] is never allowed
    let network = Network::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let basis = cohomology_basis(&network, 1, false);
    let elems: Vec<_> = (0..basis.size()).map(|i| basis.element(i)).collect();
    assert!(elems.contains(&network.path(&[0, 2])));
    assert_eq!(network.allow_time(network.path(&[0, 2])), f64::INFINITY);
    // and the 1-dimensional barcode is empty (the path a -> b -> c is contractible)
    let bars: Vec<PPHBar<Z2>> = pph_barcode(
        &network,
        1,
        Method::Cohomology,
        Algorithm::Clearing(Clearing::new()),
        true,
    );
    assert!(bars.iter().all(|bar| bar.birth.dimension() == 0));
}

#[test]
fn bi_fan_and_bi_parallel() {
    // Chowdhury and Mémoli's examples: the bi-fan has H_1 = 1, the bi-parallel motif H_1 = 0
    let bi_fan = Network::from_edges(4, &[(0, 1, 1.0), (2, 1, 1.0), (2, 3, 1.0), (0, 3, 1.0)]);
    let bi_parallel = Network::from_edges(4, &[(0, 1, 1.0), (1, 3, 1.0), (0, 2, 1.0), (2, 3, 1.0)]);
    let dim_one = |network: &Network| -> usize {
        let bars: Vec<PPHBar<Z2>> =
            pph_barcode(network, 1, Method::Cohomology, Algorithm::default(), false);
        bars.iter().filter(|bar| bar.birth.dimension() == 1).count()
    };
    assert_eq!(dim_one(&bi_fan), 1);
    assert_eq!(dim_one(&bi_parallel), 0);
}

#[test]
fn paths_round_trip_through_their_codes() {
    let mut rng = Rng(11);
    for n in [2u16, 3, 7, 300] {
        let network = Network::from_edges(n, &[]);
        let max_dim = network.max_dimension().min(8);
        for _ in 0..500 {
            let len = 1 + rng.below(max_dim as u64 + 1) as usize;
            let mut verts = vec![rng.below(n as u64) as u16];
            while verts.len() < len {
                let v = rng.below(n as u64) as u16;
                if v != *verts.last().unwrap() {
                    verts.push(v);
                }
            }
            let path = network.path(&verts);
            assert_eq!(path.dimension(), len - 1);
            assert_eq!(network.vertices(path), verts);
        }
    }
    // the codes in each dimension are exactly 0..n(n-1)^p
    let network = Network::from_edges(4, &[]);
    assert_eq!(network.max_dimension(), 15);
    let last = network.path(&[3, 2, 3, 2]);
    assert_eq!(last.as_u64() & ((1 << 60) - 1), 4 * 27 - 1);
}

#[test]
fn truncation_clips_the_barcode() {
    let mut rng = Rng(5);
    for trial in 0..30 {
        let network = random_network(&mut rng, 5, [40, 70, 100][trial % 3], 4);
        let full: Vec<PPHBar<Z2>> =
            pph_barcode(&network, 2, Method::Cohomology, Algorithm::default(), false);
        for t_max in [1.0, 2.0, 3.0] {
            let t = NotNan::new(t_max).unwrap();
            let truncated: Vec<PPHBar<Z2>> = pph_barcode(
                &network.truncate(t_max),
                2,
                Method::Cohomology,
                Algorithm::default(),
                false,
            );
            let mut expected: Vec<_> = summary(&full)
                .into_iter()
                .filter(|&(_, birth, _)| birth <= t)
                .map(|(dim, birth, death)| (dim, birth, death.filter(|&d| d <= t)))
                .collect();
            expected.sort();
            assert_eq!(
                summary(&truncated),
                expected,
                "trial {trial}, t_max {t_max}"
            );
        }
    }
}

#[test]
fn cohomology_basis_is_allowed_paths_and_their_never_allowed_faces() {
    // Compare against enumerating one dimension higher and taking faces
    let mut rng = Rng(13);
    for trial in 0..30 {
        let network = random_network(&mut rng, 6, [20, 40, 60][trial % 3], 3);
        let max_dim = 3;
        let boundary = PPHBoundary::<Z2>::new(&network);
        let higher = homology_basis(&network, max_dim, false);
        let mut expected = std::collections::BTreeSet::new();
        for i in 0..higher.size() {
            let path = higher.element(i);
            if path.dimension() <= max_dim {
                expected.insert(path);
            }
            for (_c, face) in boundary.column(path) {
                expected.insert(face);
            }
        }
        let basis = cohomology_basis(&network, max_dim, true);
        let got: std::collections::BTreeSet<_> =
            (0..basis.size()).map(|i| basis.element(i)).collect();
        assert_eq!(got.len(), basis.size(), "duplicates in the basis");
        assert_eq!(got, expected, "trial {trial}");
    }
}
