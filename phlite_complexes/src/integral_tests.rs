//! `phlite::integral`'s spectral pages against the field pages from a barcode (`phlite::spectral`): over Z/251, on
//! small random complexes, the free rank of each integral page is the dimension of the page over the field.

use phlite::barcode::{Algorithm, Method};
use phlite::fields::ZP;
use phlite::integral::{integral_pages, IntegralFiltered};
use phlite::reduction::Clearing;
use phlite::spectral::{spectral_pages, StepBar};

use crate::complex::{Chain, RawChain};

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// A random simplicial complex (the closure of a few random simplices on `n` vertices, up to dimension 3), each
/// simplex at a random step no earlier than its faces, cells in order of (dimension, vertices).
fn random_filtered(rng: &mut Rng) -> IntegralFiltered {
    use std::collections::BTreeMap;
    let n = 4 + rng.below(4) as usize;
    let mut all: Vec<Vec<usize>> = vec![];
    for _ in 0..2 + rng.below(6) {
        let top: Vec<usize> = (0..n).filter(|_| rng.below(2) == 0).take(4).collect();
        for mask in 1u32..(1 << top.len()) {
            all.push((0..top.len()).filter(|&i| mask >> i & 1 == 1).map(|i| top[i]).collect());
        }
    }
    all.sort_by_key(|s| (s.len(), s.clone()));
    all.dedup();
    let mut index: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut complex = IntegralFiltered::default();
    for s in all {
        let mut boundary = vec![];
        let mut step = rng.below(4) as usize;
        if s.len() > 1 {
            for i in 0..s.len() {
                let mut f = s.clone();
                f.remove(i);
                let face = index[&f];
                step = step.max(complex.step[face]);
                boundary.push((face, if i % 2 == 0 { 1 } else { -1 }));
            }
        }
        index.insert(s.clone(), complex.degree.len());
        complex.degree.push(s.len() - 1);
        complex.step.push(step);
        complex.boundary.push(boundary);
    }
    complex
}

#[test]
fn integral_free_ranks_match_field_pages() {
    let mut rng = Rng(99);
    for case in 0..150 {
        let complex = random_filtered(&mut rng);
        let n_steps = complex.step.iter().max().unwrap() + 1;
        let n_degrees = 4;
        let r_max = 4;
        // The field pages, from the barcode over Z/251 (values = steps)
        let mut raw = RawChain {
            dimensions: complex.degree.iter().map(|&d| d as i64).collect(),
            indptr: vec![0],
            indices: vec![],
            coefficients: Some(vec![]),
        };
        for boundary in &complex.boundary {
            for &(face, coefficient) in boundary {
                raw.indices.push(face as i64);
                raw.coefficients.as_mut().unwrap().push(coefficient);
            }
            raw.indptr.push(raw.indices.len() as i64);
        }
        let filtered = Chain::<ZP<251>>::from_raw(raw)
            .unwrap()
            .filter(Some(complex.step.iter().map(|&s| s as f64).collect()))
            .unwrap();
        let bars: Vec<StepBar> = filtered
            .barcode(None, Method::Cohomology, Algorithm::Clearing(Clearing::new()), false)
            .into_iter()
            .map(|bar| StepBar {
                degree: filtered.dimension(bar.birth),
                birth: bar.birth_value.into_inner() as usize,
                death: bar.death.map(|(_, v)| v.into_inner() as usize),
            })
            .collect();
        let field = spectral_pages(&bars, n_steps, n_degrees, r_max, None);
        let integral = integral_pages(&complex, n_degrees, r_max, None).unwrap();
        for r in 1..=r_max {
            for p in 0..n_steps {
                for k in 0..n_degrees {
                    let rank = integral[r][p][k].as_ref().unwrap().rank;
                    assert_eq!(Some(rank), field[r - 1][p][k], "case {case}: E^{r} at step {p}, degree {k}");
                }
            }
        }
    }
}
