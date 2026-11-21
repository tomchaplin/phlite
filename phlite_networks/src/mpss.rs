//! The magnitude-path spectral sequence (MPSS) of a directed graph, over a field or the integers.
//!
//! The *reachability chain complex* `RC_*(G)` has a basis of the tuples `(x_0, ..., x_k)` with `x_i ≠ x_{i+1}` and a
//! directed path from each to the next, and differential `∂ = Σ_{i=0}^{k} (-1)^i ∂_i`, where `∂_i` deletes `x_i` (and
//! terms with equal neighbours are omitted). Deleting a vertex never makes a tuple longer, so the *length*
//! `d(x_0, x_1) + ... + d(x_{k-1}, x_k)` filters `RC_*(G)`; the spectral sequence of this filtration is the MPSS
//! (Hepworth and Willerton; Asao, *Magnitude homology and path homology*, 2023; Hepworth and Roff, *Bigraded path
//! homology and the magnitude-path spectral sequence*, 2024). Its pages, in the bigrading of magnitude homology,
//! `E^r_{k,l} = E^r_{l, k-l}` (degree `k`, length `l`):
//! * `E^0` is the magnitude chain complex, `E^1` magnitude homology [`MH_{k,l}`](crate::magnitude);
//! * `E^2` is bigraded path homology, whose diagonal `E^2_{k,k}` is the path homology of `G`;
//! * it converges (for graphs whose finite distances are bounded) to reachability homology.
//!
//! **How.** A spectral sequence of a filtered complex over a field is read off its persistence barcode
//! ([`phlite::spectral`]): a bar `[b, d)` of degree `k` gives a generator of `E^r_{k,b}` and of `E^r_{k+1,d}` for
//! every `r <= d - b`, an essential bar one of `E^r_{k,b}` for every `r`. The barcode is phlite's, of `F_{l_max} RC_*(G)` up to degree `k_max + 1`, so the entries with
//! `l + r <= l_max + 1` (and `k <= k_max`) are exact: a bar born at `l` and still alive at `l_max` has length at
//! least `l_max + 1 - l >= r` whatever happens later. The others are unknown, and given as `None`.
//!
//! The complex does not split over pairs of endpoints (deleting `x_0` or `x_k` changes them), so this lists every
//! tuple up to `l_max` and is for small graphs. Over the integers ([`mpss_integral`]) there is no barcode, and each
//! entry is computed exactly from lattices ([`phlite::integral`]): far slower, for a handful of nodes.

use std::collections::HashMap;

use ordered_float::NotNan;
use phlite::barcode::{Algorithm, Method};
use phlite::integral::{ChainGenerator, Group, IntegralFiltered, Overflow, Ring};
use phlite::spectral::{BarcodeSpectral, LocalSpectral, SpectralBackend, WithGenerators};
use phlite::reduction::Clearing;
use phlite::spectral::{spectral_pages, StepBar};
use phlite_complexes::complex::{Chain, RawChain};
use phlite_complexes::field::IntField;
use rayon::prelude::*;

use crate::magnitude::{enumerate_from, Buckets, MagnitudeGraph, MODULI, UNREACHABLE};

/// The pages of the MPSS: `pages[r - 1][l][k]` is the rank of `E^r_{k,l}` (degree `k`, length `l`) for
/// `r = 1, ..., r_max`, `l <= l_max` and `k <= min(l, k_max)`, or `None` if `F_{l_max}` doesn't determine it.
pub type Pages = Vec<Vec<Vec<Option<usize>>>>;

/// The bars of the length filtration of the reachability complex, as `(degree, birth, death)` (death `None` for a
/// bar still alive at `l_max`), over `CF`, of the tuples of length at most `l_max` and degree at most `k_max + 1`.
fn reachability_bars<CF: IntField>(graph: &MagnitudeGraph, l_max: usize, k_max: usize) -> Vec<(usize, usize, Option<usize>)> {
    let (raw, values, _) = reachability_complex(graph, l_max, k_max);
    bars_of::<CF>(raw, values, k_max + 1)
}

/// The bars of the reachability complex `raw` (with `values`), over `CF`, up to degree `k_top`.
fn bars_of<CF: IntField>(raw: RawChain, values: Vec<f64>, k_top: usize) -> Vec<(usize, usize, Option<usize>)> {
    let complex = Chain::<CF>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(values)))
        .expect("the reachability complex, faces first and no longer than their cofaces");
    let algorithm = Algorithm::Clearing(Clearing::new().with_columns(phlite::columns::PivotColumns).store_r(true));
    let as_length = |v: NotNan<f64>| v.into_inner() as usize;
    complex
        .barcode(Some(k_top), Method::Cohomology, algorithm, false)
        .into_iter()
        .map(|bar| (complex.dimension(bar.birth), as_length(bar.birth_value), bar.death.map(|(_, d)| as_length(d))))
        .collect()
}

/// The reachability complex of `graph` up to length `l_max` and degree `k_max + 1`, with integer coefficients, faces
/// first, and each tuple's length as its value; and the tuples themselves, in the same order.
fn reachability_complex(graph: &MagnitudeGraph, l_max: usize, k_max: usize) -> (RawChain, Vec<f64>, Vec<Vec<u32>>) {
    let n = graph.n_vertices() as u32;
    let k_top = k_max + 1;
    let reach = vec![0u32; n as usize];
    // Every tuple, by source: [s] -> {(t, l, k): tuples}
    let by_source: Vec<Buckets> = (0..n)
        .into_par_iter()
        .map(|s| {
            let mut buckets = Buckets::new();
            buckets.entry((s, 0, 0)).or_default().push(s);
            if k_top > 0 {
                enumerate_from(graph, &[s], 0, l_max, k_top, &reach, &mut buckets);
            }
            buckets
        })
        .collect();
    // Number the cells by degree, then source, target and length, so that faces come first
    let mut first: HashMap<(u32, u32, usize, usize), usize> = HashMap::new();
    let mut keys: Vec<(usize, u32, u32, usize)> = by_source
        .iter()
        .enumerate()
        .flat_map(|(s, buckets)| buckets.keys().map(move |&(t, l, k)| (k, s as u32, t, l)))
        .collect();
    keys.sort_unstable();
    let mut n_cells = 0;
    for &(k, s, t, l) in &keys {
        first.insert((s, t, l, k), n_cells);
        n_cells += by_source[s as usize][&(t, l, k)].len() / (k + 1);
    }
    let length = |tuple: &[u32]| tuple.windows(2).map(|w| graph.distance(w[0], w[1]) as usize).sum::<usize>();
    let mut raw = RawChain {
        dimensions: Vec::with_capacity(n_cells),
        indptr: Vec::with_capacity(n_cells + 1),
        indices: vec![],
        coefficients: Some(vec![]),
    };
    raw.indptr.push(0);
    let mut values: Vec<f64> = Vec::with_capacity(n_cells);
    let mut cells: Vec<Vec<u32>> = Vec::with_capacity(n_cells);
    let mut face: Vec<u32> = Vec::with_capacity(k_top + 1);
    for &(k, s, t, l) in &keys {
        let tuples = &by_source[s as usize][&(t, l, k)];
        for tuple in tuples.chunks_exact(k + 1) {
            cells.push(tuple.to_vec());
            raw.dimensions.push(k as i64);
            values.push(l as f64);
            if k > 0 {
                for i in 0..=k {
                    // Omitted if the neighbours of x_i are equal
                    if i > 0 && i < k && tuple[i - 1] == tuple[i + 1] {
                        continue;
                    }
                    face.clear();
                    face.extend_from_slice(&tuple[..i]);
                    face.extend_from_slice(&tuple[i + 1..]);
                    let key = (face[0], face[k - 1], length(&face), k - 1);
                    let faces = &by_source[key.0 as usize][&(key.1, key.2, key.3)];
                    let (mut lo, mut hi) = (0, faces.len() / k);
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if faces[mid * k..(mid + 1) * k] < face[..] {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    debug_assert_eq!(&faces[lo * k..(lo + 1) * k], &face[..]);
                    raw.indices.push((first[&key] + lo) as i64);
                    raw.coefficients.as_mut().expect("set").push(if i % 2 == 0 { 1 } else { -1 });
                }
            }
            raw.indptr.push(raw.indices.len() as i64);
        }
    }
    (raw, values, cells)
}

/// The pages `E^1, ..., E^{r_max}` of the MPSS of `graph` over `Z/modulus` (one of [`MODULI`]), for lengths up to
/// `l_max` and degrees up to `k_max` (default `l_max`); see the [module docs](self) and [`Pages`].
///
/// # Panics
/// If the modulus isn't one of [`MODULI`].
pub fn mpss_pages(graph: &MagnitudeGraph, l_max: usize, k_max: Option<usize>, r_max: usize, modulus: u64) -> Pages {
    use phlite::fields::{Z2, ZP};
    let k_max = k_max.unwrap_or(l_max).min(l_max);
    let bars = match modulus {
        2 => reachability_bars::<Z2>(graph, l_max, k_max),
        3 => reachability_bars::<ZP<3>>(graph, l_max, k_max),
        5 => reachability_bars::<ZP<5>>(graph, l_max, k_max),
        7 => reachability_bars::<ZP<7>>(graph, l_max, k_max),
        11 => reachability_bars::<ZP<11>>(graph, l_max, k_max),
        13 => reachability_bars::<ZP<13>>(graph, l_max, k_max),
        251 => reachability_bars::<ZP<251>>(graph, l_max, k_max),
        other => panic!("unsupported modulus {other}, expected one of {MODULI:?}"),
    };
    // Lengths are the steps (every length 0..=l_max is a value), and the filtration is cut after l_max
    let bars: Vec<StepBar> =
        bars.into_iter().map(|(degree, birth, death)| StepBar { degree, birth, death }).collect();
    let pages = spectral_pages(&bars, l_max + 1, k_max + 1, r_max, Some(l_max));
    // In the bigrading of magnitude homology: degrees up to min(l, k_max) at length l
    pages
        .into_iter()
        .map(|page| page.into_iter().enumerate().map(|(l, row)| row[..=k_max.min(l)].to_vec()).collect())
        .collect()
}

/// The integral pages of the MPSS: `pages[r - 1][l][k]` is the group `E^r_{k,l}` (degree `k`, length `l`) for
/// `r = 1, ..., r_max`, `l <= l_max` and `k <= min(l, k_max)`, or `None` if `F_{l_max}` doesn't determine it.
pub type IntegralPages = Vec<Vec<Vec<Option<Group>>>>;

/// The pages `E^1, ..., E^{r_max}` of the MPSS of `graph` over the integers, as [`mpss_pages`] over a field, by exact
/// lattice computations ([`phlite::integral`]), each entry from the tuples in its window of lengths
/// ([`LocalSpectral`]): dense, so for small graphs.
///
/// # Errors
/// If an entry outgrows `i128` (see [`phlite::integral`]).
pub fn mpss_integral(
    graph: &MagnitudeGraph,
    l_max: usize,
    k_max: Option<usize>,
    r_max: usize,
) -> Result<IntegralPages, Overflow> {
    let k_max = k_max.unwrap_or(l_max).min(l_max);
    let backend = integral_backend(graph, l_max, k_max);
    (1..=r_max)
        .map(|r| {
            (0..=l_max)
                .map(|l| {
                    (0..=k_max.min(l))
                        .map(|k| if l < backend.n_steps() { backend.entry(r, l, k) } else { Ok(known_empty(r, l, l_max)) })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// An entry at a length no tuple has: zero, where the truncation determines it.
fn known_empty(r: usize, l: usize, l_max: usize) -> Option<Group> {
    (l + r <= l_max + 1).then(Group::default)
}

/// The reachability complex up to `l_max` and degree `k_max + 1` over Z, for [`LocalSpectral`].
fn integral_complex(graph: &MagnitudeGraph, l_max: usize, k_max: usize) -> IntegralFiltered {
    let (raw, values, _) = reachability_complex(graph, l_max, k_max);
    to_integral(&raw, &values)
}

fn to_integral(raw: &RawChain, values: &[f64]) -> IntegralFiltered {
    let coefficients = raw.coefficients.as_ref().expect("integer coefficients");
    IntegralFiltered {
        degree: raw.dimensions.iter().map(|&d| d as usize).collect(),
        step: values.iter().map(|&v| v as usize).collect(),
        boundary: raw
            .indptr
            .windows(2)
            .map(|w| (w[0] as usize..w[1] as usize).map(|i| (raw.indices[i] as usize, coefficients[i])).collect())
            .collect(),
    }
}

fn integral_backend(graph: &MagnitudeGraph, l_max: usize, k_max: usize) -> LocalSpectral<IntegralFiltered> {
    LocalSpectral::new(integral_complex(graph, l_max, k_max), k_max + 1, Some(l_max))
}

/// The MPSS of `graph` as a [`SpectralBackend`], for computing it a page (or an entry) at a time: over `Z/modulus`
/// (one of [`MODULI`]) from the barcode, or over the integers (`modulus` `None`) entry by entry, each from the tuples
/// in its window of lengths ([`LocalSpectral`]); generators either way (over a field, computed over `Z/p` the same
/// way when asked for), as chains of cells numbered as in the tuples returned alongside. Steps are lengths
/// `0..=l_max` and degrees `0..=k_max` (default `l_max`); entries needing tuples longer than `l_max` are `None`.
///
/// # Panics
/// If the modulus isn't one of [`MODULI`].
pub fn mpss_backend(
    graph: &MagnitudeGraph,
    l_max: usize,
    k_max: Option<usize>,
    modulus: Option<u64>,
) -> (Box<dyn SpectralBackend>, Vec<Vec<u32>>) {
    use phlite::fields::{Z2, ZP};
    let k_max = k_max.unwrap_or(l_max).min(l_max);
    let (raw, values, tuples) = reachability_complex(graph, l_max, k_max);
    let local = LocalSpectral::new(to_integral(&raw, &values), k_max + 1, Some(l_max));
    let Some(modulus) = modulus else {
        return (Box::new(Padded { inner: local, l_max }), tuples);
    };
    let k_top = k_max + 1;
    let bars = match modulus {
        2 => bars_of::<Z2>(raw, values, k_top),
        3 => bars_of::<ZP<3>>(raw, values, k_top),
        5 => bars_of::<ZP<5>>(raw, values, k_top),
        7 => bars_of::<ZP<7>>(raw, values, k_top),
        11 => bars_of::<ZP<11>>(raw, values, k_top),
        13 => bars_of::<ZP<13>>(raw, values, k_top),
        251 => bars_of::<ZP<251>>(raw, values, k_top),
        other => panic!("unsupported modulus {other}, expected one of {MODULI:?}"),
    };
    let bars: Vec<StepBar> = bars.into_iter().map(|(degree, birth, death)| StepBar { degree, birth, death }).collect();
    let backend = WithGenerators {
        entries: BarcodeSpectral::new(&bars, l_max + 1, k_max + 1, Some(l_max)),
        generators: Padded { inner: local.over(Ring::Mod(modulus)), l_max },
    };
    (Box::new(backend), tuples)
}

/// A backend over every length `0..=l_max`, though the complex may have no tuples as long as `l_max`.
struct Padded<B> {
    inner: B,
    l_max: usize,
}

impl<B: SpectralBackend> SpectralBackend for Padded<B> {
    fn n_steps(&self) -> usize {
        self.l_max + 1
    }

    fn n_degrees(&self) -> usize {
        self.inner.n_degrees()
    }

    fn entry(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if p < self.inner.n_steps() {
            self.inner.entry(r, p, k)
        } else {
            Ok(known_empty(r, p, self.l_max))
        }
    }

    fn differential(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if p < self.inner.n_steps() {
            self.inner.differential(r, p, k)
        } else {
            Ok(known_empty(r, p, self.l_max))
        }
    }

    fn generators(&self, r: usize, p: usize, k: usize) -> Result<Option<(Group, Vec<ChainGenerator>)>, Overflow> {
        if p < self.inner.n_steps() {
            self.inner.generators(r, p, k)
        } else {
            Ok(known_empty(r, p, self.l_max).map(|g| (g, vec![])))
        }
    }

    fn classify(&self, r: usize, p: usize, k: usize, chain: &[(usize, i64)]) -> Result<Option<Vec<i128>>, Overflow> {
        if p < self.inner.n_steps() {
            self.inner.classify(r, p, k, chain)
        } else {
            Ok(known_empty(r, p, self.l_max).filter(|_| chain.is_empty()).map(|_| vec![]))
        }
    }

    fn has_generators(&self) -> bool {
        self.inner.has_generators()
    }

    fn degenerates_at(&self) -> Option<usize> {
        self.inner.degenerates_at()
    }
}

/// The largest finite distance between two vertices of `graph` (0 if none).
pub fn largest_finite_distance(graph: &MagnitudeGraph) -> usize {
    let n = graph.n_vertices() as u32;
    (0..n)
        .flat_map(|a| (0..n).map(move |b| (a, b)))
        .map(|(a, b)| graph.distance(a, b))
        .filter(|&d| d != UNREACHABLE)
        .max()
        .unwrap_or(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnitude::{magnitude_ranks, total_ranks, MagnitudeQuery};

    fn graphs() -> Vec<(usize, Vec<(u32, u32)>)> {
        let cycle = |n: u32| (0..n).map(|i| (i, (i + 1) % n)).collect::<Vec<_>>();
        let undirected = |e: Vec<(u32, u32)>| e.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect::<Vec<_>>();
        vec![
            (4, cycle(4)),
            (5, cycle(5)),
            (5, undirected(cycle(5))),
            (4, vec![(0, 1), (1, 3), (0, 2), (2, 3)]),
            (5, vec![(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (1, 4)]),
        ]
    }

    /// E^1 is magnitude homology.
    #[test]
    fn first_page_is_magnitude_homology() {
        for (n, edges) in graphs() {
            let graph = MagnitudeGraph::new(n, &edges);
            let l_max = 4;
            let pages = mpss_pages(&graph, l_max, None, 1, 3);
            let mh = total_ranks(&magnitude_ranks(&graph, &MagnitudeQuery { l_max, ..Default::default() }, &[3], false), 0, l_max, None);
            for l in 0..=l_max {
                for k in 0..=l {
                    assert_eq!(pages[0][l][k], Some(mh[l][k]), "{edges:?} ({k}, {l})");
                }
            }
        }
    }

    /// Each page is a subquotient of the one before, so no rank grows.
    #[test]
    fn pages_shrink() {
        for (n, edges) in graphs() {
            let graph = MagnitudeGraph::new(n, &edges);
            let pages = mpss_pages(&graph, 5, None, 4, 2);
            for r in 1..pages.len() {
                for (l, row) in pages[r].iter().enumerate() {
                    for (k, &e) in row.iter().enumerate() {
                        if let (Some(e), Some(before)) = (e, pages[r - 1][l][k]) {
                            assert!(e <= before);
                        }
                    }
                }
            }
        }
    }

    /// Directed cycles: path homology of the directed m-cycle (m >= 3) is that of a circle (Grigor'yan, Lin, Muranov
    /// and Yau), so the E^2 diagonal has rank 1 in degrees 0 and 1 and 0 above.
    #[test]
    fn directed_cycle_path_homology() {
        for m in 3..7u32 {
            let edges: Vec<(u32, u32)> = (0..m).map(|i| (i, (i + 1) % m)).collect();
            let graph = MagnitudeGraph::new(m as usize, &edges);
            let pages = mpss_pages(&graph, 4, None, 2, 2);
            let e2 = &pages[1];
            // E^2_{0,0} = PH_0 = Z (connected), E^2_{1,1} = PH_1 = Z, E^2_{2,2} = PH_2 = 0
            assert_eq!(e2[0][0], Some(1), "m={m}");
            assert_eq!(e2[1][1], Some(1), "m={m}");
            assert_eq!(e2[2][2], Some(0), "m={m}");
        }
    }

    /// Hepworth and Roff, Theorem 8.2: for the directed m-cycle (m >= 3), magnitude homology and bigraded path
    /// homology live in bidegrees (k, l) = (2i, mi) and (2i + 1, mi + 1), of rank m and 1; E^2 = ... = E^{m-1}, and
    /// E^m is a single generator at (0, 0).
    #[test]
    fn directed_cycles_hepworth_roff() {
        for m in 3..6usize {
            let edges: Vec<(u32, u32)> = (0..m as u32).map(|i| (i, (i + 1) % m as u32)).collect();
            let graph = MagnitudeGraph::new(m, &edges);
            let l_max = 2 * m + 1;
            let pages = mpss_pages(&graph, l_max, Some(5), m, 2);
            let special = |k: usize, l: usize| (0..=2).any(|i| (k, l) == (2 * i, m * i) || (k, l) == (2 * i + 1, m * i + 1));
            for (l, row) in pages[0].iter().enumerate() {
                for (k, &e) in row.iter().enumerate() {
                    let want = if special(k, l) { m } else { 0 };
                    assert_eq!(e, Some(want), "E1 m={m} ({k}, {l})");
                }
            }
            for r in 2..m {
                for (l, row) in pages[r - 1].iter().enumerate() {
                    for (k, &e) in row.iter().enumerate() {
                        if let Some(e) = e {
                            assert_eq!(e, usize::from(special(k, l)), "E{r} m={m} ({k}, {l})");
                        }
                    }
                }
            }
            for (l, row) in pages[m - 1].iter().enumerate() {
                for (k, &e) in row.iter().enumerate() {
                    if let Some(e) = e {
                        assert_eq!(e, usize::from((k, l) == (0, 0)), "E{m} m={m} ({k}, {l})");
                    }
                }
            }
        }
    }

    /// Hepworth and Roff, Theorem 9.3: for the bi-directed cycle C_{m,n} (directed paths of lengths m and n from one
    /// vertex to another), with M = max(m, n) >= 3, bigraded path homology is a single generator in each of the
    /// bidegrees (k, l) = (0, 0), (1, 1) and (2, M); E^2 = ... = E^{M-1}, and E^M is a single generator at (0, 0).
    #[test]
    fn bidirected_cycles_hepworth_roff() {
        for (m, n) in [(3, 1), (3, 2), (3, 3), (4, 1), (4, 3), (5, 2)] {
            // 0 is the source, 1 the sink; then the inner vertices of the top path, then those of the bottom one
            let path = |len: usize, first: u32| -> Vec<(u32, u32)> {
                let mut vertices = vec![0];
                vertices.extend(first..first + len as u32 - 1);
                vertices.push(1);
                vertices.windows(2).map(|w| (w[0], w[1])).collect()
            };
            let mut edges = path(m, 2);
            edges.extend(path(n, 2 + m as u32 - 1));
            let n_vertices = 2 + (m - 1) + (n - 1);
            let graph = MagnitudeGraph::new(n_vertices, &edges);
            let big = m.max(n);
            let l_max = 2 * big + 1;
            let pages = mpss_pages(&graph, l_max, Some(4), big, 2);
            let special = |k: usize, l: usize| [(0, 0), (1, 1), (2, big)].contains(&(k, l));
            for r in 2..big {
                for (l, row) in pages[r - 1].iter().enumerate() {
                    for (k, &e) in row.iter().enumerate() {
                        if let Some(e) = e {
                            assert_eq!(e, usize::from(special(k, l)), "C_({m},{n}) E{r} ({k}, {l})");
                        }
                    }
                }
            }
            for (l, row) in pages[big - 1].iter().enumerate() {
                for (k, &e) in row.iter().enumerate() {
                    if let Some(e) = e {
                        assert_eq!(e, usize::from((k, l) == (0, 0)), "C_({m},{n}) E{big} ({k}, {l})");
                    }
                }
            }
        }
    }

    /// The prime-power (elementary) divisors of a torsion part, sorted.
    fn elementary(torsion: &[u128]) -> Vec<u128> {
        let mut out = vec![];
        for &t in torsion {
            let (mut t, mut p) = (t, 2u128);
            while t > 1 {
                let mut q = 1;
                while t % p == 0 {
                    t /= p;
                    q *= p;
                }
                if q > 1 {
                    out.push(q);
                }
                p += 1;
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn integral_pages_match_fields_and_magnitude_homology() {
        use crate::magnitude::magnitude_integral;
        for (n, edges) in graphs() {
            let graph = MagnitudeGraph::new(n, &edges);
            let l_max = 4;
            let r_max = 3;
            let integral = mpss_integral(&graph, l_max, None, r_max).unwrap();
            // Free ranks are the dimensions over Z/251
            let field = mpss_pages(&graph, l_max, None, r_max, 251);
            for r in 0..r_max {
                for l in 0..=l_max {
                    for k in 0..=l {
                        assert_eq!(
                            integral[r][l][k].as_ref().map(|g| g.rank),
                            field[r][l][k],
                            "{edges:?}: E^{} ({k}, {l})",
                            r + 1
                        );
                    }
                }
            }
            // E^1 is integral magnitude homology, the sum over pairs
            let pairs = magnitude_integral(&graph, &MagnitudeQuery { l_max, ..Default::default() }, false);
            for l in 0..=l_max {
                for k in 0..=l {
                    let (mut rank, mut torsion) = (0, vec![]);
                    for pair in &pairs {
                        if let Some(Some((r, t))) = pair.groups.get(l).and_then(|row| row.get(k)) {
                            rank += r;
                            torsion.extend(t.iter().copied());
                        }
                    }
                    let e1 = integral[0][l][k].clone().unwrap();
                    assert_eq!((e1.rank, elementary(&e1.torsion)), (rank, elementary(&torsion)), "{edges:?} ({k}, {l})");
                }
            }
        }
    }

}
