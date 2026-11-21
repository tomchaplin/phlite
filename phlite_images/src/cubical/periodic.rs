//! Periodic cubical complexes: an image on a torus (or a cylinder: any axes may wrap round), as GUDHI's
//! `PeriodicCubicalComplex` and Cubical Ripser's periodic option.
//!
//! Along a periodic axis of `n` pixels the doubled grid has `2n` cells (vertices and edges alternating round a circle)
//! rather than `2n + 1` (T-construction) or `2n - 1` (V-construction), and faces and cofaces wrap round. The complex
//! is built explicitly (every cell, its oriented boundary and its value) and reduced by `phlite_complexes`, which finds
//! dimension 0 and the top dimension by union-find; so it is exact for any number of axes, but slower and hungrier than
//! the implicit, non-periodic complexes of [`super`] on large volumes.

use phlite::barcode::{Algorithm, Bar, Method};
use phlite::columns::ColumnRepresentation;
use phlite::fields::Z2;
use phlite_complexes::complex::{Chain, RawChain};

use super::{Construction, Filtration};

/// An explicit cubical complex: every cell present (entering at a finite value), its dimension, coordinates on the
/// doubled grid, value (in the sublevel filtration of the possibly negated image) and oriented boundary.
#[derive(Debug, Clone, Default)]
pub struct ExplicitCubical {
    pub dimensions: Vec<usize>,
    pub coordinates: Vec<Vec<u32>>,
    pub values: Vec<f64>,
    /// Faces of each cell, as positions in these lists, with coefficients ±1
    pub boundary: Vec<Vec<(usize, i64)>>,
}

/// The cubical complex of the image with this `shape` (pixels in C order), with the axes in `periodic` wrapping round.
pub fn explicit_cubical(
    pixels: &[f64],
    shape: &[usize],
    periodic: &[bool],
    construction: Construction,
    filtration: Filtration,
) -> ExplicitCubical {
    let d = shape.len();
    assert_eq!(periodic.len(), d);
    if shape.contains(&0) {
        return ExplicitCubical::default();
    }
    let pixel = |p: f64| match filtration {
        _ if p.is_nan() => f64::INFINITY,
        Filtration::Sublevel => p,
        Filtration::Superlevel => -p,
    };
    // Cells along each axis of the doubled grid
    let sizes: Vec<usize> = (0..d)
        .map(|a| match (construction, periodic[a]) {
            (_, true) => 2 * shape[a],
            (Construction::T, false) => 2 * shape[a] + 1,
            (Construction::V, false) => 2 * shape[a] - 1,
        })
        .collect();
    let pixel_strides: Vec<usize> = (0..d).map(|a| shape[a + 1..].iter().product()).collect();
    let n_cells: usize = sizes.iter().product();
    // The pixels around a coordinate along an axis: the top cells (T: odd coordinates) or vertices (V: even) next to
    // it, as pixel coordinates (none past a non-periodic edge)
    let around = |a: usize, c: usize| -> Vec<usize> {
        let (n, size) = (shape[a], sizes[a]);
        let neighbours: Vec<usize> = match construction {
            Construction::T if c % 2 == 1 => vec![c],
            Construction::V if c % 2 == 0 => vec![c],
            _ => {
                let mut out = vec![];
                if c > 0 {
                    out.push(c - 1);
                } else if periodic[a] {
                    out.push(size - 1);
                }
                if c + 1 < size {
                    out.push(c + 1);
                } else if periodic[a] {
                    out.push(0);
                }
                out
            }
        };
        neighbours
            .into_iter()
            .map(|x| match construction {
                Construction::T => (x - 1) / 2,
                Construction::V => x / 2,
            })
            .filter(|&p| p < n)
            .collect()
    };
    let decode = |mut i: usize| -> Vec<usize> {
        let mut c = vec![0; d];
        for a in (0..d).rev() {
            c[a] = i % sizes[a];
            i /= sizes[a];
        }
        c
    };
    let encode = |c: &[usize]| c.iter().zip(&sizes).fold(0, |acc, (&x, &s)| acc * s + x);
    // Values: T takes the least of the top cells around, V the greatest of the vertices
    let mut values = vec![f64::INFINITY; n_cells];
    for (i, value) in values.iter_mut().enumerate() {
        let c = decode(i);
        let choices: Vec<Vec<usize>> = (0..d).map(|a| around(a, c[a])).collect();
        if choices.iter().any(Vec::is_empty) {
            continue;
        }
        let mut best = match construction {
            Construction::T => f64::INFINITY,
            Construction::V => f64::NEG_INFINITY,
        };
        // Every pixel around: one choice along each axis (a mixed-radix count)
        let total: usize = choices.iter().map(Vec::len).product();
        for combination in 0..total {
            let mut rest = combination;
            let mut p = 0;
            for a in (0..d).rev() {
                let k = choices[a].len();
                p += choices[a][rest % k] * pixel_strides[a];
                rest /= k;
            }
            let v = pixel(pixels[p]);
            best = match construction {
                Construction::T => best.min(v),
                Construction::V => best.max(v),
            };
        }
        *value = best;
    }
    // Keep the cells present, with the oriented boundary: sum over the spanned axes (the i-th spanned axis signed
    // (-1)^i) of the upper face minus the lower
    // (Listed by dimension, so that faces come before their cofaces, which ties in value need)
    let dimension_of = |i: usize| decode(i).iter().filter(|&&x| x % 2 == 1).count();
    let mut present: Vec<usize> = (0..n_cells).filter(|&i| values[i] < f64::INFINITY).collect();
    present.sort_by_key(|&i| (dimension_of(i), i));
    let mut position = vec![usize::MAX; n_cells];
    let mut out = ExplicitCubical::default();
    for i in present {
        position[i] = out.values.len();
        out.values.push(values[i]);
        let c = decode(i);
        out.dimensions.push(c.iter().filter(|&&x| x % 2 == 1).count());
        out.coordinates.push(c.iter().map(|&x| x as u32).collect());
    }
    for c in out.coordinates.iter() {
        let c: Vec<usize> = c.iter().map(|&x| x as usize).collect();
        let mut faces: Vec<(usize, i64)> = vec![];
        let mut spanned = 0;
        for a in 0..d {
            if c[a] % 2 == 1 {
                let sign = if spanned % 2 == 0 { 1 } else { -1 };
                spanned += 1;
                for (step_up, s) in [(true, sign), (false, -sign)] {
                    let mut f = c.clone();
                    f[a] = if step_up { (c[a] + 1) % sizes[a] } else { c[a] - 1 };
                    let j = position[encode(&f)];
                    debug_assert_ne!(j, usize::MAX, "a face of a present cell is present");
                    match faces.iter_mut().find(|(k, _)| *k == j) {
                        Some((_, x)) => *x += s,
                        None => faces.push((j, s)),
                    }
                }
            }
        }
        faces.retain(|&(_, x)| x != 0);
        out.boundary.push(faces);
    }
    out
}

/// A bar of a periodic cubical complex: dimension, the cells (coordinates on the doubled grid) of its birth and death,
/// and their values (sublevel, of the possibly negated image).
pub type PeriodicBar = Bar<(usize, Vec<u32>), ordered_float::NotNan<f64>, Z2>;

/// The barcode of the (partly) periodic cubical complex in dimensions `0..=max_dim`, over Z/2, by cohomology with
/// `algorithm` (bars on cells as `(dimension, coordinates)`; values sublevel, of the negated image for superlevel
/// sets). Zero-length bars are left out.
pub fn periodic_barcode<R: ColumnRepresentation>(
    pixels: &[f64],
    shape: &[usize],
    periodic: &[bool],
    construction: Construction,
    filtration: Filtration,
    max_dim: usize,
    algorithm: Algorithm<'_, R>,
) -> Vec<PeriodicBar> {
    let complex = explicit_cubical(pixels, shape, periodic, construction, filtration);
    if complex.values.is_empty() {
        return vec![];
    }
    let mut raw = RawChain { dimensions: vec![], indptr: vec![0], indices: vec![], coefficients: Some(vec![]) };
    for (k, faces) in complex.dimensions.iter().zip(&complex.boundary) {
        raw.dimensions.push(*k as i64);
        for &(f, x) in faces {
            raw.indices.push(f as i64);
            raw.coefficients.as_mut().expect("set").push(x);
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    let filtered = Chain::<Z2>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(complex.values.clone())))
        .expect("a cubical complex, faces no later than their cofaces");
    let cell = |c: phlite_complexes::complex::Cell| (complex.dimensions[c.0 as usize], complex.coordinates[c.0 as usize].clone());
    filtered
        .barcode(Some(max_dim), Method::Cohomology, algorithm, false)
        .into_iter()
        .map(|bar| Bar {
            birth: cell(bar.birth),
            birth_value: bar.birth_value,
            death: bar.death.map(|(c, v)| (cell(c), v)),
            representative: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    fn betti(pixels: &[f64], shape: &[usize], periodic: &[bool], construction: Construction) -> Vec<usize> {
        let bars = periodic_barcode(
            pixels,
            shape,
            periodic,
            construction,
            Filtration::Sublevel,
            shape.len(),
            Algorithm::Clearing(Clearing::new()),
        );
        let mut out = vec![0; shape.len() + 1];
        for bar in bars.iter().filter(|b| b.death.is_none()) {
            out[bar.birth.0] += 1;
        }
        out
    }

    #[test]
    fn tori_and_cylinders() {
        // A constant image: periodic along every axis is a torus T^d; along some, a product of circles and intervals
        for construction in [Construction::T, Construction::V] {
            assert_eq!(betti(&[0.0; 12], &[3, 4], &[true, true], construction), vec![1, 2, 1]);
            assert_eq!(betti(&[0.0; 12], &[3, 4], &[true, false], construction), vec![1, 1, 0]);
            assert_eq!(betti(&[0.0; 12], &[3, 4], &[false, false], construction), vec![1, 0, 0]);
            assert_eq!(betti(&[0.0; 24], &[2, 3, 4], &[true, true, true], construction), vec![1, 3, 3, 1]);
            assert_eq!(betti(&[0.0; 5], &[5], &[true], construction), vec![1, 1]);
        }
    }

    #[test]
    fn matches_the_implicit_complex_without_wrapping() {
        // No periodic axis: the same barcode as the implicit complex
        let mut state = 9u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..40 {
            let shape = [1 + next(5) as usize, 1 + next(5) as usize];
            let pixels: Vec<f64> = (0..shape[0] * shape[1]).map(|_| next(6) as f64).collect();
            for construction in [Construction::T, Construction::V] {
                let explicit = periodic_barcode(
                    &pixels,
                    &shape,
                    &[false, false],
                    construction,
                    Filtration::Sublevel,
                    1,
                    Algorithm::Clearing(Clearing::new()),
                );
                let complex = super::super::CubicalComplex::new(pixels.clone(), shape, construction, Filtration::Sublevel)
                    .unwrap();
                let implicit = super::super::cubical_barcode(
                    &complex,
                    1,
                    super::super::Method::Cohomology,
                    Algorithm::Clearing(Clearing::new()),
                    false,
                );
                let summary = |bars: Vec<(usize, f64, f64)>| {
                    let mut bars = bars;
                    bars.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    bars
                };
                let a = summary(
                    explicit
                        .iter()
                        .map(|b| (b.birth.0, b.birth_value.into_inner(), b.death.as_ref().map_or(f64::INFINITY, |d| d.1.into_inner())))
                        .collect(),
                );
                let b = summary(
                    implicit
                        .iter()
                        .map(|b| {
                            (complex.dimension(b.birth), b.birth_value.into_inner(), b.death.map_or(f64::INFINITY, |d| d.1.into_inner()))
                        })
                        .collect(),
                );
                assert_eq!(a, b, "{shape:?} {pixels:?} {construction:?}");
            }
        }
    }
}
