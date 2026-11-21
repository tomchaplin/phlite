//! Čech complexes of points in any dimension, as GUDHI's `CechComplex`: a simplex at the radius of the minimal
//! enclosing ball of its points (the radius at which the balls about them first all meet).
//!
//! By the nerve theorem the Čech filtration has the barcode of the union of growing balls; in the plane and in space
//! the Delaunay–Čech and alpha filtrations ([`crate::alpha`]) have the same barcode with far fewer simplices, so this is
//! for points in higher dimensions, or when the Čech complex itself is wanted. A Čech simplex of radius `r` has
//! diameter at most `2r`, so the candidates are the cliques of the Rips complex at `2r`, and each is given its minimal
//! enclosing ball: the smallest of the circumscribed balls of subsets of its points (in their affine hull) that
//! contains them all, found by trying every subset (simplices are small). In floating point; a simplex is never made
//! to enter before its faces.

use ordered_float::NotNan;
use phlite::barcode::{Algorithm, Bar, Method};
use phlite::columns::ColumnRepresentation;
use phlite::fields::Z2;
use phlite_complexes::complex::{Cell, Chain, RawChain};

/// The squared radius of the ball circumscribing `points` in their affine hull, and its centre, or `None` if they are
/// affinely dependent (numerically).
fn circumball(points: &[&[f64]]) -> Option<(f64, Vec<f64>)> {
    let p0 = points[0];
    let m = points.len() - 1;
    if m == 0 {
        return Some((0.0, p0.to_vec()));
    }
    let v: Vec<Vec<f64>> = points[1..].iter().map(|p| p.iter().zip(p0).map(|(a, b)| a - b).collect()).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    // Gram system: G λ = b, G_ij = v_i . v_j, b_i = |v_i|^2 / 2
    let mut g: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row: Vec<f64> = (0..m).map(|j| dot(&v[i], &v[j])).collect();
            row.push(dot(&v[i], &v[i]) / 2.0);
            row
        })
        .collect();
    let scale = (0..m).map(|i| g[i][i]).fold(0.0, f64::max);
    for c in 0..m {
        let pivot = (c..m).max_by(|&a, &b| g[a][c].abs().total_cmp(&g[b][c].abs()))?;
        if g[pivot][c].abs() <= 1e-12 * scale {
            return None;
        }
        g.swap(c, pivot);
        for r in 0..m {
            if r != c {
                let f = g[r][c] / g[c][c];
                for k in c..=m {
                    g[r][k] -= f * g[c][k];
                }
            }
        }
    }
    let lambda: Vec<f64> = (0..m).map(|i| g[i][m] / g[i][i]).collect();
    let mut centre = p0.to_vec();
    for (l, vi) in lambda.iter().zip(&v) {
        for (c, x) in centre.iter_mut().zip(vi) {
            *c += l * x;
        }
    }
    let r2 = centre.iter().zip(p0).map(|(c, p)| (c - p) * (c - p)).sum();
    Some((r2, centre))
}

/// The squared radius of the minimal enclosing ball of `points` (a few: every subset is tried).
pub fn minimal_enclosing_ball(points: &[&[f64]]) -> f64 {
    let k = points.len();
    if k <= 1 {
        return 0.0;
    }
    let mut best = f64::INFINITY;
    for subset in 1u32..(1 << k) {
        let chosen: Vec<&[f64]> = (0..k).filter(|i| subset >> i & 1 == 1).map(|i| points[i]).collect();
        let Some((r2, centre)) = circumball(&chosen) else { continue };
        if r2 >= best {
            continue;
        }
        let tolerance = r2 * 1e-12 + f64::MIN_POSITIVE;
        let encloses = (0..k).filter(|i| subset >> i & 1 == 0).all(|i| {
            points[i].iter().zip(&centre).map(|(p, c)| (p - c) * (p - c)).sum::<f64>() <= r2 + tolerance
        });
        if encloses {
            best = r2;
        }
    }
    best
}

/// The Čech complex: every simplex of dimension at most `dimension` whose minimal enclosing ball has radius at most
/// `max_radius` (all, if `None`), with its squared radius, faces before cofaces (by dimension, then vertices).
pub fn cech_simplices(points: &[Vec<f64>], dimension: usize, max_radius: Option<f64>) -> Vec<(f64, Vec<u32>)> {
    let n = points.len();
    let limit2 = max_radius.map_or(f64::INFINITY, |r| r * r);
    let distance2 = |a: usize, b: usize| points[a].iter().zip(&points[b]).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    // Edges of diameter at most 2 r (an edge's ball has radius half its length)
    let neighbours: Vec<Vec<u32>> = (0..n)
        .map(|a| ((a + 1)..n).filter(|&b| distance2(a, b) / 4.0 <= limit2).map(|b| b as u32).collect())
        .collect();
    let mut out: Vec<(f64, Vec<u32>)> = (0..n as u32).map(|v| (0.0, vec![v])).collect();
    let mut current: Vec<usize> = (0..n).collect();
    for _ in 0..dimension {
        let mut next = vec![];
        for &i in &current {
            let simplex = out[i].1.clone();
            let face_value = out[i].0;
            let last = *simplex.last().expect("non-empty") as usize;
            for &w in &neighbours[last] {
                if !simplex.iter().all(|&v| neighbours[v as usize].binary_search(&w).is_ok()) {
                    continue;
                }
                let mut longer = simplex.clone();
                longer.push(w);
                let coords: Vec<&[f64]> = longer.iter().map(|&v| points[v as usize].as_slice()).collect();
                // Never before a face (the faces' radii are exact lower bounds; this one is the newest)
                let r2 = minimal_enclosing_ball(&coords).max(face_value);
                if r2 <= limit2 {
                    out.push((r2, longer));
                    next.push(out.len() - 1);
                }
            }
        }
        current = next;
    }
    // A face may have been reached from another coface with a smaller rounding: raise each simplex to its faces
    let index: std::collections::HashMap<Vec<u32>, usize> =
        out.iter().enumerate().map(|(i, (_, s))| (s.clone(), i)).collect();
    for i in 0..out.len() {
        let s = out[i].1.clone();
        if s.len() > 1 {
            let faces_max = (0..s.len())
                .map(|j| {
                    let mut f = s.clone();
                    f.remove(j);
                    out[index[&f]].0
                })
                .fold(0.0, f64::max);
            out[i].0 = out[i].0.max(faces_max);
        }
    }
    out
}

/// A bar of the Čech filtration, on simplices (as their points).
pub type CechBar = Bar<Vec<u32>, NotNan<f64>, Z2>;

/// The barcode of the Čech filtration of `points` in dimensions `0..=max_dim`, up to `max_radius` (bars still alive
/// there essential), with values radii (or with `squared`, squared radii); reduced as an explicit complex. Zero-length
/// bars are left out.
pub fn cech_barcode<R: ColumnRepresentation>(
    points: &[Vec<f64>],
    max_dim: usize,
    max_radius: Option<f64>,
    squared: bool,
    algorithm: Algorithm<'_, R>,
) -> Vec<CechBar> {
    let simplices: Vec<(f64, Vec<u32>)> = cech_simplices(points, max_dim + 1, max_radius)
        .into_iter()
        .map(|(r2, s)| (if squared { r2 } else { r2.sqrt() }, s))
        .collect();
    simplicial_barcode(&simplices, max_dim, algorithm)
}

/// The barcode, in dimensions `0..=max_dim`, of the filtered simplicial complex given by its simplices with their
/// values (every face of each included, faces never later than their cofaces), reduced as an explicit complex over
/// Z/2. Zero-length bars are left out.
pub fn simplicial_barcode<R: ColumnRepresentation>(
    simplices: &[(f64, Vec<u32>)],
    max_dim: usize,
    algorithm: Algorithm<'_, R>,
) -> Vec<CechBar> {
    if simplices.is_empty() {
        return vec![];
    }
    // Faces first (by dimension), so that ties in value are broken the right way
    let mut order: Vec<usize> = (0..simplices.len()).collect();
    order.sort_by_key(|&i| (simplices[i].1.len(), i));
    let simplices: Vec<(f64, Vec<u32>)> = order.into_iter().map(|i| simplices[i].clone()).collect();
    let index: std::collections::HashMap<&[u32], usize> =
        simplices.iter().enumerate().map(|(i, (_, s))| (s.as_slice(), i)).collect();
    let mut raw = RawChain { dimensions: vec![], indptr: vec![0], indices: vec![], coefficients: None };
    for (_, s) in &simplices {
        raw.dimensions.push(s.len() as i64 - 1);
        if s.len() > 1 {
            for j in 0..s.len() {
                let f: Vec<u32> = s.iter().enumerate().filter(|&(i, _)| i != j).map(|(_, &v)| v).collect();
                raw.indices.push(index[f.as_slice()] as i64);
            }
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    let values = simplices.iter().map(|(v, _)| *v).collect();
    let complex = Chain::<Z2>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(values)))
        .expect("a simplicial complex, faces no later than their cofaces");
    let simplex = |c: Cell| simplices[c.0 as usize].1.clone();
    complex
        .barcode(Some(max_dim), Method::Cohomology, algorithm, false)
        .into_iter()
        .map(|bar| Bar {
            birth: simplex(bar.birth),
            birth_value: bar.birth_value,
            death: bar.death.map(|(c, v)| (simplex(c), v)),
            representative: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    #[test]
    fn enclosing_balls() {
        let p = |x: &[f64]| x.to_vec();
        let (a, b, c) = (p(&[0.0, 0.0]), p(&[2.0, 0.0]), p(&[1.0, 0.1]));
        // An obtuse triangle's ball is its longest edge's
        assert!((minimal_enclosing_ball(&[&a, &b, &c]) - 1.0).abs() < 1e-12);
        // An equilateral triangle's is its circumcircle
        let d = p(&[1.0, 3f64.sqrt()]);
        assert!((minimal_enclosing_ball(&[&a, &b, &d]) - 4.0 / 3.0).abs() < 1e-12);
        // A regular tetrahedron in R^4 (on four axes): circumradius^2 = 3/4 · (edge^2 / 2) ... = 3/8 · 2
        let e: Vec<Vec<f64>> = (0..4).map(|i| (0..4).map(|j| f64::from(u8::from(i == j))).collect()).collect();
        let refs: Vec<&[f64]> = e.iter().map(Vec::as_slice).collect();
        assert!((minimal_enclosing_ball(&refs) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn circle_and_sphere() {
        // Points on a circle in R^4: one H_1 bar; on a 2-sphere: one H_2 bar
        let circle: Vec<Vec<f64>> =
            (0..12).map(|i| { let t = i as f64 * std::f64::consts::TAU / 12.0; vec![t.cos(), t.sin(), 0.0, 0.0] }).collect();
        let bars = cech_barcode(&circle, 1, None, false, Algorithm::Clearing(Clearing::new()));
        let h1: Vec<&CechBar> = bars.iter().filter(|b| b.birth.len() == 2).collect();
        assert_eq!(h1.len(), 1);
        // Born when neighbours meet (half the chord), dies at the circumradius 1
        let death = h1[0].death.as_ref().unwrap().1.into_inner();
        assert!((death - 1.0).abs() < 1e-9, "{death}");
        let octahedron: Vec<Vec<f64>> = (0..6)
            .map(|i| { let mut v = vec![0.0; 3]; v[i / 2] = if i % 2 == 0 { 1.0 } else { -1.0 }; v })
            .collect();
        let bars = cech_barcode(&octahedron, 2, None, false, Algorithm::Clearing(Clearing::new()));
        assert_eq!(bars.iter().filter(|b| b.birth.len() == 3).count(), 1);
    }
}
