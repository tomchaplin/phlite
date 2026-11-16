//! Listing a small flag (Rips) complex and reducing it explicitly, as `phlite_complexes` does, rather than computing
//! its coboundary implicitly.
//!
//! The implicit route ([`rips_barcode_with`](super::rips_barcode_with)) relies on apparent pairs settling nearly every
//! column, which they do for metrics (a point cloud's distances, a graph's shortest paths). For arbitrary weights
//! (a sparse matrix of anything, a graph's own edge weights) few columns are apparent, every column is recomputed
//! each time it is added, and listing the complex once is far faster: on G(300, 0.1) with random weights, 11 ms
//! against 2.5 s.

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};

use super::Distances;

/// The cliques of `graph` (the simplices of its flag complex: Rips of the distances, whose missing entries are
/// missing edges) of dimension at most `top` and value at most `threshold`,
/// with their values, nodes first; or `None` if there are more than `budget`.
///
/// A simplex above the threshold has every coface above it too, so they aren't enumerated.
pub fn cliques_within<D: Distances + ?Sized>(
    graph: &D,
    top: usize,
    threshold: NotNan<f64>,
    budget: usize,
) -> Option<Vec<(NotNan<f64>, Vec<u32>)>> {
    let n = graph.n_points();
    let mut all: Vec<(NotNan<f64>, Vec<u32>)> = (0..n as u32)
        .map(|v| (graph.birth(v as usize), vec![v]))
        .filter(|(value, _)| *value <= threshold)
        .collect();
    if all.len() > budget {
        return None;
    }
    let mut current: Vec<usize> = (0..all.len()).collect();
    for _ in 0..top {
        let mut next = vec![];
        for &i in &current {
            let (value, simplex) = &all[i];
            let (value, simplex) = (*value, simplex.clone());
            let last = *simplex.last().expect("a simplex has a node") as usize;
            for w in graph.neighbours(last, last + 1) {
                // The new edges' values (each at least both nodes' births)
                let mut longer_value = value;
                for &v in &simplex {
                    longer_value = longer_value.max(graph.distance(v as usize, w));
                }
                // (A missing edge is at infinity: not a clique, whatever the threshold)
                if longer_value.is_finite() && longer_value <= threshold {
                    let mut longer = simplex.clone();
                    longer.push(w as u32);
                    all.push((longer_value, longer));
                    next.push(all.len() - 1);
                    if all.len() > budget {
                        return None;
                    }
                }
            }
        }
        current = next;
    }
    Some(all)
}

/// The barcode of the flag complex given by its simplices (with their values, as [`cliques_within`] gives them, every
/// face of each included) in dimensions `0..=max_dim`, by listing them as an explicit complex and reducing it with
/// `phlite_complexes`: much faster than [`rips_barcode_with`](super::rips_barcode_with) when there are few simplices
/// (a sparse graph, or a low threshold) and the distances aren't a metric (so that apparent pairs settle few columns),
/// as each column is built once rather than recomputed and pivot columns storing R reduce it. Bars are on the simplices' nodes; those still
/// alive at the largest value listed are essential.
pub fn explicit_flag_barcode<R: ColumnRepresentation>(
    simplices: &[(NotNan<f64>, Vec<u32>)],
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<phlite::barcode::Bar<Vec<u32>, NotNan<f64>, phlite::fields::Z2>> {
    use phlite_complexes::complex::{Chain, RawChain};
    use std::collections::HashMap;
    // In filtration order, faces first (a face's value is at most its coface's, and it has fewer nodes)
    let mut order: Vec<&(NotNan<f64>, Vec<u32>)> = simplices.iter().collect();
    order.sort_by(|a, b| (a.0, a.1.len(), &a.1).cmp(&(b.0, b.1.len(), &b.1)));
    let index: HashMap<&[u32], i64> = order.iter().enumerate().map(|(i, (_, s))| (s.as_slice(), i as i64)).collect();
    let mut raw = RawChain { dimensions: Vec::with_capacity(order.len()), indptr: vec![0], indices: vec![], coefficients: None };
    let mut face = Vec::new();
    for (_, s) in &order {
        raw.dimensions.push(s.len() as i64 - 1);
        if s.len() > 1 {
            for i in 0..s.len() {
                face.clear();
                face.extend(s.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v));
                raw.indices.push(index[face.as_slice()]);
            }
        }
        raw.indptr.push(raw.indices.len() as i64);
    }
    let values: Vec<f64> = order.iter().map(|(v, _)| v.into_inner()).collect();
    let complex = Chain::<phlite::fields::Z2>::from_raw(raw)
        .expect("a flag complex is a valid chain complex")
        .filter(Some(values))
        .expect("faces enter no later than their cofaces");
    let cell = |c: phlite_complexes::complex::Cell| order[c.0 as usize].1.clone();
    complex
        .barcode(Some(max_dim), method, algorithm, representatives)
        .into_iter()
        .map(|bar| phlite::barcode::Bar {
            birth: cell(bar.birth),
            birth_value: bar.birth_value,
            death: bar.death.map(|(c, v)| (cell(c), v)),
            representative: bar.representative.map(|rep| rep.into_iter().map(|(a, c)| (a, cell(c))).collect()),
        })
        .collect()
}

/// Whether the distances look like a metric, judging by the triangle inequality on up to `samples` triangles of the
/// graph of finite distances (found in order of their first node, so deterministically): `false` if more than 1% of
/// them break it. Without triangles (nothing for apparent pairs to work on in dimension 2) it says `true`.
///
/// This is what decides whether Rips' implicit computation will be fast: for a metric nearly every column is an
/// apparent pair, but for arbitrary weights few are, and listing the complex ([`cliques_within`] and
/// [`explicit_flag_barcode`]) is far faster.
pub fn looks_metric<D: Distances + ?Sized>(distances: &D, samples: usize) -> bool {
    let n = distances.n_points();
    let (mut seen, mut broken) = (0usize, 0usize);
    'search: for u in 0..n {
        for v in distances.neighbours(u, u + 1) {
            let uv = distances.distance(u, v);
            if !uv.is_finite() {
                continue;
            }
            for w in distances.neighbours(v, v + 1) {
                let (vw, uw) = (distances.distance(v, w), distances.distance(u, w));
                if !vw.is_finite() || !uw.is_finite() {
                    continue;
                }
                // The longest side is at most the sum of the others (up to rounding)
                let longest = uv.max(vw).max(uw).into_inner();
                let others = (uv + vw + uw).into_inner() - longest;
                if longest > others * (1.0 + 1e-9) {
                    broken += 1;
                }
                seen += 1;
                if seen >= samples {
                    break 'search;
                }
            }
        }
    }
    broken * 100 <= seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rips::SparseDistances;

    #[test]
    fn metrics_and_arbitrary_weights() {
        // A unit square's corners: a metric (the diagonals are sqrt 2)
        let corners = [(0.0f64, 0.0f64), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut edges = vec![];
        for i in 0..4 {
            for j in i + 1..4 {
                let (a, b) = (corners[i], corners[j]);
                edges.push((i, j, ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()));
            }
        }
        assert!(looks_metric(&SparseDistances::from_edges(4, edges.clone()).unwrap(), 1000));
        // Make a diagonal far longer than the way round the square
        edges[1].2 = 10.0;
        assert!(!looks_metric(&SparseDistances::from_edges(4, edges).unwrap(), 1000));
        // No triangles: nothing to judge by
        assert!(looks_metric(&SparseDistances::from_edges(3, vec![(0, 1, 1.0), (1, 2, 5.0)]).unwrap(), 1000));
    }
}
