//! Extended persistence (Cohen-Steiner, Edelsbrunner and Harer) of a function on the nodes of a graph, or of its flag
//! complex.
//!
//! Ordinary persistence of the lower-star filtration of `f` leaves the homology of the whole complex essential: a
//! loop of a graph is born, and never dies. Extended persistence follows the sublevel sets `K_{f ≤ a}` upwards, then
//! the pairs `(K, K_{f ≥ a})` downwards, so every class dies, and each bar is one of four kinds:
//! * **ordinary**: born and dying on the way up (as in ordinary persistence);
//! * **relative**: born and dying on the way down (the ordinary bars of `-f`, up a dimension);
//! * **extended**: born on the way up and dying on the way down, so recording where a class of the whole complex is
//!   born and dies: a component from its minimum to its maximum (**Ext⁺**, birth below death), a loop from its highest
//!   to its lowest point (**Ext⁻**, birth above death).
//!
//! It is computed, as GUDHI does, by coning: the complex `K`, then a cone vertex `ω`, and the cone `ω * σ` on every
//! simplex `σ` (whose boundary is `σ - ω * ∂σ`), entering in decreasing order of `min f(σ)`. The cone vertex enters
//! first, so its component is the only essential class, and is left out. Values are exact (GUDHI rescales them, so
//! its are within rounding of these), and bars of length zero are omitted.

use std::collections::HashMap;

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use phlite_complexes::{
    complex::{Chain, RawChain},
    field::IntField,
};

use crate::{undirected::Graph, NetworkError};

/// The four kinds of extended persistence bar (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Ordinary,
    Relative,
    ExtendedPlus,
    ExtendedMinus,
}

/// A bar of extended persistence: its kind, dimension (for a relative bar, its dimension in the cone, one more than
/// the simplices it comes from, as GUDHI reports it) and its birth and death values (in `f`'s units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedBar {
    pub kind: Kind,
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
}

/// The cliques of `graph` with at most `top + 1` nodes (so of dimension at most `top`), each as its sorted nodes,
/// nodes first.
pub fn cliques(graph: &Graph, top: usize) -> Vec<Vec<u32>> {
    crate::undirected::cliques_within(graph, top, phlite::values::INFINITY, usize::MAX)
        .expect("without a budget, every clique is listed")
        .into_iter()
        .map(|(_, simplex)| simplex)
        .collect()
}

/// The extended persistence of the lower-star filtration of `values` on the flag complex of the graph on
/// `n_vertices` nodes with these `(u, v)` edges, up to simplices of dimension `top` (1: the graph itself), over the
/// field `CF`, computed by `method` and `algorithm`. Bars in no particular order, without those of length zero.
///
/// # Errors
/// As [`Graph::new`]: an edge out of range or a self-loop, or not one finite value per node.
pub fn extended_persistence<CF: IntField, R: ColumnRepresentation>(
    n_vertices: u32,
    edges: &[(u32, u32)],
    values: &[f64],
    top: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
) -> Result<Vec<ExtendedBar>, NetworkError> {
    let weighted: Vec<_> = edges.iter().map(|&(u, v)| (u, v, f64::NEG_INFINITY)).collect();
    let graph = Graph::new(n_vertices, &weighted, Some(values))?;
    let f = |v: u32| values[v as usize];
    let simplices = cliques(&graph, top);
    let up = |s: &[u32]| s.iter().map(|&v| NotNan::new(f(v)).expect("finite")).max().expect("a node");
    let down = |s: &[u32]| s.iter().map(|&v| NotNan::new(f(v)).expect("finite")).min().expect("a node");

    // The order of the cells: the cone vertex, then K by increasing max f, then the cones by decreasing min f, ties
    // by dimension (so faces first) and then by nodes
    let mut ascending: Vec<&Vec<u32>> = simplices.iter().collect();
    ascending.sort_by(|a, b| (up(a), a.len(), *a).cmp(&(up(b), b.len(), *b)));
    let mut descending: Vec<&Vec<u32>> = simplices.iter().collect();
    descending.sort_by(|a, b| (std::cmp::Reverse(down(a)), a.len(), *a).cmp(&(std::cmp::Reverse(down(b)), b.len(), *b)));
    let m = simplices.len();
    let in_k: HashMap<&[u32], i64> = ascending.iter().enumerate().map(|(i, s)| (s.as_slice(), 1 + i as i64)).collect();
    let in_cone: HashMap<&[u32], i64> =
        descending.iter().enumerate().map(|(i, s)| (s.as_slice(), 1 + m as i64 + i as i64)).collect();
    let faces = |s: &[u32]| -> Vec<Vec<u32>> {
        (0..s.len()).map(|i| s.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &v)| v).collect()).collect()
    };

    // The boundary matrix, cell by cell
    let mut raw = RawChain { dimensions: vec![0], indptr: vec![0, 0], indices: vec![], coefficients: Some(vec![]) };
    let sign = |i: usize| if i % 2 == 0 { 1 } else { -1 };
    let push = |raw: &mut RawChain, dimension: usize, entries: Vec<(i64, i64)>| {
        raw.dimensions.push(dimension as i64);
        for (cell, coefficient) in entries {
            raw.indices.push(cell);
            raw.coefficients.as_mut().expect("coefficients").push(coefficient);
        }
        raw.indptr.push(raw.indices.len() as i64);
    };
    for s in &ascending {
        let entries = if s.len() == 1 {
            vec![]
        } else {
            faces(s).iter().enumerate().map(|(i, face)| (in_k[face.as_slice()], sign(i))).collect()
        };
        push(&mut raw, s.len() - 1, entries);
    }
    for s in &descending {
        // ∂(ω * s) = s - ω * ∂s, and ∂(ω * v) = v - ω
        let mut entries = vec![(in_k[s.as_slice()], 1)];
        if s.len() == 1 {
            entries.push((0, -1));
        } else {
            entries.extend(faces(s).iter().enumerate().map(|(i, face)| (in_cone[face.as_slice()], -sign(i))));
        }
        push(&mut raw, s.len(), entries);
    }

    let chain = Chain::<CF>::from_raw(raw).expect("the cone complex is a valid chain complex");
    // Cells enter in the order built, faces first
    let complex = chain.filter(None).expect("faces enter before their cofaces");
    let bars = complex.barcode(None, method, algorithm, false);

    // Each bar's kind and values, from the parts its cells are in
    enum Part<'a> {
        Cone,
        Up(&'a [u32]),
        Down(&'a [u32]),
    }
    let part = |cell: usize| {
        if cell == 0 {
            Part::Cone
        } else if cell <= m {
            Part::Up(ascending[cell - 1])
        } else {
            Part::Down(descending[cell - 1 - m])
        }
    };
    let mut out = vec![];
    for bar in bars {
        let Some((death, _)) = bar.death else {
            // Only the cone vertex's component is essential
            debug_assert_eq!(bar.birth.0, 0);
            continue;
        };
        let (kind, dimension, birth, death) = match (part(bar.birth.0 as usize), part(death.0 as usize)) {
            (Part::Up(b), Part::Up(d)) => (Kind::Ordinary, b.len() - 1, *up(b), *up(d)),
            (Part::Down(b), Part::Down(d)) => (Kind::Relative, b.len(), *down(b), *down(d)),
            (Part::Up(b), Part::Down(d)) => {
                let (birth, death) = (*up(b), *down(d));
                let kind = if birth < death { Kind::ExtendedPlus } else { Kind::ExtendedMinus };
                (kind, b.len() - 1, birth, death)
            }
            _ => unreachable!("the cone vertex is never paired, and cones come after K"),
        };
        if birth != death {
            out.push(ExtendedBar { kind, dimension, birth, death });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::fields::Z2;
    use phlite::reduction::Clearing;

    fn run(n: u32, edges: &[(u32, u32)], values: &[f64]) -> Vec<(Kind, usize, f64, f64)> {
        let mut bars: Vec<_> = extended_persistence::<Z2, _>(
            n,
            edges,
            values,
            1,
            Method::Cohomology,
            Algorithm::Clearing(Clearing::new()),
        )
        .unwrap()
        .into_iter()
        .map(|b| (b.kind, b.dimension, b.birth, b.death))
        .collect();
        bars.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bars
    }

    #[test]
    fn a_cycle() {
        // GUDHI: Ord (1, 2) in H0, Rel (2, 1) in H1, Ext+ (0, 3) in H0, Ext- (3, 0) in H1
        let bars = run(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], &[0.0, 2.0, 1.0, 3.0]);
        assert_eq!(
            bars,
            vec![
                (Kind::Ordinary, 0, 1.0, 2.0),
                (Kind::Relative, 1, 2.0, 1.0),
                (Kind::ExtendedPlus, 0, 0.0, 3.0),
                (Kind::ExtendedMinus, 1, 3.0, 0.0),
            ]
        );
    }

    #[test]
    fn two_components() {
        // Each component is an Ext+ bar from its minimum to its maximum
        let bars = run(4, &[(0, 1), (2, 3)], &[0.0, 1.0, 5.0, 7.0]);
        assert_eq!(bars, vec![(Kind::ExtendedPlus, 0, 0.0, 1.0), (Kind::ExtendedPlus, 0, 5.0, 7.0)]);
    }
}
