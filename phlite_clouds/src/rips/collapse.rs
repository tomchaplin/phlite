//! Edge collapse of a flag filtration: fewer edges, the same barcode.
//!
//! An edge `uv` of a graph is *dominated* by a vertex `w` (not `u` or `v`) when every common neighbour of `u` and `v`
//! is `w` or a neighbour of `w`. Removing a dominated edge doesn't change the homotopy type of the flag complex
//! (Boissonnat and Pritam, *Edge collapse and persistence of flag complexes*, SoCG 2020), and in a filtration an
//! edge that is dominated from the time it appears until some later time can be moved to that time without changing
//! the barcode (Glisse and Pritam, *Swap, shift and trim to edge collapse a filtration*, SoCG 2022). An edge that stays
//! dominated forever can be removed.
//!
//! [`collapse_edges`] does this in one backward pass over the edges, latest first, as Glisse and Pritam do, with the
//! word-parallel tests of Perseus 6 (Nanda; `collapse.hpp`): the graph at the time of the edge being processed (the
//! edges not yet processed, which keep their times) and the final graph (every edge not removed, at its new time) are
//! kept as bitsets, so that the common neighbourhood of an edge and the test for a dominator are a few word operations
//! per vertex. An edge whose dominator is adjacent, at the edge's time already, to every vertex that will ever be a
//! common neighbour stays dominated forever, so is removed without looking any further. Otherwise its common neighbours
//! are listed with the times they join, and the edge is moved to the first of those times at which it is no longer
//! dominated (or removed, if there is none).
//!
//! The result is a [`SparseDistances`] whose Rips filtration, in every dimension, has the barcode of the original's
//! (up to bars of length zero). Points must all appear at time 0 (no [`Weighted`](super::Weighted) births).

use ordered_float::NotNan;
use rustc_hash::FxHashMap;

use super::distances::{Distances, SparseDistances};
use phlite::values::INFINITY;

/// When an edge `{i < j}` appears: its length, then (for ties) `j` and `i`, as the Rips filtration orders edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Time {
    length: NotNan<f64>,
    j: u32,
    i: u32,
}

/// What [`collapse_edges`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollapseStats {
    /// The edges up to the threshold before collapsing.
    pub edges_before: usize,
    /// The edges left.
    pub edges_after: usize,
    /// The edges left that were moved to a later time.
    pub shifted: usize,
    /// The edges whose time had to be found by scanning their common neighbours (the rest were decided by bitsets).
    pub scanned: usize,
}

/// A square matrix of bits, by rows.
struct Bits {
    words: usize,
    data: Vec<u64>,
}

impl Bits {
    fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        Self {
            words,
            data: vec![0; n * words],
        }
    }

    #[inline]
    fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.words..(i + 1) * self.words]
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> bool {
        self.data[i * self.words + j / 64] >> (j % 64) & 1 != 0
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize) {
        self.data[i * self.words + j / 64] |= 1 << (j % 64);
        self.data[j * self.words + i / 64] |= 1 << (i % 64);
    }

    #[inline]
    fn clear(&mut self, i: usize, j: usize) {
        self.data[i * self.words + j / 64] &= !(1 << (j % 64));
        self.data[j * self.words + i / 64] &= !(1 << (i % 64));
    }
}

/// The set bits of `words`, in increasing order.
fn ones(words: &[u64]) -> impl Iterator<Item = usize> + '_ {
    words.iter().enumerate().flat_map(|(k, &word)| {
        let mut word = word;
        std::iter::from_fn(move || {
            (word != 0).then(|| {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                k * 64 + bit
            })
        })
    })
}

/// Collapse the edges of the Rips filtration of `distances` up to `threshold` (see the [module docs](self)).
///
/// # Panics
/// If `distances` has births ([`Distances::HAS_BIRTHS`]).
pub fn collapse_edges<D: Distances + ?Sized>(distances: &D, threshold: NotNan<f64>) -> (SparseDistances, CollapseStats) {
    assert!(!D::HAS_BIRTHS, "edge collapse needs every point to appear at time 0");
    let n = distances.n_points();
    let mut edges: Vec<Time> = vec![];
    for i in 0..n {
        for j in distances.neighbours(i, i + 1) {
            if j > i {
                let length = distances.distance(i, j);
                if length <= threshold && length != INFINITY {
                    edges.push(Time {
                        length,
                        j: j as u32,
                        i: i as u32,
                    });
                }
            }
        }
    }
    edges.sort_unstable();
    let mut stats = CollapseStats {
        edges_before: edges.len(),
        ..CollapseStats::default()
    };
    // Everything below is in terms of ranks: an edge is its position in `edges`, and so is a time (every time an edge
    // moves to is when some other edge appears), with `NEVER_RANK` for an edge removed
    let ranks = EdgeRanks::new(n, &edges);
    let mut time: Vec<u32> = (0..edges.len() as u32).collect();
    // `now`: the edges not yet processed (so at their original times, all at most the current edge's); `fin`: every
    // edge not removed
    let mut now = Bits::new(n);
    for e in &edges {
        now.set(e.i as usize, e.j as usize);
    }
    let mut fin = Bits {
        words: now.words,
        data: now.data.clone(),
    };
    // The current time of the edge {a, b} (which must be different)
    let current = |fin: &Bits, time: &[u32], a: usize, b: usize| -> u32 {
        if fin.get(a, b) {
            time[ranks.rank(a, b) as usize]
        } else {
            NEVER_RANK
        }
    };
    let words = now.words;
    let mut common = vec![0u64; words];
    let mut later = vec![0u64; words];
    let mut present = vec![0u64; words];
    // The vertices that become common neighbours of the edge being scanned after it appears, with when they join
    let mut joins: Vec<(usize, u32)> = vec![];
    // The last dominator found for an edge at each vertex
    let mut last_dominator = vec![NEVER_RANK; n];
    for rank in (0..edges.len()).rev() {
        let e = rank as u32;
        let (u, v) = (edges[rank].i as usize, edges[rank].j as usize);
        // Common neighbours at time e
        let mut any = 0;
        for k in 0..words {
            common[k] = now.row(u)[k] & now.row(v)[k];
            any |= common[k];
        }
        // Whether w (a common neighbour) dominates: every other common neighbour is a neighbour of w
        let dominates = |w: usize| {
            (0..words).all(|k| {
                let mut outside = common[k] & !now.row(w)[k];
                if w / 64 == k {
                    outside &= !(1 << (w % 64));
                }
                outside == 0
            })
        };
        let dominator = if any == 0 {
            None
        } else {
            // The last dominator found for an edge at u or v often dominates this one too. Otherwise try each common
            // neighbour: one that doesn't dominate usually fails within a word or two.
            [last_dominator[u], last_dominator[v]]
                .into_iter()
                .filter(|&w| w != NEVER_RANK && common[w as usize / 64] >> (w % 64) & 1 != 0)
                .map(|w| w as usize)
                .find(|&w| dominates(w))
                .or_else(|| ones(&common).find(|&w| dominates(w)))
        };
        now.clear(u, v);
        let Some(mut w) = dominator else {
            continue;
        };
        last_dominator[u] = w as u32;
        last_dominator[v] = w as u32;
        // Dominated at e. If w is already adjacent to every vertex that will ever be a common neighbour, the edge stays
        // dominated forever. (`now` no longer has e, but w's row never held it.)
        let mut forever = true;
        for k in 0..words {
            later[k] = fin.row(u)[k] & fin.row(v)[k] & !common[k];
            let mut outside = later[k] & !now.row(w)[k];
            if w / 64 == k {
                outside &= !(1 << (w % 64));
            }
            if outside != 0 {
                forever = false;
            }
        }
        if forever {
            fin.clear(u, v);
            time[rank] = NEVER_RANK;
            continue;
        }
        // Otherwise list the common neighbours that join after e (those present at e are in `common`), with when
        stats.scanned += 1;
        joins.clear();
        for x in ones(&later) {
            joins.push((x, current(&fin, &time, u, x).max(current(&fin, &time, v, x))));
        }
        let mut s = e;
        let new_time = loop {
            // w dominates until a common neighbour joins before its edge to w
            let mut until = NEVER_RANK;
            for &(x, joined) in &joins {
                if x != w && joined > s && joined < until && current(&fin, &time, w, x) > joined {
                    until = joined;
                }
            }
            if until == NEVER_RANK {
                break NEVER_RANK;
            }
            s = until;
            // A dominator at time s? The common neighbours present at s, then for each, those not adjacent to it
            // already at e (whose edges must have appeared by s)
            present.copy_from_slice(&common);
            for &(x, joined) in &joins {
                if joined <= s {
                    present[x / 64] |= 1 << (x % 64);
                }
            }
            let found = ones(&present).find(|&candidate| {
                (0..words).all(|k| {
                    let mut missing = present[k] & !now.row(candidate)[k];
                    if candidate / 64 == k {
                        missing &= !(1 << (candidate % 64));
                    }
                    let missing = [missing];
                    let all_there = ones(&missing).all(|bit| current(&fin, &time, candidate, k * 64 + bit) <= s);
                    all_there
                })
            });
            match found {
                Some(next) => w = next,
                None => break s,
            }
        };
        if new_time == NEVER_RANK {
            fin.clear(u, v);
        }
        time[rank] = new_time;
    }
    // The edges left, at their new times
    let mut kept = vec![];
    for (rank, e) in edges.iter().enumerate() {
        let t = time[rank];
        if t != NEVER_RANK {
            if t as usize != rank {
                stats.shifted += 1;
            }
            kept.push((e.i as usize, e.j as usize, edges[t as usize].length.into_inner()));
        }
    }
    stats.edges_after = kept.len();
    let sparse = SparseDistances::from_edges(n, kept).expect("the edges of a valid distance matrix");
    (sparse, stats)
}

const NEVER_RANK: u32 = u32::MAX;

/// The rank of each edge: in a table when there are few enough points for it to be small, otherwise in a hash map.
enum EdgeRanks {
    Dense { n: usize, ranks: Vec<u32> },
    Sparse(FxHashMap<(u32, u32), u32>),
}

impl EdgeRanks {
    /// At most this many points get a table (64 MB of it).
    const DENSE_POINTS: usize = 4096;

    fn new(n: usize, edges: &[Time]) -> Self {
        if n <= Self::DENSE_POINTS {
            let mut ranks = vec![NEVER_RANK; n * n];
            for (rank, e) in edges.iter().enumerate() {
                ranks[e.i as usize * n + e.j as usize] = rank as u32;
                ranks[e.j as usize * n + e.i as usize] = rank as u32;
            }
            Self::Dense { n, ranks }
        } else {
            Self::Sparse(edges.iter().enumerate().map(|(rank, e)| ((e.i, e.j), rank as u32)).collect())
        }
    }

    /// The rank of the edge {a, b}, which must be an edge.
    #[inline]
    fn rank(&self, a: usize, b: usize) -> u32 {
        match self {
            Self::Dense { n, ranks } => ranks[a * n + b],
            Self::Sparse(map) => {
                let (i, j) = if a < b { (a, b) } else { (b, a) };
                map[&(i as u32, j as u32)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rips::{rips_barcode, RipsBar};
    use phlite::barcode::{Algorithm, Method};
    use phlite::reduction::Clearing;
    use proptest::prelude::*;

    fn bars(bars: Vec<RipsBar>, n: usize) -> Vec<(usize, NotNan<f64>, Option<NotNan<f64>>)> {
        let mut out: Vec<_> = bars
            .into_iter()
            .map(|bar| (bar.birth.dimension(n), bar.birth_value, bar.death.map(|(_, d)| d)))
            .filter(|&(_, b, d)| d != Some(b))
            .collect();
        out.sort();
        out
    }

    fn barcode<D: Distances>(distances: D, max_dim: usize, threshold: NotNan<f64>) -> Vec<RipsBar> {
        rips_barcode::<_, usize, _>(
            distances,
            max_dim,
            threshold,
            Method::Cohomology,
            Algorithm::Clearing(Clearing::new()),
            false,
        )
    }

    fn check(points: &[[f64; 3]], levels: f64, max_dim: usize, threshold: f64) {
        let n = points.len();
        // Rounded, so that lengths tie
        let dense: Vec<Vec<NotNan<f64>>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d: f64 = (0..3).map(|k| (points[i][k] - points[j][k]).powi(2)).sum::<f64>().sqrt();
                        NotNan::new((d * levels).round() / levels).unwrap()
                    })
                    .collect()
            })
            .collect();
        let threshold = NotNan::new(threshold).unwrap();
        let (collapsed, stats) = collapse_edges(&dense, threshold);
        assert!(stats.edges_after <= stats.edges_before);
        let want = bars(barcode(&dense, max_dim, threshold), n);
        let got = bars(barcode(&collapsed, max_dim, threshold), n);
        assert_eq!(got, want, "{stats:?}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(std::env::var("PROPTEST_CASES").ok().and_then(|c| c.parse().ok()).unwrap_or(256)))]
        #[test]
        fn collapse_keeps_the_barcode(
            points in prop::collection::vec(prop::array::uniform3(0.0..1.0f64), 1..30),
            levels in prop::sample::select(vec![3.0, 10.0, 1e9]),
            max_dim in 1usize..4,
            threshold in prop::sample::select(vec![0.3, 0.6, f64::INFINITY]),
        ) {
            check(&points, levels, max_dim, threshold);
        }
    }

    /// Points on a circle: the long H1 bar must survive, and most edges go.
    #[test]
    fn circle() {
        let points: Vec<[f64; 3]> = (0..40)
            .map(|k| {
                let a = k as f64 * std::f64::consts::TAU / 40.0;
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        check(&points, 1e9, 2, f64::INFINITY);
    }
}
