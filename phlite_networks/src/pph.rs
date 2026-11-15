//! Persistent path homology (PPH) of weighted digraphs in any dimension.
//!
//! PPH is described in [Chowdhury and Mémoli (2018)](https://arxiv.org/abs/1701.00565).
//! A weighted digraph (a *network*) gives a filtration of digraphs `G_t`, with an edge `a -> b` in `G_t` once its weight is at most `t`,
//! and `PH_p` is the persistent homology of the chain complexes of ∂-invariant paths `Ω_•(G_t)`.
//!
//! # How it is computed
//!
//! The spaces `Ω_p` have no basis of paths, so we can't use them as cells. Instead the cells are the *regular* paths
//! (no vertex repeated consecutively), each with the time at which it becomes *allowed* (all of its edges are present),
//! or `+∞` if it never does. We reduce the boundary matrix of the regular path complex (irregular terms dropped), with
//! columns the allowed paths and rows ordered by allow time. Unlike a simplicial filtration, a face can be allowed
//! *after* its path (deleting `x_i` from a path needs the shortcut edge `x_{i-1} -> x_{i+1}`), so a pairing
//! `(σ, τ)` gives the bar
//!
//! `[t(σ), max(t(τ), t(σ)))`
//!
//! and essential cells give `[t(σ), ∞)`. Bars with infinite birth (a never-allowed `σ`) are discarded.
//! This is the algorithm of Chowdhury and Mémoli (see their `pph-matlab` code), phrased as a phlite matrix:
//! the columns of `V` form a filtered basis of `Ω`, with `V_j` entering `Ω` at `max(t(j), t(low R_j))`.
//!
//! Since this is the boundary matrix of a chain complex (just not ordered compatibly with a filtration), every phlite
//! algorithm applies: clearing, cohomology (the pivot pairs of a matrix and of its anti-transpose agree), apparent pairs
//! and representatives by involution. The only change is reading bars off with the `max` above, see [`pph_barcode`].
//!
//! # Paths as integers
//!
//! A [`RegularPath`] is a single `u64`: the dimension `p` in the top 4 bits and, below, the path `[x_0, ..., x_p]`
//! in a mixed radix: `x_0` (one of `n` vertices) followed by one digit in `0..n-1` per later vertex, numbering the
//! `n - 1` vertices other than its predecessor. These codes are exactly `0..n(n-1)^p`, so every regular path has
//! one; decoding needs `n`, so goes through the [`Network`] ([`Network::path`] and [`Network::vertices`]).
//! Appending a vertex is `code * (n - 1) + digit`, which makes enumerating paths cheap.
//!
//! # Example
//!
//! ```
//! use phlite::{barcode::{Algorithm, Method}, fields::Z2};
//! use phlite_networks::pph::{pph_barcode, Network, PPHBar};
//! 
//! // A directed 4-cycle a -> b -> c -> d -> a, then its chords
//! let network = Network::from_edges(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 2.0), (2, 0, 2.0)]);
//! let bars: Vec<PPHBar<Z2>> = pph_barcode(&network, 2, Method::Cohomology, Algorithm::default(), true);
//! ```
//!
//! Each bar has a `birth` path (its dimension is the bar's dimension), `birth_value`, an optional `death` and, if
//! requested, a `representative` cycle (a linear combination of allowed paths). Any phlite coefficient field works.
//!
//! # Checked against
//!
//! `tests/pph.rs` (the cycle-network theorem, the bi-fan and bi-parallel examples, the coboundary being the boundary's
//! transpose, every method and algorithm agreeing, representatives), and `matlab_compare/`: Chowdhury and Mémoli's
//! `pph-matlab` under Octave, and a brute force for networks with missing edges (see its README).

use std::marker::PhantomData;

use ordered_float::NotNan;
use phlite::{
    barcode::{self, Algorithm, Bar},
    columns::ColumnRepresentation,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{BasisElement, HasRowFiltration, MatrixOracle, SplitBasis},
};
use rayon::prelude::*;

pub use phlite::barcode::Method;

/// The largest dimension a [`RegularPath`] can have (whether it fits also depends on the number of vertices,
/// see [`Network::max_dimension`]).
pub const MAX_DIM: usize = 15;
const MAX_LEN: usize = MAX_DIM + 1;
const DIM_SHIFT: u32 = 60;
const CODE_MASK: u64 = (1 << DIM_SHIFT) - 1;

/// A regular path `[x_0, ..., x_p]` (no consecutive repeats) of dimension `p`, encoded as a single `u64`
/// (see the [module docs](self#paths-as-integers)); use [`Network::path`] and [`Network::vertices`] to convert.
///
/// The order on paths is only used to break ties between paths with the same allow time; any total order would do.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct RegularPath(u64);

impl RegularPath {
    #[inline]
    fn from_parts(dim: usize, code: u64) -> Self {
        Self(((dim as u64) << DIM_SHIFT) | code)
    }

    /// The dimension `p` of the path `[x_0, ..., x_p]`.
    #[inline]
    pub fn dimension(&self) -> usize {
        (self.0 >> DIM_SHIFT) as usize
    }

    #[inline]
    fn code(&self) -> u64 {
        self.0 & CODE_MASK
    }

    /// The underlying integer.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl BasisElement for RegularPath {}

/// The vertices of a decoded path, on the stack.
#[derive(Clone, Copy)]
struct Verts {
    len: usize,
    v: [u16; MAX_LEN + 1],
}

impl Verts {
    #[inline]
    fn as_slice(&self) -> &[u16] {
        &self.v[..self.len]
    }

    /// The path with vertex `i` deleted (which may not be regular).
    #[inline]
    fn delete(&self, i: usize) -> Self {
        let mut out = *self;
        out.v.copy_within(i + 1..self.len, i);
        out.len -= 1;
        out
    }

    /// The path with `y` inserted at position `i`.
    #[inline]
    fn insert(&self, i: usize, y: u16) -> Self {
        let mut out = *self;
        out.v.copy_within(i..self.len, i + 1);
        out.v[i] = y;
        out.len += 1;
        out
    }
}

/// The digit of `y` after `prev`: its position among the vertices other than `prev`.
#[inline]
fn digit(y: u16, prev: u16) -> u64 {
    (if y < prev { y } else { y - 1 }) as u64
}

#[inline]
fn undigit(d: u64, prev: u16) -> u16 {
    let d = d as u16;
    if d < prev {
        d
    } else {
        d + 1
    }
}

type Time = NotNan<f64>;

fn infinity() -> Time {
    NotNan::new(f64::INFINITY).unwrap()
}

/// A weighted digraph: the edge `a -> b` enters the filtration at its weight.
#[derive(Debug, Clone)]
pub struct Network {
    n_vertices: u16,
    /// `times[a * n + b]`, `None` if there is no edge (or `a == b`)
    times: Vec<Option<Time>>,
    /// `(b, time)` for each edge `a -> b`
    out_adj: Vec<Vec<(u16, Time)>>,
    /// `(a, time)` for each edge `a -> b`
    in_adj: Vec<Vec<(u16, Time)>>,
    /// The time at which vertices (0-paths) enter.
    vertex_time: Time,
}

impl Network {
    /// A network on `n_vertices` vertices with the given `(source, target, weight)` edges, all vertices present at time 0.
    /// Repeated edges keep the smallest weight; an infinite weight means no edge.
    ///
    /// # Panics
    /// If an edge is a self-loop, has a vertex out of range, or has a NaN or negative weight.
    pub fn from_edges(n_vertices: u16, edges: &[(u16, u16, f64)]) -> Self {
        let n = n_vertices as usize;
        let mut times = vec![None; n * n];
        for &(a, b, w) in edges {
            assert!(a != b, "self-loops are not supported");
            assert!(a < n_vertices && b < n_vertices, "vertex out of range");
            assert!(w >= 0.0, "edge weights must be non-negative (and not NaN)");
            if w == f64::INFINITY {
                continue;
            }
            let w = NotNan::new(w).unwrap();
            let slot: &mut Option<Time> = &mut times[a as usize * n + b as usize];
            *slot = Some(slot.map_or(w, |old| old.min(w)));
        }
        Self::from_times(n_vertices, times, NotNan::new(0.0).unwrap())
    }

    fn from_times(n_vertices: u16, times: Vec<Option<Time>>, vertex_time: Time) -> Self {
        let n = n_vertices as usize;
        let mut out_adj = vec![vec![]; n];
        let mut in_adj = vec![vec![]; n];
        for a in 0..n {
            for b in 0..n {
                if let Some(t) = times[a * n + b] {
                    out_adj[a].push((b as u16, t));
                    in_adj[b].push((a as u16, t));
                }
            }
        }
        Self {
            n_vertices,
            times,
            out_adj,
            in_adj,
            vertex_time,
        }
    }

    /// A network from a square weight matrix, `weights[a][b]` being the weight of `a -> b`.
    /// The diagonal is ignored and an infinite weight means no edge.
    pub fn from_dense(weights: &[Vec<f64>]) -> Self {
        let n = weights.len();
        let edges: Vec<_> = (0..n)
            .flat_map(|a| (0..n).map(move |b| (a, b)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| {
                assert!(weights[a].len() == n, "weight matrix must be square");
                (a as u16, b as u16, weights[a][b])
            })
            .collect();
        Self::from_edges(n as u16, &edges)
    }

    /// Set the time at which vertices (and so the 0-dimensional bars) are born, 0 by default.
    pub fn with_vertex_time(mut self, time: f64) -> Self {
        self.vertex_time = NotNan::new(time).unwrap();
        self
    }

    /// The network with every edge of weight above `t_max` removed, i.e. the filtration truncated at `t_max`.
    /// Its barcode agrees with the full barcode up to `t_max`: bars still alive at `t_max` become essential.
    /// Since allow times only grow as paths are extended, this prunes the enumeration of paths, so is much cheaper.
    pub fn truncate(&self, t_max: f64) -> Self {
        let t_max = NotNan::new(t_max).unwrap();
        let times = self
            .times
            .iter()
            .map(|t| t.filter(|&t| t <= t_max))
            .collect();
        Self::from_times(self.n_vertices, times, self.vertex_time)
    }

    pub fn n_vertices(&self) -> u16 {
        self.n_vertices
    }

    /// The largest dimension whose paths can be encoded as [`RegularPath`]s on this many vertices:
    /// at most [`MAX_DIM`], and with `n (n - 1)^p < 2^60`.
    pub fn max_dimension(&self) -> usize {
        let n = self.n_vertices as u128;
        if n <= 1 {
            // No paths of positive dimension at all
            return MAX_DIM;
        }
        let mut count = n;
        let mut p = 0;
        while p < MAX_DIM && count * (n - 1) < (1u128 << DIM_SHIFT) {
            count *= n - 1;
            p += 1;
        }
        p
    }

    /// The time at which the edge `a -> b` enters, if it ever does.
    #[inline]
    pub fn edge_time(&self, a: u16, b: u16) -> Option<Time> {
        self.times[a as usize * self.n_vertices as usize + b as usize]
    }

    #[inline]
    fn base(&self) -> u64 {
        (self.n_vertices as u64).saturating_sub(1)
    }

    #[inline]
    fn encode(&self, verts: &[u16]) -> RegularPath {
        let base = self.base();
        let mut code = verts[0] as u64;
        for w in verts.windows(2) {
            code = code * base + digit(w[1], w[0]);
        }
        RegularPath::from_parts(verts.len() - 1, code)
    }

    #[inline]
    fn decode(&self, path: RegularPath) -> Verts {
        let dim = path.dimension();
        let base = self.base();
        let mut code = path.code();
        let mut digits = [0u64; MAX_LEN];
        for k in (1..=dim).rev() {
            digits[k] = code % base;
            code /= base;
        }
        let mut v = [0u16; MAX_LEN + 1];
        v[0] = code as u16;
        for k in 1..=dim {
            v[k] = undigit(digits[k], v[k - 1]);
        }
        Verts { len: dim + 1, v }
    }

    /// The [`RegularPath`] through `vertices`.
    ///
    /// # Panics
    /// If the path is empty, repeats a vertex consecutively, has a vertex out of range, or is too long to encode
    /// (see [`max_dimension`](Self::max_dimension)).
    pub fn path(&self, vertices: &[u16]) -> RegularPath {
        assert!(!vertices.is_empty(), "paths must have a vertex");
        assert!(
            vertices.len() - 1 <= self.max_dimension(),
            "path too long to encode"
        );
        assert!(
            vertices.iter().all(|&v| v < self.n_vertices),
            "vertex out of range"
        );
        assert!(
            vertices.windows(2).all(|w| w[0] != w[1]),
            "regular paths can't repeat a vertex consecutively"
        );
        self.encode(vertices)
    }

    /// The vertices of `path`.
    pub fn vertices(&self, path: RegularPath) -> Vec<u16> {
        self.decode(path).as_slice().to_vec()
    }

    #[inline]
    fn verts_time(&self, verts: &[u16]) -> Time {
        let mut time = self.vertex_time;
        for w in verts.windows(2) {
            match self.edge_time(w[0], w[1]) {
                Some(t) => time = time.max(t),
                None => return infinity(),
            }
        }
        time
    }

    /// The time at which `path` becomes allowed (all its edges are present), `+∞` if it never does.
    #[inline]
    pub fn allow_time(&self, path: RegularPath) -> Time {
        self.verts_time(self.decode(path).as_slice())
    }

    /// The allowed paths of each dimension `0..=max_dim`, each sorted by allow time (then path), with their allow times.
    ///
    /// Each dimension is built from the last by appending out-neighbours, in parallel on the current rayon thread pool
    /// if `parallel`.
    pub fn allowed_paths(&self, max_dim: usize, parallel: bool) -> Vec<Vec<(Time, RegularPath)>> {
        assert!(
            max_dim <= self.max_dimension(),
            "dimension {max_dim} too large to encode on {} vertices",
            self.n_vertices
        );
        let base = self.base();
        // (time, code, last vertex)
        let mut frontier: Vec<(Time, u64, u16)> = (0..self.n_vertices)
            .map(|v| (self.vertex_time, v as u64, v))
            .collect();
        let mut dims = Vec::with_capacity(max_dim + 1);
        for dim in 0..=max_dim {
            let extend = |&(time, code, last): &(Time, u64, u16)| {
                self.out_adj[last as usize]
                    .iter()
                    .map(move |&(y, t)| (time.max(t), code * base + digit(y, last), y))
            };
            let next: Vec<(Time, u64, u16)> = match (dim < max_dim, parallel) {
                (false, _) => vec![],
                (true, true) => frontier.par_iter().flat_map_iter(extend).collect(),
                (true, false) => frontier.iter().flat_map(extend).collect(),
            };
            let to_path = |(time, code, _last): (Time, u64, u16)| (time, RegularPath::from_parts(dim, code));
            let mut paths: Vec<(Time, RegularPath)> = if parallel {
                frontier.into_par_iter().map(to_path).collect()
            } else {
                frontier.into_iter().map(to_path).collect()
            };
            if parallel {
                paths.par_sort_unstable();
            } else {
                paths.sort_unstable();
            }
            dims.push(paths);
            frontier = next;
        }
        dims
    }

    /// For each dimension `1..=max_dim`, the never-allowed paths which are faces of allowed paths, sorted.
    ///
    /// Such a face has exactly one missing edge `a -> c`, which is *bridged* (some `a -> b -> c` is allowed),
    /// and is an allowed path ending at `a` followed by an allowed path starting at `c`.
    /// `allowed` must hold the allowed paths in dimensions `0..max_dim` (at least).
    fn never_allowed_faces(
        &self,
        allowed: &[Vec<(Time, RegularPath)>],
        max_dim: usize,
        parallel: bool,
    ) -> Vec<Vec<RegularPath>> {
        let n = self.n_vertices;
        let bridged_from = |a| {
                let mut targets = vec![false; n as usize];
                for &(b, _) in &self.out_adj[a as usize] {
                    for &(c, _) in &self.out_adj[b as usize] {
                        if c != a && self.edge_time(a, c).is_none() {
                            targets[c as usize] = true;
                        }
                    }
                }
                (0..n)
                    .filter(move |&c| targets[c as usize])
                    .map(move |c| (a, c))
            };
        let bridged: Vec<(u16, u16)> = if parallel {
            (0..n).into_par_iter().flat_map_iter(bridged_from).collect()
        } else {
            (0..n).flat_map(bridged_from).collect()
        };
        let mut out = vec![vec![]; max_dim + 1];
        if bridged.is_empty() || max_dim == 0 {
            return out;
        }
        // Allowed paths of dimensions 0..max_dim, grouped by first and by last vertex
        let mut by_first = vec![vec![vec![]; n as usize]; max_dim];
        let mut by_last = vec![vec![vec![]; n as usize]; max_dim];
        for (dim, paths) in allowed.iter().enumerate().take(max_dim) {
            for &(_t, path) in paths {
                let verts = self.decode(path);
                by_first[dim][verts.v[0] as usize].push(verts);
                by_last[dim][verts.v[verts.len - 1] as usize].push(verts);
            }
        }
        for (dim, faces) in out.iter_mut().enumerate().skip(1) {
            // left has dimension j, right has dimension dim - 1 - j
            let faces_of = |&(a, c)| {
                    let by_first = &by_first;
                    let by_last = &by_last;
                    (0..dim).flat_map(move |j| {
                        by_last[j][a as usize].iter().flat_map(move |left| {
                            by_first[dim - 1 - j][c as usize].iter().map(move |right| {
                                let mut v = left.v;
                                v[left.len..left.len + right.len].copy_from_slice(right.as_slice());
                                self.encode(&v[..left.len + right.len])
                            })
                        })
                    })
                };
            let mut found: Vec<RegularPath> = if parallel {
                bridged.par_iter().flat_map_iter(faces_of).collect()
            } else {
                bridged.iter().flat_map(faces_of).collect()
            };
            if parallel {
                found.par_sort_unstable();
            } else {
                found.sort_unstable();
            }
            found.dedup();
            *faces = found;
        }
        out
    }
}

fn sign<CF: NonZeroCoefficient>(i: usize) -> CF {
    if i.is_multiple_of(2) {
        CF::one()
    } else {
        CF::one().additive_inverse()
    }
}

/// The boundary matrix of the regular path complex, filtered by allow time.
///
/// Columns should be allowed paths; rows are all their regular faces, including those that are never allowed (at `+∞`).
pub struct PPHBoundary<'a, CF> {
    network: &'a Network,
    phantom: PhantomData<CF>,
}

impl<'a, CF> PPHBoundary<'a, CF> {
    pub fn new(network: &'a Network) -> Self {
        Self {
            network,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> MatrixOracle for PPHBoundary<'_, CF> {
    type CoefficientField = CF;
    type ColT = RegularPath;
    type RowT = RegularPath;

    fn column(&self, col: Self::ColT) -> impl Iterator<Item = (CF, RegularPath)> {
        let network = self.network;
        let verts = network.decode(col);
        let len = if verts.len > 1 { verts.len } else { 0 };
        (0..len).filter_map(move |i| {
            if i > 0 && i + 1 < len && verts.v[i - 1] == verts.v[i + 1] {
                return None; // irregular
            }
            let face = verts.delete(i);
            Some((sign(i), network.encode(face.as_slice())))
        })
    }
}

impl<CF: NonZeroCoefficient> HasRowFiltration for PPHBoundary<'_, CF> {
    type FiltrationT = Time;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.network.allow_time(row)
    }
}

/// The coboundary matrix: the transpose of [`PPHBoundary`] (with the boundary's columns, the allowed paths, as rows).
///
/// Columns are regular paths (including never-allowed ones, at `+∞`); rows are their allowed cofaces.
pub struct PPHCoboundary<'a, CF> {
    network: &'a Network,
    phantom: PhantomData<CF>,
}

impl<'a, CF> PPHCoboundary<'a, CF> {
    pub fn new(network: &'a Network) -> Self {
        Self {
            network,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> MatrixOracle for PPHCoboundary<'_, CF> {
    type CoefficientField = CF;
    type ColT = RegularPath;
    type RowT = RegularPath;

    fn column(&self, col: Self::ColT) -> impl Iterator<Item = (CF, RegularPath)> {
        let network = self.network;
        let verts = network.decode(col);
        let len = verts.len;
        // Inserting y at position i replaces the edge x_{i-1} -> x_i by x_{i-1} -> y -> x_i, keeping the others.
        // So a coface is allowed only if col has at most one missing edge, and if it has one then y goes there.
        let mut missing = None;
        let mut n_missing = 0;
        for k in 0..len - 1 {
            if network.edge_time(verts.v[k], verts.v[k + 1]).is_none() {
                missing = Some(k);
                n_missing += 1;
            }
        }
        let positions = match (n_missing, missing) {
            (0, _) => 0..len + 1,
            (1, Some(k)) => k + 1..k + 2,
            _ => 0..0,
        };
        // Deleting vertex i from a coface gives back `col` exactly when that vertex was inserted at position i,
        // and different (position, vertex) pairs give different regular cofaces.
        positions.flat_map(move |i| {
            let candidates = if i == 0 {
                &network.in_adj[verts.v[0] as usize]
            } else {
                &network.out_adj[verts.v[i - 1] as usize]
            };
            candidates.iter().filter_map(move |&(y, _t)| {
                if i > 0 && i < len && network.edge_time(y, verts.v[i]).is_none() {
                    return None;
                }
                Some((sign(i), network.encode(verts.insert(i, y).as_slice())))
            })
        })
    }
}

impl<CF: NonZeroCoefficient> HasRowFiltration for PPHCoboundary<'_, CF> {
    type FiltrationT = Time;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.network.allow_time(row)
    }
}

/// The column basis for reducing the boundary matrix: allowed paths in dimensions `0..=max_dim + 1`, in filtration order.
pub fn homology_basis(network: &Network, max_dim: usize, parallel: bool) -> SplitBasis<RegularPath> {
    SplitBasis::from_dimensions(
        network
            .allowed_paths(max_dim + 1, parallel)
            .into_iter()
            .map(|paths| paths.into_iter().map(|(_t, path)| path)),
    )
}

/// The column basis for reducing the coboundary matrix: in each dimension `0..=max_dim`, the allowed paths and then the
/// never-allowed faces of allowed paths one dimension up (which are rows of the boundary matrix, so must be columns here,
/// at `+∞`), in filtration order. Paths of dimension `max_dim + 1` are never enumerated.
pub fn cohomology_basis(network: &Network, max_dim: usize, parallel: bool) -> SplitBasis<RegularPath> {
    let allowed = network.allowed_paths(max_dim, parallel);
    let never_allowed = network.never_allowed_faces(&allowed, max_dim, parallel);
    SplitBasis::from_dimensions(
        allowed
            .into_iter()
            .zip(never_allowed)
            .map(|(paths, faces)| paths.into_iter().map(|(_t, path)| path).chain(faces)),
    )
}

/// A bar in the PPH barcode: `birth` is a path of the bar's dimension.
/// For finite bars `death` is `(τ, max(t(τ), t(birth)))`, which may differ from the allow time of `τ`.
pub type PPHBar<CF> = Bar<RegularPath, NotNan<f64>, CF>;

/// Convert bars read off with phlite's usual rule (death = filtration value of the death cell) to PPH bars:
/// deaths become `max(t(death), t(birth))`, and bars of length zero, bars born at `+∞`
/// and bars above `max_dim` are dropped.
fn to_pph_bars<CF>(bars: Vec<PPHBar<CF>>, max_dim: usize) -> Vec<PPHBar<CF>> {
    bars.into_iter()
        .filter_map(|mut bar| {
            if bar.birth.dimension() > max_dim || bar.birth_value == infinity() {
                return None;
            }
            if let Some((_cell, death_value)) = &mut bar.death {
                *death_value = (*death_value).max(bar.birth_value);
                if *death_value == bar.birth_value {
                    return None;
                }
            }
            Some(bar)
        })
        .collect()
}

/// The PPH barcode of `network` in dimensions `0..=max_dim`, over the coefficient field `CF`.
///
/// * `method` chooses between reducing the coboundary matrix (typically much faster) and the boundary matrix.
/// * `algorithm` is any of phlite's reduction algorithms.
/// * If `representatives` is true, every bar comes with a representative cycle: a linear combination of allowed paths,
///   which is a ∂-invariant cycle from the bar's birth and becomes a boundary at its death.
///
/// Bars are returned in no particular order, and bars of length zero are omitted.
/// To truncate the filtration (and prune the enumeration of paths), use [`Network::truncate`].
///
/// # Panics
/// If paths of dimension `max_dim + 1` can't be encoded, see [`Network::max_dimension`].
pub fn pph_barcode<CF, R>(
    network: &Network,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
) -> Vec<PPHBar<CF>>
where
    CF: NonZeroCoefficient + Invertible + Send + Sync + 'static,
    R: ColumnRepresentation,
{
    #[allow(unreachable_patterns)]
    let parallel = matches!(algorithm, Algorithm::ParallelClearing(_));
    pph_barcode_with(network, max_dim, method, algorithm, representatives, parallel)
}

/// As [`pph_barcode`], choosing whether to enumerate paths in parallel on the current rayon thread pool
/// (`parallel_setup`, which [`pph_barcode`] does only for the parallel algorithm), whatever the algorithm.
pub fn pph_barcode_with<CF, R>(
    network: &Network,
    max_dim: usize,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    parallel_setup: bool,
) -> Vec<PPHBar<CF>>
where
    CF: NonZeroCoefficient + Invertible + Send + Sync + 'static,
    R: ColumnRepresentation,
{
    assert!(
        max_dim < network.max_dimension(),
        "paths of dimension {} can't be encoded on {} vertices",
        max_dim + 1,
        network.n_vertices()
    );
    let parallel = parallel_setup;
    let boundary = PPHBoundary::<CF>::new(network);
    let coboundary = PPHCoboundary::<CF>::new(network);
    let bars = match method {
        Method::Homology => {
            let boundary = boundary.by_ref().with_basis(homology_basis(network, max_dim, parallel));
            let boundary = boundary.by_ref().with_transpose(&coboundary);
            barcode::homology(boundary, algorithm, representatives)
        }
        Method::Cohomology => {
            let coboundary = coboundary.by_ref().with_basis(cohomology_basis(network, max_dim, parallel));
            let coboundary = coboundary.by_ref().with_transpose(&boundary);
            if representatives {
                barcode::cohomology_with_involution(coboundary, algorithm)
            } else {
                barcode::cohomology(coboundary, algorithm)
            }
        }
    };
    to_pph_bars(bars, max_dim)
}
