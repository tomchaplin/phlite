//! Magnitude homology of directed graphs (Hepworth and Willerton, *Categorifying the magnitude of a graph*, 2017).
//!
//! For a digraph `G` with its shortest-path (quasi-)metric `d` (every edge has length 1), the magnitude chain group
//! `MC_{k,l}` is free on the tuples `(x_0, ..., x_k)` of vertices with `x_i ≠ x_{i+1}`, every `d(x_i, x_{i+1})` finite,
//! and total length `d(x_0, x_1) + ... + d(x_{k-1}, x_k) = l`. The differential is `∂ = Σ_{i=1}^{k-1} (-1)^i ∂_i`,
//! where `∂_i` deletes `x_i` if it lies on a geodesic from `x_{i-1}` to `x_{i+1}` (so the length doesn't change) and
//! is zero otherwise. The differential keeps the endpoints and the length, so the complex splits into pieces
//! `MC_{*,l}(s, t)`, one for each pair of vertices and length, each reduced on its own.
//!
//! How it is computed:
//! * Distances by breadth-first search from every vertex.
//! * The tuples starting at `s` by depth-first search, trying vertices in increasing order, so the tuples of each
//!   piece and size come out in lexicographic order and a face is found by binary search (as Perseus 6 does). With
//!   a set of pairs, a tuple is only extended while some target is still within reach of the length left.
//! * Each piece stored once in compressed rows with its transpose, then collapsed: pairs with a free face (a cell
//!   with one coface) or a coreduction (a cell with one face) are removed, which changes nothing else since the
//!   coefficients are ±1. This removes nearly every cell (Perseus 6 does the same), and keeps the integral homology.
//! * What is left as an explicit chain complex ([`phlite_complexes`]), reduced by cohomology with clearing over each
//!   field asked for; the essential bars are the Betti numbers.
//!
//! Over a field `CF` this gives the ranks of `MH_{k,l}(s, t; CF)`. By universal coefficients, a prime `p` for which
//! the ranks over `Z/p` differ from those over `Q` (or a large prime) means `p`-torsion in the integral magnitude
//! homology, in degree `k` or `k - 1`.

use std::collections::HashMap;

use phlite::barcode::{Algorithm, Method};
use phlite::integral::elimination::{eliminate, homology_groups, CompressedComplex, ELIMINATION_COST};
use phlite::reduction::Clearing;
use phlite_complexes::complex::{Chain, RawChain};
use phlite_complexes::field::IntField;
use rayon::prelude::*;

/// An infinite distance: no directed path.
pub const UNREACHABLE: u32 = u32::MAX;

/// A digraph with its shortest-path distances (every edge of length 1).
#[derive(Debug, Clone)]
pub struct MagnitudeGraph {
    n: usize,
    /// `d(a, b)` at `a * n + b`, [`UNREACHABLE`] if there is no path.
    distances: Vec<u32>,
}

impl MagnitudeGraph {
    /// The digraph on `n` vertices with these edges `(source, target)` (repeats and self-loops are ignored).
    ///
    /// # Panics
    /// If an edge has an endpoint outside `0..n`.
    pub fn new(n: usize, edges: &[(u32, u32)]) -> Self {
        let mut out: Vec<Vec<u32>> = vec![vec![]; n];
        for &(a, b) in edges {
            assert!((a as usize) < n && (b as usize) < n, "edge ({a}, {b}) outside 0..{n}");
            if a != b {
                out[a as usize].push(b);
            }
        }
        let distances: Vec<u32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|s| {
                let mut d = vec![UNREACHABLE; n];
                d[s] = 0;
                let mut queue = std::collections::VecDeque::from([s as u32]);
                while let Some(a) = queue.pop_front() {
                    for &b in &out[a as usize] {
                        if d[b as usize] == UNREACHABLE {
                            d[b as usize] = d[a as usize] + 1;
                            queue.push_back(b);
                        }
                    }
                }
                d
            })
            .collect();
        Self { n, distances }
    }

    /// The number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.n
    }

    /// The length of a shortest path from `a` to `b`, or [`UNREACHABLE`].
    #[inline]
    pub fn distance(&self, a: u32, b: u32) -> u32 {
        self.distances[a as usize * self.n + b as usize]
    }
}

/// The ranks of the magnitude homology of one pair `(s, t)`: `ranks[m][l][k]` is the rank of `MH_{k,l}(s, t)` over
/// `Z/moduli[m]`, for `k <= min(l, k_max)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairRanks {
    pub source: u32,
    pub target: u32,
    pub ranks: Vec<Vec<Vec<usize>>>,
}

/// What to compute: lengths `0..=l_max`, degrees `0..=k_max` (default: up to `l_max`), and the pairs `(s, t)` (default:
/// every pair).
#[derive(Debug, Clone, Default)]
pub struct MagnitudeQuery {
    pub l_max: usize,
    pub k_max: Option<usize>,
    pub pairs: Option<Vec<(u32, u32)>>,
}

/// The characteristics magnitude homology can be computed over (251 is the largest prime below 256, a stand-in for
/// the rationals when looking for torsion).
pub const MODULI: [u64; 7] = [2, 3, 5, 7, 11, 13, 251];

/// The tuples of each piece `(t, l)` and size `k + 1`, found from one source, flattened: `tuples[(t, l, k)]` holds
/// the tuples one after another, `k + 1` vertices each, in lexicographic order.
pub(crate) type Buckets = HashMap<(u32, usize, usize), Vec<u32>>;

/// The tuples extending `prefix` (itself not included) of length at most `l_max` and at most `k_top + 1` vertices,
/// ending at a target (`reach[t] == 0`), into [`Buckets`]. `reach[x]` is the distance from `x` to the nearest target,
/// so tuples that can't reach one in time aren't extended.
pub(crate) fn enumerate_from(
    graph: &MagnitudeGraph,
    prefix: &[u32],
    prefix_length: usize,
    l_max: usize,
    k_top: usize,
    reach: &[u32],
    buckets: &mut Buckets,
) {
    let n = graph.n as u32;
    let mut path: Vec<u32> = prefix.to_vec();
    // The next vertex to try after the last (a manual depth-first search, in increasing order)
    let mut next: Vec<u32> = vec![0];
    let mut length: Vec<usize> = vec![prefix_length];
    while let Some(&from) = next.last() {
        let depth = path.len() - 1;
        let last = path[depth];
        let l = *length.last().expect("as deep as next");
        let mut found = None;
        if depth < k_top {
            let mut x = from;
            while x < n {
                let d = graph.distance(last, x);
                if x != last && d != UNREACHABLE {
                    let new_l = l + d as usize;
                    let r = reach[x as usize];
                    if r != UNREACHABLE && new_l + r as usize <= l_max {
                        found = Some((x, new_l));
                        break;
                    }
                }
                x += 1;
            }
        }
        match found {
            None => {
                if next.len() == 1 {
                    break;
                }
                path.pop();
                next.pop();
                length.pop();
            }
            Some((x, new_l)) => {
                *next.last_mut().expect("not empty") = x + 1;
                path.push(x);
                next.push(0);
                length.push(new_l);
                if reach[x as usize] == 0 {
                    buckets.entry((x, new_l, path.len() - 1)).or_default().extend_from_slice(&path);
                }
            }
        }
    }
}

/// The tuples starting at `s` ending at a target (`targets[t]`), into [`Buckets`]: in parallel over the second vertex
/// if `parallel` (concatenating in order keeps them lexicographic).
fn enumerate(graph: &MagnitudeGraph, s: u32, l_max: usize, k_top: usize, targets: &[bool], parallel: bool) -> Buckets {
    let n = graph.n as u32;
    let reach: Vec<u32> = (0..n)
        .map(|x| (0..n).filter(|&t| targets[t as usize]).map(|t| graph.distance(x, t)).min().unwrap_or(UNREACHABLE))
        .collect();
    let mut buckets: Buckets = HashMap::new();
    if targets[s as usize] {
        buckets.entry((s, 0, 0)).or_default().push(s);
    }
    if reach[s as usize] == UNREACHABLE || k_top == 0 {
        return buckets;
    }
    if !parallel {
        enumerate_from(graph, &[s], 0, l_max, k_top, &reach, &mut buckets);
        return buckets;
    }
    // Each second vertex's subtree on its own, then concatenated in order
    let seconds: Vec<(u32, usize)> = (0..n)
        .filter_map(|x| {
            let d = graph.distance(s, x);
            let r = reach[x as usize];
            (x != s && d != UNREACHABLE && r != UNREACHABLE && d as usize + r as usize <= l_max).then_some((x, d as usize))
        })
        .collect();
    let parts: Vec<Buckets> = seconds
        .par_iter()
        .map(|&(x, d)| {
            let mut part = Buckets::new();
            if reach[x as usize] == 0 {
                part.entry((x, d, 1)).or_default().extend_from_slice(&[s, x]);
            }
            enumerate_from(graph, &[s, x], d, l_max, k_top, &reach, &mut part);
            part
        })
        .collect();
    for part in parts {
        for (key, tuples) in part {
            buckets.entry(key).or_default().extend_from_slice(&tuples);
        }
    }
    buckets
}

/// One piece `MC_{*,l}(s, t)` in degrees `0..=k_top`, after collapsing: the cells left (each with its degree) and
/// their boundaries (restricted to the cells left), with coefficients ±1.
/// A piece after collapsing: a complex over the integers in compressed columns.
type Collapsed = CompressedComplex;

/// Build the piece `(t, l)` from its tuples and collapse it.
///
/// The boundary is stored once in compressed rows, with its transpose. Then pairs are removed while there are any:
/// a cell with one live coface and that coface (an elementary collapse), or a cell with one live face and that face
/// (a coreduction). Neither changes the boundary of any other live cell beyond dropping the pair (the coefficients
/// are ±1, and the removed face or coface had no other live incidence), so the homology of the cells left, with the
/// boundary restricted to them, is the piece's, over the integers. This removes nearly every cell (as in Perseus 6,
/// whose fill-in-free phase this is).
fn collapse_piece(graph: &MagnitudeGraph, buckets: &Buckets, t: u32, l: usize, k_top: usize) -> Collapsed {
    let size = |k: usize| buckets.get(&(t, l, k)).map_or(0, |v| v.len() / (k + 1));
    let mut offsets = vec![0usize];
    for k in 0..=k_top {
        offsets.push(offsets[k] + size(k));
    }
    let n_cells = offsets[k_top + 1];
    let mut degree = Vec::with_capacity(n_cells);
    let mut indptr: Vec<usize> = Vec::with_capacity(n_cells + 1);
    indptr.push(0);
    let mut faces: Vec<u32> = vec![];
    let mut signs: Vec<i8> = vec![];
    let mut face = Vec::with_capacity(k_top + 1);
    for k in 0..=k_top {
        let Some(tuples) = buckets.get(&(t, l, k)) else { continue };
        let below = buckets.get(&(t, l, k.wrapping_sub(1)));
        for tuple in tuples.chunks_exact(k + 1) {
            degree.push(k as u32);
            for i in 1..k {
                let (a, x, b) = (tuple[i - 1], tuple[i], tuple[i + 1]);
                // x is on a geodesic from a to b (the three distances are finite: a -> x -> b is a path)
                if graph.distance(a, x) + graph.distance(x, b) != graph.distance(a, b) {
                    continue;
                }
                face.clear();
                face.extend_from_slice(&tuple[..i]);
                face.extend_from_slice(&tuple[i + 1..]);
                let below = below.expect("a face has the same ends and length, one vertex fewer");
                // Binary search over the faces, k vertices each, in lexicographic order
                let (mut lo, mut hi) = (0, below.len() / k);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if below[mid * k..(mid + 1) * k] < face[..] {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                debug_assert_eq!(&below[lo * k..(lo + 1) * k], &face[..]);
                faces.push(u32::try_from(offsets[k - 1] + lo).expect("fewer than 2^32 cells in a piece"));
                signs.push(if i % 2 == 0 { 1 } else { -1 });
            }
            indptr.push(faces.len());
        }
    }
    // The transpose: each cell's cofaces
    let mut co_ptr = vec![0usize; n_cells + 1];
    for &f in &faces {
        co_ptr[f as usize + 1] += 1;
    }
    for i in 0..n_cells {
        co_ptr[i + 1] += co_ptr[i];
    }
    let mut cofaces = vec![0u32; faces.len()];
    let mut fill = co_ptr.clone();
    for c in 0..n_cells {
        for &f in &faces[indptr[c]..indptr[c + 1]] {
            cofaces[fill[f as usize]] = c as u32;
            fill[f as usize] += 1;
        }
    }
    drop(fill);
    // Live counts, and the cells that may be removed
    let mut alive = vec![true; n_cells];
    let mut n_faces: Vec<u32> = (0..n_cells).map(|c| (indptr[c + 1] - indptr[c]) as u32).collect();
    let mut n_cofaces: Vec<u32> = (0..n_cells).map(|c| (co_ptr[c + 1] - co_ptr[c]) as u32).collect();
    // First in, first out: removing pairs in the order they become free removes far more than last in, first out
    // (on Kaneta and Yoshinaga's graph of RP^2 at l = 10, 99.99% of the cells against 20%, the rest left to the
    // costlier second phase)
    let mut queue: std::collections::VecDeque<u32> =
        (0..n_cells as u32).filter(|&c| n_faces[c as usize] == 1 || n_cofaces[c as usize] == 1).collect();
    // Remove a cell: its live faces lose a coface and its live cofaces a face
    let remove = |c: usize,
                  alive: &mut [bool],
                  n_faces: &mut [u32],
                  n_cofaces: &mut [u32],
                  queue: &mut std::collections::VecDeque<u32>| {
        alive[c] = false;
        for &f in &faces[indptr[c]..indptr[c + 1]] {
            let f = f as usize;
            if alive[f] {
                n_cofaces[f] -= 1;
                if n_cofaces[f] == 1 {
                    queue.push_back(f as u32);
                }
            }
        }
        for &g in &cofaces[co_ptr[c]..co_ptr[c + 1]] {
            let g = g as usize;
            if alive[g] {
                n_faces[g] -= 1;
                if n_faces[g] == 1 {
                    queue.push_back(g as u32);
                }
            }
        }
    };
    while let Some(c) = queue.pop_front() {
        let c = c as usize;
        if !alive[c] {
            continue;
        }
        let partner = if n_cofaces[c] == 1 {
            cofaces[co_ptr[c]..co_ptr[c + 1]].iter().map(|&g| g as usize).find(|&g| alive[g])
        } else if n_faces[c] == 1 {
            faces[indptr[c]..indptr[c + 1]].iter().map(|&f| f as usize).find(|&f| alive[f])
        } else {
            None
        };
        if let Some(partner) = partner {
            remove(c, &mut alive, &mut n_faces, &mut n_cofaces, &mut queue);
            remove(partner, &mut alive, &mut n_faces, &mut n_cofaces, &mut queue);
        }
    }
    // The cells left, renumbered in order (so faces still come first)
    let mut new_index = vec![u32::MAX; n_cells];
    let mut out = Collapsed { degrees: vec![], indptr: vec![0], faces: vec![], coefficients: vec![] };
    for c in 0..n_cells {
        if alive[c] {
            new_index[c] = out.degrees.len() as u32;
            out.degrees.push(degree[c]);
            for e in indptr[c]..indptr[c + 1] {
                let f = faces[e] as usize;
                if alive[f] {
                    out.faces.push(new_index[f]);
                    out.coefficients.push(i64::from(signs[e]));
                }
            }
            out.indptr.push(out.faces.len());
        }
    }
    out
}

/// The ranks over `CF` of the homology of a collapsed piece, in degrees `0..ranks.len()`.
fn collapsed_ranks<CF: IntField>(piece: &Collapsed, n_degrees: usize) -> Vec<usize> {
    let mut ranks = vec![0; n_degrees];
    let n = piece.degrees.len();
    if n == 0 {
        return ranks;
    }
    // Cells with no faces and no cofaces left are generators, so cheap pieces need no reduction at all
    if piece.faces.is_empty() {
        for &k in &piece.degrees {
            if (k as usize) < n_degrees {
                ranks[k as usize] += 1;
            }
        }
        return ranks;
    }
    let raw = RawChain {
        dimensions: piece.degrees.iter().map(|&k| i64::from(k)).collect(),
        indptr: piece.indptr.iter().map(|&p| p as i64).collect(),
        indices: piece.faces.iter().map(|&f| i64::from(f)).collect(),
        coefficients: Some(piece.coefficients.clone()),
    };
    let complex = Chain::<CF>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(vec![0.0; n])))
        .expect("a magnitude chain complex, faces first");
    let algorithm = Algorithm::Clearing(Clearing::new().with_columns(phlite::columns::PivotColumns).store_r(true));
    for bar in complex.barcode_by_rank(None, Method::Cohomology, algorithm, false) {
        if bar.death.is_none() {
            let k = complex.at_rank(bar.birth).2;
            if k < n_degrees {
                ranks[k] += 1;
            }
        }
    }
    ranks
}

/// [`collapsed_ranks`] over `Z/modulus`.
fn collapsed_ranks_mod(piece: &Collapsed, n_degrees: usize, modulus: u64) -> Vec<usize> {
    use phlite::fields::{Z2, ZP};
    match modulus {
        2 => collapsed_ranks::<Z2>(piece, n_degrees),
        3 => collapsed_ranks::<ZP<3>>(piece, n_degrees),
        5 => collapsed_ranks::<ZP<5>>(piece, n_degrees),
        7 => collapsed_ranks::<ZP<7>>(piece, n_degrees),
        11 => collapsed_ranks::<ZP<11>>(piece, n_degrees),
        13 => collapsed_ranks::<ZP<13>>(piece, n_degrees),
        251 => collapsed_ranks::<ZP<251>>(piece, n_degrees),
        other => panic!("unsupported modulus {other}, expected one of {MODULI:?}"),
    }
}

/// Run `f` on every piece `MC_{*,l}(s, t)` asked for, once collapsed and eliminated (with the number of degrees
/// wanted, `min(l, k_max) + 1`): `(s, t, l, f(piece))`, sorted. In parallel on the current rayon pool if `parallel`:
/// over sources, and, when there are few sources, also over each source's tuples and pieces.
fn for_each_piece<T: Send>(
    graph: &MagnitudeGraph,
    query: &MagnitudeQuery,
    parallel: bool,
    f: impl Fn(&Collapsed, usize) -> T + Sync,
) -> Vec<(u32, u32, usize, T)> {
    let n = graph.n;
    let l_max = query.l_max;
    let k_max = query.k_max.unwrap_or(l_max).min(l_max);
    let mut targets: Vec<Vec<bool>> = vec![];
    let sources: Vec<u32> = match &query.pairs {
        None => (0..n as u32).collect(),
        Some(pairs) => {
            targets = vec![vec![false; n]; n];
            for &(s, t) in pairs {
                targets[s as usize][t as usize] = true;
            }
            let mut sources: Vec<u32> = pairs.iter().map(|&(s, _)| s).collect();
            sources.sort_unstable();
            sources.dedup();
            sources
        }
    };
    let all = vec![true; n];
    // Few sources: parallelise within each instead
    let inner = parallel && sources.len() < 2 * rayon::current_num_threads();
    let one_source = |&s: &u32| -> Vec<(u32, u32, usize, T)> {
        let wanted = if query.pairs.is_some() { &targets[s as usize] } else { &all };
        let buckets = enumerate(graph, s, l_max, k_max + 1, wanted, inner);
        let pieces: Vec<(u32, usize)> = (0..n as u32)
            .filter(|&t| wanted[t as usize] && graph.distance(s, t) != UNREACHABLE)
            .flat_map(|t| (graph.distance(s, t) as usize..=l_max).map(move |l| (t, l)))
            .collect();
        let piece = |&(t, l): &(u32, usize)| {
            let (collapsed, _) = eliminate(collapse_piece(graph, &buckets, t, l, (k_max + 1).min(l)), None, ELIMINATION_COST);
            (s, t, l, f(&collapsed, k_max.min(l) + 1))
        };
        if inner {
            pieces.par_iter().map(piece).collect()
        } else {
            pieces.iter().map(piece).collect()
        }
    };
    let mut out: Vec<(u32, u32, usize, T)> = if parallel && !inner {
        sources.par_iter().flat_map_iter(one_source).collect()
    } else {
        sources.iter().flat_map(one_source).collect()
    };
    out.sort_unstable_by_key(|&(s, t, l, _)| (s, t, l));
    out
}

/// The ranks of the magnitude homology of `graph` over `Z/m` for each `m` in `moduli` (each one of [`MODULI`]), for
/// each pair asked for (or every pair), with lengths up to `query.l_max` and degrees up to `query.k_max` (or
/// `l_max`). Pairs whose ranks are all zero are left out. Each piece is collapsed once, whatever the moduli.
///
/// Work runs in parallel on the current rayon pool if `parallel`: over sources, and, when there are few sources, also
/// over each source's tuples and pieces.
///
/// # Panics
/// If a modulus isn't one of [`MODULI`].
pub fn magnitude_ranks(graph: &MagnitudeGraph, query: &MagnitudeQuery, moduli: &[u64], parallel: bool) -> Vec<PairRanks> {
    for m in moduli {
        assert!(MODULI.contains(m), "unsupported modulus {m}, expected one of {MODULI:?}");
    }
    let l_max = query.l_max;
    let k_max = query.k_max.unwrap_or(l_max).min(l_max);
    let pieces = for_each_piece(graph, query, parallel, |piece, n_degrees| {
        moduli.iter().map(|&m| collapsed_ranks_mod(piece, n_degrees, m)).collect::<Vec<Vec<usize>>>()
    });
    let mut out: Vec<PairRanks> = vec![];
    for (s, t, l, ranks) in pieces {
        if out.last().is_none_or(|p| (p.source, p.target) != (s, t)) {
            out.push(PairRanks {
                source: s,
                target: t,
                ranks: moduli.iter().map(|_| (0..=l_max).map(|l| vec![0; k_max.min(l) + 1]).collect()).collect(),
            });
        }
        let pair = out.last_mut().expect("pushed");
        for (m, r) in ranks.into_iter().enumerate() {
            pair.ranks[m][l] = r;
        }
    }
    out.retain(|p| p.ranks.iter().flatten().flatten().any(|&r| r > 0));
    out
}

/// An integral magnitude homology group `Z^free ⊕ Z/t_1 ⊕ ... ⊕ Z/t_m`, with the invariant factors `t_i > 1` (each
/// dividing the next), or `None` if the Smith form overflowed or was too large (see [`magnitude_integral`]).
pub type Group = Option<(usize, Vec<u128>)>;

/// The integral homology of a collapsed piece in degrees `0..n_degrees` ([`homology_groups`]), as [`Group`]s.
fn integral_groups(piece: &Collapsed, n_degrees: usize) -> Vec<Group> {
    homology_groups(piece, n_degrees).into_iter().map(|g| g.map(|g| (g.rank, g.torsion))).collect()
}

/// The integral magnitude homology of one pair: `groups[l][k]` is `MH_{k,l}(s, t; Z)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairGroups {
    pub source: u32,
    pub target: u32,
    pub groups: Vec<Vec<Group>>,
}

/// The integral magnitude homology of `graph`, for each pair asked for (or every pair), with lengths up to
/// `query.l_max` and degrees up to `query.k_max` (or `l_max`): each piece collapsed and eliminated over the integers,
/// then the Smith forms of what is left (nearly always tiny). Pairs whose groups are all zero are left out.
pub fn magnitude_integral(graph: &MagnitudeGraph, query: &MagnitudeQuery, parallel: bool) -> Vec<PairGroups> {
    let l_max = query.l_max;
    let k_max = query.k_max.unwrap_or(l_max).min(l_max);
    let pieces = for_each_piece(graph, query, parallel, integral_groups);
    let mut out: Vec<PairGroups> = vec![];
    for (s, t, l, groups) in pieces {
        if out.last().is_none_or(|p| (p.source, p.target) != (s, t)) {
            out.push(PairGroups {
                source: s,
                target: t,
                groups: (0..=l_max).map(|l| vec![Some((0, vec![])); k_max.min(l) + 1]).collect(),
            });
        }
        out.last_mut().expect("pushed").groups[l] = groups;
    }
    out.retain(|p| p.groups.iter().flatten().any(|g| g != &Some((0, vec![]))));
    out
}

/// The ranks over the `m`-th modulus summed over the pairs: `total[l][k]`.
pub fn total_ranks(pairs: &[PairRanks], m: usize, l_max: usize, k_max: Option<usize>) -> Vec<Vec<usize>> {
    let k_max = k_max.unwrap_or(l_max).min(l_max);
    let mut total: Vec<Vec<usize>> = (0..=l_max).map(|l| vec![0; k_max.min(l) + 1]).collect();
    for pair in pairs {
        for (l, row) in pair.ranks[m].iter().enumerate() {
            for (k, &r) in row.iter().enumerate() {
                total[l][k] += r;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn cycle(n: u32, directed: bool) -> Vec<(u32, u32)> {
        let mut edges: Vec<(u32, u32)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
        if !directed {
            edges.extend((0..n).map(|i| ((i + 1) % n, i)));
        }
        edges
    }

    /// The magnitude homology by brute force over Z/2: every tuple, every differential as a dense matrix.
    fn brute_force(graph: &MagnitudeGraph, l_max: usize) -> Vec<Vec<usize>> {
        let n = graph.n as u32;
        let mut tuples: HashMap<(usize, usize), Vec<Vec<u32>>> = HashMap::new();
        let mut stack: Vec<(Vec<u32>, usize)> = (0..n).map(|s| (vec![s], 0)).collect();
        while let Some((path, l)) = stack.pop() {
            tuples.entry((path.len() - 1, l)).or_default().push(path.clone());
            for x in 0..n {
                let last = *path.last().unwrap();
                let d = graph.distance(last, x);
                if x != last && d != UNREACHABLE && l + d as usize <= l_max {
                    let mut p = path.clone();
                    p.push(x);
                    stack.push((p, l + d as usize));
                }
            }
        }
        let rank = |rows: &[Vec<u32>], cols: &[Vec<u32>]| -> usize {
            // Rank over Z/2 of the differential from cols (size k+1) to rows (size k)
            let index: HashMap<&Vec<u32>, usize> = rows.iter().enumerate().map(|(i, r)| (r, i)).collect();
            let mut matrix: Vec<Vec<bool>> = cols
                .iter()
                .map(|c| {
                    let mut v = vec![false; rows.len()];
                    for i in 1..c.len() - 1 {
                        if graph.distance(c[i - 1], c[i]) + graph.distance(c[i], c[i + 1]) == graph.distance(c[i - 1], c[i + 1]) {
                            let mut f = c.clone();
                            f.remove(i);
                            v[index[&f]] ^= true;
                        }
                    }
                    v
                })
                .collect();
            let mut rank = 0;
            let n_rows = rows.len();
            for r in 0..n_rows {
                if let Some(p) = (rank..matrix.len()).find(|&j| matrix[j][r]) {
                    matrix.swap(rank, p);
                    for j in 0..matrix.len() {
                        if j != rank && matrix[j][r] {
                            let pivot = matrix[rank].clone();
                            for (a, b) in matrix[j].iter_mut().zip(pivot) {
                                *a ^= b;
                            }
                        }
                    }
                    rank += 1;
                }
            }
            rank
        };
        let empty = vec![];
        (0..=l_max)
            .map(|l| {
                (0..=l)
                    .map(|k| {
                        let c = tuples.get(&(k, l)).unwrap_or(&empty);
                        let below = if k == 0 { 0 } else { rank(tuples.get(&(k - 1, l)).unwrap_or(&empty), c) };
                        let above = rank(c, tuples.get(&(k + 1, l)).unwrap_or(&empty));
                        c.len() - below - above
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn small_graphs_match_brute_force() {
        let graphs: Vec<(usize, Vec<(u32, u32)>)> = vec![
            (5, cycle(5, false)),
            (6, cycle(6, true)),
            (4, vec![(0, 1), (1, 2), (0, 3), (3, 2), (2, 0)]),
            (5, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 1), (2, 4), (4, 0), (0, 2)]),
        ];
        for (n, edges) in graphs {
            let graph = MagnitudeGraph::new(n, &edges);
            for l_max in 0..6 {
                let query = MagnitudeQuery { l_max, ..Default::default() };
                for parallel in [false, true] {
                    let got = total_ranks(&magnitude_ranks(&graph, &query, &[2], parallel), 0, l_max, None);
                    assert_eq!(got, brute_force(&graph, l_max), "{edges:?} l_max {l_max} parallel {parallel}");
                }
            }
        }
    }

    /// Known values: Hepworth and Willerton's table for the 5-cycle (MH_{k,l} has rank 10 at (1,1), and so on); here
    /// just the Euler characteristic of each row against the magnitude series' coefficients, and diagonality of the
    /// complete graph.
    #[test]
    fn complete_graphs_are_diagonal() {
        let n = 4;
        let edges: Vec<(u32, u32)> = (0..n).flat_map(|a| (0..n).filter(move |&b| b != a).map(move |b| (a, b))).collect();
        let graph = MagnitudeGraph::new(n as usize, &edges);
        let query = MagnitudeQuery { l_max: 4, ..Default::default() };
        let total = total_ranks(&magnitude_ranks(&graph, &query, &[3], true), 0, 4, None);
        for (l, row) in total.iter().enumerate() {
            for (k, &r) in row.iter().enumerate() {
                if k != l {
                    assert_eq!(r, 0, "MH_{k},{l}");
                }
            }
            // MH_{l,l}(K_n) has rank n (n-1)^l
            assert_eq!(row[l], n as usize * (n as usize - 1).pow(l as u32));
        }
    }

    /// The integral groups give every field's ranks by universal coefficients.
    #[test]
    fn integral_groups_agree_with_fields() {
        let graphs: Vec<(usize, Vec<(u32, u32)>)> = vec![
            (5, cycle(5, false)),
            (6, cycle(6, true)),
            (5, vec![(0, 1), (1, 0), (1, 2), (2, 3), (3, 1), (2, 4), (4, 0), (0, 2)]),
        ];
        for (n, edges) in graphs {
            let graph = MagnitudeGraph::new(n, &edges);
            let query = MagnitudeQuery { l_max: 5, ..Default::default() };
            let integral = magnitude_integral(&graph, &query, false);
            for (m, p) in [(0usize, 2u128), (1, 3)] {
                let ranks = magnitude_ranks(&graph, &query, &[2, 3], false);
                for pair in &integral {
                    let field = ranks.iter().find(|r| (r.source, r.target) == (pair.source, pair.target));
                    for (l, row) in pair.groups.iter().enumerate() {
                        for (k, group) in row.iter().enumerate() {
                            let (free, torsion) = group.clone().expect("small");
                            let divisible = |g: &Group| g.as_ref().map_or(0, |(_, t)| t.iter().filter(|&&d| d % p == 0).count());
                            let below = if k > 0 { divisible(&row[k - 1]) } else { 0 };
                            let want = free + torsion.iter().filter(|&&d| d % p == 0).count() + below;
                            let got = field.map_or(0, |f| f.ranks[m][l][k]);
                            assert_eq!(got, want, "{edges:?} ({k}, {l})");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn smith_forms() {
        assert_eq!(phlite::integral::smith_diagonal(vec![vec![2, 4], vec![6, 8]]), Ok(vec![2, 4]));
        assert_eq!(phlite::integral::smith_diagonal(vec![vec![2, 0], vec![0, 3]]), Ok(vec![1, 6]));
        assert_eq!(phlite::integral::smith_diagonal(vec![vec![0, 0], vec![0, 0]]), Ok(vec![]));
        assert_eq!(phlite::integral::smith_diagonal(vec![vec![4, 6, 0]]), Ok(vec![2]));
    }

    /// Restricting to pairs, and to degrees, gives the same ranks as computing everything.
    #[test]
    fn pairs_and_degrees_restrict() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (3, 1)];
        let graph = MagnitudeGraph::new(4, &edges);
        let all = magnitude_ranks(&graph, &MagnitudeQuery { l_max: 5, ..Default::default() }, &[2], false);
        for pair in &all {
            let query = MagnitudeQuery { l_max: 5, k_max: Some(3), pairs: Some(vec![(pair.source, pair.target)]) };
            for parallel in [false, true] {
                let one = magnitude_ranks(&graph, &query, &[2], parallel);
                assert_eq!(one.len(), 1);
                for (l, row) in one[0].ranks[0].iter().enumerate() {
                    assert_eq!(row[..], pair.ranks[0][l][..row.len()]);
                }
            }
        }
    }
}
