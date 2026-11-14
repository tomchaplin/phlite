//! Where the Rips filtration gets its distances from: a dense matrix, or a sparse one.
//!
//! The Rips oracles only ever ask for the distance between two vertices, and for the vertices that could be added to
//! a simplex to make a cofacet. A dense matrix answers the second with every vertex; a sparse matrix (where a missing
//! entry means the edge never appears) answers it with the neighbours of the simplex's vertex of smallest degree, so
//! the work per simplex scales with the degrees rather than with the number of points, and the memory with the number
//! of edges rather than its square.

use ordered_float::NotNan;

/// The distances between the points of a Rips filtration.
///
/// The distance between two different points may be infinite, meaning the edge between them never appears (in a
/// [`SparseDistances`], a missing entry). The distance from a point to itself is never asked for.
pub trait Distances: Sync {
    /// The number of points.
    fn n_points(&self) -> usize;

    /// The distance between the points `i` and `j` (with `i != j`).
    fn distance(&self, i: usize, j: usize) -> NotNan<f64>;

    /// Whether any point may appear after 0 (see [`birth`](Self::birth)): if not, births are left out of every
    /// computation.
    const HAS_BIRTHS: bool = false;

    /// When point `v` appears: 0 unless the points are weighted (see [`Weighted`]). The distance between two points
    /// must be at least both their births.
    #[inline(always)]
    fn birth(&self, _v: usize) -> NotNan<f64> {
        NotNan::new(0.0).unwrap()
    }

    /// Every vertex `w >= from` (other than `v`) whose distance to `v` might be finite, in increasing order (it may
    /// also list `v` itself, and vertices at infinite distance).
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_;

    /// How many vertices [`neighbours`](Self::neighbours) lists for `v` (to pick the shortest list).
    fn degree(&self, v: usize) -> usize;

    /// The enclosing radius `min_i max_{j != i} d(i, j)`: at this diameter the Rips complex is a cone on some point, so
    /// nothing changes afterwards. Infinite if there are no points, or if no point is joined to every other.
    fn enclosing_radius(&self) -> NotNan<f64> {
        let n = self.n_points();
        (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| self.distance(i, j))
                    .max()
                    .unwrap_or(NotNan::new(0.0).unwrap())
            })
            .min()
            .unwrap_or(NotNan::new(f64::INFINITY).unwrap())
    }

    /// The vertex of smallest degree among `vertices` (the first, if they tie), or `None` if there are none.
    #[inline]
    fn sparsest(&self, vertices: &[usize]) -> Option<usize> {
        vertices.iter().copied().min_by_key(|&v| self.degree(v))
    }

    /// The vertices `w >= from` that could be added to a simplex to make a cofacet, in increasing order: the
    /// neighbours of its [`sparsest`](Self::sparsest) vertex, or every point if it has none. (Two steps rather than
    /// one, so that the iterator borrows only the distances, not the simplex's vertices.)
    #[inline]
    fn cofacet_candidates(
        &self,
        sparsest: Option<usize>,
        from: usize,
    ) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        match sparsest {
            Some(v) => Candidates::Neighbours(self.neighbours(v, from)),
            None => Candidates::All(from..self.n_points()),
        }
    }
}

/// Either every point, or some point's neighbours (two iterator types as one).
#[derive(Clone)]
enum Candidates<N> {
    All(std::ops::Range<usize>),
    Neighbours(N),
}

impl<N: Iterator<Item = usize>> Iterator for Candidates<N> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            Candidates::All(range) => range.next(),
            Candidates::Neighbours(neighbours) => neighbours.next(),
        }
    }
}

// ====== Dense distance matrices ==========================================

/// A dense matrix: every vertex is a neighbour of every other (some perhaps at infinite distance).
impl Distances for [Vec<NotNan<f64>>] {
    #[inline(always)]
    fn n_points(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        self[i][j]
    }

    #[inline(always)]
    fn neighbours(&self, _v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        from..self.len()
    }

    #[inline(always)]
    fn degree(&self, _v: usize) -> usize {
        self.len()
    }

    // Every vertex is a candidate, so skip choosing one to take neighbours from
    #[inline(always)]
    fn sparsest(&self, _vertices: &[usize]) -> Option<usize> {
        None
    }

    #[inline(always)]
    fn cofacet_candidates(
        &self,
        _sparsest: Option<usize>,
        from: usize,
    ) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        from..self.len()
    }
}

impl Distances for Vec<Vec<NotNan<f64>>> {
    #[inline(always)]
    fn n_points(&self) -> usize {
        self.as_slice().n_points()
    }

    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        self.as_slice().distance(i, j)
    }

    #[inline(always)]
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        self.as_slice().neighbours(v, from)
    }

    #[inline(always)]
    fn degree(&self, v: usize) -> usize {
        self.as_slice().degree(v)
    }

    #[inline(always)]
    fn sparsest(&self, vertices: &[usize]) -> Option<usize> {
        self.as_slice().sparsest(vertices)
    }

    #[inline(always)]
    fn cofacet_candidates(
        &self,
        sparsest: Option<usize>,
        from: usize,
    ) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        self.as_slice().cofacet_candidates(sparsest, from)
    }
}

/// So that a boundary and a coboundary oracle can share one matrix.
impl<D: Distances + ?Sized> Distances for &D {
    const HAS_BIRTHS: bool = D::HAS_BIRTHS;

    #[inline(always)]
    fn n_points(&self) -> usize {
        (**self).n_points()
    }

    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        (**self).distance(i, j)
    }

    #[inline(always)]
    fn birth(&self, v: usize) -> NotNan<f64> {
        (**self).birth(v)
    }

    #[inline(always)]
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        (**self).neighbours(v, from)
    }

    #[inline(always)]
    fn degree(&self, v: usize) -> usize {
        (**self).degree(v)
    }

    fn enclosing_radius(&self) -> NotNan<f64> {
        (**self).enclosing_radius()
    }

    #[inline(always)]
    fn sparsest(&self, vertices: &[usize]) -> Option<usize> {
        (**self).sparsest(vertices)
    }

    #[inline(always)]
    fn cofacet_candidates(
        &self,
        sparsest: Option<usize>,
        from: usize,
    ) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        (**self).cofacet_candidates(sparsest, from)
    }
}

// ====== Sparse distance matrices =========================================

/// A sparse distance matrix: each point's finite distances to the others, by row (compressed sparse rows, symmetric,
/// columns increasing in each row). A missing entry is an edge that never appears.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseDistances {
    indptr: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<NotNan<f64>>,
}

impl SparseDistances {
    /// The sparse matrix on `n_points` points with these edges `(i, j, d)`, in any order.
    ///
    /// Each edge may be given once (either way round) or both ways round with the same distance; an edge from a
    /// point to itself is ignored (vertices always enter at 0), as is an infinite distance (the edge never appears).
    pub fn from_edges(
        n_points: usize,
        edges: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Result<Self, String> {
        if u32::try_from(n_points).is_err() {
            return Err(format!("Too many points ({n_points})"));
        }
        let mut entries: Vec<(u32, u32, NotNan<f64>)> = vec![];
        for (i, j, d) in edges {
            if i >= n_points || j >= n_points {
                return Err(format!(
                    "Edge ({i}, {j}) is out of range for {n_points} points"
                ));
            }
            if d.is_nan() || d < 0.0 {
                return Err(format!(
                    "The distance between {i} and {j} is {d}, but distances must be non-negative (or infinite)"
                ));
            }
            if i == j || d.is_infinite() {
                continue;
            }
            let d = NotNan::new(d).unwrap();
            entries.push((i as u32, j as u32, d));
            entries.push((j as u32, i as u32, d));
        }
        entries.sort_unstable();
        // An edge given twice (both ways round, or repeated) must have one distance
        let mut deduped: Vec<(u32, u32, NotNan<f64>)> = Vec::with_capacity(entries.len());
        for entry in entries {
            match deduped.last() {
                Some(&(i, j, d)) if (i, j) == (entry.0, entry.1) => {
                    if d != entry.2 {
                        return Err(format!(
                            "The distance between {i} and {j} is given as both {d} and {}",
                            entry.2
                        ));
                    }
                }
                _ => deduped.push(entry),
            }
        }
        let mut indptr = vec![0; n_points + 1];
        for &(i, _, _) in &deduped {
            indptr[i as usize + 1] += 1;
        }
        for i in 0..n_points {
            indptr[i + 1] += indptr[i];
        }
        Ok(Self {
            indptr,
            indices: deduped.iter().map(|&(_, j, _)| j).collect(),
            values: deduped.iter().map(|&(_, _, d)| d).collect(),
        })
    }

    /// The edges with distance at most `threshold` of the points `points` (one point per row, any dimension), under
    /// the metric `metric`, computed without ever storing every distance.
    ///
    /// Computed serially: rayon's global pool would hang a process forked after it started (as Python's multiprocessing
    /// does), and this is quick next to the reduction.
    pub fn from_points(
        points: &[Vec<f64>],
        metric: Metric,
        threshold: NotNan<f64>,
    ) -> Result<Self, String> {
        let n = points.len();
        if u32::try_from(n).is_err() {
            return Err(format!("Too many points ({n})"));
        }
        // Each pair once, then each edge into both rows (in increasing order, since i and then j increase)
        let mut rows: Vec<Vec<(u32, NotNan<f64>)>> = vec![vec![]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = metric.distance(&points[i], &points[j]);
                if d <= threshold {
                    rows[i].push((j as u32, d));
                    rows[j].push((i as u32, d));
                }
            }
        }
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for row in &rows {
            indptr.push(indptr.last().unwrap() + row.len());
        }
        Ok(Self {
            indptr,
            indices: rows.iter().flatten().map(|&(j, _)| j).collect(),
            values: rows.iter().flatten().map(|&(_, d)| d).collect(),
        })
    }

    /// The number of edges.
    pub fn n_edges(&self) -> usize {
        self.indices.len() / 2
    }

    fn row(&self, i: usize) -> (&[u32], &[NotNan<f64>]) {
        let range = self.indptr[i]..self.indptr[i + 1];
        (&self.indices[range.clone()], &self.values[range])
    }
}

impl Distances for SparseDistances {
    #[inline]
    fn n_points(&self) -> usize {
        self.indptr.len() - 1
    }

    #[inline]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        let (indices, values) = self.row(i);
        match indices.binary_search(&(j as u32)) {
            Ok(k) => values[k],
            Err(_) => NotNan::new(f64::INFINITY).unwrap(),
        }
    }

    #[inline]
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        let row = self.row(v).0;
        let start = row.partition_point(|&w| (w as usize) < from);
        row[start..].iter().map(|&w| w as usize)
    }

    #[inline]
    fn degree(&self, v: usize) -> usize {
        self.indptr[v + 1] - self.indptr[v]
    }

    fn enclosing_radius(&self) -> NotNan<f64> {
        let n = self.n_points();
        // Only a point joined to every other has a finite eccentricity
        (0..n)
            .filter(|&i| self.degree(i) == n - 1)
            .map(|i| {
                self.row(i)
                    .1
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(NotNan::new(0.0).unwrap())
            })
            .min()
            .unwrap_or(NotNan::new(f64::INFINITY).unwrap())
    }
}

// ====== Weighted points ===================================================

/// Weighted Rips, as GUDHI's `WeightedRipsComplex` (the weighted Rips filtration of Anai et al.'s DTM-filtrations,
/// with p = 1, doubled like Rips): point `i` with weight `w_i` appears at `2 w_i`, and the edge between `i` and `j` at
/// `max(2 w_i, 2 w_j, d(i, j) + w_i + w_j)`. Missing (infinite) distances stay missing. With the distance to measure
/// as the weights, this is DTM-Rips.
#[derive(Debug, Clone)]
pub struct Weighted<D> {
    distances: D,
    /// Twice the weights: the births
    births: Vec<NotNan<f64>>,
    weights: Vec<NotNan<f64>>,
}

impl<D: Distances> Weighted<D> {
    /// `distances` with a finite weight for every point.
    pub fn new(distances: D, weights: &[f64]) -> Result<Self, String> {
        let n = distances.n_points();
        if weights.len() != n {
            return Err(format!("There are {} weights for {n} points", weights.len()));
        }
        if let Some(i) = weights.iter().position(|w| !w.is_finite()) {
            return Err(format!("Weights must be finite, but weight {i} is {}", weights[i]));
        }
        Ok(Self {
            distances,
            births: weights.iter().map(|&w| NotNan::new(2.0 * w).unwrap()).collect(),
            weights: weights.iter().map(|&w| NotNan::new(w).unwrap()).collect(),
        })
    }
}

impl<D: Distances> Distances for Weighted<D> {
    const HAS_BIRTHS: bool = true;

    #[inline(always)]
    fn n_points(&self) -> usize {
        self.distances.n_points()
    }

    #[inline(always)]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        let d = self.distances.distance(i, j);
        if d.is_infinite() {
            return d;
        }
        // Add in the same order whichever way round the edge is asked for: floating point addition isn't associative,
        // and an edge whose length depended on the order could be in the complex from one end but not the other.
        // (NaN only from inf + -inf, impossible with finite weights.)
        let (lo, hi) = (i.min(j), i.max(j));
        // (The larger index's weight first, as GUDHI adds them, so that the values agree to the last bit)
        let sum = NotNan::new(*d + *self.weights[hi] + *self.weights[lo]).unwrap();
        sum.max(self.births[i]).max(self.births[j])
    }

    #[inline(always)]
    fn birth(&self, v: usize) -> NotNan<f64> {
        self.births[v]
    }

    #[inline(always)]
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        self.distances.neighbours(v, from)
    }

    #[inline(always)]
    fn degree(&self, v: usize) -> usize {
        self.distances.degree(v)
    }

    #[inline(always)]
    fn sparsest(&self, vertices: &[usize]) -> Option<usize> {
        self.distances.sparsest(vertices)
    }

    #[inline(always)]
    fn cofacet_candidates(
        &self,
        sparsest: Option<usize>,
        from: usize,
    ) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        self.distances.cofacet_candidates(sparsest, from)
    }

    /// The enclosing radius of the weighted distances (a point joined to every other within it is a cone point, and
    /// has appeared, since each edge appears after its ends), looking only at each point's neighbours.
    fn enclosing_radius(&self) -> NotNan<f64> {
        let n = self.n_points();
        let infinity = NotNan::new(f64::INFINITY).unwrap();
        (0..n)
            .map(|i| {
                let mut others = 0;
                let mut farthest = self.birth(i);
                for j in self.neighbours(i, 0).filter(|&j| j != i) {
                    others += 1;
                    farthest = farthest.max(self.distance(i, j));
                }
                if others == n - 1 {
                    farthest
                } else {
                    infinity
                }
            })
            .min()
            .unwrap_or(infinity)
    }
}

/// The distance to measure of each point (Chazal, Cohen-Steiner and Mérigot), as GUDHI's `DistanceToMeasure`:
/// `(sum of d^q over its k nearest points, itself included) / k)^(1/q)`, from each point's distances to every point
/// (`row(i)` gives point `i`'s, its own included). With `k = 1` every weight is 0.
pub fn distance_to_measure(n: usize, row: impl Fn(usize) -> Vec<f64>, k: usize, q: f64) -> Result<Vec<f64>, String> {
    if k == 0 || k > n {
        return Err(format!("k must be between 1 and the number of points ({n}), not {k}"));
    }
    if !(q.is_finite() && q > 0.0) {
        return Err(format!("q must be positive and finite, not {q}"));
    }
    Ok((0..n)
        .map(|i| {
            let mut distances = row(i);
            distances[i] = 0.0;
            distances.select_nth_unstable_by(k - 1, f64::total_cmp);
            let sum: f64 = distances[..k].iter().map(|d| d.powf(q)).sum();
            (sum / k as f64).powf(1.0 / q)
        })
        .collect())
}

// ====== Points ============================================================

/// A metric on points in `R^d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// The usual distance, `sqrt(sum (x_i - y_i)^2)`.
    Euclidean,
    /// `sum |x_i - y_i|`.
    Manhattan,
    /// `max |x_i - y_i|`.
    Chebyshev,
}

impl Metric {
    /// The distance between `x` and `y` (of the same length, with finite coordinates).
    pub fn distance(self, x: &[f64], y: &[f64]) -> NotNan<f64> {
        let diffs = x.iter().zip(y).map(|(a, b)| (a - b).abs());
        let d = match self {
            Metric::Euclidean => diffs.map(|d| d * d).sum::<f64>().sqrt(),
            Metric::Manhattan => diffs.sum(),
            Metric::Chebyshev => diffs.fold(0.0, f64::max),
        };
        NotNan::new(d).expect("coordinates must be finite")
    }

    /// The dense distance matrix of `points`.
    ///
    /// Computed serially, as for [`SparseDistances::from_points`], and each pair once.
    pub fn distance_matrix(self, points: &[Vec<f64>]) -> Vec<Vec<NotNan<f64>>> {
        let n = points.len();
        let mut matrix = vec![vec![NotNan::new(0.0).unwrap(); n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(&points[i], &points[j]);
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    #[test]
    fn sparse_matches_dense() {
        let inf = f64::INFINITY;
        let dense = vec![
            vec![0.0, 1.0, inf, 2.0],
            vec![1.0, 0.0, 3.0, inf],
            vec![inf, 3.0, 0.0, 0.5],
            vec![2.0, inf, 0.5, 0.0],
        ];
        let edges = (0..4).flat_map(|i| (0..4).map(move |j| (i, j))).filter(|(i, j)| i < j).map(|(i, j)| (i, j, dense[i][j]));
        let sparse = SparseDistances::from_edges(4, edges).unwrap();
        let dense: Vec<Vec<NotNan<f64>>> = dense.iter().map(|row| row.iter().map(|&d| nn(d)).collect()).collect();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert_eq!(sparse.distance(i, j), dense.distance(i, j), "({i}, {j})");
                }
            }
        }
        assert_eq!(sparse.n_edges(), 4);
        assert_eq!(sparse.neighbours(0, 0).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(sparse.neighbours(0, 2).collect::<Vec<_>>(), vec![3]);
        let sparsest = sparse.sparsest(&[2, 1]);
        assert_eq!(sparsest, Some(2));
        assert_eq!(sparse.cofacet_candidates(sparsest, 0).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(dense.sparsest(&[2, 1]), None);
        assert_eq!(dense.cofacet_candidates(None, 1).count(), 3);
        assert_eq!(sparse.enclosing_radius(), dense.enclosing_radius());
    }

    #[test]
    fn sparse_rejects_conflicts_and_reads_either_way_round() {
        assert!(SparseDistances::from_edges(2, [(0, 1, 1.0), (1, 0, 2.0)]).is_err());
        assert!(SparseDistances::from_edges(2, [(0, 2, 1.0)]).is_err());
        assert!(SparseDistances::from_edges(2, [(0, 1, -1.0)]).is_err());
        let one_way = SparseDistances::from_edges(3, [(2, 0, 1.5), (1, 1, 7.0)]).unwrap();
        let both = SparseDistances::from_edges(3, [(0, 2, 1.5), (2, 0, 1.5)]).unwrap();
        assert_eq!(one_way, both);
        assert_eq!(one_way.distance(0, 2), nn(1.5));
        assert_eq!(one_way.distance(0, 1), nn(f64::INFINITY));
    }

    #[test]
    fn points_to_sparse_keeps_edges_up_to_the_threshold() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![0.0, 1.0]];
        let sparse = SparseDistances::from_points(&points, Metric::Euclidean, nn(4.5)).unwrap();
        assert_eq!(sparse.n_edges(), 2);
        assert_eq!(sparse.distance(0, 1), nn(f64::INFINITY));
        assert_eq!(sparse.distance(1, 2), nn((9.0f64 + 9.0).sqrt()));
        assert_eq!(Metric::Manhattan.distance(&points[0], &points[1]), nn(7.0));
        assert_eq!(Metric::Chebyshev.distance(&points[0], &points[1]), nn(4.0));
    }
}
