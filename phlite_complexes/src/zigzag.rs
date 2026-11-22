//! Zigzag persistence, by FastZigzag (Dey and Hou, *Fast computation of zigzag persistence*, ESA 2022): a cell-wise
//! zigzag filtration `∅ = K_0 ↔ K_1 ↔ ... ↔ K_m` (each step inserting or deleting one cell) is converted into an
//! ordinary filtration of a complex twice its length, whose barcode, computed by phlite's usual reductions (cohomology,
//! clearing, union-find), maps back to the zigzag barcode interval by interval.
//!
//! 1. Each insertion makes a new cell (a copy, if the same cell is inserted again later), whose faces are the current
//!    copies of its faces; each deletion deletes the current copy. If the zigzag doesn't end empty, the cells still
//!    there are deleted at the end (cofaces first), and intervals reaching those deletions are cut at `m`.
//! 2. The up-down filtration `U`: all the insertions in order, then all the deletions in order (Dey and Hou,
//!    Proposition 9 and Fact 10); with `τ_0, ..., τ_{n-1}` the inserted cells and `τ_n, ..., τ_{2n-1}` the deleted ones.
//! 3. The extended filtration `E` of `U` by coning: a vertex `ω` at 0, `τ_j` at `j + 1`, and the cone `ω·τ_i` on each
//!    deleted cell at `3n - i`. Its barcode, less the one infinite bar, is that of `E`.
//! 4. An interval `[b, d]` of `E` (complexes `b` to `d`) in dimension `p` is an interval of `U` (Proposition 19):
//!    ordinary (`d < n`): `[b, d]`, closed-open, dimension `p`; relative (`b > n`): `[3n - d, 3n - b]`, open-closed,
//!    dimension `p - 1`; extended: `[b, 3n - d - 1]`, closed-closed, dimension `p`.
//! 5. An interval `[b, d]` of `U` is one of the input (Proposition 15), with `↓τ` and `↑τ` the positions of a cell's
//!    insertion and deletion in the input: closed-open `[↓τ_{b-1} + 1, ↓τ_d]`; open-closed `[↑τ_{b-1} + 1, ↑τ_d]`;
//!    closed-closed `[↓τ_{b-1} + 1, ↑τ_d]` if `↓τ_{b-1} < ↑τ_d`, and otherwise open-open `[↑τ_d + 1, ↓τ_{b-1}]` in
//!    dimension `p - 1`.
//!
//! Over Z/2. A birth at `b` is closed if `K_{b-1} -> K_b` is an insertion (open if a deletion); a death at `d` is closed
//! if `K_d <- K_{d+1}` is a deletion (open if an insertion).
//!
//! # Zigzags of complexes
//!
//! Most zigzags are sequences of complexes `K_0, K_1, ..., K_s` (windows of a point cloud, frames of a video,
//! snapshots of a network), joined either by unions `K_0 -> K_0 ∪ K_1 <- K_1 -> ...` or by intersections
//! `K_0 <- K_0 ∩ K_1 -> K_1 <- ...`. Anything implementing [`ZigzagComplex`] (its cells, named by keys that are the
//! same from one complex to the next, and their faces) can be pushed onto a [`ComplexZigzag`], which works out the
//! operations and reads the barcode back in *levels* ([`LevelBar`]: `K_i` at level `2 i`, the join of `K_i` and
//! `K_{i+1}` at `2 i + 1`). Implemented here for simplicial complexes ([`SimplicialComplex`], and sets of vertex
//! lists, sorted or not, so directed flag complexes too); `phlite_clouds`, `phlite_networks` and `phlite_images` build
//! their complexes (Rips, Čech, alpha, witness; clique and directed flag complexes; cubical complexes of images) as
//! these or implement the trait.
//!
//! ```
//! use phlite::barcode::Algorithm;
//! use phlite::reduction::Clearing;
//! use phlite_complexes::zigzag::{ComplexZigzag, Join, LevelBar, SimplicialComplex};
//!
//! // Two edges of a triangle, then the other two: the loop needs the union of both
//! let a = SimplicialComplex::from_maximal([vec![0, 1], vec![1, 2]]);
//! let b = SimplicialComplex::from_maximal([vec![1, 2], vec![0, 2]]);
//! let bars = ComplexZigzag::new(Join::Union).push(&a).push(&b).barcode(Algorithm::Clearing(Clearing::new()))?;
//! assert!(bars.contains(&LevelBar { dimension: 1, birth: 1, death: 1 }));
//! # Ok::<(), phlite_complexes::zigzag::ZigzagError>(())
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

use phlite::barcode::{Algorithm, Method};
use phlite::columns::ColumnRepresentation;
use phlite::fields::Z2;

use crate::complex::{Chain, RawChain};

/// One step of a zigzag filtration: insert a cell (with its faces, which must be there), or delete one (which must be
/// there, with no coface).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation<K> {
    Insert { cell: K, dimension: usize, faces: Vec<K> },
    Delete(K),
}

/// An interval of a zigzag barcode: homology of `dimension` alive in `K_birth, ..., K_death`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZigzagBar {
    pub dimension: usize,
    pub birth: usize,
    pub death: usize,
    /// Born by an insertion (`K_{birth-1} -> K_birth`), rather than a deletion
    pub birth_closed: bool,
    /// Dying by a deletion (`K_death <- K_{death+1}`), rather than an insertion; also true for an interval alive at
    /// the end, cut at `m`
    pub death_closed: bool,
}

/// What can be wrong with the operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZigzagError {
    /// Operation `index` inserts a cell already there
    AlreadyThere(usize),
    /// Operation `index` deletes a cell not there
    NotThere(usize),
    /// Operation `index` inserts a cell with a face not there
    MissingFace(usize),
    /// Operation `index` deletes a cell with a coface still there
    CofaceThere(usize),
}

impl std::fmt::Display for ZigzagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyThere(i) => write!(f, "operation {i} inserts a cell that is already there"),
            Self::NotThere(i) => write!(f, "operation {i} deletes a cell that isn't there"),
            Self::MissingFace(i) => write!(f, "operation {i} inserts a cell with a face that isn't there"),
            Self::CofaceThere(i) => write!(f, "operation {i} deletes a cell whose coface is still there"),
        }
    }
}

impl std::error::Error for ZigzagError {}

/// The zigzag barcode of `operations` (starting from the empty complex), by FastZigzag with `algorithm` (see the
/// [module docs](self)). Intervals are in no particular order.
///
/// # Errors
/// If an operation is impossible (see [`ZigzagError`]).
pub fn zigzag_barcode<K: Hash + Eq + Clone, R: ColumnRepresentation>(
    operations: &[Operation<K>],
    algorithm: Algorithm<'_, R>,
) -> Result<Vec<ZigzagBar>, ZigzagError> {
    zigzag_barcode_with(operations, Method::Cohomology, algorithm, true)
}

/// As [`zigzag_barcode`] (which reduces by cohomology with union-find), reducing the coned filtration by `method`, and
/// (by cohomology) finding dimension 0 and the top dimension by union-find where they allow it if `union_find`.
///
/// # Errors
/// If an operation is impossible (see [`ZigzagError`]).
pub fn zigzag_barcode_with<K: Hash + Eq + Clone, R: ColumnRepresentation>(
    operations: &[Operation<K>],
    method: Method,
    algorithm: Algorithm<'_, R>,
    union_find: bool,
) -> Result<Vec<ZigzagBar>, ZigzagError> {
    let m = operations.len();
    // 1. Cells: one per insertion (its position in the input), with faces as cells
    let mut current: HashMap<K, usize> = HashMap::new();
    let mut dimension: Vec<usize> = vec![];
    let mut faces: Vec<Vec<usize>> = vec![];
    let mut inserted_at: Vec<usize> = vec![];
    let mut deleted: Vec<(usize, usize)> = vec![]; // (cell, position)
    let mut cofaces: Vec<usize> = vec![]; // number of cofaces present
    for (i, op) in operations.iter().enumerate() {
        match op {
            Operation::Insert { cell, dimension: d, faces: f } => {
                if current.contains_key(cell) {
                    return Err(ZigzagError::AlreadyThere(i));
                }
                let mut cell_faces = Vec::with_capacity(f.len());
                for face in f {
                    let &id = current.get(face).ok_or(ZigzagError::MissingFace(i))?;
                    cell_faces.push(id);
                    cofaces[id] += 1;
                }
                current.insert(cell.clone(), dimension.len());
                dimension.push(*d);
                faces.push(cell_faces);
                inserted_at.push(i);
                cofaces.push(0);
            }
            Operation::Delete(cell) => {
                let id = current.remove(cell).ok_or(ZigzagError::NotThere(i))?;
                if cofaces[id] > 0 {
                    return Err(ZigzagError::CofaceThere(i));
                }
                for &f in &faces[id] {
                    cofaces[f] -= 1;
                }
                deleted.push((id, i));
            }
        }
    }
    // Delete what is left at the end, cofaces first (in reverse order of insertion)
    let mut left: Vec<usize> = current.into_values().collect();
    left.sort_unstable_by(|a, b| b.cmp(a));
    for (k, id) in left.into_iter().enumerate() {
        deleted.push((id, m + k));
    }
    let n = dimension.len();
    if n == 0 {
        return Ok(vec![]);
    }
    // 2-3. Ê: ω, then the cells in order of insertion, then cones on the deleted cells (last deleted first)
    let omega = 0;
    let cell_index = |id: usize| id + 1;
    let mut cone_index = vec![0usize; n];
    let mut raw = RawChain { dimensions: vec![0], indptr: vec![0, 0], indices: vec![], coefficients: None };
    let mut values: Vec<f64> = vec![0.0];
    for id in 0..n {
        raw.dimensions.push(dimension[id] as i64);
        raw.indices.extend(faces[id].iter().map(|&f| cell_index(f) as i64));
        raw.indptr.push(raw.indices.len() as i64);
        values.push((id + 1) as f64);
    }
    for (k, &(id, _)) in deleted.iter().enumerate().rev() {
        // τ_i with i = n + k, coned at 3n - i = 2n - k
        let index = values.len();
        cone_index[id] = index;
        raw.dimensions.push(dimension[id] as i64 + 1);
        raw.indices.push(cell_index(id) as i64);
        if faces[id].is_empty() {
            raw.indices.push(omega as i64);
        } else {
            raw.indices.extend(faces[id].iter().map(|&f| cone_index[f] as i64));
        }
        raw.indptr.push(raw.indices.len() as i64);
        values.push((2 * n - k) as f64);
    }
    let complex = Chain::<Z2>::from_raw(raw)
        .and_then(|chain| chain.filter(Some(values)))
        .expect("the coned up-down filtration is a filtered complex");
    let bars = complex.barcode_with(None, method, algorithm, false, union_find);
    // 4-5. Back to the input
    let up = |k: usize| inserted_at[k]; // ↓τ_k, k < n
    let down = |k: usize| deleted[k - n].1; // ↑τ_k, k >= n
    let mut out = vec![];
    for bar in bars {
        let p = complex.dimension(bar.birth);
        let b = bar.birth_value.into_inner() as usize;
        let Some((_, death)) = bar.death else {
            continue; // the cone point's class
        };
        let d = death.into_inner() as usize - 1;
        let (u_b, u_d, dim, kind) = if d < n {
            (b, d, p, (true, false))
        } else if b > n {
            (3 * n - d, 3 * n - b, p - 1, (false, true))
        } else {
            (b, 3 * n - d - 1, p, (true, true))
        };
        // Creator τ_{u_b - 1} and destroyer τ_{u_d}
        let interval = match kind {
            (true, false) => ZigzagBar { dimension: dim, birth: up(u_b - 1) + 1, death: up(u_d), birth_closed: true, death_closed: false },
            (false, true) => {
                ZigzagBar { dimension: dim, birth: down(u_b - 1) + 1, death: down(u_d), birth_closed: false, death_closed: true }
            }
            _ => {
                let (insert, delete) = (up(u_b - 1), down(u_d));
                if insert < delete {
                    ZigzagBar { dimension: dim, birth: insert + 1, death: delete, birth_closed: true, death_closed: true }
                } else {
                    ZigzagBar { dimension: dim - 1, birth: delete + 1, death: insert, birth_closed: false, death_closed: false }
                }
            }
        };
        // The deletions added at the end: cut intervals at m, drop those after it
        if interval.birth > m {
            continue;
        }
        let interval = if interval.death >= m { ZigzagBar { death: m, death_closed: true, ..interval } } else { interval };
        out.push(interval);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------
// Zigzags of complexes

/// A finite cell complex that can be a term of a zigzag: its cells, each named by a key that names the same cell in
/// every complex of the zigzag (the vertices of a simplex, the position of a cube on the grid), with their dimensions
/// and faces. Only which faces a cell has matters (FastZigzag is over Z/2).
pub trait ZigzagComplex {
    type Key: Hash + Ord + Clone;

    /// Every cell, with its dimension, in any order.
    fn cells(&self) -> Vec<(Self::Key, usize)>;

    /// The faces of a cell of the complex (the cells of one dimension less in its boundary), each in the complex.
    fn faces(&self, cell: &Self::Key) -> Vec<Self::Key>;
}

/// The facets of a simplex given by its vertices: the simplex without each vertex in turn (none for a vertex).
/// The vertices needn't be sorted: an ordered simplex (of a directed flag complex) has ordered facets.
pub fn simplex_facets<V: Clone>(simplex: &[V]) -> Vec<Vec<V>> {
    if simplex.len() < 2 {
        return vec![];
    }
    (0..simplex.len())
        .map(|i| simplex.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, v)| v.clone()).collect())
        .collect()
}

/// A simplicial complex as the set of its simplices, each a list of vertices: sorted for an ordinary simplicial
/// complex, or ordered (each a directed clique, `x_i -> x_j` for `i < j`) for a directed flag complex, whose facets
/// keep the order. Every facet of a simplex must be in the set ([`SimplicialComplex::from_maximal`] adds them).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SimplicialComplex<V: Ord> {
    pub simplices: BTreeSet<Vec<V>>,
}

impl<V: Ord + Clone> SimplicialComplex<V> {
    /// The simplices given, which must include every face of each.
    pub fn new(simplices: impl IntoIterator<Item = Vec<V>>) -> Self {
        Self { simplices: simplices.into_iter().collect() }
    }

    /// The simplices given and all their faces (keeping the order of each simplex's vertices).
    pub fn from_maximal(simplices: impl IntoIterator<Item = Vec<V>>) -> Self {
        let mut all = BTreeSet::new();
        let mut stack: Vec<Vec<V>> = simplices.into_iter().filter(|s| !s.is_empty()).collect();
        while let Some(s) = stack.pop() {
            if !all.contains(&s) {
                stack.extend(simplex_facets(&s));
                all.insert(s);
            }
        }
        Self { simplices: all }
    }

    /// The simplices of a filtration (value, vertices) with value at most `threshold`: a sublevel set, as listed by
    /// `phlite_clouds`' Rips, Čech, alpha and witness complexes.
    pub fn sublevel<T: PartialOrd>(simplices: impl IntoIterator<Item = (T, Vec<V>)>, threshold: T) -> Self {
        Self::new(simplices.into_iter().filter(|(value, _)| *value <= threshold).map(|(_, s)| s))
    }

    /// The complex with each vertex renamed by `name` (e.g. from indices into a window to indices into the whole
    /// point cloud), the order of each simplex's vertices kept.
    pub fn relabel<W: Ord + Clone>(&self, name: impl Fn(&V) -> W) -> SimplicialComplex<W> {
        SimplicialComplex::new(self.simplices.iter().map(|s| s.iter().map(&name).collect()))
    }

    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }
}

impl<V: Ord + Hash + Clone> ZigzagComplex for SimplicialComplex<V> {
    type Key = Vec<V>;
    fn cells(&self) -> Vec<(Vec<V>, usize)> {
        self.simplices.iter().map(|s| (s.clone(), s.len() - 1)).collect()
    }
    fn faces(&self, cell: &Vec<V>) -> Vec<Vec<V>> {
        simplex_facets(cell)
    }
}

impl<V: Ord + Hash + Clone> ZigzagComplex for BTreeSet<Vec<V>> {
    type Key = Vec<V>;
    fn cells(&self) -> Vec<(Vec<V>, usize)> {
        self.iter().map(|s| (s.clone(), s.len() - 1)).collect()
    }
    fn faces(&self, cell: &Vec<V>) -> Vec<Vec<V>> {
        simplex_facets(cell)
    }
}

/// How consecutive complexes of a [`ComplexZigzag`] are joined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Join {
    /// `K_i -> K_i ∪ K_{i+1} <- K_{i+1}`: a feature lasts while it overlaps itself from one complex to the next.
    #[default]
    Union,
    /// `K_i <- K_i ∩ K_{i+1} -> K_{i+1}`: a feature lasts while its common part with the next complex carries it.
    Intersection,
}

/// An interval of the zigzag barcode of a sequence of complexes, in *levels*: complex `i` is level `2 i`, the join of
/// complexes `i` and `i + 1` level `2 i + 1`. The class is alive at every level from `birth` to `death` (inclusive);
/// one alive in the last complex has `death == 2 (s - 1)` for `s` complexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LevelBar {
    pub dimension: usize,
    pub birth: usize,
    pub death: usize,
}

/// A zigzag of complexes, built a complex at a time: each [`push`](Self::push) moves from the last complex, through
/// its join with the new one, to the new one, inserting cells faces first and deleting them cofaces first.
#[derive(Debug, Clone)]
pub struct ComplexZigzag<K: Ord> {
    join: Join,
    operations: Vec<Operation<K>>,
    /// The number of operations done at each level
    levels: Vec<usize>,
    /// (dimension, key) of the cells now there
    current: BTreeSet<(usize, K)>,
}

impl<K: Hash + Ord + Clone> ComplexZigzag<K> {
    pub fn new(join: Join) -> Self {
        Self { join, operations: vec![], levels: vec![], current: BTreeSet::new() }
    }

    /// The zigzag of `complexes`, joined by `join`.
    pub fn of<'a, C: ZigzagComplex<Key = K> + ?Sized + 'a>(join: Join, complexes: impl IntoIterator<Item = &'a C>) -> Self {
        let mut zigzag = Self::new(join);
        for complex in complexes {
            zigzag.push(complex);
        }
        zigzag
    }

    /// Append a complex (its join with the last first, unless it is the first).
    pub fn push<C: ZigzagComplex<Key = K> + ?Sized>(&mut self, complex: &C) -> &mut Self {
        let next: BTreeMap<K, usize> = complex.cells().into_iter().collect();
        let next_set: BTreeSet<(usize, K)> = next.iter().map(|(k, &d)| (d, k.clone())).collect();
        let first = self.levels.is_empty();
        let insert = |ops: &mut Vec<Operation<K>>, current: &BTreeSet<(usize, K)>| {
            for (dimension, cell) in next_set.difference(current) {
                ops.push(Operation::Insert { cell: cell.clone(), dimension: *dimension, faces: complex.faces(cell) });
            }
        };
        let delete = |ops: &mut Vec<Operation<K>>, current: &BTreeSet<(usize, K)>| {
            for (_, cell) in current.difference(&next_set).collect::<Vec<_>>().into_iter().rev() {
                ops.push(Operation::Delete(cell.clone()));
            }
        };
        match self.join {
            Join::Union => {
                insert(&mut self.operations, &self.current);
                if !first {
                    self.levels.push(self.operations.len());
                }
                delete(&mut self.operations, &self.current);
            }
            Join::Intersection => {
                delete(&mut self.operations, &self.current);
                if !first {
                    self.levels.push(self.operations.len());
                }
                insert(&mut self.operations, &self.current);
            }
        }
        self.levels.push(self.operations.len());
        self.current = next_set;
        self
    }

    /// The number of complexes pushed.
    pub fn n_complexes(&self) -> usize {
        self.levels.len().div_ceil(2)
    }

    pub fn join(&self) -> Join {
        self.join
    }

    /// The cell-wise zigzag: what [`zigzag_barcode`] takes.
    pub fn operations(&self) -> &[Operation<K>] {
        &self.operations
    }

    /// The number of operations done by each level (level `l` is the complex after `levels()[l]` operations).
    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    /// The barcode, in levels, by cohomology with union-find (see [`zigzag_barcode`]).
    ///
    /// # Errors
    /// If a complex isn't closed under faces ([`ZigzagError::MissingFace`]).
    pub fn barcode<R: ColumnRepresentation>(&self, algorithm: Algorithm<'_, R>) -> Result<Vec<LevelBar>, ZigzagError> {
        self.barcode_with(Method::Cohomology, algorithm, true)
    }

    /// The barcode, in levels, reducing by `method` (see [`zigzag_barcode_with`]). Sorted.
    ///
    /// # Errors
    /// If a complex isn't closed under faces ([`ZigzagError::MissingFace`]).
    pub fn barcode_with<R: ColumnRepresentation>(
        &self,
        method: Method,
        algorithm: Algorithm<'_, R>,
        union_find: bool,
    ) -> Result<Vec<LevelBar>, ZigzagError> {
        let bars = zigzag_barcode_with(&self.operations, method, algorithm, union_find)?;
        let mut out: Vec<LevelBar> = bars.into_iter().filter_map(|bar| to_levels(&self.levels, &bar)).collect();
        out.sort_unstable();
        Ok(out)
    }
}

impl<K: Hash + Ord + Clone> Default for ComplexZigzag<K> {
    fn default() -> Self {
        Self::new(Join::default())
    }
}

/// An interval over positions (`K_p` after `p` operations) as one over levels (`levels[l]` the position of level
/// `l`, non-decreasing): the levels whose complexes it is alive in, if any.
pub fn to_levels(levels: &[usize], bar: &ZigzagBar) -> Option<LevelBar> {
    let first = levels.partition_point(|&p| p < bar.birth);
    let last = levels.partition_point(|&p| p <= bar.death);
    (first < last).then(|| LevelBar { dimension: bar.dimension, birth: first, death: last - 1 })
}

#[cfg(test)]
mod complex_tests {
    use super::*;
    use phlite::reduction::Clearing;

    fn bars(join: Join, complexes: &[SimplicialComplex<u32>]) -> Vec<LevelBar> {
        ComplexZigzag::of(join, complexes).barcode(Algorithm::Clearing(Clearing::new())).unwrap()
    }

    /// Ranks of homology at each level, from bars, match those computed directly
    fn betti(bars: &[LevelBar], level: usize, dim: usize) -> usize {
        bars.iter().filter(|b| b.dimension == dim && b.birth <= level && level <= b.death).count()
    }

    #[test]
    fn union_and_intersection() {
        // A path 0-1-2 then 1-2-0: union is the triangle's boundary (a loop at level 1), intersection the edge 1-2
        // and the vertex 0 (a second component at level 1)
        let a = SimplicialComplex::from_maximal([vec![0, 1], vec![1, 2]]);
        let b = SimplicialComplex::from_maximal([vec![1, 2], vec![0, 2]]);
        let u = bars(Join::Union, &[a.clone(), b.clone()]);
        assert_eq!(u, vec![LevelBar { dimension: 0, birth: 0, death: 2 }, LevelBar { dimension: 1, birth: 1, death: 1 }]);
        let i = bars(Join::Intersection, &[a, b]);
        assert_eq!(i, vec![LevelBar { dimension: 0, birth: 0, death: 2 }, LevelBar { dimension: 0, birth: 1, death: 1 }]);
        // Two components joined only through the union
        let c = SimplicialComplex::from_maximal([vec![0], vec![5]]);
        let d = SimplicialComplex::from_maximal([vec![0, 5]]);
        let u = bars(Join::Union, &[c.clone(), d.clone(), c.clone()]);
        assert_eq!(betti(&u, 0, 0), 2);
        assert_eq!(betti(&u, 2, 0), 1);
        assert_eq!(betti(&u, 4, 0), 2);
        // Intersection: c ∩ d = c
        let i = bars(Join::Intersection, &[c.clone(), d, c]);
        assert_eq!(betti(&i, 1, 0), 2);
        assert_eq!(betti(&i, 2, 0), 1);
    }

    #[test]
    fn directed_flag_simplices_keep_order() {
        // Ordered simplices (0,1,2) and (2,1,0) are different cells
        let a = SimplicialComplex::from_maximal([vec![0u32, 1, 2]]);
        assert!(a.simplices.contains(&vec![0, 2]) && !a.simplices.contains(&vec![2, 0]));
        let b = SimplicialComplex::from_maximal([vec![2u32, 1, 0]]);
        let u = bars(Join::Union, &[a, b]);
        // Union: two filled triangles on the same three vertices, sharing only vertices: H1 of rank 2 at level 1
        assert_eq!(betti(&u, 1, 1), 2);
        assert_eq!(betti(&u, 0, 1), 0);
    }

    /// Random sequences of complexes: Betti numbers of every level and ranks of every map against direct computation
    #[test]
    fn random_sequences_betti() {
        let mut seed = 12345u64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let n_complexes = 2 + (rand() % 4) as usize;
            let complexes: Vec<SimplicialComplex<u32>> = (0..n_complexes)
                .map(|_| {
                    let k = 1 + rand() % 6;
                    SimplicialComplex::from_maximal((0..k).map(|_| {
                        let size = 1 + rand() % 3;
                        let mut s: Vec<u32> = (0..size).map(|_| (rand() % 6) as u32).collect();
                        s.sort_unstable();
                        s.dedup();
                        s
                    }))
                })
                .collect();
            for join in [Join::Union, Join::Intersection] {
                let b = bars(join, &complexes);
                for (i, c) in complexes.iter().enumerate() {
                    for dim in 0..3 {
                        assert_eq!(betti(&b, 2 * i, dim), direct_betti(&c.simplices, dim), "{join:?} {complexes:?}");
                    }
                    if i + 1 < complexes.len() {
                        let joined: BTreeSet<Vec<u32>> = match join {
                            Join::Union => c.simplices.union(&complexes[i + 1].simplices).cloned().collect(),
                            Join::Intersection => c.simplices.intersection(&complexes[i + 1].simplices).cloned().collect(),
                        };
                        for dim in 0..3 {
                            assert_eq!(betti(&b, 2 * i + 1, dim), direct_betti(&joined, dim));
                        }
                    }
                }
            }
        }
    }

    /// Betti number over Z/2 by Gaussian elimination
    fn direct_betti(simplices: &BTreeSet<Vec<u32>>, dim: usize) -> usize {
        let rank = |d: usize| -> usize {
            // rank of the boundary from dimension d to d - 1
            if d == 0 {
                return 0;
            }
            let rows: Vec<&Vec<u32>> = simplices.iter().filter(|s| s.len() == d).collect();
            let index: HashMap<&Vec<u32>, usize> = rows.iter().enumerate().map(|(i, s)| (*s, i)).collect();
            let mut cols: Vec<BTreeSet<usize>> = simplices
                .iter()
                .filter(|s| s.len() == d + 1)
                .map(|s| simplex_facets(s).iter().map(|f| index[f]).collect())
                .collect();
            let mut pivots: HashMap<usize, usize> = HashMap::new();
            let mut r = 0;
            for j in 0..cols.len() {
                while let Some(&low) = cols[j].iter().next_back() {
                    if let Some(&k) = pivots.get(&low) {
                        let other = cols[k].clone();
                        cols[j] = cols[j].symmetric_difference(&other).copied().collect();
                    } else {
                        pivots.insert(low, j);
                        r += 1;
                        break;
                    }
                }
            }
            r
        };
        let n = simplices.iter().filter(|s| s.len() == dim + 1).count();
        n - rank(dim) - rank(dim + 1)
    }
}
