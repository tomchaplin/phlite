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

use std::collections::HashMap;
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
