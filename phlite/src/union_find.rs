//! Reducing a matrix whose columns have at most two entries, by union-find.
//!
//! When every column in a dimension has at most two non-zero entries, reducing that dimension is a union-find on its
//! rows: this is how the barcode in dimension 0 is found from the edges (each edge's boundary is `v - w`), and, by
//! duality, the barcode in the top dimension `d - 1` of a complex in which every `(d - 1)`-cell has at most two
//! cofaces (a grid, a mesh, a triangulation of a region of space) from its coboundary. Both are far cheaper than
//! reducing, and need no storage beyond two words per row.
//!
//! [`IncidenceReduction`] takes the columns one at a time, in the order a reduction would visit them, and says what
//! the reduced column's pivot would be (or that it would reduce to zero). Rows are numbered `0..n`, in the order of
//! the matrix's rows: the pivot of a column is its *largest* row, as in [`reduction`](crate::reduction) (or, with
//! [`add_column_by`](IncidenceReduction::add_column_by), in any other order).
//!
//! # Why this works, over any field
//! Think of the rows as the nodes of a graph and each column with two entries as an edge between its rows (a column
//! with one entry is an edge to an extra node, the *ground*, which is never a pivot). The columns reduced so far span,
//! on each connected component of the graph,
//! * if the component is a tree (or contains only cycles whose columns are dependent), the kernel of one linear
//!   functional `λ` on the component's rows, with every `λ(v)` non-zero; we keep `λ` relative to the component's
//!   earliest row (its *root*), so `λ(root) = 1`;
//! * otherwise every vector on the component (it is *grounded*: joined to the ground, or by a cycle whose columns
//!   are independent).
//!
//! The pivots of the earlier columns are every row of the component but its root (and the root too, once grounded).
//! So a new column `y` reduces, on each component it touches, to `λ(y) · e_root` (or to zero, on a grounded
//! component), and its pivot is the largest root left with a non-zero coefficient. Then:
//! * two different roots: the later is the pivot, and the components merge, rescaling the dying root's `λ` so that
//!   `y` is in the kernel (so the merged component keeps the earlier root);
//! * one root, `λ(y) = 0`: the column reduces to zero (it closes a cycle);
//! * one root, `λ(y) ≠ 0` (a column with one entry, or a cycle of independent columns, which only happens off Z/2, e.g.
//!   the dual graph of a non-orientable surface over Z/3): the root is the pivot and the component becomes grounded.
//!
//! Over Z/2, and whenever each column's two entries have opposite signs (as for the boundary `v - w` of an edge), `λ`
//! is 1 everywhere and only columns with one entry ground a component.

use crate::fields::Invertible;

/// What reducing a column (of at most two entries) by [`IncidenceReduction::add_column`] gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncidenceOutcome {
    /// The column reduces to zero.
    Zero,
    /// The reduced column has this pivot (a row number), which no earlier column had.
    Pivot(u32),
}

/// The state of reducing, one at a time, columns with at most two non-zero entries, see the [module docs](self).
#[derive(Debug, Clone)]
pub struct IncidenceReduction<CF> {
    /// `parent[v]`, with the top bit set at a grounded root. A root is its own parent.
    parent: Vec<u32>,
    /// `λ(v) = gain[v] · λ(parent[v])`. Zero-sized for [`Z2`](crate::fields::Z2), so free.
    gain: Vec<CF>,
}

const GROUNDED: u32 = 1 << 31;

impl<CF: Invertible> IncidenceReduction<CF> {
    /// Rows `0..n_rows`, each its own component. At most `2^31 - 1` rows.
    pub fn new(n_rows: usize) -> Self {
        assert!(n_rows < GROUNDED as usize, "at most 2^31 - 1 rows");
        Self {
            parent: (0..n_rows as u32).collect(),
            gain: vec![CF::one(); n_rows],
        }
    }

    /// The number of rows.
    pub fn n_rows(&self) -> usize {
        self.parent.len()
    }

    /// Make room for rows `0..n_rows` (each new row its own component), if there isn't already.
    pub fn grow(&mut self, n_rows: usize) {
        assert!(n_rows < GROUNDED as usize, "at most 2^31 - 1 rows");
        let old = self.parent.len();
        if n_rows > old {
            self.parent.extend(old as u32..n_rows as u32);
            self.gain.resize(n_rows, CF::one());
        }
    }

    #[inline]
    fn parent_of(&self, v: u32) -> u32 {
        self.parent[v as usize] & !GROUNDED
    }

    /// The root of `v`'s component, and `λ(v)` (relative to the root), halving the path on the way.
    #[inline]
    fn find(&mut self, mut v: u32) -> (u32, CF) {
        let mut lambda = CF::one();
        loop {
            let p = self.parent_of(v);
            if p == v {
                return (v, lambda);
            }
            let g = self.parent_of(p);
            if g != p {
                // Point v at its grandparent: λ(v) = gain[v] · gain[p] · λ(g)
                self.gain[v as usize] = self.gain[v as usize] * self.gain[p as usize];
                self.parent[v as usize] = g;
            }
            lambda = lambda * self.gain[v as usize];
            v = self.parent_of(v);
        }
    }

    /// Whether the root `r` is grounded.
    #[inline]
    fn grounded(&self, r: u32) -> bool {
        self.parent[r as usize] & GROUNDED != 0
    }

    #[inline]
    fn ground(&mut self, r: u32) {
        self.parent[r as usize] |= GROUNDED;
    }

    /// Reduce the column with these entries (at most two, with distinct rows), given in any order, where the rows'
    /// order is that of their numbers.
    ///
    /// # Panics
    /// If there are more than two entries.
    #[inline]
    pub fn add_column(&mut self, entries: &[(CF, u32)]) -> IncidenceOutcome {
        self.add_column_by(entries, |a, b| a > b)
    }

    /// As [`add_column`](Self::add_column), but with the rows in another order: `later(a, b)` says whether row `a`
    /// comes after row `b` (so would be the pivot of a column holding both). Every call must use the same order.
    pub fn add_column_by(&mut self, entries: &[(CF, u32)], later: impl Fn(u32, u32) -> bool) -> IncidenceOutcome {
        match *entries {
            [] => IncidenceOutcome::Zero,
            [(_, v)] => {
                let (r, _) = self.find(v);
                if self.grounded(r) {
                    IncidenceOutcome::Zero
                } else {
                    self.ground(r);
                    IncidenceOutcome::Pivot(r)
                }
            }
            [(alpha, a), (beta, b)] => {
                let (ra, la) = self.find(a);
                let (rb, lb) = self.find(b);
                // The column reduces to ca · e_ra + cb · e_rb
                let (ca, cb) = (alpha * la, beta * lb);
                if ra == rb {
                    if self.grounded(ra) {
                        return IncidenceOutcome::Zero;
                    }
                    return match ca + cb {
                        None => IncidenceOutcome::Zero,
                        Some(_) => {
                            self.ground(ra);
                            IncidenceOutcome::Pivot(ra)
                        }
                    };
                }
                match (self.grounded(ra), self.grounded(rb)) {
                    (true, true) => IncidenceOutcome::Zero,
                    // A grounded component reduces away, leaving the other's root
                    (true, false) => {
                        self.parent[rb as usize] = ra | GROUNDED;
                        self.parent[ra as usize] = ra | GROUNDED;
                        IncidenceOutcome::Pivot(rb)
                    }
                    (false, true) => {
                        self.parent[ra as usize] = rb;
                        IncidenceOutcome::Pivot(ra)
                    }
                    (false, false) => {
                        // The larger root dies; rescale its component so that ca · 1 + t · cb · 1 = 0 (or vice versa)
                        let (keep, ck, die, cd) = if later(rb, ra) { (ra, ca, rb, cb) } else { (rb, cb, ra, ca) };
                        let t = (ck * cd.mult_inverse()).additive_inverse();
                        self.parent[die as usize] = keep;
                        self.gain[die as usize] = t;
                        IncidenceOutcome::Pivot(die)
                    }
                }
            }
            _ => panic!("a column of an incidence dimension has at most two entries"),
        }
    }

    /// Whether row `v` is the root of its component and not grounded: then no column so far has `v` as its pivot,
    /// and no reduced column touching its component reaches `v` with a non-zero coefficient other than via its root.
    /// In other words, whether `v` was never a pivot.
    pub fn is_free_root(&self, v: u32) -> bool {
        self.parent[v as usize] == v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{NonZeroCoefficient, Z2, Z3};
    use proptest::prelude::*;

    /// The pivots of reducing `columns` (each a list of `(coefficient, row)`, rows `0..n`) with the standard algorithm,
    /// done by hand over a field given as integers mod p.
    fn reference_pivots(columns: &[Vec<(i64, u32)>], p: i64) -> Vec<Option<u32>> {
        let mut reduced: Vec<Vec<i64>> = vec![];
        let mut owner: std::collections::HashMap<u32, usize> = Default::default();
        let mut out = vec![];
        let n = columns.iter().flatten().map(|&(_, r)| r as usize + 1).max().unwrap_or(0);
        for col in columns {
            let mut dense = vec![0i64; n];
            for &(c, r) in col {
                dense[r as usize] = (dense[r as usize] + c).rem_euclid(p);
            }
            loop {
                let Some(piv) = (0..n).rev().find(|&r| dense[r] != 0) else {
                    out.push(None);
                    break;
                };
                match owner.get(&(piv as u32)) {
                    None => {
                        owner.insert(piv as u32, reduced.len());
                        reduced.push(dense);
                        out.push(Some(piv as u32));
                        break;
                    }
                    Some(&j) => {
                        // dense -= (dense[piv] / reduced[j][piv]) · reduced[j]
                        let inv = modinv(reduced[j][piv], p);
                        let factor = dense[piv] * inv % p;
                        for r in 0..n {
                            dense[r] = (dense[r] - factor * reduced[j][r]).rem_euclid(p);
                        }
                    }
                }
            }
        }
        out
    }

    fn modinv(a: i64, p: i64) -> i64 {
        (1..p).find(|&x| a * x % p == 1).expect("invertible")
    }

    fn z3(c: i64) -> Z3 {
        match c.rem_euclid(3) {
            1 => Z3::one(),
            2 => Z3::one().additive_inverse(),
            _ => unreachable!("non-zero"),
        }
    }

    fn columns_strategy() -> impl Strategy<Value = (u32, Vec<Vec<(i64, u32)>>)> {
        (1u32..12).prop_flat_map(|n| {
            let entry = (1i64..3, 0..n);
            let column = prop::collection::vec(entry, 0..=2).prop_filter("distinct rows", |c| {
                c.len() < 2 || c[0].1 != c[1].1
            });
            (Just(n), prop::collection::vec(column, 0..30))
        })
    }

    proptest! {
        /// Over Z/3, with arbitrary signs (so cycles of independent columns occur), union-find gives the same pivots as
        /// reducing.
        #[test]
        fn matches_reduction_over_z3((n, columns) in columns_strategy()) {
            let expected = reference_pivots(&columns, 3);
            let mut uf = IncidenceReduction::<Z3>::new(n as usize);
            let got: Vec<Option<u32>> = columns
                .iter()
                .map(|col| {
                    let entries: Vec<(Z3, u32)> = col.iter().map(|&(c, r)| (z3(c), r)).collect();
                    match uf.add_column(&entries) {
                        IncidenceOutcome::Zero => None,
                        IncidenceOutcome::Pivot(r) => Some(r),
                    }
                })
                .collect();
            prop_assert_eq!(got, expected);
        }

        /// Over Z/2 too.
        #[test]
        fn matches_reduction_over_z2((n, columns) in columns_strategy()) {
            let ones: Vec<Vec<(i64, u32)>> =
                columns.iter().map(|col| col.iter().map(|&(_, r)| (1, r)).collect()).collect();
            let expected = reference_pivots(&ones, 2);
            let mut uf = IncidenceReduction::<Z2>::new(n as usize);
            let got: Vec<Option<u32>> = columns
                .iter()
                .map(|col| {
                    let entries: Vec<(Z2, u32)> = col.iter().map(|&(_, r)| (Z2, r)).collect();
                    match uf.add_column(&entries) {
                        IncidenceOutcome::Zero => None,
                        IncidenceOutcome::Pivot(r) => Some(r),
                    }
                })
                .collect();
            prop_assert_eq!(got, expected);
        }
    }
}
