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

/// A row: its parent, the gain `λ(v) = gain · λ(parent)` (zero-sized for [`Z2`](crate::fields::Z2)), and a key to
/// order roots by, side by side so that one cache line has all three.
#[derive(Debug, Clone, Copy)]
struct Node<CF, K> {
    /// A root is its own parent.
    parent: u32,
    gain: CF,
    key: K,
}

/// The state of reducing, one at a time, columns with at most two non-zero entries, see the [module docs](self).
///
/// Rows are ordered by their numbers ([`add_column`](Self::add_column)), by a function
/// ([`add_column_by`](Self::add_column_by)), or by keys stored with them, then their numbers
/// ([`with_keys`](Self::with_keys), [`add_column_keyed`](Self::add_column_keyed) and [`join`](Self::join)), which is
/// fastest when the order comes from values: comparing two roots then reads nothing but the two rows.
///
/// The ground is a node of its own after the rows, never a pivot: a grounded component hangs below it, so finding a
/// root is the classic one-branch loop, and whether a root is grounded is a comparison of numbers.
#[derive(Debug, Clone)]
pub struct IncidenceReduction<CF, K = ()> {
    /// The rows, then the ground
    nodes: Vec<Node<CF, K>>,
}

/// What [`IncidenceReduction::root`] gives for a grounded component.
pub const GROUND: u32 = u32::MAX;

/// The root of `v`, halving the path, without the gains (all 1 for [`IncidenceReduction::join`]): the classic loop.
#[inline(always)]
fn root_plain<CF, K>(nodes: &mut [Node<CF, K>], mut v: u32) -> u32 {
    loop {
        let p = nodes[v as usize].parent;
        if p == v {
            return v;
        }
        let g = nodes[p as usize].parent;
        nodes[v as usize].parent = g;
        v = g;
    }
}

impl<CF: Invertible> IncidenceReduction<CF, ()> {
    /// Rows `0..n_rows`, each its own component. Fewer than `2^32 - 2` rows.
    pub fn new(n_rows: usize) -> Self {
        Self::with_keys(std::iter::repeat_n((), n_rows))
    }

    /// Make room for rows `0..n_rows` (each new row its own component), if there isn't already. Grows by at least
    /// doubling, so that rows added one by one cost linear time in all.
    pub fn grow(&mut self, n_rows: usize) {
        self.grow_with(n_rows, ());
    }
}

impl<CF: Invertible, K: Copy + Default> IncidenceReduction<CF, K> {
    /// Rows `0, 1, ...`, each its own component, with these keys: with [`add_column_keyed`](Self::add_column_keyed)
    /// and [`join`](Self::join), rows are ordered by `(key, number)`. Fewer than `2^32 - 2` rows.
    pub fn with_keys(keys: impl IntoIterator<Item = K>) -> Self {
        let mut nodes: Vec<Node<CF, K>> = keys
            .into_iter()
            .enumerate()
            .map(|(v, key)| Node { parent: v as u32, gain: CF::one(), key })
            .collect();
        assert!(nodes.len() + 1 < GROUND as usize, "fewer than 2^32 - 2 rows");
        let ground = nodes.len() as u32;
        nodes.push(Node { parent: ground, gain: CF::one(), key: K::default() });
        Self { nodes }
    }

    /// Make room for rows `0..n_rows` (each new row its own component, with key `key`), if there isn't already.
    /// Grows by at least doubling (the ground moves to the end, so every row below it is re-pointed).
    pub fn grow_with(&mut self, n_rows: usize, key: K) {
        let old = self.n_rows();
        if n_rows <= old {
            return;
        }
        let new = n_rows.max(2 * old);
        assert!(new + 1 < GROUND as usize, "fewer than 2^32 - 2 rows");
        let (old_ground, ground) = (old as u32, new as u32);
        for node in &mut self.nodes {
            if node.parent == old_ground {
                node.parent = ground;
            }
        }
        // The old ground's place becomes a row like the others
        self.nodes[old] = Node { parent: old_ground, gain: CF::one(), key };
        self.nodes.extend((old + 1..new).map(|v| Node { parent: v as u32, gain: CF::one(), key }));
        self.nodes.push(Node { parent: ground, gain: CF::one(), key: K::default() });
    }
}

impl<CF: Invertible, K: Copy> IncidenceReduction<CF, K> {
    /// The number of rows.
    pub fn n_rows(&self) -> usize {
        self.nodes.len() - 1
    }

    #[inline]
    fn ground(&self) -> u32 {
        (self.nodes.len() - 1) as u32
    }

    /// The root of `v`'s component and `λ(v)` relative to it (meaningless below the ground), halving the path.
    #[inline]
    fn find(&mut self, mut v: u32) -> (u32, CF) {
        let mut lambda = CF::one();
        loop {
            let p = self.nodes[v as usize].parent;
            if p == v {
                return (v, lambda);
            }
            // Point v at its grandparent: λ(v) = gain[v] · gain[p] · λ(g) (when p is the root, g = p and its gain is 1)
            let g = self.nodes[p as usize].parent;
            let gain = if g == p {
                self.nodes[v as usize].gain
            } else {
                self.nodes[v as usize].gain * self.nodes[p as usize].gain
            };
            let node = &mut self.nodes[v as usize];
            node.parent = g;
            node.gain = gain;
            lambda = lambda * gain;
            v = g;
        }
    }

    /// Reduce the column with these entries (at most two, with distinct rows), given in any order, where the rows'
    /// order is that of their numbers.
    ///
    /// # Panics
    /// If there are more than two entries.
    #[inline]
    pub fn add_column(&mut self, entries: &[(CF, u32)]) -> IncidenceOutcome {
        self.reduce(entries, |_, a, b| a > b)
    }

    /// As [`add_column`](Self::add_column), but with the rows in another order: `later(a, b)` says whether row `a`
    /// comes after row `b` (so would be the pivot of a column holding both). Every call must use the same order.
    #[inline]
    pub fn add_column_by(&mut self, entries: &[(CF, u32)], later: impl Fn(u32, u32) -> bool) -> IncidenceOutcome {
        self.reduce(entries, |_, a, b| later(a, b))
    }

    #[inline]
    fn reduce(&mut self, entries: &[(CF, u32)], later: impl Fn(&Self, u32, u32) -> bool) -> IncidenceOutcome {
        let ground = self.ground();
        match *entries {
            [] => IncidenceOutcome::Zero,
            [(_, v)] => {
                let (r, _) = self.find(v);
                if r == ground {
                    IncidenceOutcome::Zero
                } else {
                    self.nodes[r as usize].parent = ground;
                    IncidenceOutcome::Pivot(r)
                }
            }
            [(alpha, a), (beta, b)] => {
                let (ra, la) = self.find(a);
                let (rb, lb) = self.find(b);
                if ra == rb {
                    if ra == ground {
                        return IncidenceOutcome::Zero;
                    }
                    // The column reduces to (ca + cb) · e_ra
                    return match alpha * la + beta * lb {
                        None => IncidenceOutcome::Zero,
                        Some(_) => {
                            self.nodes[ra as usize].parent = ground;
                            IncidenceOutcome::Pivot(ra)
                        }
                    };
                }
                // A grounded component reduces away, leaving the other's root (which joins the ground)
                if ra == ground || rb == ground {
                    let die = if ra == ground { rb } else { ra };
                    self.nodes[die as usize].parent = ground;
                    return IncidenceOutcome::Pivot(die);
                }
                // The column reduces to ca · e_ra + cb · e_rb: the later root dies, and its component is rescaled so
                // that ck · 1 + t · cd · 1 = 0
                let (ca, cb) = (alpha * la, beta * lb);
                let (keep, ck, die, cd) = if later(self, rb, ra) { (ra, ca, rb, cb) } else { (rb, cb, ra, ca) };
                let t = (ck * cd.mult_inverse()).additive_inverse();
                let node = &mut self.nodes[die as usize];
                node.parent = keep;
                node.gain = t;
                IncidenceOutcome::Pivot(die)
            }
            _ => panic!("a column of an incidence dimension has at most two entries"),
        }
    }

    /// Reduce a column `±(e_a - e_b)` (the boundary of an edge, with `b`), or `±e_a` (without), with rows ordered by
    /// `(key, number)`: the root that dies (the pivot), if any. These columns keep every gain 1 over any field, so
    /// this needs none of [`add_column`](Self::add_column)'s arithmetic, and is the fast path for graphs. Don't mix it
    /// with columns of other shapes on the same rows.
    #[inline(always)]
    pub fn join(&mut self, a: u32, b: Option<u32>) -> Option<u32>
    where
        K: Ord,
    {
        let ground = self.ground();
        // On a local slice, so that its address and length stay in registers
        let nodes = &mut self.nodes[..];
        let ra = root_plain(nodes, a);
        let rb = match b {
            Some(b) => root_plain(nodes, b),
            None => ground,
        };
        if ra == rb {
            return None;
        }
        let (keep, die) = if ra == ground {
            (ra, rb)
        } else if rb == ground || (nodes[ra as usize].key, ra) > (nodes[rb as usize].key, rb) {
            (rb, ra)
        } else {
            (ra, rb)
        };
        nodes[die as usize].parent = keep;
        Some(die)
    }

    /// The root of `v`'s component, or [`GROUND`] if it is grounded, halving the path on the way.
    #[inline]
    pub fn root(&mut self, v: u32) -> u32 {
        let ground = self.ground();
        match root_plain(&mut self.nodes[..], v) {
            r if r == ground => GROUND,
            r => r,
        }
    }

    /// Whether row `v` is the root of its component and not grounded: then no column so far has `v` as its pivot,
    /// and no reduced column touching its component reaches `v` with a non-zero coefficient other than via its root.
    /// In other words, whether `v` was never a pivot.
    pub fn is_free_root(&self, v: u32) -> bool {
        self.nodes[v as usize].parent == v
    }

    /// The key of row `v`.
    #[inline]
    pub fn key(&self, v: u32) -> K {
        self.nodes[v as usize].key
    }
}

impl<CF: Invertible, K: Copy + Ord> IncidenceReduction<CF, K> {
    /// As [`add_column`](Self::add_column), with rows ordered by `(key, number)` (see
    /// [`with_keys`](Self::with_keys)).
    #[inline]
    pub fn add_column_keyed(&mut self, entries: &[(CF, u32)]) -> IncidenceOutcome {
        self.reduce(entries, |this, a, b| (this.nodes[a as usize].key, a) > (this.nodes[b as usize].key, b))
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

    proptest! {
        /// The fast path for graphs agrees with the general one on columns e_a - e_b and e_a, with keys.
        #[test]
        fn join_matches_add_column(
            keys in prop::collection::vec(0u8..4, 1..12),
            columns in prop::collection::vec((0usize..12, prop::option::of(0usize..12)), 0..30),
        ) {
            let n = keys.len();
            let mut fast = IncidenceReduction::<Z3, u8>::with_keys(keys.iter().copied());
            let mut slow = IncidenceReduction::<Z3, u8>::with_keys(keys.iter().copied());
            let minus = Z3::one().additive_inverse();
            for (a, b) in columns {
                let a = (a % n) as u32;
                let b = b.map(|b| (b % n) as u32).filter(|&b| b != a);
                let entries: Vec<(Z3, u32)> = match b {
                    Some(b) => vec![(Z3::one(), a), (minus, b)],
                    None => vec![(Z3::one(), a)],
                };
                let want = match slow.add_column_keyed(&entries) {
                    IncidenceOutcome::Zero => None,
                    IncidenceOutcome::Pivot(r) => Some(r),
                };
                prop_assert_eq!(fast.join(a, b), want);
            }
            for v in 0..n as u32 {
                prop_assert_eq!(fast.is_free_root(v), slow.is_free_root(v));
            }
        }
    }
}
