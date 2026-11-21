//! Homology over the integers: the spectral sequence of a filtered chain complex of free abelian groups, page by
//! page, and its persistent homology groups, as finitely generated abelian groups (free rank and torsion).
//!
//! Over a field everything is read off the persistence barcode ([`crate::spectral`]), but a filtered complex over Z
//! doesn't decompose into intervals, so this works with the subgroups themselves, as lattices in `Z^n` given by bases:
//! for cells in degree `k`, with `F_p` the cells of filtration step at most `p` and `d` the boundary,
//!
//! * `Z^r_p = {x ∈ F_p C_k : dx ∈ F_{p-r} C_{k-1}}` (a kernel), and
//! * `E^r_p = Z^r_p / (Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1})` (a quotient of lattices, read off a Smith normal form),
//!
//! so that `E^0_p = F_p C / F_{p-1} C`, `E^1_p = H(F_p C / F_{p-1} C)` and the differential `d^r` goes from step `p` to
//! step `p - r`, as in [`crate::spectral`]; and the persistent homology group from step `i` to step `j` is
//! `H^{i,j}_k = Z_k(F_i) / d{x ∈ F_j C_{k+1} : dx ∈ F_i C_k}`.
//!
//! The arithmetic is exact, dense, in `i128` and checked: every function returns [`Overflow`] rather than a wrong
//! answer if an entry outgrows it. Elimination picks a least-modulus pivot at every step, which keeps entries small in
//! practice. Dense means this is for complexes of up to a few thousand cells per degree.

use std::fmt;

/// An entry outgrew `i128` (see the [module docs](self)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "integer overflow in exact integral linear algebra")
    }
}

impl std::error::Error for Overflow {}

/// A finitely generated abelian group `Z^rank ⊕ Z/t_1 ⊕ ... ⊕ Z/t_m`, with `1 < t_1 | t_2 | ... | t_m`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Group {
    pub rank: usize,
    pub torsion: Vec<u128>,
}

impl Group {
    /// Whether the group is trivial.
    pub fn is_zero(&self) -> bool {
        self.rank == 0 && self.torsion.is_empty()
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = vec![];
        match self.rank {
            0 => {}
            1 => parts.push("Z".to_string()),
            r => parts.push(format!("Z^{r}")),
        }
        parts.extend(self.torsion.iter().map(|t| format!("Z/{t}")));
        if parts.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", parts.join(" + "))
        }
    }
}

/// The ring the computations are over: the integers, or `Z/p` for a prime `p` (a field, so no torsion, but the same
/// window-by-window computation, which gives generators over a field too).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ring {
    #[default]
    Integers,
    /// `Z/p`, `p` prime (entries kept in `0..p`)
    Mod(u64),
}

impl Ring {
    #[inline]
    fn reduce(self, x: i128) -> i128 {
        match self {
            Ring::Integers => x,
            Ring::Mod(p) => x.rem_euclid(i128::from(p)),
        }
    }

    /// The Euclidean size of a non-zero `x`: a pivot of least size is used.
    #[inline]
    fn size(self, x: i128) -> u128 {
        match self {
            Ring::Integers => x.unsigned_abs(),
            Ring::Mod(_) => 1,
        }
    }

    /// `a / b` rounded so that `a - (a / b) b` is smaller than `b` (exact if `b` divides `a`; over `Z/p` always exact).
    #[inline]
    fn div(self, a: i128, b: i128) -> i128 {
        match self {
            Ring::Integers => a.div_euclid(b),
            Ring::Mod(p) => {
                let p = i128::from(p);
                a * mod_inverse(b, p) % p
            }
        }
    }

    #[inline]
    fn divides(self, b: i128, a: i128) -> bool {
        match self {
            Ring::Integers => a % b == 0,
            Ring::Mod(_) => true,
        }
    }

    /// The order of the cyclic group `R / (d)`: `None` if infinite (`d = 0` over Z), 1 if `d` is a unit.
    fn order(self, d: i128) -> Option<u128> {
        match self {
            Ring::Integers if d == 0 => None,
            Ring::Integers => Some(d.unsigned_abs()),
            Ring::Mod(_) => Some(1),
        }
    }
}

/// The inverse of `b` modulo the prime `p`.
fn mod_inverse(b: i128, p: i128) -> i128 {
    let (mut r0, mut r1, mut s0, mut s1) = (b.rem_euclid(p), p, 1i128, 0i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
    }
    debug_assert_eq!(r0, 1, "not invertible");
    s0.rem_euclid(p)
}

type Vector = Vec<i128>;

fn sub_multiple(ring: Ring, target: &mut [i128], source: &[i128], q: i128, from: usize) -> Result<(), Overflow> {
    if q == 0 {
        return Ok(());
    }
    for (t, &s) in target[from..].iter_mut().zip(&source[from..]) {
        if s != 0 {
            *t = ring.reduce(t.checked_sub(q.checked_mul(s).ok_or(Overflow)?).ok_or(Overflow)?);
        }
    }
    Ok(())
}

/// Bring `rows` to echelon form on their first `width` coordinates by unimodular row operations (so they span the same
/// lattice), returning the number of rows with a pivot: those come first, each with its pivot (first non-zero
/// coordinate below `width`) strictly right of the previous one's, and the rest are zero on the first `width`
/// coordinates.
fn echelon(ring: Ring, rows: &mut [Vector], width: usize) -> Result<usize, Overflow> {
    let mut t = 0;
    for c in 0..width {
        if t == rows.len() {
            break;
        }
        // The least non-zero entry in column c among rows t.., moved to row t
        while let Some(best) = (t..rows.len()).filter(|&i| rows[i][c] != 0).min_by_key(|&i| ring.size(rows[i][c])) {
            rows.swap(t, best);
            let (pivot_row, rest) = rows[t..].split_at_mut(1);
            let pivot_row = &pivot_row[0];
            let p = pivot_row[c];
            let mut clean = true;
            for row in rest.iter_mut() {
                if row[c] != 0 {
                    let q = ring.div(row[c], p);
                    sub_multiple(ring, row, pivot_row, q, c)?;
                    clean &= row[c] == 0;
                }
            }
            if clean {
                t += 1;
                break;
            }
        }
    }
    Ok(t)
}

/// A basis of the kernel `{x ∈ R^n : M x = 0}` of the matrix with these `columns` (each of length `m`), as vectors of
/// length `n`.
fn kernel_in(ring: Ring, columns: &[Vector], m: usize) -> Result<Vec<Vector>, Overflow> {
    let n = columns.len();
    // Each row: a column of M, then the unit vector recording which combination of columns it is
    let mut rows: Vec<Vector> = columns
        .iter()
        .enumerate()
        .map(|(j, column)| {
            let mut row = Vec::with_capacity(m + n);
            row.extend(column.iter().map(|&x| ring.reduce(x)));
            row.resize(m + n, 0);
            row[m + j] = 1;
            row
        })
        .collect();
    let pivots = echelon(ring, &mut rows, m)?;
    Ok(rows.drain(pivots..).map(|row| row[m..].to_vec()).collect())
}

/// A basis of the kernel `{x ∈ Z^n : M x = 0}` of the matrix with these `columns` (each of length `m`), as vectors of
/// length `n`.
pub fn kernel(columns: &[Vector], m: usize) -> Result<Vec<Vector>, Overflow> {
    kernel_in(Ring::Integers, columns, m)
}

/// A lattice given by a basis in echelon form, for expressing its members in that basis.
struct Echelon {
    ring: Ring,
    rows: Vec<Vector>,
    /// Each row's pivot coordinate
    pivots: Vec<usize>,
}

impl Echelon {
    /// The lattice spanned by `generators` (vectors of length `n`).
    fn new(ring: Ring, mut generators: Vec<Vector>, n: usize) -> Result<Self, Overflow> {
        let independent = echelon(ring, &mut generators, n)?;
        generators.truncate(independent);
        let pivots = generators.iter().map(|row| row.iter().position(|&x| x != 0).expect("a pivot row")).collect();
        Ok(Self { ring, rows: generators, pivots })
    }

    /// The coordinates of `v` in the basis, or `None` if it isn't in the lattice.
    fn try_coordinates(&self, v: &[i128]) -> Result<Option<Vector>, Overflow> {
        let mut v: Vector = v.iter().map(|&x| self.ring.reduce(x)).collect();
        let mut out = vec![0; self.rows.len()];
        for (i, (row, &c)) in self.rows.iter().zip(&self.pivots).enumerate() {
            if v[c] != 0 {
                if !self.ring.divides(row[c], v[c]) {
                    return Ok(None);
                }
                let q = self.ring.div(v[c], row[c]);
                sub_multiple(self.ring, &mut v, row, q, c)?;
                out[i] = q;
            }
        }
        Ok(v.iter().all(|&x| x == 0).then_some(out))
    }

    /// The coordinates of `v` in the basis.
    ///
    /// # Panics
    /// If `v` isn't in the lattice.
    fn coordinates(&self, v: Vector) -> Result<Vector, Overflow> {
        Ok(self.try_coordinates(&v)?.expect("not in the lattice"))
    }
}

/// The Smith normal form of the matrix with these rows, by unimodular row and column operations: its diagonal (each
/// entry dividing the next, possibly negative), and if `track`, the column operations as an invertible matrix `Q`
/// (columns) with its inverse `Q⁻¹`, so that `P A Q` is diagonal for some invertible `P`.
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
fn smith(ring: Ring, mut a: Vec<Vector>, cols: usize, track: bool) -> Result<(Vector, Option<(Vec<Vector>, Vec<Vector>)>), Overflow> {
    let rows = a.len();
    let identity = || (0..cols).map(|i| (0..cols).map(|j| i128::from(i == j)).collect::<Vector>()).collect::<Vec<_>>();
    // q: the columns operations applied to the identity (Q, rows of Q are its rows); inverse: Q⁻¹
    let (mut q, mut inverse) = if track { (identity(), identity()) } else { (vec![], vec![]) };
    let swap_columns = |a: &mut Vec<Vector>, q: &mut Vec<Vector>, inverse: &mut Vec<Vector>, i: usize, j: usize| {
        for row in a.iter_mut() {
            row.swap(i, j);
        }
        if track {
            for row in q.iter_mut() {
                row.swap(i, j);
            }
            inverse.swap(i, j);
        }
    };
    // Column j -= m * column t (and the same to Q; Q⁻¹ gets row t += m * row j)
    let sub_column = |a: &mut Vec<Vector>, q: &mut Vec<Vector>, inverse: &mut Vec<Vector>, j: usize, t: usize, m: i128| -> Result<(), Overflow> {
        for row in a.iter_mut() {
            row[j] = ring.reduce(row[j].checked_sub(m.checked_mul(row[t]).ok_or(Overflow)?).ok_or(Overflow)?);
        }
        if track {
            for row in q.iter_mut() {
                row[j] = ring.reduce(row[j].checked_sub(m.checked_mul(row[t]).ok_or(Overflow)?).ok_or(Overflow)?);
            }
            let (low, high) = (t.min(j), t.max(j));
            let (head, tail) = inverse.split_at_mut(high);
            let (target, source) = if t < j { (&mut head[low], &tail[0]) } else { (&mut tail[0], &head[low]) };
            sub_multiple(ring, target, source, ring.reduce(-m), 0)?;
        }
        Ok(())
    };
    for row in &mut a {
        for x in row.iter_mut() {
            *x = ring.reduce(*x);
        }
    }
    let mut diagonal = vec![];
    let mut t = 0;
    while t < rows.min(cols) {
        // The entry of least size in the rest
        let mut best: Option<(u128, usize, usize)> = None;
        for (i, row) in a.iter().enumerate().skip(t) {
            for (j, &x) in row.iter().enumerate().skip(t) {
                if x != 0 && best.is_none_or(|(m, _, _)| ring.size(x) < m) {
                    best = Some((ring.size(x), i, j));
                }
            }
        }
        let Some((_, i, j)) = best else { break };
        a.swap(t, i);
        swap_columns(&mut a, &mut q, &mut inverse, t, j);
        loop {
            let p = a[t][t];
            let mut clean = true;
            for i in t + 1..rows {
                if a[i][t] != 0 {
                    let m = ring.div(a[i][t], p);
                    let (top, rest) = a.split_at_mut(i);
                    sub_multiple(ring, &mut rest[0], &top[t], m, t)?;
                    clean &= a[i][t] == 0;
                }
            }
            for j in t + 1..cols {
                if a[t][j] != 0 {
                    let m = ring.div(a[t][j], p);
                    sub_column(&mut a, &mut q, &mut inverse, j, t, m)?;
                    clean &= a[t][j] == 0;
                }
            }
            if clean {
                // The pivot must divide the rest; if not, add the offending row to row t and go round again
                let bad = (t + 1..rows).find(|&i| (t + 1..cols).any(|j| !ring.divides(p, a[i][j])));
                match bad {
                    None => break,
                    Some(i) => {
                        let (top, rest) = a.split_at_mut(i);
                        sub_multiple(ring, &mut top[t], &rest[0], -1, t)?;
                    }
                }
            } else {
                // A smaller remainder appeared: move the least entry of row and column t to the pivot
                let mut best = (ring.size(a[t][t]), t, t);
                for i in t + 1..rows {
                    if a[i][t] != 0 && ring.size(a[i][t]) < best.0 {
                        best = (ring.size(a[i][t]), i, t);
                    }
                }
                for j in t + 1..cols {
                    if a[t][j] != 0 && ring.size(a[t][j]) < best.0 {
                        best = (ring.size(a[t][j]), t, j);
                    }
                }
                let (_, i, j) = best;
                a.swap(t, i);
                swap_columns(&mut a, &mut q, &mut inverse, t, j);
            }
        }
        diagonal.push(a[t][t]);
        t += 1;
    }
    Ok((diagonal, track.then_some((q, inverse))))
}

/// The non-zero diagonal of the Smith normal form of the matrix with these rows (the invariant factors, each dividing
/// the next). A least-modulus pivot at every step, as Perseus 6 does.
pub fn smith_diagonal(a: Vec<Vector>) -> Result<Vec<u128>, Overflow> {
    let cols = a.first().map_or(0, Vec::len);
    Ok(smith(Ring::Integers, a, cols, false)?.0.into_iter().map(i128::unsigned_abs).collect())
}

/// The group `A / B`, for `A` spanned by `a` and `B ⊆ A` spanned by `b` (vectors of length `n`).
///
/// # Panics
/// If some generator of `B` isn't in `A`.
pub fn quotient(a: Vec<Vector>, b: Vec<Vector>, n: usize) -> Result<Group, Overflow> {
    Quotient::new(Ring::Integers, a, b, n, false).map(|q| q.group())
}

/// A generator of one cyclic summand of a quotient `A / B`: its order (`None` if infinite) and a representative in
/// `A`, as a vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generator {
    pub order: Option<u128>,
    pub vector: Vector,
}

/// A quotient `A / B` of lattices (or of vector spaces over `Z/p`), with (if asked for) a basis of `A` adapted to `B`:
/// `A` has basis `a'_1, ..., a'_s` and `B` is spanned by `d_1 a'_1, ..., d_m a'_m`.
struct Quotient {
    ring: Ring,
    a: Echelon,
    diagonal: Vector,
    /// `Q` and `Q⁻¹`: the adapted basis is `Q⁻¹ · (the echelon basis of A)`
    change: Option<(Vec<Vector>, Vec<Vector>)>,
}

impl Quotient {
    fn new(ring: Ring, a: Vec<Vector>, b: Vec<Vector>, n: usize, track: bool) -> Result<Self, Overflow> {
        let a = Echelon::new(ring, a, n)?;
        let coordinates = b.into_iter().map(|v| a.coordinates(v)).collect::<Result<Vec<_>, _>>()?;
        let (diagonal, change) = smith(ring, coordinates, a.rows.len(), track)?;
        Ok(Self { ring, a, diagonal, change })
    }

    fn group(&self) -> Group {
        Group {
            rank: self.a.rows.len() - self.diagonal.len(),
            torsion: self.diagonal.iter().filter_map(|&d| self.ring.order(d)).filter(|&o| o > 1).collect(),
        }
    }

    /// The order of the `i`-th summand of the adapted basis: 1 if trivial, `None` if infinite.
    fn order(&self, i: usize) -> Option<u128> {
        self.diagonal.get(i).and_then(|&d| self.ring.order(d))
    }

    /// The generators of the non-trivial summands, in the order of the group's invariants (torsion, then free).
    fn generators(&self) -> Result<Vec<Generator>, Overflow> {
        let (_, inverse) = self.change.as_ref().expect("tracked");
        let n = self.a.rows.first().map_or(0, Vec::len);
        let mut out = vec![];
        for (i, row) in inverse.iter().enumerate() {
            let order = self.order(i);
            if order == Some(1) {
                continue;
            }
            let mut vector = vec![0; n];
            for (&c, basis) in row.iter().zip(&self.a.rows) {
                if c != 0 {
                    sub_multiple(self.ring, &mut vector, basis, self.ring.reduce(-c), 0)?;
                }
            }
            out.push(Generator { order, vector });
        }
        // Torsion (in increasing order, as found) first, then free
        out.sort_by_key(|g| g.order.is_none());
        Ok(out)
    }

    /// The class of `v` in the quotient, as its coordinates on [`generators`](Self::generators) (reduced modulo each
    /// order), or `None` if `v` isn't in `A`.
    #[allow(clippy::needless_range_loop)]
    fn classify(&self, v: &[i128]) -> Result<Option<Vector>, Overflow> {
        let (q, _) = self.change.as_ref().expect("tracked");
        let Some(c) = self.a.try_coordinates(v)? else {
            return Ok(None);
        };
        // Coordinates on the adapted basis: c Q
        let mut torsion = vec![];
        let mut free = vec![];
        for i in 0..q.len() {
            let mut x: i128 = 0;
            for (j, &cj) in c.iter().enumerate() {
                x = x.checked_add(cj.checked_mul(q[j][i]).ok_or(Overflow)?).ok_or(Overflow)?;
            }
            x = self.ring.reduce(x);
            match self.order(i) {
                Some(1) => {}
                Some(o) => torsion.push(x.rem_euclid(o as i128)),
                None => free.push(x),
            }
        }
        torsion.extend(free);
        Ok(Some(torsion))
    }
}

/// A filtered chain complex of free abelian groups: each cell's degree, filtration step and boundary (faces as cell
/// indices, with integer coefficients). Faces must be in the degree below and enter no later.
#[derive(Debug, Clone, Default)]
pub struct IntegralFiltered {
    pub degree: Vec<usize>,
    pub step: Vec<usize>,
    pub boundary: Vec<Vec<(usize, i64)>>,
}

/// The cells of one degree, in order of step, and where each cell is among them.
struct Degree {
    /// Cell indices in order of step (then index)
    cells: Vec<usize>,
    /// For each step `p`, how many cells have step at most `p`
    up_to: Vec<usize>,
}

/// [`IntegralFiltered`] arranged by degree.
struct Arranged<'a> {
    ring: Ring,
    complex: &'a IntegralFiltered,
    degrees: Vec<Degree>,
    /// Each cell's position within its degree
    position: Vec<usize>,
    n_steps: usize,
}

impl<'a> Arranged<'a> {
    fn new(complex: &'a IntegralFiltered) -> Self {
        Self::new_in(complex, Ring::Integers)
    }

    fn new_in(complex: &'a IntegralFiltered, ring: Ring) -> Self {
        let n = complex.degree.len();
        let n_degrees = complex.degree.iter().max().map_or(0, |&d| d + 1);
        let n_steps = complex.step.iter().max().map_or(0, |&s| s + 1);
        let mut degrees: Vec<Degree> = (0..n_degrees + 1).map(|_| Degree { cells: vec![], up_to: vec![] }).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&c| (complex.step[c], c));
        let mut position = vec![0; n];
        for c in order {
            let degree = &mut degrees[complex.degree[c]];
            position[c] = degree.cells.len();
            degree.cells.push(c);
        }
        for degree in &mut degrees {
            degree.up_to = (0..n_steps)
                .map(|p| degree.cells.partition_point(|&c| complex.step[c] <= p))
                .collect();
        }
        Self { ring, complex, degrees, position, n_steps }
    }

    fn size(&self, k: usize) -> usize {
        self.degrees.get(k).map_or(0, |d| d.cells.len())
    }

    /// How many cells of degree `k` have step at most `p` (`p` may be negative or past the last step).
    fn up_to(&self, k: usize, p: isize) -> usize {
        match self.degrees.get(k) {
            None => 0,
            Some(d) if p < 0 || d.cells.is_empty() => 0,
            Some(d) => d.up_to[(p as usize).min(self.n_steps - 1)],
        }
    }

    /// The boundary of the cell at `position` in degree `k`, as a vector over the cells of degree `k - 1`.
    fn boundary(&self, k: usize, position: usize) -> Vector {
        let mut out = vec![0; if k == 0 { 0 } else { self.size(k - 1) }];
        let cell = self.degrees[k].cells[position];
        for &(face, coefficient) in &self.complex.boundary[cell] {
            let x = &mut out[self.position[face]];
            *x = self.ring.reduce(*x + i128::from(coefficient));
        }
        out
    }

    /// `{x ∈ F_p C_k : dx ∈ F_q C_{k-1}}`, as vectors over all the cells of degree `k`.
    fn cycles(&self, k: usize, p: isize, q: isize) -> Result<Vec<Vector>, Overflow> {
        let n = self.up_to(k, p);
        let size = self.size(k);
        // The rows of d (faces) past F_q, restricted to the first n columns
        let skip = if k == 0 { 0 } else { self.up_to(k - 1, q) };
        let columns: Vec<Vector> = (0..n).map(|j| self.boundary(k, j).split_off(skip)).collect();
        let m = if k == 0 { 0 } else { self.size(k - 1) - skip };
        let mut basis = kernel_in(self.ring, &columns, m)?;
        for v in &mut basis {
            v.resize(size, 0);
        }
        Ok(basis)
    }

    /// The boundaries of `vectors` (over the cells of degree `k`), over the cells of degree `k - 1`.
    fn apply_boundary(&self, k: usize, vectors: &[Vector]) -> Result<Vec<Vector>, Overflow> {
        let columns: Vec<Vector> = (0..self.size(k)).map(|j| self.boundary(k, j)).collect();
        let m = self.size(k - 1);
        vectors
            .iter()
            .map(|v| {
                let mut out = vec![0i128; m];
                for (x, column) in v.iter().zip(&columns) {
                    if *x != 0 {
                        sub_multiple(self.ring, &mut out, column, -*x, 0)?;
                    }
                }
                Ok(out)
            })
            .collect()
    }

    /// `E^r_p` in degree `k`.
    fn page_entry(&self, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
        self.page_quotient(r, p, k, false).map(|q| q.group())
    }

    /// `E^r_p` in degree `k` as a quotient, with an adapted basis if `track`.
    fn page_quotient(&self, r: usize, p: usize, k: usize, track: bool) -> Result<Quotient, Overflow> {
        let (p, r) = (p as isize, r as isize);
        let numerator = self.cycles(k, p, p - r)?;
        let mut denominator = if r == 0 {
            // Z^{-1}_{p-1} is F_{p-1}
            (0..self.up_to(k, p - 1))
                .map(|j| {
                    let mut v = vec![0; self.size(k)];
                    v[j] = 1;
                    v
                })
                .collect()
        } else {
            self.cycles(k, p - 1, p - r)?
        };
        if r >= 1 {
            let above = self.cycles(k + 1, p + r - 1, p)?;
            denominator.extend(self.apply_boundary(k + 1, &above)?);
        }
        Quotient::new(self.ring, numerator, denominator, self.size(k), track)
    }

    /// A vector over the cells of degree `k` as a chain of cell indices (non-zero terms only).
    fn chain(&self, k: usize, vector: &[i128]) -> Vec<(usize, i128)> {
        vector.iter().enumerate().filter(|(_, &x)| x != 0).map(|(i, &x)| (self.degrees[k].cells[i], x)).collect()
    }

    /// A chain of cell indices as a vector over the cells of degree `k`, or `None` if it has a cell of another degree.
    fn vector(&self, k: usize, chain: &[(usize, i64)]) -> Option<Vector> {
        let mut out = vec![0; self.size(k)];
        for &(cell, x) in chain {
            if self.complex.degree.get(cell) != Some(&k) {
                return None;
            }
            let slot = &mut out[self.position[cell]];
            *slot = self.ring.reduce(*slot + i128::from(x));
        }
        Some(out)
    }
}

impl Arranged<'_> {
    /// The image of `d^r` out of `E^r_p` in degree `k` (into `E^r_{p-r}` in degree `k - 1`), for `r >= 1`:
    /// `Z^r_p / (Z^{r+1}_p + Z^{r-1}_{p-1})`, `E^r_p` modulo the kernel of `d^r`.
    fn differential_image(&self, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
        assert!(r >= 1, "d^0 is the differential of the associated graded");
        let (p, r) = (p as isize, r as isize);
        let numerator = self.cycles(k, p, p - r)?;
        let mut kernel = self.cycles(k, p, p - r - 1)?;
        kernel.extend(self.cycles(k, p - 1, p - r)?);
        Quotient::new(self.ring, numerator, kernel, self.size(k), false).map(|q| q.group())
    }
}

/// `E^r_p` in degree `k` of `complex` (see the [module docs](self)).
pub fn page_entry(complex: &IntegralFiltered, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
    Arranged::new(complex).page_entry(r, p, k)
}

/// The image of the differential `d^r` (`r >= 1`) out of `E^r_p` in degree `k`, a subgroup of `E^r_{p-r}` in degree
/// `k - 1`: `Z^r_p / (Z^{r+1}_p + Z^{r-1}_{p-1})`.
pub fn differential_image(complex: &IntegralFiltered, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
    Arranged::new(complex).differential_image(r, p, k)
}

/// As [`page_entry`], over `ring`.
pub fn page_entry_in(complex: &IntegralFiltered, ring: Ring, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
    Arranged::new_in(complex, ring).page_entry(r, p, k)
}

/// As [`differential_image`], over `ring`.
pub fn differential_image_in(
    complex: &IntegralFiltered,
    ring: Ring,
    r: usize,
    p: usize,
    k: usize,
) -> Result<Group, Overflow> {
    Arranged::new_in(complex, ring).differential_image(r, p, k)
}

/// A generator of one cyclic summand of a group of a spectral sequence: its order (`None` if infinite; over `Z/p`
/// every generator is of a free summand), and a chain representing it, as `(cell, coefficient)` with cells indices of
/// the complex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainGenerator {
    pub order: Option<u128>,
    pub chain: Vec<(usize, i128)>,
}

/// `E^r_p` in degree `k` of `complex` over `ring`, with a generator for each cyclic summand (in the order of its
/// invariants: torsion, each order dividing the next, then free). Each generator is a chain `x` of degree `k` with
/// `x ∈ F_p` and `dx ∈ F_{p-r}`, determined modulo `Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1}`.
pub fn page_generators(
    complex: &IntegralFiltered,
    ring: Ring,
    r: usize,
    p: usize,
    k: usize,
) -> Result<(Group, Vec<ChainGenerator>), Overflow> {
    let arranged = Arranged::new_in(complex, ring);
    let quotient = arranged.page_quotient(r, p, k, true)?;
    let generators = quotient
        .generators()?
        .into_iter()
        .map(|g| ChainGenerator { order: g.order, chain: arranged.chain(k, &g.vector) })
        .collect();
    Ok((quotient.group(), generators))
}

/// The class in `E^r_p` (degree `k`, over `ring`) of the chain `chain` (`(cell, coefficient)`), as its coordinates on
/// the generators [`page_generators`] gives (reduced modulo the orders), or `None` if the chain doesn't represent an
/// element (it isn't of degree `k`, in `F_p`, with boundary in `F_{p-r}`).
pub fn page_classify(
    complex: &IntegralFiltered,
    ring: Ring,
    r: usize,
    p: usize,
    k: usize,
    chain: &[(usize, i64)],
) -> Result<Option<Vec<i128>>, Overflow> {
    let arranged = Arranged::new_in(complex, ring);
    let Some(vector) = arranged.vector(k, chain) else {
        return Ok(None);
    };
    arranged.page_quotient(r, p, k, true)?.classify(&vector)
}

/// Integral pages `E^0, ..., E^{r_max}`: `pages[r][p][k]` is `E^r_p` in degree `k`, or `None` where the complex, if
/// it is the truncation of a larger filtration after step `truncated_at`, doesn't determine it (as in
/// [`crate::spectral::spectral_pages`]: page `r` at step `p` needs `p + r <= truncated_at + 1`).
pub type IntegralPages = Vec<Vec<Vec<Option<Group>>>>;

/// The pages `E^0, ..., E^{r_max}` of the spectral sequence of `complex` (see the [module docs](self)), in steps
/// `0..n_steps` (the complex's) and degrees `0..n_degrees`.
pub fn integral_pages(
    complex: &IntegralFiltered,
    n_degrees: usize,
    r_max: usize,
    truncated_at: Option<usize>,
) -> Result<IntegralPages, Overflow> {
    let arranged = Arranged::new(complex);
    let n_steps = arranged.n_steps;
    (0..=r_max)
        .map(|r| {
            (0..n_steps)
                .map(|p| {
                    (0..n_degrees)
                        .map(|k| {
                            if truncated_at.is_some_and(|t| r >= 1 && p + r > t + 1) {
                                return Ok(None);
                            }
                            arranged.page_entry(r, p, k).map(Some)
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// The persistent homology group `H^{i,j}_k = Z_k(F_i) / d{x ∈ F_j C_{k+1} : dx ∈ F_i C_k}` of `complex` (the image
/// of `H_k(F_i)` in `H_k(F_j)`), for steps `i <= j`.
pub fn persistent_group(complex: &IntegralFiltered, k: usize, i: usize, j: usize) -> Result<Group, Overflow> {
    assert!(i <= j, "a persistent group needs i <= j");
    let arranged = Arranged::new(complex);
    if k >= arranged.degrees.len() {
        return Ok(Group::default());
    }
    let cycles = arranged.cycles(k, i as isize, -1)?;
    let above = arranged.cycles(k + 1, j as isize, i as isize)?;
    let boundaries = arranged.apply_boundary(k + 1, &above)?;
    quotient(cycles, boundaries, arranged.size(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simplicial complex given by its top simplices, every face at the step `step(face)` (by default 0), with the
    /// usual alternating signs.
    fn simplicial(tops: &[&[usize]], step: impl Fn(&[usize]) -> usize) -> IntegralFiltered {
        use std::collections::BTreeMap;
        let mut faces: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
        let mut all: Vec<Vec<usize>> = vec![];
        for top in tops {
            let n = top.len();
            for mask in 1u32..(1 << n) {
                let mut s: Vec<usize> = (0..n).filter(|&i| mask >> i & 1 == 1).map(|i| top[i]).collect();
                s.sort_unstable();
                all.push(s);
            }
        }
        all.sort_by_key(|s| (s.len(), s.clone()));
        all.dedup();
        let mut complex = IntegralFiltered::default();
        for s in all {
            let index = complex.degree.len();
            let boundary = if s.len() == 1 {
                vec![]
            } else {
                (0..s.len())
                    .map(|i| {
                        let mut f = s.clone();
                        f.remove(i);
                        (faces[&f], if i % 2 == 0 { 1 } else { -1 })
                    })
                    .collect()
            };
            complex.degree.push(s.len() - 1);
            complex.step.push(step(&s));
            complex.boundary.push(boundary);
            faces.insert(s, index);
        }
        complex
    }

    /// The minimal (6-vertex) triangulation of the real projective plane.
    const RP2: [&[usize]; 10] = [
        &[0, 1, 2],
        &[0, 2, 3],
        &[0, 3, 4],
        &[0, 4, 5],
        &[0, 5, 1],
        &[1, 2, 4],
        &[2, 3, 5],
        &[3, 4, 1],
        &[4, 5, 2],
        &[5, 1, 3],
    ];

    fn group(rank: usize, torsion: &[u128]) -> Group {
        Group { rank, torsion: torsion.to_vec() }
    }

    #[test]
    fn homology_of_the_projective_plane() {
        let complex = simplicial(&RP2, |_| 0);
        let pages = integral_pages(&complex, 3, 1, None).unwrap();
        // E^1 at the only step is the homology
        let h: Vec<Group> = pages[1][0].iter().map(|g| g.clone().unwrap()).collect();
        assert_eq!(h, vec![group(1, &[]), group(0, &[2]), group(0, &[])]);
        assert_eq!(persistent_group(&complex, 1, 0, 0).unwrap(), group(0, &[2]));
        assert_eq!(h[1].to_string(), "Z/2");
    }

    #[test]
    fn filtration_by_dimension_converges_to_homology() {
        // Steps by dimension: E^1 is the chain groups, d^1 the boundary, so E^2 = E^∞ is the homology
        let complex = simplicial(&RP2, |s| s.len() - 1);
        let pages = integral_pages(&complex, 3, 3, None).unwrap();
        let entry = |r: usize, p: usize, k: usize| pages[r][p][k].clone().unwrap();
        assert_eq!(entry(1, 0, 0), group(6, &[]));
        assert_eq!(entry(1, 1, 1), group(15, &[]));
        assert_eq!(entry(1, 2, 2), group(10, &[]));
        assert_eq!(entry(1, 1, 0), group(0, &[]));
        for r in 2..=3 {
            assert_eq!(entry(r, 0, 0), group(1, &[]));
            assert_eq!(entry(r, 1, 1), group(0, &[2]));
            assert_eq!(entry(r, 2, 2), group(0, &[]));
        }
        // E^0 is the associated graded: the cells of each step
        assert_eq!(entry(0, 1, 1), group(15, &[]));
    }

    #[test]
    fn persistent_groups_of_a_growing_projective_plane() {
        // The 1-skeleton at step 0, the triangles at step 1: H_1 goes from Z^10 to Z/2
        let complex = simplicial(&RP2, |s| usize::from(s.len() == 3));
        assert_eq!(persistent_group(&complex, 1, 0, 0).unwrap(), group(10, &[]));
        assert_eq!(persistent_group(&complex, 1, 0, 1).unwrap(), group(0, &[2]));
        assert_eq!(persistent_group(&complex, 1, 1, 1).unwrap(), group(0, &[2]));
        assert_eq!(persistent_group(&complex, 0, 0, 1).unwrap(), group(1, &[]));
    }

    #[test]
    fn homology_agrees_with_smith_normal_forms() {
        // H_k = Z^{n_k - rank d_k - rank d_{k+1}} ⊕ the torsion of d_{k+1}, on random small complexes
        let mut state = 7u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..40 {
            let n = 4 + next(4) as usize;
            let tops: Vec<Vec<usize>> = (0..3 + next(6))
                .map(|_| {
                    let mut s: Vec<usize> = (0..n).filter(|_| next(2) == 0).collect();
                    s.truncate(4);
                    if s.is_empty() {
                        s.push(0);
                    }
                    s
                })
                .collect();
            let refs: Vec<&[usize]> = tops.iter().map(Vec::as_slice).collect();
            let complex = simplicial(&refs, |_| 0);
            let arranged = Arranged::new(&complex);
            let n_degrees = arranged.degrees.len() - 1;
            let d = |k: usize| -> Vec<Vector> {
                if k == 0 || k > n_degrees || arranged.size(k) == 0 {
                    return vec![];
                }
                (0..arranged.size(k)).map(|j| arranged.boundary(k, j)).collect()
            };
            for k in 0..n_degrees {
                let low = smith_diagonal(d(k)).unwrap();
                let high = smith_diagonal(d(k + 1)).unwrap();
                let expected = group(
                    arranged.size(k) - low.len() - high.len(),
                    &high.iter().copied().filter(|&t| t > 1).collect::<Vec<_>>(),
                );
                assert_eq!(persistent_group(&complex, k, 0, 0).unwrap(), expected);
            }
        }
    }

    #[test]
    fn differentials_account_for_the_next_page() {
        // Over Z, rank E^{r+1}_p = rank E^r_p - rank(d^r out) - rank(d^r in), and filtered by dimension d^1 is the
        // boundary: on RP^2, d^1 from (2, 2) is injective (H_2 = 0), image Z^10 in the Z^10 of 1-cycles, leaving Z/2
        let complex = simplicial(&RP2, |s| s.len() - 1);
        assert_eq!(differential_image(&complex, 1, 2, 2).unwrap(), group(10, &[]));
        assert_eq!(differential_image(&complex, 1, 1, 1).unwrap(), group(5, &[]));
        let mut state = 3u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..30 {
            let n = 4 + next(3) as usize;
            let tops: Vec<Vec<usize>> = (0..2 + next(5))
                .map(|_| (0..n).filter(|_| next(2) == 0).take(4).collect::<Vec<_>>())
                .filter(|s| !s.is_empty())
                .collect();
            let refs: Vec<&[usize]> = tops.iter().map(Vec::as_slice).collect();
            let complex = simplicial(&refs, |s| s.iter().max().copied().unwrap_or(0) + s.len() % 2);
            let n_steps = complex.step.iter().max().map_or(0, |s| s + 1);
            for r in 1..4 {
                for p in 0..n_steps {
                    for k in 0..4 {
                        let out = if k >= 1 { differential_image(&complex, r, p, k).unwrap().rank } else { 0 };
                        let into = if p + r < n_steps { differential_image(&complex, r, p + r, k + 1).unwrap().rank } else { 0 };
                        assert_eq!(
                            page_entry(&complex, r + 1, p, k).unwrap().rank,
                            page_entry(&complex, r, p, k).unwrap().rank - out - into,
                            "r {r} p {p} k {k}"
                        );
                    }
                }
            }
        }
    }

    /// Check the generators of every entry: their orders are the group's invariants, each is a chain of degree `k` in
    /// `F_p` with boundary in `F_{p-r}`, and classifying it gives its unit vector (and twice a torsion generator of
    /// order 2 classifies to zero).
    fn check_generators(complex: &IntegralFiltered, ring: Ring, r_max: usize, n_degrees: usize) -> usize {
        let n_steps = complex.step.iter().max().map_or(0, |s| s + 1);
        let mut torsion = 0;
        for r in 1..=r_max {
            for p in 0..n_steps {
                for k in 0..n_degrees {
                    let (group, generators) = page_generators(complex, ring, r, p, k).unwrap();
                    assert_eq!(group, page_entry_in(complex, ring, r, p, k).unwrap());
                    let orders: Vec<u128> = generators.iter().filter_map(|g| g.order).collect();
                    assert_eq!(orders, group.torsion, "r {r} p {p} k {k}");
                    assert_eq!(generators.iter().filter(|g| g.order.is_none()).count(), group.rank);
                    torsion += group.torsion.len();
                    for (i, g) in generators.iter().enumerate() {
                        // In F_p, of degree k, with boundary in F_{p-r}
                        let mut boundary: std::collections::BTreeMap<usize, i128> = Default::default();
                        for &(cell, x) in &g.chain {
                            assert_eq!(complex.degree[cell], k);
                            assert!(complex.step[cell] <= p);
                            for &(face, c) in &complex.boundary[cell] {
                                *boundary.entry(face).or_default() += x * i128::from(c);
                            }
                        }
                        for (face, x) in boundary {
                            if ring.reduce(x) != 0 {
                                assert!(complex.step[face] + r <= p, "boundary of a generator outside F_(p-r)");
                            }
                        }
                        let chain: Vec<(usize, i64)> = g.chain.iter().map(|&(c, x)| (c, x as i64)).collect();
                        let class = page_classify(complex, ring, r, p, k, &chain).unwrap().unwrap();
                        let unit: Vec<i128> = (0..generators.len()).map(|j| i128::from(i == j)).collect();
                        assert_eq!(class, unit, "r {r} p {p} k {k}");
                        if g.order == Some(2) {
                            let twice: Vec<(usize, i64)> = chain.iter().map(|&(c, x)| (c, 2 * x)).collect();
                            assert!(page_classify(complex, ring, r, p, k, &twice).unwrap().unwrap().iter().all(|&x| x == 0));
                        }
                    }
                }
            }
        }
        torsion
    }

    #[test]
    fn generators_of_the_projective_plane() {
        // All at step 0: H_1 = Z/2, generated by a 1-cycle
        let complex = simplicial(&RP2, |_| 0);
        let (h1, generators) = page_generators(&complex, Ring::Integers, 1, 0, 1).unwrap();
        assert_eq!(h1, group(0, &[2]));
        assert_eq!(generators.len(), 1);
        // A chain of the wrong degree, or not a cycle, isn't an element
        assert_eq!(page_classify(&complex, Ring::Integers, 1, 0, 1, &[(0, 1)]).unwrap(), None);
        let edge = complex.degree.iter().position(|&d| d == 1).unwrap();
        assert_eq!(page_classify(&complex, Ring::Integers, 1, 0, 1, &[(edge, 1)]).unwrap(), None);
        assert!(check_generators(&complex, Ring::Integers, 1, 3) > 0);
        // Over Z/2: H_1 = H_2 = Z/2 as vector spaces, generators and all
        let (h2, generators) = page_generators(&complex, Ring::Mod(2), 1, 0, 2).unwrap();
        assert_eq!((h2.rank, generators.len()), (1, 1));
        check_generators(&complex, Ring::Mod(2), 1, 3);
        // Filtered by its vertices: torsion on later pages
        let complex = simplicial(&RP2, |s| *s.iter().max().unwrap());
        assert!(check_generators(&complex, Ring::Integers, 6, 3) > 0);
        check_generators(&complex, Ring::Mod(3), 6, 3);
    }

    #[test]
    fn generators_of_random_complexes() {
        let mut state = 11u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..25 {
            let n = 4 + next(3) as usize;
            let tops: Vec<Vec<usize>> = (0..2 + next(5))
                .map(|_| (0..n).filter(|_| next(2) == 0).take(4).collect::<Vec<_>>())
                .filter(|s| !s.is_empty())
                .collect();
            let refs: Vec<&[usize]> = tops.iter().map(Vec::as_slice).collect();
            let complex = simplicial(&refs, |s| s.iter().max().copied().unwrap_or(0) + s.len() % 2);
            check_generators(&complex, Ring::Integers, 3, 4);
            check_generators(&complex, Ring::Mod(5), 3, 4);
        }
    }

    #[test]
    fn smith_normal_form() {
        assert_eq!(smith_diagonal(vec![vec![2, 4, 4], vec![-6, 6, 12], vec![10, -4, -16]]).unwrap(), vec![2, 6, 12]);
        assert_eq!(smith_diagonal(vec![vec![2, 0], vec![0, 3]]).unwrap(), vec![1, 6]);
    }
}
