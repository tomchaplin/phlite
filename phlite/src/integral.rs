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

type Vector = Vec<i128>;

fn sub_multiple(target: &mut [i128], source: &[i128], q: i128, from: usize) -> Result<(), Overflow> {
    if q == 0 {
        return Ok(());
    }
    for (t, &s) in target[from..].iter_mut().zip(&source[from..]) {
        if s != 0 {
            *t = t.checked_sub(q.checked_mul(s).ok_or(Overflow)?).ok_or(Overflow)?;
        }
    }
    Ok(())
}

/// Bring `rows` to echelon form on their first `width` coordinates by unimodular row operations (so they span the same
/// lattice), returning the number of rows with a pivot: those come first, each with its pivot (first non-zero
/// coordinate below `width`) strictly right of the previous one's, and the rest are zero on the first `width`
/// coordinates.
fn echelon(rows: &mut [Vector], width: usize) -> Result<usize, Overflow> {
    let mut t = 0;
    for c in 0..width {
        if t == rows.len() {
            break;
        }
        // The least non-zero entry in column c among rows t.., moved to row t
        while let Some(best) =
            (t..rows.len()).filter(|&i| rows[i][c] != 0).min_by_key(|&i| rows[i][c].unsigned_abs())
        {
            rows.swap(t, best);
            let (pivot_row, rest) = rows[t..].split_at_mut(1);
            let pivot_row = &pivot_row[0];
            let p = pivot_row[c];
            let mut clean = true;
            for row in rest.iter_mut() {
                if row[c] != 0 {
                    let q = row[c].div_euclid(p);
                    sub_multiple(row, pivot_row, q, c)?;
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

/// A basis of the kernel `{x ∈ Z^n : M x = 0}` of the matrix with these `columns` (each of length `m`), as vectors of
/// length `n`.
pub fn kernel(columns: &[Vector], m: usize) -> Result<Vec<Vector>, Overflow> {
    let n = columns.len();
    // Each row: a column of M, then the unit vector recording which combination of columns it is
    let mut rows: Vec<Vector> = columns
        .iter()
        .enumerate()
        .map(|(j, column)| {
            let mut row = Vec::with_capacity(m + n);
            row.extend_from_slice(column);
            row.resize(m + n, 0);
            row[m + j] = 1;
            row
        })
        .collect();
    let pivots = echelon(&mut rows, m)?;
    Ok(rows.drain(pivots..).map(|row| row[m..].to_vec()).collect())
}

/// A lattice given by a basis in echelon form, for expressing its members in that basis.
struct Echelon {
    rows: Vec<Vector>,
    /// Each row's pivot coordinate
    pivots: Vec<usize>,
}

impl Echelon {
    /// The lattice spanned by `generators` (vectors of length `n`).
    fn new(mut generators: Vec<Vector>, n: usize) -> Result<Self, Overflow> {
        let independent = echelon(&mut generators, n)?;
        generators.truncate(independent);
        let pivots = generators.iter().map(|row| row.iter().position(|&x| x != 0).expect("a pivot row")).collect();
        Ok(Self { rows: generators, pivots })
    }

    /// The coordinates of `v` in the basis.
    ///
    /// # Panics
    /// If `v` isn't in the lattice.
    fn coordinates(&self, mut v: Vector) -> Result<Vector, Overflow> {
        let mut out = vec![0; self.rows.len()];
        for (i, (row, &c)) in self.rows.iter().zip(&self.pivots).enumerate() {
            if v[c] != 0 {
                assert!(v[c] % row[c] == 0, "not in the lattice");
                let q = v[c] / row[c];
                sub_multiple(&mut v, row, q, c)?;
                out[i] = q;
            }
        }
        assert!(v.iter().all(|&x| x == 0), "not in the lattice");
        Ok(out)
    }
}

/// The non-zero diagonal of the Smith normal form of the matrix with these rows (the invariant factors, each dividing
/// the next). A least-modulus pivot at every step, as Perseus 6 does.
#[allow(clippy::needless_range_loop)]
pub fn smith_diagonal(mut a: Vec<Vector>) -> Result<Vec<u128>, Overflow> {
    let rows = a.len();
    let cols = if rows == 0 { 0 } else { a[0].len() };
    let mut diagonal = vec![];
    let mut t = 0;
    while t < rows.min(cols) {
        // The entry of least modulus in the rest
        let mut best: Option<(u128, usize, usize)> = None;
        for (i, row) in a.iter().enumerate().skip(t) {
            for (j, &x) in row.iter().enumerate().skip(t) {
                if x != 0 && best.is_none_or(|(m, _, _)| x.unsigned_abs() < m) {
                    best = Some((x.unsigned_abs(), i, j));
                }
            }
        }
        let Some((_, i, j)) = best else { break };
        a.swap(t, i);
        for row in a.iter_mut() {
            row.swap(t, j);
        }
        loop {
            let p = a[t][t];
            let mut clean = true;
            for i in t + 1..rows {
                if a[i][t] != 0 {
                    let q = a[i][t] / p;
                    let (top, rest) = a.split_at_mut(i);
                    sub_multiple(&mut rest[0], &top[t], q, t)?;
                    clean &= a[i][t] == 0;
                }
            }
            for j in t + 1..cols {
                if a[t][j] != 0 {
                    let q = a[t][j] / p;
                    for row in a.iter_mut().skip(t) {
                        row[j] = row[j].checked_sub(q.checked_mul(row[t]).ok_or(Overflow)?).ok_or(Overflow)?;
                    }
                    clean &= a[t][j] == 0;
                }
            }
            if clean {
                // The pivot must divide the rest; if not, add the offending row to row t and go round again
                let bad = (t + 1..rows).find(|&i| (t + 1..cols).any(|j| a[i][j] % p != 0));
                match bad {
                    None => break,
                    Some(i) => {
                        let (top, rest) = a.split_at_mut(i);
                        sub_multiple(&mut top[t], &rest[0], -1, t)?;
                    }
                }
            } else {
                // A smaller remainder appeared: move the least entry of row and column t to the pivot
                let mut best = (a[t][t].unsigned_abs(), t, t);
                for i in t + 1..rows {
                    if a[i][t] != 0 && a[i][t].unsigned_abs() < best.0 {
                        best = (a[i][t].unsigned_abs(), i, t);
                    }
                }
                for j in t + 1..cols {
                    if a[t][j] != 0 && a[t][j].unsigned_abs() < best.0 {
                        best = (a[t][j].unsigned_abs(), t, j);
                    }
                }
                let (_, i, j) = best;
                a.swap(t, i);
                for row in a.iter_mut() {
                    row.swap(t, j);
                }
            }
        }
        diagonal.push(a[t][t].unsigned_abs());
        t += 1;
    }
    Ok(diagonal)
}

/// The group `A / B`, for `A` spanned by `a` and `B ⊆ A` spanned by `b` (vectors of length `n`).
///
/// # Panics
/// If some generator of `B` isn't in `A`.
pub fn quotient(a: Vec<Vector>, b: Vec<Vector>, n: usize) -> Result<Group, Overflow> {
    let a = Echelon::new(a, n)?;
    let coordinates = b.into_iter().map(|v| a.coordinates(v)).collect::<Result<Vec<_>, _>>()?;
    let diagonal = smith_diagonal(coordinates)?;
    Ok(Group {
        rank: a.rows.len() - diagonal.len(),
        torsion: diagonal.into_iter().filter(|&d| d > 1).collect(),
    })
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
    complex: &'a IntegralFiltered,
    degrees: Vec<Degree>,
    /// Each cell's position within its degree
    position: Vec<usize>,
    n_steps: usize,
}

impl<'a> Arranged<'a> {
    fn new(complex: &'a IntegralFiltered) -> Self {
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
        Self { complex, degrees, position, n_steps }
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
            out[self.position[face]] += i128::from(coefficient);
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
        let mut basis = kernel(&columns, m)?;
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
                        sub_multiple(&mut out, column, -*x, 0)?;
                    }
                }
                Ok(out)
            })
            .collect()
    }

    /// `E^r_p` in degree `k`.
    fn page_entry(&self, r: usize, p: usize, k: usize) -> Result<Group, Overflow> {
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
        quotient(numerator, denominator, self.size(k))
    }
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
    fn smith_normal_form() {
        assert_eq!(smith_diagonal(vec![vec![2, 4, 4], vec![-6, 6, 12], vec![10, -4, -16]]).unwrap(), vec![2, 6, 12]);
        assert_eq!(smith_diagonal(vec![vec![2, 0], vec![0, 3]]).unwrap(), vec![1, 6]);
    }
}
