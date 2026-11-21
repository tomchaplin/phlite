//! Persistent homology over the integers, as far as it has a barcode: the barcode over Q, and the finitely many
//! primes `p` whose barcode over `F_p` differs from it (the primes of the torsion, in the homology of the steps or in
//! the maps between them), with their barcodes.
//!
//! A filtered complex over Z has no barcode (its persistence module, over `Z[t]`, needn't decompose into intervals).
//! What can be computed exactly, for any filtration, is its barcode over every field: over Q, and over each `F_p`,
//! which agrees with Q's for all but finitely many `p`. Those *bad primes* are found as Perseus does: by a
//! fraction-free standard reduction over Z (each column reduced by integer combinations, `a·c - b·R_k` with `a` the
//! pivot of `R_k`, and divided by its content), whose pivot pattern is that over Q. A prime dividing none of the pivots
//! used and none of the contents divided out sees exactly the same reduction mod p (every operation is invertible mod
//! p), so the same barcode: the primes dividing them are the only candidates, and each is checked by reducing mod p.
//!
//! Note that torsion in the homology of the steps alone doesn't find them all: a loop `x`, then a loop `y` and a cell
//! with boundary `p·y - x` has no torsion at any step, but the map `Z -> Z` from the first step to the last is
//! multiplication by `p`, so over `F_p` the first bar dies where over Q it doesn't.
//!
//! The arithmetic is exact in checked `i128` ([`Overflow`] rather than a wrong answer).

use std::collections::{BTreeMap, BTreeSet};

use crate::integral::Overflow;

/// A barcode as pairs of cells (indices in filtration order): `(birth, Some(death))` or `(birth, None)` if essential.
pub type CellBarcode = Vec<(usize, Option<usize>)>;

/// The persistence of a filtered complex over Z: its barcode over Q, and each bad prime with its barcode over `F_p`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegralPersistence {
    pub rational: CellBarcode,
    /// The primes whose barcode differs from Q's, increasing, each with its barcode
    pub torsion: Vec<(u64, CellBarcode)>,
    /// Every candidate checked (the primes dividing a pivot or content of the reduction), increasing
    pub candidates: Vec<u64>,
}

type Column = BTreeMap<usize, i128>;

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Record the prime factors of `|x|` into `primes`.
fn factor_into(x: i128, primes: &mut BTreeSet<u64>) {
    let mut n = x.unsigned_abs();
    if n <= 1 {
        return;
    }
    let mut p = 2u128;
    while p * p <= n && p < 1 << 20 {
        if n.is_multiple_of(p) {
            primes.insert(p as u64);
            while n.is_multiple_of(p) {
                n /= p;
            }
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        factor_large(n, primes);
    }
}

fn mul_mod(a: u128, b: u128, m: u128) -> u128 {
    // Double and add, as a * b may not fit in u128
    let (mut a, mut b, mut out) = (a % m, b, 0u128);
    while b > 0 {
        if b & 1 == 1 {
            out = (out + a) % m;
        }
        a = (a << 1) % m;
        b >>= 1;
    }
    out
}

fn pow_mod(mut a: u128, mut e: u128, m: u128) -> u128 {
    let mut out = 1 % m;
    a %= m;
    while e > 0 {
        if e & 1 == 1 {
            out = mul_mod(out, a, m);
        }
        a = mul_mod(a, a, m);
        e >>= 1;
    }
    out
}

/// Miller–Rabin with the first twelve primes as witnesses (deterministic below 3.3·10^24, and overwhelmingly likely
/// beyond).
fn is_prime(n: u128) -> bool {
    if n < 2 {
        return false;
    }
    let witnesses = [2u128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &w in &witnesses {
        if n.is_multiple_of(w) {
            return n == w;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'witness: for &a in &witnesses {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Factor `n` (no factor below 2^20) by Pollard's rho, recording its primes; any factor that doesn't fit a `u64` is
/// left out (no reduction mod a prime that large could be done anyway: such a prime is reported as unchecked).
fn factor_large(n: u128, primes: &mut BTreeSet<u64>) {
    if n == 1 {
        return;
    }
    if is_prime(n) {
        if let Ok(p) = u64::try_from(n) {
            primes.insert(p);
        }
        return;
    }
    let mut c = 1u128;
    loop {
        let f = |x: u128| (mul_mod(x, x, n) + c) % n;
        let (mut x, mut y, mut d) = (2u128, 2u128, 1u128);
        while d == 1 {
            x = f(x);
            y = f(f(y));
            d = gcd(x.abs_diff(y) as i128, n as i128) as u128;
        }
        if d != n {
            factor_large(d, primes);
            factor_large(n / d, primes);
            return;
        }
        c += 1;
    }
}

/// The barcode, pairs from the low of each reduced column, given the lows (`low[j]`, `None` if column j reduced to
/// zero).
fn barcode_from_lows(low: &[Option<usize>]) -> CellBarcode {
    let n = low.len();
    let mut dies = vec![false; n];
    let mut out = vec![];
    for (j, l) in low.iter().enumerate() {
        if let Some(i) = l {
            dies[*i] = true;
            out.push((*i, Some(j)));
        }
    }
    for j in 0..n {
        if low[j].is_none() && !dies[j] {
            out.push((j, None));
        }
    }
    out.sort_unstable();
    out
}

/// The barcode over `F_p` of the filtered complex with these columns (`boundary[j]`: the faces of cell `j`, cells in
/// filtration order, faces earlier), by the standard reduction with clearing (by dimension, highest first).
pub fn barcode_mod(boundary: &[Vec<(usize, i64)>], degree: &[usize], p: u64) -> CellBarcode {
    let p = u128::from(p);
    let n = boundary.len();
    let reduce = |x: i64| -> u128 { (i128::from(x).rem_euclid(p as i128)) as u128 };
    let inverse = |a: u128| pow_mod(a, p - 2, p);
    let mut low: Vec<Option<usize>> = vec![None; n];
    let mut pivot_of: Vec<Option<usize>> = vec![None; n];
    let mut reduced: Vec<BTreeMap<usize, u128>> = vec![BTreeMap::new(); n];
    let mut cleared = vec![false; n];
    let top = degree.iter().copied().max().unwrap_or(0);
    for d in (0..=top).rev() {
        for j in (0..n).filter(|&j| degree[j] == d) {
            if cleared[j] {
                continue;
            }
            let mut col: BTreeMap<usize, u128> = BTreeMap::new();
            for &(f, x) in &boundary[j] {
                let e = col.entry(f).or_insert(0);
                *e = (*e + reduce(x)) % p;
            }
            col.retain(|_, x| *x != 0);
            while let Some((&i, &b)) = col.iter().next_back() {
                let Some(k) = pivot_of[i] else { break };
                // col -= (b / a) R_k
                let factor = mul_mod(b, inverse(reduced[k][&i]), p);
                for (&r, &x) in &reduced[k] {
                    let e = col.entry(r).or_insert(0);
                    *e = (*e + p - mul_mod(factor, x, p)) % p;
                }
                col.retain(|_, x| *x != 0);
            }
            if let Some((&i, _)) = col.iter().next_back() {
                low[j] = Some(i);
                pivot_of[i] = Some(j);
                cleared[i] = true;
            }
            reduced[j] = col;
        }
    }
    barcode_from_lows(&low)
}

/// The persistence over Z of the filtered complex with these columns (`boundary[j]`: the faces of cell `j` with
/// integer coefficients, cells in filtration order, faces earlier; `degree[j]` its dimension): see the
/// [module docs](self).
pub fn integral_persistence(boundary: &[Vec<(usize, i64)>], degree: &[usize]) -> Result<IntegralPersistence, Overflow> {
    let n = boundary.len();
    let mut low: Vec<Option<usize>> = vec![None; n];
    let mut pivot_of: Vec<Option<usize>> = vec![None; n];
    let mut reduced: Vec<Column> = vec![Column::new(); n];
    let mut cleared = vec![false; n];
    let mut candidates: BTreeSet<u64> = BTreeSet::new();
    let top = degree.iter().copied().max().unwrap_or(0);
    for d in (0..=top).rev() {
        for j in (0..n).filter(|&j| degree[j] == d) {
            if cleared[j] {
                continue;
            }
            let mut col = Column::new();
            for &(f, x) in &boundary[j] {
                *col.entry(f).or_insert(0) += i128::from(x);
            }
            col.retain(|_, x| *x != 0);
            while let Some((&i, &b)) = col.iter().next_back() {
                let Some(k) = pivot_of[i] else { break };
                let a = reduced[k][&i];
                let g = gcd(a, b);
                let (sa, sb) = (a / g, b / g);
                // col := sa·col - sb·R_k (sa is a unit mod p unless p | a, a candidate already)
                for x in col.values_mut() {
                    *x = x.checked_mul(sa).ok_or(Overflow)?;
                }
                for (&r, &x) in &reduced[k] {
                    let e = col.entry(r).or_insert(0);
                    *e = e.checked_sub(sb.checked_mul(x).ok_or(Overflow)?).ok_or(Overflow)?;
                }
                col.retain(|_, x| *x != 0);
                // Divide out the content
                let content = col.values().fold(0, |c, &x| gcd(c, x));
                if content > 1 {
                    factor_into(content, &mut candidates);
                    for x in col.values_mut() {
                        *x /= content;
                    }
                }
            }
            if let Some((&i, &a)) = col.iter().next_back() {
                factor_into(a, &mut candidates);
                low[j] = Some(i);
                pivot_of[i] = Some(j);
                cleared[i] = true;
            }
            reduced[j] = col;
        }
    }
    let rational = barcode_from_lows(&low);
    let candidates: Vec<u64> = candidates.into_iter().collect();
    let torsion = candidates
        .iter()
        .filter_map(|&p| {
            let bars = barcode_mod(boundary, degree, p);
            (bars != rational).then_some((p, bars))
        })
        .collect();
    Ok(IntegralPersistence { rational, torsion, candidates })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The columns of a simplicial complex given by its simplices in filtration order (faces earlier), with signs.
    fn simplicial(simplices: &[Vec<usize>]) -> (Vec<Vec<(usize, i64)>>, Vec<usize>) {
        let index: BTreeMap<&Vec<usize>, usize> = simplices.iter().enumerate().map(|(i, s)| (s, i)).collect();
        let boundary = simplices
            .iter()
            .map(|s| {
                if s.len() == 1 {
                    return vec![];
                }
                (0..s.len())
                    .map(|i| {
                        let mut f = s.clone();
                        f.remove(i);
                        (index[&f], if i % 2 == 0 { 1 } else { -1 })
                    })
                    .collect()
            })
            .collect();
        (boundary, simplices.iter().map(|s| s.len() - 1).collect())
    }

    fn rp2() -> Vec<Vec<usize>> {
        let tops = [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 1, 5], [1, 2, 4], [2, 3, 5], [1, 3, 4], [2, 4, 5], [1, 3, 5],
        ];
        let mut all: BTreeSet<Vec<usize>> = BTreeSet::new();
        for t in tops {
            for mask in 1u32..8 {
                all.insert((0..3).filter(|&i| mask >> i & 1 == 1).map(|i| t[i]).collect());
            }
        }
        let mut all: Vec<Vec<usize>> = all.into_iter().collect();
        all.sort_by_key(|s| (s.len(), s.clone()));
        all
    }

    #[test]
    fn projective_plane() {
        // Over Q: H_0 = Q only; over F_2: H_1 = H_2 = F_2 too, so 2 is the one bad prime
        let (boundary, degree) = simplicial(&rp2());
        let result = integral_persistence(&boundary, &degree).unwrap();
        let essential = |bars: &CellBarcode| bars.iter().filter(|(_, d)| d.is_none()).map(|(b, _)| degree[*b]).collect::<Vec<_>>();
        assert_eq!(essential(&result.rational), vec![0]);
        assert_eq!(result.torsion.len(), 1);
        assert_eq!(result.torsion[0].0, 2);
        let mut over_f2 = essential(&result.torsion[0].1);
        over_f2.sort_unstable();
        assert_eq!(over_f2, vec![0, 1, 2]);
        assert_eq!(result.rational, barcode_mod(&boundary, &degree, 3));
    }

    #[test]
    fn torsion_only_in_a_map() {
        // A loop x (vertex v, edge x), then a loop y and a 2-cell c with dc = p y - x: no step has torsion, but the
        // map Z -> Z from H_1 at the second step to the last is multiplication by p, so the barcodes differ at p
        for p in [2i64, 3, 7, 13] {
            let boundary = vec![vec![], vec![], vec![], vec![(2, p), (1, -1)]];
            let degree = vec![0, 1, 1, 2];
            let result = integral_persistence(&boundary, &degree).unwrap();
            // Over Q, c kills the younger loop y; over F_p its boundary is -x, so it kills x
            assert_eq!(result.rational, vec![(0, None), (1, None), (2, Some(3))]);
            assert_eq!(result.torsion.iter().map(|(q, _)| *q as i64).collect::<Vec<_>>(), vec![p]);
            assert_eq!(result.torsion[0].1, vec![(0, None), (1, Some(3)), (2, None)]);
        }
    }

    #[test]
    fn rational_barcode_matches_a_large_prime_and_bad_primes_are_complete() {
        // Random filtered cellular complexes (elementary pieces, mixed by a unitriangular change of basis): the
        // rational barcode is that over a large prime, and every prime below 50 whose barcode differs is reported
        let mut state = 5u64;
        let mut next = move |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for _ in 0..200 {
            // Elementary pieces in filtration order: a free cell, or d e = m e'
            let mut boundary: Vec<Vec<(usize, i64)>> = vec![];
            let mut degree: Vec<usize> = vec![];
            for _ in 0..1 + next(8) {
                let k = next(3) as usize;
                degree.push(k);
                boundary.push(vec![]);
                if next(3) > 0 {
                    let m = [1i64, 2, 3, 4, 6, 10][next(6) as usize];
                    degree.push(k + 1);
                    boundary.push(vec![(degree.len() - 2, m)]);
                }
            }
            // Mix: each cell gains multiples of earlier cells of its dimension (b_j = e_j + a e_i), which changes the
            // boundaries of the cofaces of e_j... keep it simple: add earlier same-dimension cycles' combinations to
            // boundaries of later cells only where the result is still a cycle: multiply boundaries by units
            for col in &mut boundary {
                for (_, x) in col.iter_mut() {
                    if next(2) == 0 {
                        *x = -*x;
                    }
                }
            }
            let result = integral_persistence(&boundary, &degree).unwrap();
            assert_eq!(result.rational, barcode_mod(&boundary, &degree, 1_000_003));
            let reported: BTreeSet<u64> = result.torsion.iter().map(|(p, _)| *p).collect();
            for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
                let differs = barcode_mod(&boundary, &degree, p) != result.rational;
                assert_eq!(differs, reported.contains(&p), "p = {p}");
            }
        }
    }

    #[test]
    fn factoring() {
        let mut primes = BTreeSet::new();
        factor_into(2 * 3 * 3 * 1_000_003 * 1_000_033, &mut primes);
        assert_eq!(primes.into_iter().collect::<Vec<_>>(), vec![2, 3, 1_000_003, 1_000_033]);
        assert!(is_prime(1_000_000_007) && !is_prime(1_000_000_007 * 3));
    }
}
