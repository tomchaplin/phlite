//! Shrinking a complex over the integers without changing its homology: algebraic Morse theory by elimination, and
//! the homology of what is left by Smith forms.
//!
//! A pair `(σ, τ)`, `σ` a face of `τ` with incidence `ε = [τ : σ] = ±1`, is eliminated by Gaussian elimination:
//! every other coface `ρ` of `σ` gets `∂ρ - [ρ : σ] ε ∂τ`, `τ` is dropped from its cofaces' boundaries, and `σ` and
//! `τ` go. The complex left is chain homotopy equivalent to the one before, over any principal ideal domain, provided
//! the incidence is a unit (Kozlov; Sköldberg): over Z, `±1` only, which this checks. In a filtration, pairs are made
//! only between cells entering at the same step, and then the complex left has the same persistent homology groups
//! over the PID (Mischaikow and Nanda, *Morse theory for filtrations and efficient computation of persistent
//! homology*, DCG 2013: their Definitions 2.2 and 4.1 and Theorem 4.3). This is the second phase of Perseus 6's
//! reduction: pairs with no fill-in first (a cell with one face, or one coface), then the rest by least Markowitz
//! cost.
//!
//! Only groups survive: generators of the smaller complex aren't chains of the original one.

use super::{smith_diagonal, Group};

/// A complex over the integers in compressed columns: cell `c` has degree `degrees[c]` and boundary
/// `Σ coefficients[e] · faces[e]` for `e` in `indptr[c]..indptr[c + 1]`, its faces sorted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompressedComplex {
    pub degrees: Vec<u32>,
    pub indptr: Vec<usize>,
    pub faces: Vec<u32>,
    pub coefficients: Vec<i64>,
}

/// The default cap on the cost of a pair (in entries changed): pairs costlier than this are left alone.
pub const ELIMINATION_COST: u64 = 1 << 12;

/// Eliminate unit pairs of `piece` (see the [module docs](self)), with `steps` (one per cell) if it is filtered,
/// pairs costing more than `max_cost` left alone. Returns the complex left, cells renumbered in their order, and for
/// each the index it had.
///
/// A pair `(σ, τ)` with `[τ : σ] = ε = ±1` is eliminated by Gaussian elimination: every other coface `ρ` of `σ` gets
/// `∂ρ - [ρ : σ] ε ∂τ` (which no longer has `σ`), `τ` is dropped from the boundary of its cofaces, and `σ` and `τ`
/// go. The complex left is chain homotopy equivalent to the one before, over the integers. The cost of a pair is
/// `(|∂τ| - 1)(|δσ| - 1)`, the most entries it can change, so the fill-in-free pairs of the first phase (a cell with
/// one face, or one coface) cost nothing and still come first. A pair costing more than `max_cost`, or
/// whose coefficients would overflow, is left alone.
pub fn eliminate(
    piece: CompressedComplex,
    steps: Option<&[usize]>,
    max_cost: u64,
) -> (CompressedComplex, Vec<usize>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = piece.degrees.len();
    if piece.faces.is_empty() {
        return (piece, (0..n).collect());
    }
    // Cells may be paired only within a step (Mischaikow and Nanda's Definition 4.1)
    let same_step = |a: usize, b: usize| steps.is_none_or(|s| s[a] == s[b]);
    let mut faces: Vec<Vec<(u32, i64)>> = (0..n)
        .map(|c| {
            let range = piece.indptr[c]..piece.indptr[c + 1];
            let mut row: Vec<(u32, i64)> =
                piece.faces[range.clone()].iter().copied().zip(piece.coefficients[range].iter().copied()).collect();
            row.sort_unstable();
            row
        })
        .collect();
    drop(piece.faces);
    drop(piece.coefficients);
    let mut cofaces: Vec<Vec<u32>> = vec![vec![]; n];
    for (c, row) in faces.iter().enumerate() {
        for &(f, _) in row {
            cofaces[f as usize].push(c as u32);
        }
    }
    let mut alive = vec![true; n];
    let unit = |x: i64| x == 1 || x == -1;
    // The cheapest pair with tau = c: a unit face with the fewest cofaces
    let best = |c: usize, faces: &[Vec<(u32, i64)>], cofaces: &[Vec<u32>]| -> Option<(u64, u32, u32)> {
        let row = &faces[c];
        row.iter()
            .filter(|&&(f, x)| unit(x) && same_step(f as usize, c))
            .map(|&(f, _)| ((row.len() as u64 - 1) * (cofaces[f as usize].len() as u64 - 1), c as u32, f))
            .min()
    };
    let remove_from = |list: &mut Vec<u32>, x: u32| {
        if let Some(i) = list.iter().position(|&y| y == x) {
            list.swap_remove(i);
        }
    };
    // Cells to look at for a free pair: one face (a unit), or one coface (with a unit coefficient)
    let mut work: std::collections::VecDeque<u32> = (0..n as u32).collect();
    let mut heap: BinaryHeap<Reverse<(u64, u32, u32)>> = BinaryHeap::new();
    let mut heap_built = false;
    let mut new_row: Vec<(u32, i64)> = vec![];
    let mut updated: Vec<(u32, Vec<(u32, i64)>)> = vec![];
    loop {
        // A free pair, if the work list has one: (sigma, tau)
        let mut free = None;
        while let Some(c) = work.pop_front() {
            let c = c as usize;
            if !alive[c] {
                continue;
            }
            if let [(f, x)] = faces[c][..] {
                if unit(x) && same_step(f as usize, c) {
                    free = Some((f as usize, c));
                    break;
                }
            }
            if let [g] = cofaces[c][..] {
                let x = faces[g as usize].iter().find(|&&(f, _)| f as usize == c).expect("a coface").1;
                if unit(x) && same_step(c, g as usize) {
                    free = Some((c, g as usize));
                    break;
                }
            }
        }
        let (sg, t) = match free {
            Some(pair) => pair,
            None => {
                // Otherwise the cheapest pair, by Markowitz cost
                if !heap_built {
                    heap = (0..n).filter(|&c| alive[c]).filter_map(|c| best(c, &faces, &cofaces)).map(Reverse).collect();
                    heap_built = true;
                }
                let mut found = None;
                while let Some(Reverse((cost, tau, sigma))) = heap.pop() {
                    if !alive[tau as usize] || !alive[sigma as usize] {
                        continue;
                    }
                    match best(tau as usize, &faces, &cofaces) {
                        Some(now) if now == (cost, tau, sigma) => {
                            if cost <= max_cost {
                                found = Some((sigma as usize, tau as usize));
                            }
                            break;
                        }
                        Some(now) => heap.push(Reverse(now)),
                        None => {}
                    }
                }
                match found {
                    Some(pair) => pair,
                    None => break,
                }
            }
        };
        let (tau, sigma) = (t as u32, sg as u32);
        let epsilon = faces[t].iter().find(|&&(f, _)| f == sigma).expect("a face").1;
        // The new boundaries of sigma's other cofaces: row - [row : sigma] epsilon faces[tau], computed before any is
        // changed (so an overflow leaves everything as it was). With tau's only face sigma, that just drops sigma.
        updated.clear();
        let mut overflow = false;
        for &rho in &cofaces[sg] {
            if rho == tau {
                continue;
            }
            let row = &faces[rho as usize];
            let c = row.iter().find(|&&(f, _)| f == sigma).expect("a coface").1;
            let Some(factor) = c.checked_mul(epsilon).and_then(i64::checked_neg) else {
                overflow = true;
                break;
            };
            new_row.clear();
            let (a, b) = (row, &faces[t]);
            let (mut i, mut j) = (0, 0);
            while i < a.len() || j < b.len() {
                if j == b.len() || (i < a.len() && a[i].0 < b[j].0) {
                    new_row.push(a[i]);
                    i += 1;
                } else if i == a.len() || b[j].0 < a[i].0 {
                    let Some(x) = b[j].1.checked_mul(factor) else {
                        overflow = true;
                        break;
                    };
                    new_row.push((b[j].0, x));
                    j += 1;
                } else {
                    let Some(x) = b[j].1.checked_mul(factor).and_then(|x| x.checked_add(a[i].1)) else {
                        overflow = true;
                        break;
                    };
                    if x != 0 {
                        new_row.push((a[i].0, x));
                    }
                    i += 1;
                    j += 1;
                }
            }
            if overflow {
                break;
            }
            updated.push((rho, new_row.clone()));
        }
        if overflow {
            // Leave this pair: it won't come back from the heap, since its entry is gone
            continue;
        }
        for (rho, row) in updated.drain(..) {
            let old = std::mem::replace(&mut faces[rho as usize], row);
            for &(f, _) in &old {
                if faces[rho as usize].binary_search_by_key(&f, |&(g, _)| g).is_err() {
                    remove_from(&mut cofaces[f as usize], rho);
                    work.push_back(f);
                }
            }
            for &(f, _) in &faces[rho as usize] {
                if old.binary_search_by_key(&f, |&(g, _)| g).is_err() {
                    cofaces[f as usize].push(rho);
                }
            }
            work.push_back(rho);
            if heap_built {
                if let Some(pair) = best(rho as usize, &faces, &cofaces) {
                    heap.push(Reverse(pair));
                }
            }
        }
        // Drop tau (from its cofaces' boundaries) and sigma
        alive[t] = false;
        alive[sg] = false;
        for rho in std::mem::take(&mut cofaces[t]) {
            let row = &mut faces[rho as usize];
            if let Ok(i) = row.binary_search_by_key(&tau, |&(g, _)| g) {
                row.remove(i);
            }
            work.push_back(rho);
            if heap_built {
                if let Some(pair) = best(rho as usize, &faces, &cofaces) {
                    heap.push(Reverse(pair));
                }
            }
        }
        for cell in [t, sg] {
            for (f, _) in std::mem::take(&mut faces[cell]) {
                remove_from(&mut cofaces[f as usize], cell as u32);
                work.push_back(f);
            }
        }
        cofaces[sg].clear();
    }
    // The cells left, renumbered in order
    let mut new_index = vec![u32::MAX; n];
    let mut out = CompressedComplex { degrees: vec![], indptr: vec![0], faces: vec![], coefficients: vec![] };
    let mut kept = vec![];
    for c in 0..n {
        if alive[c] {
            new_index[c] = out.degrees.len() as u32;
            kept.push(c);
            out.degrees.push(piece.degrees[c]);
            for &(f, x) in &faces[c] {
                out.faces.push(new_index[f as usize]);
                out.coefficients.push(x);
            }
            out.indptr.push(out.faces.len());
        }
    }
    (out, kept)
}

/// Matrices of a piece left after elimination with more entries than this are not put in Smith form (their groups
/// are `None`); after elimination they are nearly always tiny.
pub const SMITH_ENTRIES: usize = 1 << 24;

/// The homology over the integers of `piece` in degrees `0..n_degrees`, by Smith forms of its boundary matrices: `None`
/// for a degree whose boundary matrices have more than [`SMITH_ENTRIES`] entries, or overflow.
pub fn homology_groups(piece: &CompressedComplex, n_degrees: usize) -> Vec<Option<Group>> {
    let mut groups: Vec<Option<Group>> = vec![Some(Group::default()); n_degrees];
    let n = piece.degrees.len();
    // The cells of each degree, and each cell's position among them
    let top = piece.degrees.iter().copied().max().map_or(0, |d| d as usize);
    let mut cells: Vec<Vec<usize>> = vec![vec![]; top + 2];
    let mut position = vec![0usize; n];
    for (c, &k) in piece.degrees.iter().enumerate() {
        position[c] = cells[k as usize].len();
        cells[k as usize].push(c);
    }
    // The diagonal of the Smith form of the boundary from degree k to k - 1, for k = 1..=n_degrees
    let diagonal = |k: usize| -> Option<Vec<u128>> {
        if k == 0 || k > top {
            return Some(vec![]);
        }
        let (rows, cols) = (cells[k - 1].len(), cells[k].len());
        if rows.saturating_mul(cols) > SMITH_ENTRIES {
            return None;
        }
        let mut a = vec![vec![0i128; cols]; rows];
        for (j, &c) in cells[k].iter().enumerate() {
            for e in piece.indptr[c]..piece.indptr[c + 1] {
                a[position[piece.faces[e] as usize]][j] = i128::from(piece.coefficients[e]);
            }
        }
        smith_diagonal(a).ok()
    };
    let diagonals: Vec<Option<Vec<u128>>> = (0..=n_degrees).map(diagonal).collect();
    for k in 0..n_degrees {
        let size = cells.get(k).map_or(0, Vec::len);
        groups[k] = match (&diagonals[k], &diagonals[k + 1]) {
            (Some(out), Some(into)) => {
                let free = size - out.len() - into.len();
                Some(Group { rank: free, torsion: into.iter().copied().filter(|&d| d > 1).collect() })
            }
            _ => None,
        };
    }
    groups
}

