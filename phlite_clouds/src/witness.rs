//! Witness complexes, as GUDHI's `EuclideanWitnessComplex` (relaxed) and `EuclideanStrongWitnessComplex`: simplices on
//! a few *landmarks*, entering when enough of the other points (*witnesses*) vouch for them, a small complex that
//! approximates the topology of all the points.
//!
//! With `d(w, l)` the distance from a witness to a landmark, and values `α²`:
//!
//! * **strong**: `w` witnesses `σ` at `max_{l ∈ σ} d(w, l)² - d(w, L)²` (how far `σ`'s points are beyond `w`'s nearest
//!   landmark), and `σ` enters at the least of these over the witnesses;
//! * **relaxed** (the witness complex of de Silva and Carlsson, relaxed by `α`): `w` witnesses `σ` at
//!   `max_{l ∈ σ} d(w, l)² - min_{l ∉ σ} d(w, l)²` (how far `σ`'s points are beyond the nearest landmark outside it),
//!   the least over the witnesses, and `σ` enters once all its faces are witnessed too (the largest of their values).
//!
//! Both are found witness by witness: only landmarks within `max_alpha_square` of what matters to `w` can be in a
//! simplex it witnesses (for the strong complex, its nearest landmark; for the relaxed one, its `(k + 2)`-th nearest,
//! for simplices of dimension at most `k`, since a simplex's first missing landmark must be among those).

use std::collections::HashMap;

/// Which witness complex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Witness {
    Strong,
    Relaxed,
}

/// The witness complex: every simplex of dimension at most `dimension` on the landmarks (by index) with value at most
/// `max_alpha_square`, with its value, faces before cofaces. Witnesses are processed in parallel, each enumerating
/// the subsets of its candidate landmarks without allocating, with simplices keyed by integers.
pub fn witness_simplices(
    landmarks: &[Vec<f64>],
    witnesses: &[Vec<f64>],
    kind: Witness,
    dimension: usize,
    max_alpha_square: f64,
) -> Vec<(f64, Vec<u32>)> {
    use rayon::prelude::*;
    let n_landmarks = landmarks.len();
    let k_max = dimension + 1;
    // Simplices as base-(n + 1) numbers (vertices + 1 as digits, increasing), when they fit 128 bits
    let radix = n_landmarks as u128 + 1;
    let fits = k_max <= MAX_VERTICES && radix.checked_pow(k_max as u32).is_some();
    if !fits {
        return witness_simplices_slow(landmarks, witnesses, kind, dimension, max_alpha_square);
    }
    let encode = |vertices: &[u32]| vertices.iter().fold(0u128, |acc, &v| acc * radix + v as u128 + 1);
    let decode = |mut key: u128| {
        let mut out = vec![];
        while key > 0 {
            out.push((key % radix) as u32 - 1);
            key /= radix;
        }
        out.reverse();
        out
    };
    let distance2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    let witness_values = |w: &Vec<f64>, best: &mut HashMap<u128, f64>| {
        let d2: Vec<f64> = landmarks.iter().map(|l| distance2(w, l)).collect();
        let mut sorted: Vec<u32> = (0..n_landmarks as u32).collect();
        sorted.sort_by(|&a, &b| d2[a as usize].total_cmp(&d2[b as usize]).then(a.cmp(&b)));
        let Some(&nearest) = sorted.first() else { return };
        let reference = match kind {
            Witness::Strong => d2[nearest as usize],
            Witness::Relaxed => sorted.get(dimension + 1).map_or(f64::INFINITY, |&l| d2[l as usize]),
        };
        let bound = reference + max_alpha_square;
        let n = sorted.iter().take_while(|&&l| d2[l as usize] <= bound).count();
        // Every subset of the first n landmarks in w's order, of at most k_max of them, as increasing positions
        let mut positions = [0usize; MAX_VERTICES];
        let mut len = 1;
        if n == 0 {
            return;
        }
        loop {
            let farthest = d2[sorted[positions[len - 1]] as usize];
            let value = match kind {
                Witness::Strong => farthest - d2[nearest as usize],
                Witness::Relaxed => {
                    let missing = (0..len).find(|&i| positions[i] != i).unwrap_or(len);
                    sorted.get(missing).map_or(f64::NEG_INFINITY, |&l| farthest - d2[l as usize])
                }
            }
            .max(0.0);
            if value <= max_alpha_square {
                let mut vertices = [0u32; MAX_VERTICES];
                for i in 0..len {
                    vertices[i] = sorted[positions[i]];
                }
                vertices[..len].sort_unstable();
                let entry = best.entry(encode(&vertices[..len])).or_insert(f64::INFINITY);
                *entry = entry.min(value);
            }
            // The next subset: extend if we can, else advance the last position (dropping the exhausted ones)
            if len < k_max && positions[len - 1] + 1 < n {
                positions[len] = positions[len - 1] + 1;
                len += 1;
                continue;
            }
            loop {
                positions[len - 1] += 1;
                if positions[len - 1] < n {
                    break;
                }
                len -= 1;
                if len == 0 {
                    return;
                }
            }
        }
    };
    let best: HashMap<u128, f64> = witnesses
        .par_chunks(64)
        .fold(HashMap::new, |mut best, chunk| {
            for w in chunk {
                witness_values(w, &mut best);
            }
            best
        })
        .reduce(HashMap::new, |a, b| {
            if a.len() < b.len() {
                return merge(b, a);
            }
            merge(a, b)
        });
    fn merge(mut into: HashMap<u128, f64>, from: HashMap<u128, f64>) -> HashMap<u128, f64> {
        for (k, v) in from {
            let entry = into.entry(k).or_insert(f64::INFINITY);
            *entry = entry.min(v);
        }
        into
    }
    finish(best.into_iter().map(|(k, v)| (v, decode(k))).collect(), max_alpha_square)
}

/// The most vertices a simplex can have on the fast path.
const MAX_VERTICES: usize = 16;

/// Faces first; a relaxed simplex enters once all its faces are witnessed (the strong values are already monotone).
fn finish(mut simplices: Vec<(f64, Vec<u32>)>, max_alpha_square: f64) -> Vec<(f64, Vec<u32>)> {
    simplices.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then(a.1.cmp(&b.1)));
    let mut value_of: HashMap<Vec<u32>, f64> = HashMap::with_capacity(simplices.len());
    let mut out = Vec::with_capacity(simplices.len());
    for (value, simplex) in simplices {
        let mut value = value;
        let mut complete = true;
        if simplex.len() > 1 {
            for j in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(j);
                match value_of.get(&face) {
                    Some(&v) => value = value.max(v),
                    None => complete = false,
                }
            }
        }
        if complete && value <= max_alpha_square {
            value_of.insert(simplex.clone(), value);
            out.push((value, simplex));
        }
    }
    out
}

/// [`witness_simplices`] for simplices whose keys don't fit 128 bits: serially, simplices as vectors.
fn witness_simplices_slow(
    landmarks: &[Vec<f64>],
    witnesses: &[Vec<f64>],
    kind: Witness,
    dimension: usize,
    max_alpha_square: f64,
) -> Vec<(f64, Vec<u32>)> {
    let n_landmarks = landmarks.len();
    let mut best: HashMap<Vec<u32>, f64> = HashMap::new();
    let distance2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    for w in witnesses {
        let d2: Vec<f64> = landmarks.iter().map(|l| distance2(w, l)).collect();
        let mut sorted: Vec<u32> = (0..n_landmarks as u32).collect();
        sorted.sort_by(|&a, &b| d2[a as usize].total_cmp(&d2[b as usize]).then(a.cmp(&b)));
        let Some(&nearest) = sorted.first() else { break };
        let reference = match kind {
            Witness::Strong => d2[nearest as usize],
            Witness::Relaxed => sorted.get(dimension + 1).map_or(f64::INFINITY, |&l| d2[l as usize]),
        };
        let bound = reference + max_alpha_square;
        let candidates: Vec<u32> = sorted.iter().copied().take_while(|&l| d2[l as usize] <= bound).collect();
        // Every subset of the candidates of at most dimension + 1 landmarks, by increasing position
        let mut stack: Vec<Vec<usize>> = (0..candidates.len()).map(|i| vec![i]).collect();
        while let Some(positions) = stack.pop() {
            let farthest = d2[candidates[*positions.last().expect("non-empty")] as usize];
            let value = match kind {
                Witness::Strong => farthest - d2[nearest as usize],
                Witness::Relaxed => {
                    // The first landmark in w's order not in the simplex (positions are increasing)
                    let missing = positions.iter().enumerate().find(|&(i, &p)| p != i).map_or(positions.len(), |(i, _)| i);
                    sorted.get(missing).map_or(f64::NEG_INFINITY, |&l| farthest - d2[l as usize])
                }
            }
            .max(0.0);
            if value <= max_alpha_square {
                let mut simplex: Vec<u32> = positions.iter().map(|&p| candidates[p]).collect();
                simplex.sort_unstable();
                let entry = best.entry(simplex).or_insert(f64::INFINITY);
                *entry = entry.min(value);
            }
            if positions.len() <= dimension {
                let last = *positions.last().expect("non-empty");
                for next in last + 1..candidates.len() {
                    let mut longer = positions.clone();
                    longer.push(next);
                    stack.push(longer);
                }
            }
        }
    }
    finish(best.into_iter().map(|(s, v)| (v, s)).collect(), max_alpha_square)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast path (integer keys, in parallel) agrees with the plain one.
    #[test]
    fn fast_matches_slow() {
        let mut x: u64 = 7;
        let mut rand = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 1000) as f64 / 1000.0
        };
        for trial in 0..6 {
            let witnesses: Vec<Vec<f64>> = (0..80).map(|_| vec![rand(), rand(), rand()]).collect();
            let landmarks: Vec<Vec<f64>> = witnesses.iter().take(8 + trial).cloned().collect();
            for kind in [Witness::Strong, Witness::Relaxed] {
                for bound in [0.05, f64::INFINITY] {
                    let mut fast = witness_simplices(&landmarks, &witnesses, kind, 2, bound);
                    let mut slow = witness_simplices_slow(&landmarks, &witnesses, kind, 2, bound);
                    fast.sort_by(|a, b| a.1.cmp(&b.1));
                    slow.sort_by(|a, b| a.1.cmp(&b.1));
                    assert_eq!(fast, slow, "{kind:?} {bound}");
                }
            }
        }
    }

    #[test]
    fn landmarks_witnessing_themselves() {
        // Witnesses at the landmarks: every landmark at 0; an edge strongly witnessed by w at d(w, far end)^2
        let landmarks = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]];
        let simplices = witness_simplices(&landmarks, &landmarks, Witness::Strong, 1, f64::INFINITY);
        let value = |s: &[u32]| simplices.iter().find(|(_, t)| t == s).map(|(v, _)| *v);
        assert_eq!(value(&[0]), Some(0.0));
        assert_eq!(value(&[0, 1]), Some(1.0));
        assert_eq!(value(&[0, 2]), Some(4.0));
        // (1-2 is best witnessed by landmark 0, at max(1, 4) - 0)
        assert_eq!(value(&[1, 2]), Some(4.0));
        // Relaxed: the edge 0-1 is witnessed by 0 at d(0,1)^2 - d(0,2)^2 < 0, so at 0
        let relaxed = witness_simplices(&landmarks, &landmarks, Witness::Relaxed, 1, f64::INFINITY);
        assert_eq!(relaxed.iter().find(|(_, t)| t == &[0, 1]).map(|(v, _)| *v), Some(0.0));
    }
}
