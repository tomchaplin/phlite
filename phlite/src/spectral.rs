//! The spectral sequence of a filtered chain complex over a field, read off its barcode.
//!
//! A filtration `F_0 C ⊆ F_1 C ⊆ ...` of a chain complex `C` has a spectral sequence whose page `E^0` is the
//! associated graded `F_p C / F_{p-1} C`, `E^1` its homology, and so on, converging to `H(C)`. Over a field the whole
//! spectral sequence is determined by the persistence barcode of the filtration (the bars of length zero only matter
//! for `E^0`): a bar `[b, d)` in degree `k`, with `b` and `d` counted in filtration *steps*, is a generator of `E^r` in
//! degree `k` at step `b` and one in degree `k + 1` at step `d` for every page `r <= d - b`, and the differential
//! `d^{d-b}` cancels the two; an essential bar is a generator of every page in degree `k` at step `b`. So
//!
//! `dim E^r_p` (degree `k`) = #{bars of degree `k` born at `p` lasting at least `r` steps}
//! \+ #{bars of degree `k - 1` dying at `p` that lasted at least `r` steps}.
//!
//! Steps are positions among the distinct filtration values ([`step_of`]), so page `r` concerns differentials that
//! cross `r` values. If the barcode is of a truncation of the filtration at step `T` (so that a bar still alive at `T`
//! may die later), an entry at step `p` is known for pages `r <= T + 1 - p` only ([`spectral_pages`]).
//!
//! For the spectral sequence of the length filtration of the reachability complex of a digraph (whose first page is
//! magnitude homology), see `phlite_networks::mpss`.
//!
//! # Computing entries lazily
//!
//! [`SpectralBackend`] answers for one entry at a time, `E^r_p` in degree `k` and the image of the differential `d^r`
//! out of it, so pages can be computed one by one (as the Python `SpectralSequence` does):
//!
//! * [`BarcodeSpectral`]: over a field, from the barcode (computed once, by any reduction), each entry a lookup.
//! * [`LocalSpectral`]: over the integers (or wherever there is no barcode), for any [`StepFiltered`] complex, a
//!   boundary oracle that can list its cells by degree and step. `E^r_p` only depends on the quotient complex
//!   `F_{p+r-1} / F_{p-r}`, a window of `2r - 1` steps, and the image of `d^r` out of it on `F_p / F_{p-r-1}`, so each
//!   entry is computed exactly ([`crate::integral`]) from the cells in its window alone.
//!
//! A spectral sequence over a field is determined by a barcode, so (up to isomorphism) every one with finite pages is
//! that of a filtered complex; this covers those given by a filtered chain complex (double complexes by their total
//! complex, the MPSS, cellular Serre and Atiyah–Hirzebruch spectral sequences, ...), but not those known only from an
//! exact couple.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use crate::fields::Integer;
use crate::integral::{self, Group, IntegralFiltered, Overflow};
use crate::matrices::MatrixOracle;

/// A filtered chain complex given lazily: its boundary as a [`MatrixOracle`] (columns and rows both cells), each cell's
/// degree and *step* (its filtration value coarsened to `0, 1, ...`; faces never come after their cofaces), and the
/// cells of a degree in a window of steps, which is all [`LocalSpectral`] ever asks for.
pub trait StepFiltered: MatrixOracle<ColT = <Self as MatrixOracle>::RowT> {
    /// The number of steps: every cell has step below it.
    fn n_steps(&self) -> usize;
    /// The degree of `cell`.
    fn degree(&self, cell: &Self::RowT) -> usize;
    /// The step of `cell`.
    fn step(&self, cell: &Self::RowT) -> usize;
    /// The cells of degree `degree` whose step is in `steps`, in any order.
    fn cells(&self, degree: usize, steps: RangeInclusive<usize>) -> Vec<Self::RowT>;
}

/// An explicit filtered complex over Z is a [`StepFiltered`] matrix (cells by index).
impl MatrixOracle for IntegralFiltered {
    type CoefficientField = Integer;
    type ColT = usize;
    type RowT = usize;

    fn column(&self, col: usize) -> impl Iterator<Item = (Integer, usize)> {
        self.boundary[col].iter().filter_map(|&(face, c)| Integer::new(c).map(|c| (c, face)))
    }
}

impl StepFiltered for IntegralFiltered {
    fn n_steps(&self) -> usize {
        self.step.iter().max().map_or(0, |&s| s + 1)
    }

    fn degree(&self, cell: &usize) -> usize {
        self.degree[*cell]
    }

    fn step(&self, cell: &usize) -> usize {
        self.step[*cell]
    }

    fn cells(&self, degree: usize, steps: RangeInclusive<usize>) -> Vec<usize> {
        (0..self.degree.len()).filter(|&c| self.degree[c] == degree && steps.contains(&self.step[c])).collect()
    }
}

/// Entries of a spectral sequence, one at a time (see the [module docs](self#computing-entries-lazily)).
pub trait SpectralBackend: Send + Sync {
    /// The number of steps `p`.
    fn n_steps(&self) -> usize;
    /// The number of degrees `k`.
    fn n_degrees(&self) -> usize;
    /// `E^r_p` in degree `k` (`r >= 1`), or `None` if not determined (by a truncated filtration).
    fn entry(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow>;
    /// The image of `d^r` out of `E^r_p` in degree `k` (into `E^r_{p-r}` in degree `k - 1`), or `None` if not
    /// determined. Zero when `p < r` or `k = 0`.
    fn differential(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow>;
    /// A page from which every differential is zero (so it is `E^∞`), if known.
    fn degenerates_at(&self) -> Option<usize>;
}

/// The spectral sequence over a field, from its barcode in steps: each entry and differential a lookup.
#[derive(Debug, Clone)]
pub struct BarcodeSpectral {
    n_steps: usize,
    n_degrees: usize,
    truncated_at: Option<usize>,
    /// For each (degree, birth step): the lengths of the bars born there (`usize::MAX` if essential), sorted
    born: HashMap<(usize, usize), Vec<usize>>,
    /// For each (degree of the bar, death step): the lengths of the bars dying there, sorted
    died: HashMap<(usize, usize), Vec<usize>>,
    longest: usize,
}

impl BarcodeSpectral {
    /// From `bars` (those of length zero are ignored), in steps `0..n_steps` and degrees `0..n_degrees`; if the bars are
    /// of a filtration truncated after step `truncated_at`, entries needing later steps are `None`.
    pub fn new(bars: &[StepBar], n_steps: usize, n_degrees: usize, truncated_at: Option<usize>) -> Self {
        let mut born: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut died: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut longest = 0;
        for bar in bars {
            let length = bar.death.map_or(usize::MAX, |d| d.saturating_sub(bar.birth));
            if length == 0 {
                continue;
            }
            born.entry((bar.degree, bar.birth)).or_default().push(length);
            if let Some(d) = bar.death {
                died.entry((bar.degree, d)).or_default().push(length);
                longest = longest.max(length);
            }
        }
        for lengths in born.values_mut().chain(died.values_mut()) {
            lengths.sort_unstable();
        }
        Self { n_steps, n_degrees, truncated_at, born, died, longest }
    }

    fn known(&self, r: usize, p: usize) -> bool {
        self.truncated_at.is_none_or(|t| p + r <= t + 1)
    }

    /// How many of `lengths` (sorted) are at least `r`.
    fn at_least(lengths: Option<&Vec<usize>>, r: usize) -> usize {
        lengths.map_or(0, |l| l.len() - l.partition_point(|&x| x < r))
    }
}

impl SpectralBackend for BarcodeSpectral {
    fn n_steps(&self) -> usize {
        self.n_steps
    }

    fn n_degrees(&self) -> usize {
        self.n_degrees
    }

    fn entry(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if !self.known(r, p) {
            return Ok(None);
        }
        let mut rank = Self::at_least(self.born.get(&(k, p)), r);
        if k >= 1 {
            rank += Self::at_least(self.died.get(&(k - 1, p)), r);
        }
        Ok(Some(Group { rank, torsion: vec![] }))
    }

    fn differential(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if !self.known(r, p) {
            return Ok(None);
        }
        // d^r out of (p, k) kills the bars of degree k - 1 from p - r to p
        let rank = if k == 0 || p < r {
            0
        } else {
            self.died.get(&(k - 1, p)).map_or(0, |l| l.iter().filter(|&&x| x == r).count())
        };
        Ok(Some(Group { rank, torsion: vec![] }))
    }

    fn degenerates_at(&self) -> Option<usize> {
        Some(self.longest + 1)
    }
}

/// The spectral sequence of a [`StepFiltered`] complex over Z, each entry computed exactly from the cells in its
/// window of steps (see the [module docs](self#computing-entries-lazily)).
#[derive(Debug, Clone)]
pub struct LocalSpectral<S> {
    complex: S,
    n_degrees: usize,
    truncated_at: Option<usize>,
}

impl<S: StepFiltered<CoefficientField = Integer> + Send + Sync> LocalSpectral<S> {
    /// In degrees `0..n_degrees`; if `complex` is a filtration truncated after step `truncated_at`, entries needing later
    /// steps are `None`.
    pub fn new(complex: S, n_degrees: usize, truncated_at: Option<usize>) -> Self {
        Self { complex, n_degrees, truncated_at }
    }

    /// The complex.
    pub fn complex(&self) -> &S {
        &self.complex
    }

    /// The quotient complex of the cells of these degrees with steps in `steps`, as an [`IntegralFiltered`] (faces
    /// outside the window are dropped: those below are quotiented out, and none are above).
    fn window(&self, degrees: RangeInclusive<usize>, steps: RangeInclusive<usize>) -> IntegralFiltered
    where
        S::RowT: std::hash::Hash,
    {
        let cells: Vec<S::RowT> = degrees.flat_map(|d| self.complex.cells(d, steps.clone())).collect();
        let index: HashMap<S::RowT, usize> = cells.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
        let mut out = IntegralFiltered::default();
        for cell in &cells {
            out.degree.push(self.complex.degree(cell));
            out.step.push(self.complex.step(cell));
            out.boundary.push(
                self.complex
                    .column(cell.clone())
                    .filter_map(|(c, face)| index.get(&face).map(|&f| (f, c.get())))
                    .collect(),
            );
        }
        out
    }
}

impl<S> SpectralBackend for LocalSpectral<S>
where
    S: StepFiltered<CoefficientField = Integer> + Send + Sync,
    S::RowT: std::hash::Hash,
{
    fn n_steps(&self) -> usize {
        self.complex.n_steps()
    }

    fn n_degrees(&self) -> usize {
        self.n_degrees
    }

    fn entry(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if self.truncated_at.is_some_and(|t| r >= 1 && p + r > t + 1) {
            return Ok(None);
        }
        let reach = r.max(1);
        let steps = (p + 1).saturating_sub(reach)..=p + reach - 1;
        let window = self.window(k.saturating_sub(1)..=k + 1, steps);
        integral::page_entry(&window, r, p, k).map(Some)
    }

    fn differential(&self, r: usize, p: usize, k: usize) -> Result<Option<Group>, Overflow> {
        if self.truncated_at.is_some_and(|t| p + r > t + 1) {
            return Ok(None);
        }
        if k == 0 || p < r {
            return Ok(Some(Group::default()));
        }
        let window = self.window(k - 1..=k, p - r..=p);
        integral::differential_image(&window, r, p, k).map(Some)
    }

    fn degenerates_at(&self) -> Option<usize> {
        // d^r goes down r steps, so from r = n_steps it has nowhere to go
        Some(self.complex.n_steps().max(1))
    }
}

/// A bar in filtration steps: its degree, the step it is born at, and the step it dies at (`None` if essential).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBar {
    pub degree: usize,
    pub birth: usize,
    pub death: Option<usize>,
}

/// The step of `value` among `steps`, the distinct filtration values in increasing order.
///
/// # Panics
/// If `value` is not one of `steps`.
pub fn step_of<F: PartialOrd + std::fmt::Debug>(steps: &[F], value: &F) -> usize {
    let i = steps.partition_point(|s| s < value);
    assert!(i < steps.len() && &steps[i] == value, "{value:?} is not a filtration value");
    i
}

/// Pages `E^1, ..., E^{r_max}`: `pages[r - 1][p][k]` is the dimension of `E^r_p` in degree `k`, for steps
/// `p < n_steps` and degrees `k < n_degrees`, or `None` if the barcode doesn't determine it (when it is of the
/// filtration truncated at step `truncated_at`, see the [module docs](self)).
pub type Pages = Vec<Vec<Vec<Option<usize>>>>;

/// The pages `E^1, ..., E^{r_max}` of the spectral sequence of a filtration with these bars (any bars of length zero
/// are ignored), in steps `0..n_steps` and degrees `0..n_degrees`. If the bars are of the filtration truncated after
/// step `truncated_at` (bars still alive there given as essential), entries at step `p` on pages
/// `r > truncated_at + 1 - p` are `None`.
pub fn spectral_pages(
    bars: &[StepBar],
    n_steps: usize,
    n_degrees: usize,
    r_max: usize,
    truncated_at: Option<usize>,
) -> Pages {
    let mut pages: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; n_degrees]; n_steps]; r_max];
    for bar in bars {
        let lasts = bar.death.map(|d| d.saturating_sub(bar.birth));
        if lasts == Some(0) {
            continue;
        }
        // Pages 1..=min(length, r_max) (every page for an essential bar)
        let until = lasts.map_or(r_max, |l| l.min(r_max));
        for page in pages.iter_mut().take(until) {
            if bar.birth < n_steps && bar.degree < n_degrees {
                page[bar.birth][bar.degree] += 1;
            }
            if let Some(d) = bar.death {
                if d < n_steps && bar.degree + 1 < n_degrees {
                    page[d][bar.degree + 1] += 1;
                }
            }
        }
    }
    pages
        .into_iter()
        .enumerate()
        .map(|(r0, page)| {
            let r = r0 + 1;
            page.into_iter()
                .enumerate()
                .map(|(p, row)| {
                    let known = truncated_at.is_none_or(|t| p + r <= t + 1);
                    row.into_iter().map(|x| known.then_some(x)).collect()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn local_entries_match_whole_complex() {
        // RP^2 by the order of its vertices (lower star), with signs: every entry and differential computed from its
        // window agrees with the whole complex
        let tops: [[usize; 3]; 10] = [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 1, 5], [1, 2, 4], [2, 3, 5], [1, 3, 4], [2, 4, 5], [1, 3, 5],
        ];
        let mut simplices: Vec<Vec<usize>> = vec![];
        for top in &tops {
            for mask in 1u32..8 {
                simplices.push((0..3).filter(|&i| mask >> i & 1 == 1).map(|i| top[i]).collect());
            }
        }
        simplices.sort_by_key(|s| (s.len(), s.clone()));
        simplices.dedup();
        let index: HashMap<Vec<usize>, usize> = simplices.iter().cloned().enumerate().map(|(i, s)| (s, i)).collect();
        let complex = IntegralFiltered {
            degree: simplices.iter().map(|s| s.len() - 1).collect(),
            step: simplices.iter().map(|s| *s.iter().max().unwrap()).collect(),
            boundary: simplices
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
                .collect(),
        };
        let local = LocalSpectral::new(complex.clone(), 3, None);
        let mut torsion = 0;
        for r in 1..=6 {
            for p in 0..6 {
                for k in 0..3 {
                    let whole = integral::page_entry(&complex, r, p, k).unwrap();
                    torsion += usize::from(!whole.torsion.is_empty());
                    assert_eq!(local.entry(r, p, k).unwrap(), Some(whole), "E^{r}_{p} degree {k}");
                    if k >= 1 && p >= r {
                        let image = integral::differential_image(&complex, r, p, k).unwrap();
                        assert_eq!(local.differential(r, p, k).unwrap(), Some(image), "d^{r} at {p}, {k}");
                    }
                }
            }
        }
        assert!(torsion > 0);
    }

    #[test]
    fn barcode_backend_matches_pages() {
        let bars = [
            StepBar { degree: 0, birth: 0, death: None },
            StepBar { degree: 0, birth: 1, death: Some(3) },
            StepBar { degree: 1, birth: 2, death: Some(3) },
        ];
        let pages = spectral_pages(&bars, 4, 3, 3, None);
        let backend = BarcodeSpectral::new(&bars, 4, 3, None);
        for r in 1..=3 {
            for p in 0..4 {
                for k in 0..3 {
                    assert_eq!(backend.entry(r, p, k).unwrap().map(|g| g.rank), pages[r - 1][p][k]);
                }
            }
        }
        assert_eq!(backend.differential(2, 3, 1).unwrap().unwrap().rank, 1);
        assert_eq!(backend.differential(1, 3, 2).unwrap().unwrap().rank, 1);
        assert_eq!(backend.degenerates_at(), Some(3));
    }

    #[test]
    fn a_single_bar() {
        // A bar [1, 3) in degree 0: generators at (step 1, degree 0) and (step 3, degree 1) on E^1 and E^2, gone on E^3
        let bars = [StepBar { degree: 0, birth: 1, death: Some(3) }];
        let pages = spectral_pages(&bars, 4, 2, 3, None);
        for r in 0..2 {
            assert_eq!(pages[r][1][0], Some(1));
            assert_eq!(pages[r][3][1], Some(1));
        }
        assert!(pages[2].iter().flatten().all(|&x| x == Some(0)));
    }

    #[test]
    fn essential_bars_and_truncation() {
        let bars = [StepBar { degree: 1, birth: 2, death: None }];
        let pages = spectral_pages(&bars, 4, 2, 3, Some(3));
        assert_eq!(pages[0][2][1], Some(1));
        assert_eq!(pages[1][2][1], Some(1));
        // At step 2 page 3 needs the filtration up to step 4
        assert_eq!(pages[2][2][1], None);
        assert_eq!(step_of(&[0.5, 1.0, 2.0], &1.0), 1);
    }
}
