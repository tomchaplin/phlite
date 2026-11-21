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
mod tests {
    use super::*;

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
