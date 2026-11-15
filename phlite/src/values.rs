//! Constant filtration values, as [`NotNan<f64>`](ordered_float::NotNan) (which can't otherwise be built in a
//! `const`, or without an `unwrap`).

use ordered_float::NotNan;

/// `0.0`.
// SAFETY (all three): the values are not NaN
pub const ZERO: NotNan<f64> = unsafe { NotNan::new_unchecked(0.0) };
/// `f64::INFINITY`: a value no finite filtration value reaches (a cell that never enters).
pub const INFINITY: NotNan<f64> = unsafe { NotNan::new_unchecked(f64::INFINITY) };
/// `f64::MAX`, the largest finite value.
pub const MAX_FINITE: NotNan<f64> = unsafe { NotNan::new_unchecked(f64::MAX) };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        assert_eq!(ZERO, NotNan::new(0.0).unwrap());
        assert!(INFINITY.is_infinite() && *INFINITY > 0.0);
        assert_eq!(MAX_FINITE.into_inner(), f64::MAX);
    }
}
