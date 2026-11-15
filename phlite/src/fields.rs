//! Traits for types that represent non-zero coefficients in a matrix.
//! Implementations of the rational numbers and finite fields up to Z13 are provided.
//!
//! There is no implementaion of the field of real numbers.
//! Due to floating point error, this would require additional support in the [`reduction`](crate::reduction) module to ensure that pivots are properly cleared.

use num::Integer;
use std::fmt::Debug;
use std::num::NonZeroU8;
use std::ops::{Add, Mul};

/// Represents a **non-zero** coefficient in a matrix.
///
/// Ensure that you are unable to construct an element that represents `0`.
/// Instead, `0` will be represented by the absence of a summand.
/// We avoid requiring an element for `0` to make [`Z2`] calculations more efficient.
///
/// In order to implement `Add<Opton<Self>>` you may wish to use [`impl_add_options`](crate::impl_add_options).
pub trait NonZeroCoefficient:
    Eq
    + Copy
    + Add<Self, Output = Option<Self>>
    + Add<Option<Self>, Output = Option<Self>>
    + Mul<Self, Output = Self>
{
    /// Return the multiplicative unit, i.e. `1`.
    fn one() -> Self;
    /// Return the additive inverse of `self`, i.e. `-self`.
    fn additive_inverse(self) -> Self;
}

/// Represents the ability to take a multiplicative inverse.
pub trait Invertible: NonZeroCoefficient {
    /// Return the multiplicative inverse of `self`, i.e. `1/self`.
    fn mult_inverse(self) -> Self;
}

/// The finite field with 2 elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Z2;

impl Add<Z2> for Z2 {
    type Output = Option<Z2>;

    fn add(self, _rhs: Z2) -> Self::Output {
        None
    }
}

impl Mul<Z2> for Z2 {
    type Output = Z2;

    // 1 * 1 = 1
    fn mul(self, _rhs: Z2) -> Self::Output {
        Z2
    }
}

impl Invertible for Z2 {
    fn mult_inverse(self) -> Self {
        Self
    }
}

/// Helper macro for creating structs that implement [`NonZeroCoefficient`].
///
/// Takes as input a single struct idenitifier `CF` and levarages a pre-existing implementation of `Add<CF>` in order to implement `Add<Option<CF>>`.
#[macro_export]
macro_rules! impl_add_options {
    ($cf:ident) => {
        impl Add<Option<$cf>> for $cf {
            type Output = Option<$cf>;

            fn add(self, rhs: Option<$cf>) -> Self::Output {
                match rhs {
                    None => Some(self),
                    Some(rhs) => self + rhs,
                }
            }
        }
    };
}

impl_add_options!(Z2);

impl NonZeroCoefficient for Z2 {
    fn one() -> Self {
        Self
    }

    fn additive_inverse(self) -> Self {
        Self
    }
}

/// The finite field `Z/P`, for a prime `P` (at most 251, the largest prime that fits in a `u8`).
///
/// Elements are the residues `1..P` (zero is the absence of an entry, as for every [`NonZeroCoefficient`]); build them
/// with [`ZP::new`] or from [`one`](NonZeroCoefficient::one), and read them with [`ZP::value`]. Arithmetic is exact and
/// never panics. That `P` is prime is checked at compile time, when the field is first used: `ZP::<4>::one()` does not
/// compile. For `P = 2` prefer [`Z2`], which stores nothing.
///
/// ```
/// use phlite::fields::{Invertible, Z7};
/// let three = Z7::new(3).unwrap();
/// assert_eq!((three * three.mult_inverse()).value(), 1);
/// assert_eq!(three + Z7::new(4).unwrap(), None); // 3 + 4 = 0 in Z/7
/// ```
///
/// ```compile_fail
/// use phlite::fields::{NonZeroCoefficient, ZP};
/// let _ = ZP::<4>::one(); // 4 isn't prime
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZP<const P: u8>(NonZeroU8);

impl<const P: u8> Debug for ZP<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Whether `p` is prime.
const fn is_prime(p: u8) -> bool {
    if p < 2 {
        return false;
    }
    let mut d: u16 = 2;
    while d * d <= p as u16 {
        if (p as u16).is_multiple_of(d) {
            return false;
        }
        d += 1;
    }
    true
}

/// `table[a]` is the inverse of `a` modulo the prime `p`, for `a` in `1..p` (and 0 elsewhere).
const fn inverses(p: u8) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut a: u16 = 1;
    while a < p as u16 {
        let mut b: u16 = 1;
        while (a * b) % p as u16 != 1 {
            b += 1;
        }
        table[a as usize] = b as u8;
        a += 1;
    }
    table
}

impl<const P: u8> ZP<P> {
    /// Evaluated (at compile time) wherever an element is made, so that `ZP<P>` can only be used with `P` prime.
    const PRIME: () = assert!(is_prime(P), "ZP<P> is a field only for P prime");
    const INVERSES: [u8; 256] = inverses(P);

    /// The residue of `value` modulo `P`, or `None` if it is zero.
    pub const fn new(value: u8) -> Option<Self> {
        #[allow(clippy::let_unit_value)]
        let () = Self::PRIME;
        match NonZeroU8::new(value % P) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// The representative of `self` in `1..P`.
    pub const fn value(self) -> u8 {
        self.0.get()
    }

    /// An element from a residue known to be in `1..P`.
    #[inline]
    fn from_residue(value: u16) -> Self {
        debug_assert!(value > 0 && value < P as u16);
        // SAFETY: a non-zero residue modulo a prime (every caller passes one, see above) is non-zero
        Self(unsafe { NonZeroU8::new_unchecked(value as u8) })
    }
}

impl<const P: u8> Add<ZP<P>> for ZP<P> {
    type Output = Option<ZP<P>>;

    #[inline]
    fn add(self, rhs: ZP<P>) -> Self::Output {
        // In u16, so no overflow for any P
        let sum = (self.0.get() as u16 + rhs.0.get() as u16) % P as u16;
        (sum != 0).then(|| Self::from_residue(sum))
    }
}

impl<const P: u8> Add<Option<ZP<P>>> for ZP<P> {
    type Output = Option<ZP<P>>;

    #[inline]
    fn add(self, rhs: Option<ZP<P>>) -> Self::Output {
        match rhs {
            None => Some(self),
            Some(rhs) => self + rhs,
        }
    }
}

impl<const P: u8> Mul<ZP<P>> for ZP<P> {
    type Output = ZP<P>;

    #[inline]
    fn mul(self, rhs: ZP<P>) -> Self::Output {
        // Non-zero, as P is prime; in u16, so no overflow for any P
        Self::from_residue((self.0.get() as u16 * rhs.0.get() as u16) % P as u16)
    }
}

impl<const P: u8> NonZeroCoefficient for ZP<P> {
    #[inline]
    fn one() -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::PRIME;
        Self::from_residue(1)
    }

    #[inline]
    fn additive_inverse(self) -> Self {
        Self::from_residue((P - self.0.get()) as u16)
    }
}

impl<const P: u8> Invertible for ZP<P> {
    #[inline]
    fn mult_inverse(self) -> Self {
        Self::from_residue(Self::INVERSES[self.0.get() as usize] as u16)
    }
}

/// The finite field with 3 elements.
pub type Z3 = ZP<3>;
/// The finite field with 5 elements.
pub type Z5 = ZP<5>;
/// The finite field with 7 elements.
pub type Z7 = ZP<7>;
/// The finite field with 11 elements.
pub type Z11 = ZP<11>;
/// The finite field with 13 elements.
pub type Z13 = ZP<13>;

/// The field of rational numbers, as a fraction in lowest terms with a positive denominator.
///
/// # Panics
/// Arithmetic panics if a numerator or denominator overflows (`isize` and `usize`), rather than give a wrong answer.
/// Reductions over the rationals can grow entries quickly: prefer a finite field unless you need torsion-free results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Q(isize, usize);

const Q_OVERFLOW: &str = "a rational coefficient overflowed (see the Panics section of `Q`)";

impl Q {
    fn reduce(self) -> Self {
        let gcd = self.0.unsigned_abs().gcd(&self.1);
        Q(self.0 / (gcd as isize), self.1 / gcd)
    }
}

impl Add for Q {
    type Output = Option<Q>;

    // Dividing the common denominator by each denominator is part of adding fractions
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        let lowest_common_denom = self.1.lcm(&rhs.1);
        let scale = |q: Q| {
            isize::try_from(lowest_common_denom / q.1)
                .ok()
                .and_then(|s| s.checked_mul(q.0))
                .expect(Q_OVERFLOW)
        };
        let numerator = scale(self).checked_add(scale(rhs)).expect(Q_OVERFLOW);
        if numerator == 0 {
            None
        } else {
            Some(Q(numerator, lowest_common_denom).reduce())
        }
    }
}

impl_add_options!(Q);

impl Mul for Q {
    type Output = Q;

    fn mul(self, rhs: Self) -> Self::Output {
        // Cancel across first, which keeps the products small
        let g1 = self.0.unsigned_abs().gcd(&rhs.1);
        let g2 = rhs.0.unsigned_abs().gcd(&self.1);
        let numerator = (self.0 / g1 as isize).checked_mul(rhs.0 / g2 as isize).expect(Q_OVERFLOW);
        let denominator = (self.1 / g2).checked_mul(rhs.1 / g1).expect(Q_OVERFLOW);
        Q(numerator, denominator).reduce()
    }
}

impl NonZeroCoefficient for Q {
    fn one() -> Self {
        Q(1, 1)
    }

    fn additive_inverse(self) -> Self {
        Q(self.0.checked_neg().expect(Q_OVERFLOW), self.1)
    }
}

impl Invertible for Q {
    fn mult_inverse(self) -> Self {
        let sign = self.0.signum();
        Q(sign * isize::try_from(self.1).expect(Q_OVERFLOW), self.0.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {

    use crate::fields::{Invertible, NonZeroCoefficient, Q, Z13, Z3, ZP};

    use super::Z2;

    #[test]
    fn test_add_mod_2() {
        assert_eq!(Z2 + Z2, None);
        assert_eq!(Z2 + None, Some(Z2));
    }

    #[test]
    fn test_prod_mod_2() {
        assert_eq!(Z2 * Z2, Z2);
    }

    #[test]
    fn test_add_mod_3() {
        assert_eq!(Z3::one() + Z3::one(), Z3::new(2));
        let two = Z3::one() + Z3::one();
        assert_eq!(Z3::one() + two, None);
    }

    #[test]
    fn test_prod_mod_3() {
        let two = (Z3::one() + Z3::one()).unwrap();
        let one = Z3::one();
        assert_eq!(two * one, two);
        assert_eq!(one * two, two);
        assert_eq!(two * two, one);
        assert_eq!(one * one, one);
    }

    /// Every field axiom we rely on, exhaustively, for a small and the largest prime.
    fn check_field<const P: u8>() {
        let elems: Vec<ZP<P>> = (1..P).map(|v| ZP::new(v).unwrap()).collect();
        assert_eq!(ZP::<P>::new(0), None);
        assert_eq!(ZP::<P>::new(P), None);
        assert_eq!(ZP::<P>::one().value(), 1);
        for &a in &elems {
            assert_eq!(a * a.mult_inverse(), ZP::one());
            assert_eq!(a + a.additive_inverse(), None);
            for &b in &elems {
                let (x, y) = (a.value() as u32, b.value() as u32);
                assert_eq!((a + b).map(|c| c.value() as u32), Some((x + y) % P as u32).filter(|&r| r != 0));
                assert_eq!((a * b).value() as u32, x * y % P as u32);
            }
        }
    }

    #[test]
    fn prime_fields() {
        check_field::<3>();
        check_field::<5>();
        check_field::<7>();
        check_field::<11>();
        check_field::<13>();
        check_field::<17>(); // overflowed u8 before
        check_field::<251>();
        assert_eq!(Z13::new(2).unwrap().mult_inverse().value(), 7);
    }

    #[test]
    fn rationals() {
        let half = Q::one().mult_inverse() * Q(1, 2);
        assert_eq!(half + half, Some(Q::one()));
        assert_eq!(Q(-3, 4).mult_inverse(), Q(-4, 3));
        assert_eq!(Q(2, 3) * Q(3, 4), Q(1, 2));
        assert_eq!(Q(1, 3) + Q(-1, 3), None);
    }

    #[test]
    #[should_panic(expected = "overflowed")]
    fn rational_overflow_panics() {
        let big = Q(isize::MAX, 1);
        let _ = big * Q(2, 1);
    }
}
