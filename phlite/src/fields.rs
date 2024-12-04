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

// TODO: Make ZP generic over underlying representation
//       Add option to macro

/// Const generic struct for the finite field `Z_p`.
/// Should ensure that `p` is prime and that `(p-1)^2` does not overflow `NonZeroU8`
/// For `p=2` use [`Z2`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ZP<const P: u8>(NonZeroU8);

impl<const P: u8> Debug for ZP<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// TODO: Replace checked_ operations with unchecked since we can guarantee in the cases we implement?

impl<const P: u8> Add<ZP<P>> for ZP<P> {
    type Output = Option<ZP<P>>;

    fn add(self, rhs: ZP<P>) -> Self::Output {
        let result = self
            .0
            .checked_add(rhs.0.get())
            .expect("Should be able to add Zp entries within U8")
            .get()
            .rem_euclid(P);
        Some(ZP(NonZeroU8::new(result)?))
    }
}

impl<const P: u8> Add<Option<ZP<P>>> for ZP<P> {
    type Output = Option<ZP<P>>;

    fn add(self, rhs: Option<ZP<P>>) -> Self::Output {
        match rhs {
            None => Some(self),
            Some(rhs) => self + rhs,
        }
    }
}

impl<const P: u8> Mul<ZP<P>> for ZP<P> {
    type Output = ZP<P>;

    fn mul(self, rhs: ZP<P>) -> Self::Output {
        let product = self
            .0
            .checked_mul(rhs.0)
            .expect("Should be able to multiply Zp entries within U8")
            .get()
            .rem_euclid(P);
        ZP(NonZeroU8::new(product)
            .expect("Product of two non-zero should be non-zero, is P prime?"))
    }
}

impl<const P: u8> NonZeroCoefficient for ZP<P> {
    fn one() -> Self {
        Self(unsafe { NonZeroU8::new_unchecked(1) })
    }

    fn additive_inverse(self) -> Self {
        Self(unsafe { NonZeroU8::new_unchecked(P - self.0.get()) })
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

impl Invertible for Z3 {
    fn mult_inverse(self) -> Self {
        let inner: u8 = self.0.into();
        match inner {
            1 => ZP(unsafe { NonZeroU8::new_unchecked(1) }),
            2 => ZP(unsafe { NonZeroU8::new_unchecked(2) }),
            _ => panic!("Not in Z3"),
        }
    }
}

impl Invertible for Z5 {
    fn mult_inverse(self) -> Self {
        let inner: u8 = self.0.into();
        match inner {
            1 => ZP(unsafe { NonZeroU8::new_unchecked(1) }),
            2 => ZP(unsafe { NonZeroU8::new_unchecked(3) }),
            3 => ZP(unsafe { NonZeroU8::new_unchecked(2) }),
            4 => ZP(unsafe { NonZeroU8::new_unchecked(4) }),
            _ => panic!("Not in Z5"),
        }
    }
}

impl Invertible for Z7 {
    fn mult_inverse(self) -> Self {
        let inner: u8 = self.0.into();
        match inner {
            1 => ZP(unsafe { NonZeroU8::new_unchecked(1) }),
            2 => ZP(unsafe { NonZeroU8::new_unchecked(4) }),
            3 => ZP(unsafe { NonZeroU8::new_unchecked(5) }),
            4 => ZP(unsafe { NonZeroU8::new_unchecked(2) }),
            5 => ZP(unsafe { NonZeroU8::new_unchecked(3) }),
            6 => ZP(unsafe { NonZeroU8::new_unchecked(6) }),
            _ => panic!("Not in Z7"),
        }
    }
}

impl Invertible for Z11 {
    fn mult_inverse(self) -> Self {
        let inner: u8 = self.0.into();
        match inner {
            1 => ZP(unsafe { NonZeroU8::new_unchecked(1) }),
            2 => ZP(unsafe { NonZeroU8::new_unchecked(6) }),
            3 => ZP(unsafe { NonZeroU8::new_unchecked(4) }),
            4 => ZP(unsafe { NonZeroU8::new_unchecked(3) }),
            5 => ZP(unsafe { NonZeroU8::new_unchecked(9) }),
            6 => ZP(unsafe { NonZeroU8::new_unchecked(2) }),
            7 => ZP(unsafe { NonZeroU8::new_unchecked(8) }),
            8 => ZP(unsafe { NonZeroU8::new_unchecked(7) }),
            9 => ZP(unsafe { NonZeroU8::new_unchecked(5) }),
            10 => ZP(unsafe { NonZeroU8::new_unchecked(10) }),
            _ => panic!("Not in Z11"),
        }
    }
}

impl Invertible for Z13 {
    fn mult_inverse(self) -> Self {
        let inner: u8 = self.0.into();
        match inner {
            1 => ZP(unsafe { NonZeroU8::new_unchecked(1) }),
            2 => ZP(unsafe { NonZeroU8::new_unchecked(7) }),
            3 => ZP(unsafe { NonZeroU8::new_unchecked(9) }),
            4 => ZP(unsafe { NonZeroU8::new_unchecked(10) }),
            5 => ZP(unsafe { NonZeroU8::new_unchecked(8) }),
            6 => ZP(unsafe { NonZeroU8::new_unchecked(11) }),
            7 => ZP(unsafe { NonZeroU8::new_unchecked(2) }),
            8 => ZP(unsafe { NonZeroU8::new_unchecked(5) }),
            9 => ZP(unsafe { NonZeroU8::new_unchecked(3) }),
            10 => ZP(unsafe { NonZeroU8::new_unchecked(4) }),
            11 => ZP(unsafe { NonZeroU8::new_unchecked(6) }),
            12 => ZP(unsafe { NonZeroU8::new_unchecked(12) }),
            _ => panic!("Not in Z13"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The field of rational numbers.
pub struct Q(isize, usize);

impl Q {
    fn reduce(self) -> Self {
        let gcd = self.0.unsigned_abs().gcd(&self.1);
        Q(self.0 / (gcd as isize), self.1 / gcd)
    }
}

impl Add for Q {
    type Output = Option<Q>;

    fn add(self, rhs: Self) -> Self::Output {
        let lowest_common_denom = self.1.lcm(&rhs.1);
        let numerator = (lowest_common_denom / self.1) as isize * self.0
            + (lowest_common_denom / rhs.1) as isize * rhs.0;
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
        Q(self.0 * rhs.0, self.1 * rhs.1).reduce()
    }
}

impl NonZeroCoefficient for Q {
    fn one() -> Self {
        Q(1, 1)
    }

    fn additive_inverse(self) -> Self {
        Q(-self.0, self.1)
    }
}

impl Invertible for Q {
    fn mult_inverse(self) -> Self {
        let sign = self.0.signum();
        Q(sign * self.1 as isize, self.0.unsigned_abs())
    }
}

#[cfg(test)]
mod tests {

    use std::num::NonZeroU8;

    use crate::fields::{NonZeroCoefficient, Z3, ZP};

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
        assert_eq!(
            Z3::one() + Z3::one(),
            Some(ZP(unsafe { NonZeroU8::new_unchecked(2) }))
        );
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
}
