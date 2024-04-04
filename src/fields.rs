use std::fmt::Debug;
use std::num::NonZeroU8;
use std::ops::Add;

// TODO: Get additive inverse and multiply

/// Ensure that you are unable to construct an element that represents `0`.
/// Instead, `0` will be represented by the absence of a summand.
/// We avoid requiring an element for `0` to make [`Z2`] calculations more efficient.
pub trait NonZeroCoefficient:
    Sized + Copy + Add<Option<Self>, Output = Option<Self>> + Add<Self, Output = Option<Self>> + Eq
{
    /// Is this needed?
    fn one() -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Z2;

impl Add<Z2> for Z2 {
    type Output = Option<Z2>;

    fn add(self, _rhs: Z2) -> Self::Output {
        None
    }
}

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
}

/// Const generic struct for the finite field `Z_p`.
/// Should ensure that `p` is prime and that `p^2` does not overflow NonZeroU8
/// For `p=2` use [`Z2`]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ZP<const P: u8>(NonZeroU8);

impl<const P: u8> Debug for ZP<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<const P: u8> Add<ZP<P>> for ZP<P> {
    type Output = Option<ZP<P>>;

    fn add(self, rhs: ZP<P>) -> Self::Output {
        let res = self
            .0
            .checked_add(rhs.0.get())
            .expect("Should be able to add Zp entries within U8")
            .get()
            .rem_euclid(P);
        Some(ZP(NonZeroU8::new(res)?))
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

impl<const P: u8> NonZeroCoefficient for ZP<P> {
    fn one() -> Self {
        Self(unsafe { NonZeroU8::new_unchecked(1) })
    }
}

#[macro_export]
macro_rules! instantiate_zp {
    (  $p:expr, $struct_name:ident  ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $struct_name(ZP<$p>);

        impl Add<$struct_name> for $struct_name {
            type Output = Option<$struct_name>;
            fn add(self, rhs: $struct_name) -> Self::Output {
                Some($struct_name(self.0.add(rhs.0)?))
            }
        }

        impl_add_options!($struct_name);

        impl NonZeroCoefficient for $struct_name {
            fn one() -> Self {
                $struct_name(ZP::<$p>::one())
            }
        }
    };
}

instantiate_zp!(3, Z3);
instantiate_zp!(5, Z5);
instantiate_zp!(7, Z7);
instantiate_zp!(11, Z11);
instantiate_zp!(13, Z13);
