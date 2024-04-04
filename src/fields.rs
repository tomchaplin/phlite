use std::num::NonZeroU8;

/// Ensure that you are unable to construct an element that represents `0`.
/// Instead, `0` will be represented by the absence of a summand.
/// We avoid requiring an element for `0` to make [`Z2`] calculations more efficient.
pub trait CoefficientField: Sized {
    /// Return None if sum is `0`
    fn add(&self, other: &Self) -> Option<Self>;
}

pub struct Z2;

impl Z2 {
    pub fn one() -> Self {
        Self
    }
}

impl CoefficientField for Z2 {
    fn add(&self, _other: &Self) -> Option<Self> {
        None
    }
}

/// Const generic struct for the finite field `Z_p`.
/// Should ensure that `p` is prime and that `2*p` does not overflow NonZeroU8
/// For `p=2` use [`Z2`]
pub struct ZP<const P: u8>(NonZeroU8);

impl<const P: u8> ZP<P> {
    pub fn one() -> Self {
        Self(unsafe { NonZeroU8::new_unchecked(1) })
    }
}

impl<const P: u8> CoefficientField for ZP<P> {
    fn add(&self, other: &Self) -> Option<Self> {
        let res = self
            .0
            .checked_add(other.0.get())
            .expect("Should be able to add Zp entries within U8")
            .get()
            .rem_euclid(P);
        Some(ZP(NonZeroU8::new(res)?))
    }
}

#[macro_export]
macro_rules! instantiate_zp {
    (  $p:expr, $struct_name:ident  ) => {
        pub struct $struct_name(ZP<$p>);
        impl $struct_name {
            pub fn one() -> Self {
                $struct_name(ZP::<$p>::one())
            }
        }

        impl CoefficientField for $struct_name {
            fn add(&self, other: &Self) -> Option<Self> {
                self.0.add(&other.0).map($struct_name)
            }
        }
    };
}

instantiate_zp!(3, Z3);
instantiate_zp!(5, Z5);
instantiate_zp!(7, Z7);
instantiate_zp!(11, Z11);
instantiate_zp!(13, Z13);
instantiate_zp!(17, Z17);
instantiate_zp!(19, Z19);
instantiate_zp!(23, Z23);
instantiate_zp!(29, Z29);
instantiate_zp!(31, Z31);
instantiate_zp!(37, Z37);
instantiate_zp!(41, Z41);
instantiate_zp!(43, Z43);
instantiate_zp!(47, Z47);
instantiate_zp!(53, Z53);
instantiate_zp!(59, Z59);
instantiate_zp!(61, Z61);
instantiate_zp!(67, Z67);
instantiate_zp!(71, Z71);
instantiate_zp!(73, Z73);
instantiate_zp!(79, Z79);
instantiate_zp!(83, Z83);
instantiate_zp!(89, Z89);
instantiate_zp!(97, Z97);
instantiate_zp!(101, Z101);
instantiate_zp!(103, Z103);
instantiate_zp!(107, Z107);
instantiate_zp!(109, Z109);
